"""Stratified, quality-weighted subsample of the matrix_5M article corpus.

Streams dataset/db/matrix_5M.jsonl (multi-GB, ~5M records) in two passes so
the full file never has to fit in memory:

  Pass 1 - count records per (adm0_code, label) stratum to compute
  per-stratum sample quotas proportional to each stratum's share of the
  corpus (largest-remainder allocation, sums to exactly --sample-size).

  Pass 2 - stream again (in parallel across --workers processes); for each
  record that clears the text-quality heuristic and isn't a near-duplicate
  of an already-selected article, run weighted reservoir sampling
  (Efraimidis-Spirakis) into that record's stratum bucket, using the
  quality score as the sampling weight so higher-quality articles are more
  likely to survive. Reservoir/dedup state is inherently sequential and
  lives in the main process; workers only do the parallelizable per-record
  scoring (quality heuristic, shingles, identity keys).

Output preserves the original record schema (id, label, adm0_code,
risk_factors, source{...}) with a "quality_score" field added.

Usage:
    python scripts/data/download/sample.py \
        dataset/db/matrix_5M.jsonl dataset/db/matrix_5M.sample_20k.jsonl \
        --sample-size 20000
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import multiprocessing
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v3.sample_articles import (
    article_identity_keys,
    article_identity_seen,
    empty_article_identity_seen,
    remember_article_identity,
)
from scripts.data.generation_v3.io_utils import resolve_path

StratumKey = tuple[str, str]

# Quality/dedup heuristics only look at the first ANALYSIS_CHARS characters of
# each article. Scanning full article bodies (avg ~18KB in this corpus) makes
# a single pass over 5M records take hours; a bounded prefix is representative
# for these signals (paragraph/sentence counts, all-caps ratio, lead-shingle
# overlap) and matches the preview size already used by the LLM relevance
# gate (relevance_filter.py's default max_chars=2000).
ANALYSIS_CHARS = 2000
SHINGLE_SIZE = 5
SHINGLE_SAMPLE_SIZE = 200
# Cap per-shingle posting list length (mirrors generation_v3/sample_articles.py)
# to bound near-duplicate lookup cost for common boilerplate 5-grams.
SHINGLE_POSTING_CAP = 8

# Cap on records per multiprocessing.Pool.imap task, independent of corpus size.
# Raw input lines carry the full article body (avg ~18KB), so a count-based
# chunksize scaled to a multi-million-record corpus would ship hundreds of MB
# per task; since imap preserves order, a few in-flight/buffered chunks can
# multiply that into tens of GB. 500 records/task keeps per-task IPC payload
# in the single-digit-MB range while amortizing per-task scheduling overhead.
CHUNK_RECORD_CAP = 500

_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fast_quality_score(text: str) -> float:
    """Cheap text-quality heuristic over a bounded prefix (see ANALYSIS_CHARS)."""
    if len(text) < 500:
        return -1.0
    sample = text[:ANALYSIS_CHARS]
    paragraphs = [p for p in re.split(r"\n\s*\n", sample) if p.strip()]
    sentences = _SENTENCE_RE.findall(sample)
    # Character classification pushed into C-level map()/compress() calls instead of a
    # per-character Python loop -- this function runs on ~every record in the corpus
    # (nearly all strata get a nonzero quota), so the loop was the dominant CPU cost
    # of pass 2. Equivalence verified: upper is only counted among alpha chars, same
    # as the original `elif ch.isalpha(): ... if ch.isupper()` gating.
    bad_chars = sample.count("�")
    alpha_mask = list(map(str.isalpha, sample))
    alpha = sum(alpha_mask)
    upper = sum(map(str.isupper, itertools.compress(sample, alpha_mask)))
    uppercase_ratio = upper / max(alpha, 1)
    return len(paragraphs) * 0.5 + len(sentences) * 0.1 - bad_chars * 2 - uppercase_ratio


def fast_shingles(text: str) -> frozenset[str]:
    """Stride-sampled 5-gram shingles over a bounded prefix, for near-dup detection."""
    tokens = _TOKEN_RE.findall(text[:ANALYSIS_CHARS].lower())
    n = len(tokens)
    if n < SHINGLE_SIZE:
        return frozenset(tokens)
    total = n - SHINGLE_SIZE + 1
    stride = max(1, total // SHINGLE_SAMPLE_SIZE)
    return frozenset(" ".join(tokens[i : i + SHINGLE_SIZE]) for i in range(0, total, stride))


def stratum_key(record: dict) -> StratumKey:
    return (str(record.get("adm0_code") or ""), str(record.get("label") or ""))


def _count_line(raw: bytes) -> tuple[StratumKey | None, int]:
    n = len(raw)
    line = raw.strip()
    if not line:
        return None, n
    record = orjson.loads(line)
    return stratum_key(record), n


def count_strata(input_path: Path, workers: int) -> Counter[StratumKey]:
    counts: Counter[StratumKey] = Counter()
    file_size = input_path.stat().st_size
    # Parallelized like pass 2: fully JSON-parsing every record (including its large
    # text body) just to read adm0_code/label is CPU-bound, and single-threaded this
    # pass over a multi-GB, 5M-record corpus dominated runtime.
    with input_path.open("rb") as fh, multiprocessing.Pool(workers) as pool, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Pass 1/2: counting strata"
    ) as bar:
        for key, n in pool.imap(_count_line, fh, chunksize=CHUNK_RECORD_CAP):
            bar.update(n)
            if key is not None:
                counts[key] += 1
    return counts


def _allocate_proportional(counts: Counter[StratumKey], n: int) -> dict[StratumKey, int]:
    """Largest-remainder allocation of `n` across strata, proportional to `counts`."""
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {key: 0 for key in counts}
    raw = {key: n * count / total for key, count in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remainder = n - sum(quotas.values())
    if remainder > 0:
        order = sorted(counts, key=lambda key: raw[key] - quotas[key], reverse=True)
        for key in order[:remainder]:
            quotas[key] += 1
    return quotas


def allocate_quotas(
    counts: Counter[StratumKey],
    sample_size: int,
    *,
    pos_ratio: float | None = None,
) -> dict[StratumKey, int]:
    """Allocate `sample_size` across (adm0_code, label) strata.

    Without --pos-ratio, positive/negative mix follows whatever ratio already
    exists in the corpus (country proportions preserved within each label).
    With --pos-ratio, the positive/negative split is forced to that ratio
    first, then country quotas are allocated proportionally within each half
    independently -- so a country's positive and negative shares no longer
    have to match the label's corpus-wide country distribution.
    """
    if pos_ratio is None:
        return _allocate_proportional(counts, sample_size)

    pos_counts = Counter({key: n for key, n in counts.items() if key[1] == "positive"})
    neg_counts = Counter({key: n for key, n in counts.items() if key[1] != "positive"})
    n_pos = round(sample_size * pos_ratio)
    n_neg = sample_size - n_pos
    quotas = _allocate_proportional(pos_counts, n_pos)
    quotas.update(_allocate_proportional(neg_counts, n_neg))
    return quotas


# ── Multiprocessing worker (module-level for pickling) ────────────────────────
# Reservoir/dedup state is stateful and sequential, so it stays in the main
# process; workers only do the parallelizable per-record scoring.

_wstate: dict = {}


def _worker_init(min_quality_score: float, quotas: dict[StratumKey, int]) -> None:
    _wstate["min_quality_score"] = min_quality_score
    _wstate["quotas"] = quotas


def _score_line(raw: bytes) -> dict | None:
    line = raw.strip()
    if not line:
        return None
    record = orjson.loads(line)
    key = stratum_key(record)
    if _wstate["quotas"].get(key, 0) <= 0:
        return {"status": "no_quota"}

    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    text = str(source.get("text") or "")
    title = str(source.get("title") or "")

    qs = fast_quality_score(text)
    if qs < _wstate["min_quality_score"]:
        return {"status": "low_quality"}

    identity_keys = article_identity_keys(
        {
            "title": title,
            "text": text[:ANALYSIS_CHARS],
            "source_url": source.get("source_url") or "",
            "publish_date": source.get("published_at") or "",
        }
    )
    shingles = fast_shingles(text)
    return {
        "record": record,
        "status": "ok",
        "qs": qs,
        "identity_keys": identity_keys,
        "shingles": shingles,
    }


def _remember_identity(
    identity_keys: dict[str, str | None],
    shingles: frozenset[str],
    seen: dict[str, set[str]],
    seen_shingles: list[frozenset[str]],
    shingle_index: dict[str, list[int]],
) -> None:
    remember_article_identity(identity_keys, seen)
    idx = len(seen_shingles)
    seen_shingles.append(shingles)
    for s in shingles:
        posting = shingle_index.setdefault(s, [])
        if len(posting) < SHINGLE_POSTING_CAP:
            posting.append(idx)


def _update_risk_factor_coverage(covered: Counter[str], risk_factors: list, delta: int) -> None:
    for rf in risk_factors:
        covered[rf] += delta
        if covered[rf] <= 0:
            del covered[rf]


def sample_reservoirs(
    input_path: Path,
    quotas: dict[StratumKey, int],
    *,
    seed: int,
    min_quality_score: float,
    workers: int,
    expected_total: int,
    risk_factor_diversity_weight: float = 0.0,
) -> tuple[dict[StratumKey, list], Counter[str]]:
    rng = random.Random(seed)
    reservoirs: dict[StratumKey, list] = {key: [] for key, quota in quotas.items() if quota > 0}
    # Risk factors currently held in each stratum's reservoir, kept in sync as
    # entries are pushed/evicted -- used to boost the sampling weight of
    # articles that add risk-factor tags not yet covered in that stratum, so
    # the positive quota doesn't just fill up with the most common tags.
    risk_factor_coverage: dict[StratumKey, Counter[str]] = {key: Counter() for key in reservoirs}
    seen = empty_article_identity_seen()
    seen_shingles: list[frozenset[str]] = []
    shingle_index: dict[str, list[int]] = {}
    seq = itertools.count()
    stats: Counter[str] = Counter()

    chunksize = max(1, min(CHUNK_RECORD_CAP, expected_total // (workers * 8))) if expected_total else 1
    with input_path.open("rb") as fh, multiprocessing.Pool(
        workers, initializer=_worker_init, initargs=(min_quality_score, quotas)
    ) as pool:
        results = pool.imap(_score_line, fh, chunksize=chunksize)
        for result in tqdm(
            results, total=expected_total or None, desc="Pass 2/2: sampling", unit="rec"
        ):
            if result is None:
                continue
            stats["input"] += 1
            status = result["status"]
            if status == "no_quota":
                continue
            if status == "low_quality":
                stats["quality_filtered"] += 1
                continue

            record = result["record"]
            key = stratum_key(record)
            identity_keys = result["identity_keys"]
            shingles = result["shingles"]
            if article_identity_seen(identity_keys, shingles, seen, seen_shingles, shingle_index):
                stats["duplicate"] += 1
                continue

            heap = reservoirs[key]
            quota = quotas[key]
            qs = result["qs"]
            risk_factors = record.get("risk_factors") or []

            weight = max(qs, 1e-6)
            if risk_factor_diversity_weight and risk_factors:
                covered = risk_factor_coverage[key]
                new_factors = sum(1 for rf in risk_factors if covered.get(rf, 0) == 0)
                novelty_fraction = new_factors / len(risk_factors)
                weight *= 1 + risk_factor_diversity_weight * novelty_fraction

            # Efraimidis-Spirakis weighted reservoir sampling: keys with larger
            # weight are (probabilistically) closer to 1, so keeping the k
            # largest keys samples proportional to weight, without replacement.
            # (The risk-factor boost above makes weight depend on reservoir
            # state, so this is a greedy coverage heuristic, not exact
            # weighted-without-replacement sampling.)
            es_key = rng.random() ** (1.0 / weight)
            entry = (es_key, next(seq), qs, record)

            if len(heap) < quota:
                heapq.heappush(heap, entry)
                _remember_identity(identity_keys, shingles, seen, seen_shingles, shingle_index)
                _update_risk_factor_coverage(risk_factor_coverage[key], risk_factors, +1)
                stats["selected"] += 1
            elif es_key > heap[0][0]:
                evicted = heapq.heapreplace(heap, entry)
                _remember_identity(identity_keys, shingles, seen, seen_shingles, shingle_index)
                evicted_record = evicted[3]
                _update_risk_factor_coverage(
                    risk_factor_coverage[key], evicted_record.get("risk_factors") or [], -1
                )
                _update_risk_factor_coverage(risk_factor_coverage[key], risk_factors, +1)

    return reservoirs, stats


def flatten_output(reservoirs: dict[StratumKey, list]) -> list[dict]:
    output: list[dict] = []
    for heap in reservoirs.values():
        for _es_key, _seq, qs, record in heap:
            row = dict(record)
            row["quality_score"] = qs
            output.append(row)
    output.sort(
        key=lambda row: (
            str(row.get("adm0_code")),
            str(row.get("label")),
            -row["quality_score"],
            str(row.get("id")),
        )
    )
    return output


def print_summary(
    counts: Counter[StratumKey],
    quotas: dict[StratumKey, int],
    reservoirs: dict[StratumKey, list],
    stats: Counter[str],
    sample_size: int,
    output_rows: list[dict],
) -> None:
    filled = sum(len(heap) for heap in reservoirs.values())
    distinct_risk_factors = {rf for row in output_rows for rf in row.get("risk_factors") or []}
    print("Sampling summary:")
    print(f"  input_records: {stats['input']:,}")
    print(f"  quality_filtered: {stats['quality_filtered']:,}")
    print(f"  duplicate_filtered: {stats['duplicate']:,}")
    print(f"  strata: {len(counts):,}")
    print(f"  requested_sample_size: {sample_size:,}")
    print(f"  output_records: {filled:,}")
    print(f"  distinct_risk_factors_covered: {len(distinct_risk_factors):,}")
    underfilled = [
        (key, quotas[key], len(reservoirs.get(key, [])))
        for key in quotas
        if quotas[key] > 0 and len(reservoirs.get(key, [])) < quotas[key]
    ]
    if underfilled:
        underfilled.sort(key=lambda item: item[1] - item[2], reverse=True)
        print(f"  strata_below_quota: {len(underfilled)} (showing up to 10)")
        for (adm0_code, label), quota, actual in underfilled[:10]:
            print(f"    {adm0_code}/{label}: quota={quota} actual={actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified, quality-weighted subsample of matrix_5M.jsonl.",
    )
    parser.add_argument("input", type=Path, nargs="?", default=Path("dataset/db/matrix_5M.jsonl"))
    parser.add_argument("output", type=Path)
    parser.add_argument("-n", "--sample-size", type=int, required=True, help="Target output record count.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--pos-ratio",
        type=float,
        default=None,
        help=(
            "Force this fraction of output records to label=positive (0-1 exclusive). "
            "Default: keep whatever positive/negative ratio already exists in the corpus."
        ),
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Reject articles scoring below this (fast_quality_score returns -1.0 for articles under 500 chars).",
    )
    parser.add_argument(
        "--risk-factor-diversity-weight",
        type=float,
        default=1.0,
        help=(
            "Boost the sampling weight of positive articles that add risk-factor tags not yet "
            "covered in their (country, label) stratum's reservoir, so quotas don't just fill up "
            "with the most common tags. 0 disables the boost (pure quality weighting). "
            "1.0 means an article whose tags are all novel gets 2x its quality weight."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes for parallel scoring in pass 2 (default: all CPU cores).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pos_ratio is not None and not 0.0 < args.pos_ratio < 1.0:
        print("--pos-ratio must be strictly between 0 and 1")
        return 1
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Output already exists, use --overwrite to replace: {output_path}")
        return 1

    workers = args.workers or os.cpu_count() or 1
    counts = count_strata(input_path, workers)
    quotas = allocate_quotas(counts, args.sample_size, pos_ratio=args.pos_ratio)
    reservoirs, stats = sample_reservoirs(
        input_path,
        quotas,
        seed=args.seed,
        min_quality_score=args.min_quality_score,
        workers=workers,
        expected_total=sum(counts.values()),
        risk_factor_diversity_weight=args.risk_factor_diversity_weight,
    )
    output_rows = flatten_output(reservoirs)
    print_summary(counts, quotas, reservoirs, stats, args.sample_size, output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for row in output_rows:
            fh.write(orjson.dumps(row))
            fh.write(b"\n")
    print(f"Wrote {len(output_rows):,} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
