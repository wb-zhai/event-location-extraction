"""Stratified, quality-weighted subsample of the matrix_5M article corpus.

Streams dataset/db/matrix_5M.jsonl (multi-GB, ~5M records) in two passes so
the full file never has to fit in memory:

  Pass 1 - count records per stratum to compute per-stratum sample quotas
  proportional to each stratum's share of the corpus (largest-remainder
  allocation, sums to exactly --sample-size). The stratum key is always
  (adm0_code, label), optionally extended with a risk-factor cluster
  (--stratify-by-cluster, mapping risk_factors -> cluster via
  ontologies/zhai/science_clusters.json) and/or a publication decade
  (--stratify-by-decade).

  Pass 2 - stream again (in parallel across --workers processes); for each
  record that clears the text-quality heuristic and isn't a near-duplicate
  of an already-selected article, run weighted reservoir sampling
  (Efraimidis-Spirakis) into that record's stratum bucket, using the
  quality score as the sampling weight so higher-quality articles are more
  likely to survive. Reservoir/dedup state is inherently sequential and
  lives in the main process; workers only do the parallelizable per-record
  scoring (quality heuristic, shingles, identity keys).

--stratify-by-decade forces an equal sample share per decade present in the
corpus (rather than proportional to how decade-skewed the corpus is), so
older, sparser decades aren't crowded out by recent years.

Output preserves the original record schema (id, label, adm0_code,
risk_factors, source{...}) with a "quality_score" field added.

Usage:
    python scripts/data/download/sample.py \
        dataset/db/matrix_5M.jsonl dataset/db/matrix_5M.sample_20k.jsonl \
        --sample-size 20000

    python scripts/data/download/sample.py \
        dataset/db/matrix_5M.jsonl dataset/db/matrix_5M.sample_20k.jsonl \
        --sample-size 20000 --stratify-by-cluster --stratify-by-decade
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
from collections.abc import Iterator
from pathlib import Path

import orjson
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation.sample_articles import (
    article_identity_keys,
    article_identity_seen,
    empty_article_identity_seen,
    remember_article_identity,
)
from scripts.data.generation.io_utils import resolve_path

StratumKey = tuple[str, str, str, str]  # (adm0_code, label, cluster, decade)
DECADE_AXIS = 3

DEFAULT_SCIENCE_CLUSTERS_PATH = Path("ontologies/zhai/science_clusters.json")

# Quality/dedup heuristics only look at the first ANALYSIS_CHARS characters of
# each article. Scanning full article bodies (avg ~18KB in this corpus) makes
# a single pass over 5M records take hours; a bounded prefix is representative
# for these signals (paragraph/sentence counts, all-caps ratio, lead-shingle
# overlap) and matches the preview size already used by the LLM relevance
# gate (relevance_filter.py's default max_chars=2000).
ANALYSIS_CHARS = 2000
SHINGLE_SIZE = 5
SHINGLE_SAMPLE_SIZE = 200
# Cap per-shingle posting list length (mirrors generation/sample_articles.py)
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
    return (
        len(paragraphs) * 0.5 + len(sentences) * 0.1 - bad_chars * 2 - uppercase_ratio
    )


def fast_shingles(text: str) -> frozenset[str]:
    """Stride-sampled 5-gram shingles over a bounded prefix, for near-dup detection."""
    tokens = _TOKEN_RE.findall(text[:ANALYSIS_CHARS].lower())
    n = len(tokens)
    if n < SHINGLE_SIZE:
        return frozenset(tokens)
    total = n - SHINGLE_SIZE + 1
    stride = max(1, total // SHINGLE_SAMPLE_SIZE)
    return frozenset(
        " ".join(tokens[i : i + SHINGLE_SIZE]) for i in range(0, total, stride)
    )


def load_cluster_map(path: Path) -> dict[str, str]:
    """Map risk_factor name -> cluster name, from a science_clusters.json export."""
    entries = orjson.loads(path.read_bytes())
    return {entry["name"]: entry["cluster"] for entry in entries}


def record_cluster(record: dict, cluster_map: dict[str, str]) -> str:
    """Plurality-vote cluster among a record's risk_factors, ties broken by first occurrence.

    Records can carry several risk_factors spanning different clusters; a single
    scalar is needed for the stratum key, so we pick the cluster that covers the
    most of the record's tags rather than e.g. always taking the first tag.
    """
    clusters = [
        cluster_map[rf] for rf in record.get("risk_factors") or [] if rf in cluster_map
    ]
    if not clusters:
        return ""
    counts = Counter(clusters)
    best = max(counts.values())
    return next(c for c in clusters if counts[c] == best)


def record_decade(record: dict) -> str:
    source = record.get("source")
    source = source if isinstance(source, dict) else {}
    published_at = str(source.get("published_at") or "")
    if len(published_at) < 4 or not published_at[:4].isdigit():
        return "unknown"
    return str((int(published_at[:4]) // 10) * 10)


def stratum_key(
    record: dict, cluster_map: dict[str, str] | None, use_decade: bool
) -> StratumKey:
    cluster = record_cluster(record, cluster_map) if cluster_map is not None else ""
    decade = record_decade(record) if use_decade else ""
    return (
        str(record.get("adm0_code") or ""),
        str(record.get("label") or ""),
        cluster,
        decade,
    )


_cwstate: dict = {}


def _count_worker_init(cluster_map: dict[str, str] | None, use_decade: bool) -> None:
    _cwstate["cluster_map"] = cluster_map
    _cwstate["use_decade"] = use_decade


def _count_line(raw: bytes) -> tuple[StratumKey | None, int]:
    n = len(raw)
    line = raw.strip()
    if not line:
        return None, n
    record = orjson.loads(line)
    return stratum_key(record, _cwstate["cluster_map"], _cwstate["use_decade"]), n


def count_strata(
    input_path: Path,
    workers: int,
    *,
    cluster_map: dict[str, str] | None,
    use_decade: bool,
) -> Counter[StratumKey]:
    counts: Counter[StratumKey] = Counter()
    file_size = input_path.stat().st_size
    # Parallelized like pass 2: fully JSON-parsing every record (including its large
    # text body) just to read adm0_code/label is CPU-bound, and single-threaded this
    # pass over a multi-GB, 5M-record corpus dominated runtime.
    with input_path.open("rb") as fh, multiprocessing.Pool(
        workers, initializer=_count_worker_init, initargs=(cluster_map, use_decade)
    ) as pool, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Pass 1/2: counting strata"
    ) as bar:
        for key, n in pool.imap(_count_line, fh, chunksize=CHUNK_RECORD_CAP):
            bar.update(n)
            if key is not None:
                counts[key] += 1
    return counts


def _allocate_proportional(
    counts: Counter[StratumKey], n: int
) -> dict[StratumKey, int]:
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


def _allocate_balanced_by_axis(
    counts: Counter[StratumKey], n: int, axis: int
) -> dict[StratumKey, int]:
    """Split `n` equally across distinct values of key[axis] seen in `counts`, then
    allocate each share proportionally across the strata within that value.

    Used for --stratify-by-decade: an equal quota per decade (rather than
    proportional to how decade-skewed the corpus is) keeps sparse older
    decades from being crowded out by recent years.
    """
    values = sorted({key[axis] for key in counts})
    if not values or n <= 0:
        return {key: 0 for key in counts}
    base, extra = divmod(n, len(values))
    quotas: dict[StratumKey, int] = {}
    for i, value in enumerate(values):
        group_counts = Counter(
            {key: c for key, c in counts.items() if key[axis] == value}
        )
        group_n = base + (1 if i < extra else 0)
        quotas.update(_allocate_proportional(group_counts, group_n))
    return quotas


def allocate_quotas(
    counts: Counter[StratumKey],
    sample_size: int,
    *,
    pos_ratio: float | None = None,
    balance_decades: bool = False,
) -> dict[StratumKey, int]:
    """Allocate `sample_size` across strata: (adm0_code, label), optionally
    extended with a risk-factor cluster and/or a publication decade.

    Without --pos-ratio, positive/negative mix follows whatever ratio already
    exists in the corpus (country proportions preserved within each label).
    With --pos-ratio, the positive/negative split is forced to that ratio
    first, then country quotas are allocated proportionally within each half
    independently -- so a country's positive and negative shares no longer
    have to match the label's corpus-wide country distribution.

    Without --stratify-by-decade, decade mix (if the key carries one) follows
    the corpus's natural distribution, same as country/cluster. With it, each
    decade present gets an equal share of its group's quota before country/
    cluster proportions are applied within that decade.
    """

    def _allocate(group_counts: Counter[StratumKey], n: int) -> dict[StratumKey, int]:
        if balance_decades:
            return _allocate_balanced_by_axis(group_counts, n, axis=DECADE_AXIS)
        return _allocate_proportional(group_counts, n)

    if pos_ratio is None:
        return _allocate(counts, sample_size)

    pos_counts = Counter({key: n for key, n in counts.items() if key[1] == "positive"})
    neg_counts = Counter({key: n for key, n in counts.items() if key[1] != "positive"})
    n_pos = round(sample_size * pos_ratio)
    n_neg = sample_size - n_pos
    quotas = _allocate(pos_counts, n_pos)
    quotas.update(_allocate(neg_counts, n_neg))
    return quotas


# ── Multiprocessing worker (module-level for pickling) ────────────────────────
# Reservoir/dedup state is stateful and sequential, so it stays in the main
# process; workers only do the parallelizable per-record scoring.

_wstate: dict = {}


def _worker_init(
    min_quality_score: float,
    quotas: dict[StratumKey, int],
    cluster_map: dict[str, str] | None,
    use_decade: bool,
) -> None:
    _wstate["min_quality_score"] = min_quality_score
    _wstate["quotas"] = quotas
    _wstate["cluster_map"] = cluster_map
    _wstate["use_decade"] = use_decade


def _iter_offset_lines(fh) -> "Iterator[tuple[int, bytes]]":
    """Yield (byte_offset, raw_line) pairs, tracking position ourselves so
    workers can report back a seek point instead of the full parsed record.
    """
    offset = 0
    for raw in fh:
        yield offset, raw
        offset += len(raw)


def _score_line(item: "tuple[int, bytes]") -> dict | None:
    offset, raw = item
    line = raw.strip()
    if not line:
        return None
    record = orjson.loads(line)
    key = stratum_key(record, _wstate["cluster_map"], _wstate["use_decade"])
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
    # Deliberately excludes the parsed `record` (its "text" field averages
    # several KB): pickling that back to the main process for every candidate
    # that clears quota+quality -- most of which never end up in the output
    # sample -- was the dominant cost in the main process's single-threaded
    # loop. `offset`/`length` let the tiny final set of actually-selected
    # records be re-read directly from disk in flatten_output instead.
    return {
        "status": "ok",
        "key": key,
        "qs": qs,
        "identity_keys": identity_keys,
        "shingles": shingles,
        "risk_factors": record.get("risk_factors") or [],
        "offset": offset,
        "length": len(raw),
    }


def _remember_identity(
    seq_id: int,
    identity_keys: dict[str, str | None],
    shingles: frozenset[str],
    seen: dict[str, set[str]],
    seen_shingles: dict[int, frozenset[str]],
    shingle_index: dict[str, list[int]],
) -> None:
    remember_article_identity(identity_keys, seen)
    seen_shingles[seq_id] = shingles
    for s in shingles:
        posting = shingle_index.setdefault(s, [])
        if len(posting) < SHINGLE_POSTING_CAP:
            posting.append(seq_id)


def _forget_identity(
    seq_id: int,
    identity_keys: dict[str, str | None],
    shingles: frozenset[str],
    seen: dict[str, set[str]],
    seen_shingles: dict[int, frozenset[str]],
    shingle_index: dict[str, list[int]],
) -> None:
    """Undo _remember_identity for a reservoir entry that just got evicted.

    Without this, `seen`/`seen_shingles`/`shingle_index` would grow for every
    record ever pushed into a reservoir across the whole corpus pass instead
    of staying bounded by what's currently held (matching the module
    docstring's "already-selected article" semantics) -- on a multi-GB, 5M
    record corpus that unbounded growth is what drove RAM past 17GB and
    slowed the sequential main-process loop to a crawl.
    """
    for key, value in identity_keys.items():
        if value is not None:
            seen[key].discard(value)
    seen_shingles.pop(seq_id, None)
    for s in shingles:
        posting = shingle_index.get(s)
        if posting is None:
            continue
        try:
            posting.remove(seq_id)
        except ValueError:
            pass
        if not posting:
            del shingle_index[s]


def _update_risk_factor_coverage(
    covered: Counter[str], risk_factors: list, delta: int
) -> None:
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
    cluster_map: dict[str, str] | None = None,
    use_decade: bool = False,
    risk_factor_diversity_weight: float = 0.0,
) -> tuple[dict[StratumKey, list], Counter[str]]:
    rng = random.Random(seed)
    reservoirs: dict[StratumKey, list] = {
        key: [] for key, quota in quotas.items() if quota > 0
    }
    # Risk factors currently held in each stratum's reservoir, kept in sync as
    # entries are pushed/evicted -- used to boost the sampling weight of
    # articles that add risk-factor tags not yet covered in that stratum, so
    # the positive quota doesn't just fill up with the most common tags.
    risk_factor_coverage: dict[StratumKey, Counter[str]] = {
        key: Counter() for key in reservoirs
    }
    seen = empty_article_identity_seen()
    seen_shingles: dict[int, frozenset[str]] = {}
    shingle_index: dict[str, list[int]] = {}
    seq = itertools.count()
    stats: Counter[str] = Counter()

    chunksize = (
        max(1, min(CHUNK_RECORD_CAP, expected_total // (workers * 8)))
        if expected_total
        else 1
    )
    with input_path.open("rb") as fh, multiprocessing.Pool(
        workers,
        initializer=_worker_init,
        initargs=(min_quality_score, quotas, cluster_map, use_decade),
    ) as pool:
        results = pool.imap(_score_line, _iter_offset_lines(fh), chunksize=chunksize)
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

            key = result["key"]
            identity_keys = result["identity_keys"]
            shingles = result["shingles"]
            heap = reservoirs[key]
            quota = quotas[key]
            qs = result["qs"]
            risk_factors = result["risk_factors"]

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

            # is_near_duplicate_text is the most expensive step per candidate
            # (profiling on the real 5M-record corpus showed the main process
            # pinned inside it): it scores shingle overlap against every
            # currently-live reservoir entry sharing one of ~200 shingles. A
            # record that can't beat the reservoir's current worst member is
            # getting discarded regardless of whether it's a duplicate, so
            # skip the check for it -- doesn't change which records end up
            # selected, only how many candidates pay for the duplicate check
            # (in practice most of the corpus, once reservoirs fill up).
            if len(heap) >= quota and es_key <= heap[0][0]:
                continue

            if article_identity_seen(
                identity_keys, shingles, seen, seen_shingles, shingle_index
            ):
                stats["duplicate"] += 1
                continue

            seq_id = next(seq)
            entry = (
                es_key,
                seq_id,
                qs,
                result["offset"],
                result["length"],
                identity_keys,
                shingles,
                risk_factors,
            )

            if len(heap) < quota:
                heapq.heappush(heap, entry)
                _remember_identity(
                    seq_id, identity_keys, shingles, seen, seen_shingles, shingle_index
                )
                _update_risk_factor_coverage(
                    risk_factor_coverage[key], risk_factors, +1
                )
                stats["selected"] += 1
            else:
                evicted = heapq.heapreplace(heap, entry)
                _remember_identity(
                    seq_id, identity_keys, shingles, seen, seen_shingles, shingle_index
                )
                _forget_identity(
                    evicted[1],
                    evicted[5],
                    evicted[6],
                    seen,
                    seen_shingles,
                    shingle_index,
                )
                _update_risk_factor_coverage(risk_factor_coverage[key], evicted[7], -1)
                _update_risk_factor_coverage(
                    risk_factor_coverage[key], risk_factors, +1
                )

    return reservoirs, stats


def flatten_output(input_path: Path, reservoirs: dict[StratumKey, list]) -> list[dict]:
    # Only the tiny final set of selected records (bounded by --sample-size)
    # gets fully re-parsed here, by seeking back into the source file -- see
    # the comment in _score_line for why full records aren't carried through
    # pass 2 itself.
    entries = [
        (offset, length, qs)
        for heap in reservoirs.values()
        for _es_key, _seq, qs, offset, length, _identity_keys, _shingles, _risk_factors in heap
    ]
    entries.sort(
        key=lambda entry: entry[0]
    )  # ascending offset for sequential disk access

    output: list[dict] = []
    with input_path.open("rb") as fh:
        for offset, length, qs in entries:
            fh.seek(offset)
            record = orjson.loads(fh.read(length).strip())
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
    distinct_risk_factors = {
        rf for row in output_rows for rf in row.get("risk_factors") or []
    }
    distinct_clusters = {key[2] for key in quotas if quotas[key] > 0 and key[2]}
    distinct_decades = {key[3] for key in quotas if quotas[key] > 0 and key[3]}
    print("Sampling summary:")
    print(f"  input_records: {stats['input']:,}")
    print(f"  quality_filtered: {stats['quality_filtered']:,}")
    print(f"  duplicate_filtered: {stats['duplicate']:,}")
    print(f"  strata: {len(counts):,}")
    print(f"  requested_sample_size: {sample_size:,}")
    print(f"  output_records: {filled:,}")
    print(f"  distinct_risk_factors_covered: {len(distinct_risk_factors):,}")
    if distinct_clusters:
        print(f"  distinct_clusters_covered: {len(distinct_clusters):,}")
    if distinct_decades:
        print(f"  distinct_decades_covered: {sorted(distinct_decades)}")
    underfilled = [
        (key, quotas[key], len(reservoirs.get(key, [])))
        for key in quotas
        if quotas[key] > 0 and len(reservoirs.get(key, [])) < quotas[key]
    ]
    if underfilled:
        underfilled.sort(key=lambda item: item[1] - item[2], reverse=True)
        print(f"  strata_below_quota: {len(underfilled)} (showing up to 10)")
        for (adm0_code, label, cluster, decade), quota, actual in underfilled[:10]:
            stratum_desc = "/".join(
                part for part in (adm0_code, label, cluster, decade) if part
            )
            print(f"    {stratum_desc}: quota={quota} actual={actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified, quality-weighted subsample of matrix_5M.jsonl.",
    )
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("dataset/db/matrix_5M.jsonl")
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        required=True,
        help="Target output record count.",
    )
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
    parser.add_argument(
        "--stratify-by-cluster",
        action="store_true",
        help=(
            "Add a risk-factor cluster dimension to the stratum key (via --science-clusters), "
            "on top of (adm0_code, label). Records are assigned the cluster that covers the "
            "plurality of their risk_factors; records with no mapped risk_factors (e.g. all "
            "negatives) fall into a single unclustered stratum."
        ),
    )
    parser.add_argument(
        "--science-clusters",
        type=Path,
        default=DEFAULT_SCIENCE_CLUSTERS_PATH,
        help="Path to the science_clusters.json risk_factor->cluster mapping (used with --stratify-by-cluster).",
    )
    parser.add_argument(
        "--stratify-by-decade",
        action="store_true",
        help=(
            "Add a publication-decade dimension to the stratum key, and give each decade "
            "present in the corpus an equal share of the sample (rather than proportional to "
            "how decade-skewed the corpus is), so sparse older decades aren't crowded out."
        ),
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

    cluster_map = None
    if args.stratify_by_cluster:
        cluster_map = load_cluster_map(resolve_path(args.science_clusters))

    workers = args.workers or os.cpu_count() or 1
    counts = count_strata(
        input_path, workers, cluster_map=cluster_map, use_decade=args.stratify_by_decade
    )
    quotas = allocate_quotas(
        counts,
        args.sample_size,
        pos_ratio=args.pos_ratio,
        balance_decades=args.stratify_by_decade,
    )
    reservoirs, stats = sample_reservoirs(
        input_path,
        quotas,
        seed=args.seed,
        min_quality_score=args.min_quality_score,
        workers=workers,
        expected_total=sum(counts.values()),
        cluster_map=cluster_map,
        use_decade=args.stratify_by_decade,
        risk_factor_diversity_weight=args.risk_factor_diversity_weight,
    )
    output_rows = flatten_output(input_path, reservoirs)
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
