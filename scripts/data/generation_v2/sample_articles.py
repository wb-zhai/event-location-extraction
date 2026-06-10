"""Article quality scoring and simple bucketed sampling."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import random
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2.io_utils import iter_jsonl, load_json_tolerant, load_records, resolve_path, write_jsonl

FOOD_INSECURITY_KEYWORDS: tuple[tuple[str, float], ...] = (
    ("food insecurity", 4.0),
    ("acute food insecurity", 4.0),
    ("food crisis", 4.0),
    ("food scarcity", 4.0),
    ("food shortage", 4.0),
    ("food shortages", 4.0),
    ("hunger crisis", 4.0),
    ("famine", 4.0),
    ("starvation", 4.0),
    ("malnutrition", 3.5),
    ("malnourished", 3.5),
    ("ipc phase", 3.5),
    ("emergency phase", 3.0),
    ("crisis phase", 3.0),
    ("food aid", 3.0),
    ("food assistance", 3.0),
    ("food ration", 3.0),
    ("food rations", 3.0),
    ("crop failure", 2.5),
    ("crop failures", 2.5),
    ("poor harvest", 2.5),
    ("poor harvests", 2.5),
    ("low crop yield", 2.5),
    ("low crop yields", 2.5),
    ("reduced harvest", 2.5),
    ("drought", 2.0),
    ("water shortage", 2.0),
    ("water shortages", 2.0),
    ("livelihood crisis", 2.0),
    ("humanitarian crisis", 2.0),
    ("humanitarian access", 2.0),
    ("aid blockade", 2.0),
    ("market disruption", 1.5),
    ("food prices", 1.5),
    ("staple prices", 1.5),
    ("inflation", 1.5),
    ("conflict", 1.0),
    ("displacement", 1.0),
)

RISK_FACTOR_SEED_WORDS = {
    "aid",
    "crop",
    "drought",
    "food",
    "hunger",
    "humanitarian",
    "insecurity",
    "livelihood",
    "market",
    "nutrition",
    "scarcity",
    "shortage",
    "water",
}


GENERIC_TITLES = {"", "english", "untitled", "news", "article"}
TEXT_SHINGLE_SIZE = 5
TEXT_NEAR_DUPLICATE_THRESHOLD = 0.9


@dataclass
class SamplingSummary:
    input_count: int = 0
    quality_filtered_count: int = 0
    scored_count: int = 0
    duplicate_removed_count: int = 0
    unique_count: int = 0
    selected_count: int = 0
    limit: int | None = None
    order_by_score: bool = False
    bucket_counts: Counter[str] = field(default_factory=Counter)
    selected_bucket_counts: Counter[str] = field(default_factory=Counter)


def _compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalized_text_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def normalized_title_key(title: str) -> str | None:
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    normalized = re.sub(r"^[\\W_]+|[\\W_]+$", "", normalized)
    if normalized in GENERIC_TITLES or len(normalized) < 12:
        return None
    return normalized


def normalized_source_key(source_url: str) -> str | None:
    source_url = source_url.strip()
    if not source_url:
        return None
    parsed = urlsplit(source_url)
    if not parsed.scheme and not parsed.netloc:
        return source_url.lower()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def source_key_is_article_specific(source_key: str | None) -> bool:
    if not source_key:
        return False
    parsed = urlsplit(source_key)
    if not parsed.scheme and not parsed.netloc:
        return "/" in source_key.strip("/")
    return bool(parsed.path.strip("/") or parsed.query)


def normalized_publish_date_key(publish_date: str) -> str | None:
    normalized = _compact_whitespace(publish_date)
    if not normalized:
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", normalized)
    if match:
        return match.group(1)
    return normalized.lower()


def article_identity_keys(record: dict) -> dict[str, str | None]:
    title_key = normalized_title_key(str(record.get("title") or ""))
    text = str(record.get("text") or "")
    source_key = normalized_source_key(str(record.get("source_url") or ""))
    if not source_key_is_article_specific(source_key):
        source_key = None
    date_key = normalized_publish_date_key(str(record.get("publish_date") or ""))
    return {
        "text": normalized_text_key(text) if text else None,
        "title": title_key,
        "source": source_key,
        "title_date": f"{title_key}::{date_key}" if title_key and date_key else None,
        "source_date": f"{source_key}::{date_key}" if source_key and date_key else None,
    }


def text_shingles(text: str, size: int = TEXT_SHINGLE_SIZE) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def is_near_duplicate_text(
    shingles: frozenset[str],
    seen_shingles: list[frozenset[str]],
    threshold: float = TEXT_NEAR_DUPLICATE_THRESHOLD,
) -> bool:
    if not shingles:
        return False
    for previous in seen_shingles:
        if not previous:
            continue
        overlap = len(shingles & previous) / min(len(shingles), len(previous))
        if overlap >= threshold:
            return True
    return False


def article_identity_seen(
    keys: dict[str, str | None],
    shingles: frozenset[str],
    seen: dict[str, set[str]],
    seen_shingles: list[frozenset[str]],
) -> bool:
    return any(value is not None and value in seen[key] for key, value in keys.items()) or is_near_duplicate_text(
        shingles,
        seen_shingles,
    )


def remember_article_identity(keys: dict[str, str | None], seen: dict[str, set[str]]) -> None:
    for key, value in keys.items():
        if value is not None:
            seen[key].add(value)


def empty_article_identity_seen() -> dict[str, set[str]]:
    return {key: set() for key in ("text", "title", "source", "title_date", "source_date")}


def quality_score(text: str) -> float:
    if len(text) < 500:
        return -1.0
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = re.findall(r"[.!?](?:\s|$)", text)
    bad_chars = sum(1 for char in text if char == "\ufffd")
    uppercase_ratio = sum(1 for char in text if char.isupper()) / max(sum(1 for char in text if char.isalpha()), 1)
    return len(paragraphs) * 0.5 + len(sentences) * 0.1 - bad_chars * 2 - uppercase_ratio


def ontology_terms(ontology_path: Path) -> list[str]:
    raw = load_json_tolerant(ontology_path)
    events = raw.get("events", raw) if isinstance(raw, dict) else {}
    terms: set[str] = set()
    for label, description in events.items():
        terms.update(str(label).lower().split())
        terms.update(re.findall(r"[a-z]{5,}", str(description).lower()))
    return sorted(term for term in terms if len(term) > 4)


def keyword_terms(ontology_path: Path) -> list[tuple[str, float]]:
    raw = load_json_tolerant(ontology_path)
    events = raw.get("events", raw) if isinstance(raw, dict) else {}
    terms = {term: weight for term, weight in FOOD_INSECURITY_KEYWORDS}

    for label, description in events.items():
        label_text = str(label).lower()
        description_text = str(description).lower()
        if any(seed in f"{label_text} {description_text}" for seed in RISK_FACTOR_SEED_WORDS):
            terms.setdefault(label_text, 2.5)
            for phrase in re.findall(r"[a-z]+(?:\s+[a-z]+){1,3}", description_text):
                if any(seed in phrase for seed in RISK_FACTOR_SEED_WORDS):
                    terms.setdefault(phrase, 1.0)

    return sorted(terms.items(), key=lambda item: (-item[1], item[0]))


def keyword_quality_score(title: str, text: str, terms: list[tuple[str, float]]) -> float:
    haystack = f"{title}\n{text}".lower()
    score = 0.0
    for term, weight in terms:
        pattern = rf"(?<![a-z]){re.escape(term)}(?![a-z])"
        hits = len(re.findall(pattern, haystack))
        if hits:
            score += weight * min(hits, 3)
    return score


def bucket_record(record: dict, terms: list[str]) -> str:
    text = f"{record.get('title', '')}\n{record.get('text', '')}".lower()
    hits = sum(1 for term in terms if term in text)
    if hits >= 3:
        return "event_seeded_positive"
    if hits >= 1:
        return "hard_negative"
    return "random_negative"


def compact_sample_record(record: dict) -> dict:
    return {
        "id": str(record.get("id") or ""),
        "title": str(record.get("title") or ""),
        "text": str(record.get("text") or ""),
        "source_url": record.get("source_url") or "",
        "publish_date": record.get("publish_date") or "",
    }


def overall_score(quality_score_value: float, keyword_quality_score_value: float = 0.0) -> float:
    return quality_score_value + keyword_quality_score_value


def score_sort_key(row: dict) -> tuple[float, str]:
    return (
        -float(row.get("overall_score", 0.0)),
        str(row.get("id") or ""),
    )


def sample_records(
    records: list[dict],
    *,
    ontology_path: Path,
    limit: int | None,
    seed: int,
    keyword: bool = False,
    order_by_score: bool = False,
    summary: SamplingSummary | None = None,
) -> list[dict]:
    if summary is not None:
        summary.input_count = len(records)
        summary.limit = limit
        summary.order_by_score = order_by_score

    rng = random.Random(seed)
    terms = ontology_terms(ontology_path)
    keywords = keyword_terms(ontology_path) if keyword else []
    buckets: dict[str, list[dict]] = {
        "keyword_risk_factor": [],
        "event_seeded_positive": [],
        "hard_negative": [],
        "sibling_negative": [],
        "random_negative": [],
    }
    candidates: list[dict] = []
    for record in tqdm(records, desc="Scoring articles"):
        text = str(record.get("text") or "")
        score = quality_score(text)
        if score < 0:
            if summary is not None:
                summary.quality_filtered_count += 1
            continue
        bucket = bucket_record(record, terms)
        row = {
            **compact_sample_record(record),
            "source_bucket": bucket,
            "quality_score": score,
            "overall_score": overall_score(score),
        }
        if keyword:
            keyword_score = keyword_quality_score(str(record.get("title") or ""), text, keywords)
            row["keyword_quality_score"] = keyword_score
            row["overall_score"] = overall_score(score, keyword_score)
            if keyword_score >= 4.0:
                row["source_bucket"] = "keyword_risk_factor"
        candidates.append(row)

    if summary is not None:
        summary.scored_count = len(candidates)

    seen = empty_article_identity_seen()
    seen_shingles: list[frozenset[str]] = []
    candidates.sort(key=score_sort_key)
    for row in candidates:
        identity_keys = article_identity_keys(row)
        shingles = text_shingles(str(row.get("text") or ""))
        if article_identity_seen(identity_keys, shingles, seen, seen_shingles):
            if summary is not None:
                summary.duplicate_removed_count += 1
            continue
        remember_article_identity(identity_keys, seen)
        seen_shingles.append(shingles)
        buckets[row["source_bucket"]].append(row)

    if summary is not None:
        summary.bucket_counts = Counter(
            {bucket: len(rows) for bucket, rows in buckets.items() if rows}
        )
        summary.unique_count = sum(summary.bucket_counts.values())

    for rows in buckets.values():
        rng.shuffle(rows)
    if keyword:
        buckets["keyword_risk_factor"].sort(key=lambda row: row["keyword_quality_score"], reverse=True)

    if limit is None:
        selected = [row for bucket in buckets.values() for row in bucket]
        selected = sorted(selected, key=score_sort_key) if order_by_score else selected
        if summary is not None:
            summary.selected_count = len(selected)
            summary.selected_bucket_counts = Counter(str(row["source_bucket"]) for row in selected)
        return selected

    mix = (
        {
            "keyword_risk_factor": 0.45,
            "event_seeded_positive": 0.35,
            "hard_negative": 0.10,
            "sibling_negative": 0.05,
            "random_negative": 0.05,
        }
        if keyword
        else {
            "event_seeded_positive": 0.50,
            "hard_negative": 0.20,
            "sibling_negative": 0.15,
            "random_negative": 0.15,
        }
    )
    selected: list[dict] = []
    for bucket, fraction in mix.items():
        take = int(limit * fraction)
        selected.extend(buckets[bucket][:take])
    remainder = [row for rows in buckets.values() for row in rows if row not in selected]
    selected.extend(remainder[: max(limit - len(selected), 0)])
    selected = selected[:limit]
    selected = sorted(selected, key=score_sort_key) if order_by_score else selected
    if summary is not None:
        summary.selected_count = len(selected)
        summary.selected_bucket_counts = Counter(str(row["source_bucket"]) for row in selected)
    return selected


def print_sampling_summary(summary: SamplingSummary) -> None:
    print("Sampling summary:")
    print(f"  input_articles: {summary.input_count}")
    print(f"  quality_filtered_articles: {summary.quality_filtered_count}")
    print(f"  scored_articles: {summary.scored_count}")
    print(f"  duplicate_articles_removed: {summary.duplicate_removed_count}")
    print(f"  unique_articles_after_dedup: {summary.unique_count}")
    print(f"  output_articles: {summary.selected_count}")
    if summary.limit is not None:
        print(f"  limit: {summary.limit}")
    else:
        print("  limit: none")
    print(f"  ordered_by_score: {str(summary.order_by_score).lower()}")
    if summary.bucket_counts:
        print("  unique_articles_by_bucket:")
        for bucket, count in summary.bucket_counts.items():
            print(f"    {bucket}: {count}")
    if summary.selected_bucket_counts:
        print("  output_articles_by_bucket:")
        for bucket, count in summary.selected_bucket_counts.items():
            print(f"    {bucket}: {count}")


def append_sample_records_to_limit(output_path: Path, selected: list[dict], limit: int | None) -> int:
    existing = list(iter_jsonl(output_path))
    if limit is not None and len(existing) >= limit:
        print(f"Output already has {len(existing)} sampled rows, meeting limit {limit}: {output_path}")
        return 0

    target = limit if limit is not None else len(existing) + len(selected)
    existing_ids = {str(row.get("id")) for row in existing}
    seen = empty_article_identity_seen()
    seen_shingles: list[frozenset[str]] = []
    for row in existing:
        remember_article_identity(article_identity_keys(row), seen)
        seen_shingles.append(text_shingles(str(row.get("text") or "")))

    appended = []
    for row in selected:
        if len(appended) >= max(target - len(existing), 0):
            break
        if str(row.get("id")) in existing_ids:
            continue
        identity_keys = article_identity_keys(row)
        shingles = text_shingles(str(row.get("text") or ""))
        if article_identity_seen(identity_keys, shingles, seen, seen_shingles):
            continue
        appended.append(row)
        existing_ids.add(str(row.get("id")))
        remember_article_identity(identity_keys, seen)
        seen_shingles.append(shingles)
    if not appended:
        print(f"No new sampled rows to append: {output_path}")
        return 0

    write_jsonl(output_path, appended, overwrite=False)
    print(f"Appended {len(appended)} sampled rows to {output_path}")
    return len(appended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample clean articles for generation_v2.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("ontologies/zhai/ontology.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Boost likely food-insecurity risk-factor articles.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append-to-limit",
        action="store_true",
        help="Append new sampled rows when output exists and --limit is larger than the current row count.",
    )
    parser.add_argument(
        "--order-by-score",
        action="store_true",
        help="Order output rows by overall_score descending, then id after sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite and not args.append_to_limit:
        print(f"Output already exists, skipping: {output_path}")
        return 0
    records = load_records(resolve_path(args.input))
    summary = SamplingSummary()
    selected = sample_records(
        records,
        ontology_path=resolve_path(args.ontology),
        limit=args.limit,
        seed=args.seed,
        keyword=args.keyword,
        order_by_score=args.order_by_score,
        summary=summary,
    )
    print_sampling_summary(summary)
    if output_path.exists() and not args.overwrite and args.append_to_limit:
        append_sample_records_to_limit(output_path, selected, args.limit)
        return 0
    write_jsonl(output_path, selected, overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
