"""Article quality scoring and simple bucketed sampling."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

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


def normalized_text_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def normalized_title_key(title: str) -> str | None:
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    normalized = re.sub(r"^[\\W_]+|[\\W_]+$", "", normalized)
    if normalized in GENERIC_TITLES or len(normalized) < 12:
        return None
    return normalized


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


def sample_records(
    records: list[dict],
    *,
    ontology_path: Path,
    limit: int | None,
    seed: int,
    keyword: bool = False,
) -> list[dict]:
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
            continue
        bucket = bucket_record(record, terms)
        row = {
            **compact_sample_record(record),
            "source_bucket": bucket,
            "quality_score": score,
        }
        if keyword:
            keyword_score = keyword_quality_score(str(record.get("title") or ""), text, keywords)
            row["keyword_quality_score"] = keyword_score
            if keyword_score >= 4.0:
                row["source_bucket"] = "keyword_risk_factor"
        candidates.append(row)

    seen_texts: set[str] = set()
    seen_titles: set[str] = set()
    candidates.sort(
        key=lambda row: (
            -float(row.get("keyword_quality_score", 0.0)),
            -float(row.get("quality_score", 0.0)),
            str(row.get("id") or ""),
        )
    )
    for row in candidates:
        text_key = normalized_text_key(str(row.get("text") or ""))
        title_key = normalized_title_key(str(row.get("title") or ""))
        if text_key in seen_texts or (title_key is not None and title_key in seen_titles):
            continue
        seen_texts.add(text_key)
        if title_key is not None:
            seen_titles.add(title_key)
        buckets[row["source_bucket"]].append(row)

    for rows in buckets.values():
        rng.shuffle(rows)
    if keyword:
        buckets["keyword_risk_factor"].sort(key=lambda row: row["keyword_quality_score"], reverse=True)

    if limit is None:
        return [row for bucket in buckets.values() for row in bucket]

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
    return selected[:limit]


def append_sample_records_to_limit(output_path: Path, selected: list[dict], limit: int | None) -> int:
    existing = list(iter_jsonl(output_path))
    if limit is not None and len(existing) >= limit:
        print(f"Output already has {len(existing)} sampled rows, meeting limit {limit}: {output_path}")
        return 0

    target = limit if limit is not None else len(existing) + len(selected)
    existing_ids = {str(row.get("id")) for row in existing}
    appended = [row for row in selected if str(row.get("id")) not in existing_ids][: max(target - len(existing), 0)]
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite and not args.append_to_limit:
        print(f"Output already exists, skipping: {output_path}")
        return 0
    records = load_records(resolve_path(args.input))
    selected = sample_records(
        records,
        ontology_path=resolve_path(args.ontology),
        limit=args.limit,
        seed=args.seed,
        keyword=args.keyword,
    )
    if output_path.exists() and not args.overwrite and args.append_to_limit:
        append_sample_records_to_limit(output_path, selected, args.limit)
        return 0
    write_jsonl(output_path, selected, overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
