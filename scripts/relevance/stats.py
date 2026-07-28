"""Print summary stats for relevance-labeled JSONL/CSV data.

By default only prints the core relevant/not-relevant breakdown: label
(positive/negative) and relevance.decision distributions. Pass --complete to
also print the top-N most frequent risk_factors and countries, and how
positive/negative labels break down against the relevance decision (and vice
versa).

--input accepts a single file, a local folder (searched recursively for
"*.jsonl"/"*.csv"), a single GCS object (gs://bucket/path/to/file.jsonl), or a
GCS "folder" (gs://bucket/prefix/, listed for "*.jsonl"/"*.csv" objects).
".jsonl" files are the usual relevance-labeled records. ".csv" files are
two-column (uri, decision) files where decision is "relevant"/"not relevant";
these only feed the relevance decision counts, since they carry no
label/country/risk_factors fields. Each file is streamed one line at a time
(never loaded whole into memory), and --workers files are read concurrently
-- useful since GCS reads are network-bound -- so memory stays flat
regardless of how many rows/files are processed (built for runs in the
hundreds of millions of rows).

    # Local file, default (relevant/not relevant only) stats
    python scripts/relevance/stats.py --input dataset/relevance/matrix.jsonl

    # Local folder, full breakdown
    python scripts/relevance/stats.py --input dataset/relevance/shards/ --complete

    # GCS folder, mix of .jsonl and .csv shards, 16 concurrent file readers
    python scripts/relevance/stats.py --input gs://my-bucket/relevance/shards/ --workers 16
"""

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator
from urllib.parse import urlparse

from tqdm import tqdm

CSV_DECISION_VALUES = {"relevant", "not relevant", "irrelevant"}


def get_decision(rec: dict) -> str:
    rel = rec.get("relevance") or {}
    return rel.get("decision") or "none"


# ── Input resolution (local files/folders and gs:// files/folders) ────────────

def is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URI: {uri}. Expected gs://bucket/path")
    return parsed.netloc, parsed.path.lstrip("/")


def _require_gcs_storage():
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
        sys.exit(1)
    return storage


def iter_input_files(input_path: str) -> list[str]:
    """Resolve --input to a sorted list of .jsonl/.csv file paths/URIs."""
    if is_gcs_uri(input_path):
        storage = _require_gcs_storage()
        bucket_name, prefix = parse_gcs_uri(input_path)
        if prefix.endswith(".jsonl") or prefix.endswith(".csv"):
            return [input_path]
        prefix = prefix.rstrip("/") + "/" if prefix else prefix
        client = storage.Client()
        files = sorted(
            f"gs://{bucket_name}/{blob.name}"
            for blob in client.list_blobs(bucket_name, prefix=prefix)
            if blob.name.endswith(".jsonl") or blob.name.endswith(".csv")
        )
        if not files:
            raise SystemExit(f"No .jsonl/.csv files found under {input_path}")
        return files

    path = Path(input_path)
    if path.is_dir():
        files = sorted(str(p) for p in (*path.rglob("*.jsonl"), *path.rglob("*.csv")))
        if not files:
            raise SystemExit(f"No .jsonl/.csv files found in {input_path}")
        return files
    return [str(path)]


def open_input(path: str) -> IO[str]:
    if is_gcs_uri(path):
        storage = _require_gcs_storage()
        bucket_name, blob_name = parse_gcs_uri(path)
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        return blob.open("r", encoding="utf-8")
    return Path(path).open(encoding="utf-8")


def display_name(path: str) -> str:
    return path if is_gcs_uri(path) else Path(path).name


def _iter_csv_records(f: IO[str]) -> Iterator[dict]:
    """Two-column (uri, decision) CSV, decision in {"relevant", "not relevant"}."""
    for row_idx, row in enumerate(csv.reader(f)):
        if not row or len(row) < 2:
            continue
        decision = row[1].strip().lower()
        if row_idx == 0 and decision not in CSV_DECISION_VALUES:
            continue  # header row
        yield {"relevance": {"decision": decision}}


def _iter_jsonl_records(f: IO[str]) -> Iterator[dict]:
    for line in f:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


@dataclass
class FileStats:
    total: int = 0
    labels: Counter = field(default_factory=Counter)
    decisions: Counter = field(default_factory=Counter)
    risk_factors: Counter = field(default_factory=Counter)
    countries: Counter = field(default_factory=Counter)
    label_by_decision: Counter = field(default_factory=Counter)
    decision_by_country: Counter = field(default_factory=Counter)


def process_file(path: str, complete: bool, pbar: tqdm) -> FileStats:
    """Stream one file and tally it in isolation (thread-local, no shared state)."""
    stats = FileStats()
    with open_input(path) as f:
        records = _iter_csv_records(f) if path.endswith(".csv") else _iter_jsonl_records(f)
        for rec in records:
            stats.total += 1
            label = rec.get("label") or "none"
            decision = get_decision(rec)
            stats.labels[label] += 1
            stats.decisions[decision] += 1

            if complete:
                country = rec.get("country_name") or "none"
                stats.countries[country] += 1
                stats.label_by_decision[(label, decision)] += 1
                stats.decision_by_country[(country, decision)] += 1
                for rf in rec.get("risk_factors") or []:
                    stats.risk_factors[rf] += 1
            pbar.update(1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats for relevance-labeled JSONL data.")
    parser.add_argument("--input", required=True,
        help="Path to a JSONL file/folder, local or gs://bucket/prefix.")
    parser.add_argument("--complete", action="store_true",
        help="Also print risk_factors/countries breakdowns and label-vs-decision cross tabs.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top risk_factors/countries to show")
    parser.add_argument("--workers", type=int, default=8,
        help="Number of files to read concurrently (default 8; helps most with GCS input).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    files = iter_input_files(args.input)

    labels: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    risk_factors: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    label_by_decision: Counter[tuple[str, str]] = Counter()
    decision_by_country: Counter[tuple[str, str]] = Counter()
    total = 0
    files_done = 0

    with tqdm(desc="rows", unit="rows", unit_scale=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, path, args.complete, pbar): path for path in files}
            for future in as_completed(futures):
                stats = future.result()
                files_done += 1
                pbar.set_postfix_str(f"files {files_done}/{len(files)}: {display_name(futures[future])}")

                total += stats.total
                labels.update(stats.labels)
                decisions.update(stats.decisions)
                if args.complete:
                    risk_factors.update(stats.risk_factors)
                    countries.update(stats.countries)
                    label_by_decision.update(stats.label_by_decision)
                    decision_by_country.update(stats.decision_by_country)

    print(f"Total records: {total} ({args.input}, {len(files)} file(s))")
    print()

    print("Label counts (positive/negative):")
    for label, count in labels.most_common():
        pct = count / total * 100 if total else 0.0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

    print("Relevance decision counts:")
    for decision, count in decisions.most_common():
        pct = count / total * 100 if total else 0.0
        print(f"  {decision}: {count} ({pct:.1f}%)")

    if not args.complete:
        return
    print()

    print(f"Top {args.top_n} risk_factors:")
    for rf, count in risk_factors.most_common(args.top_n):
        print(f"  {rf}: {count}")
    print()

    print(f"Top {args.top_n} countries:")
    for country, count in countries.most_common(args.top_n):
        pct = count / total * 100 if total else 0.0
        print(f"  {country}: {count} ({pct:.1f}%)")
    print()

    print(f"Relevance decision by country (top {args.top_n}):")
    for country, country_total in countries.most_common(args.top_n):
        print(f"  {country} ({country_total}):")
        for decision in sorted(decisions):
            count = decision_by_country[(country, decision)]
            if count == 0:
                continue
            pct = count / country_total * 100 if country_total else 0.0
            print(f"    {decision}: {count} ({pct:.1f}%)")
    print()

    print("Label vs. relevance decision:")
    for label in sorted(labels):
        label_total = labels[label]
        print(f"  {label} ({label_total}):")
        for decision in sorted(decisions):
            count = label_by_decision[(label, decision)]
            if count == 0:
                continue
            pct = count / label_total * 100 if label_total else 0.0
            print(f"    {decision}: {count} ({pct:.1f}%)")
    print()

    print("Relevance decision vs. label:")
    for decision in sorted(decisions):
        decision_total = decisions[decision]
        print(f"  {decision} ({decision_total}):")
        for label in sorted(labels):
            count = label_by_decision[(label, decision)]
            if count == 0:
                continue
            pct = count / decision_total * 100 if decision_total else 0.0
            print(f"    {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
