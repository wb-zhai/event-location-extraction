"""Check for duplicate (article_uri, relevance_version) keys across relevance CSVs.

Reads every file under the given path(s) — local folder or gs:// prefix, same
expansion/column-detection rules as db_ingestion.py — and reports how many rows
share a key with another row, both within a single shard and across shards.
This is the same key `db_ingestion.py` upserts on, so the numbers here explain
gaps between "rows read" and "rows upserted" in that script's output.

Only the uri/version columns are read; labels are ignored.

Usage:
    python scripts/relevance/check_shard_duplicates.py gs://my-bucket/relevance/output/run-001 \
        --version gemini-2.5-flash-v1

    # Quick sample: only check the first 5 expanded files
    python scripts/relevance/check_shard_duplicates.py gs://my-bucket/relevance/output/run-001 \
        --version gemini-2.5-flash-v1 --limit-shards 5

Requires the same deps as db_ingestion.py (google-cloud-storage for gs:// paths).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_ingestion import (  # noqa: E402
    IngestionError,
    columns_from_width,
    expand_inputs,
    open_input,
    sniff_header,
)


def read_keys(path_or_uri: str, cli_version: str | None):
    """Yield (article_uri, relevance_version) for every data row in one file."""
    with open_input(path_or_uri) as f:
        reader = csv.reader(f)
        try:
            first_row = next(reader)
        except StopIteration:
            return

        try:
            columns = sniff_header(first_row)
            if columns is None:
                columns = columns_from_width(first_row)
                rows = itertools.chain([first_row], reader)
            else:
                rows = reader
        except ValueError as exc:
            raise IngestionError(f"{path_or_uri}: {exc}") from exc

        if columns["version"] is None and not cli_version:
            raise IngestionError(
                f"{path_or_uri} has no version column and no --version was "
                f"provided. Pass --version <value> to match what db_ingestion.py would use."
            )

        uri_idx, version_idx = columns["uri"], columns["version"]
        for lineno, row in enumerate(rows, start=1):
            if not row:
                continue
            try:
                uri = row[uri_idx].strip()
                row_version = row[version_idx].strip() if version_idx is not None else ""
            except IndexError:
                raise IngestionError(f"{path_or_uri} line {lineno}: malformed row {row!r}")
            yield uri, row_version or cli_version


@dataclass
class ShardResult:
    path: str
    rows: int = 0
    counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    error: str | None = None


def process_shard(path_or_uri: str, version: str | None) -> ShardResult:
    result = ShardResult(path=path_or_uri)
    try:
        for key in read_keys(path_or_uri, version):
            result.counts[key] += 1
            result.rows += 1
    except IngestionError as exc:
        result.error = str(exc)
    except Exception as exc:
        result.error = f"unexpected error: {exc}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="CSV file(s)/folder(s), local or gs://bucket/path")
    parser.add_argument("--version", default=None, help="Fallback relevance_version for files without one.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent file reads (default 8).")
    parser.add_argument("--top", type=int, default=20, help="Show the N most-repeated keys (default 20).")
    parser.add_argument("--limit-shards", type=int, default=None, help="Only check the first N expanded files.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> None:
    args = parse_args()

    inputs = expand_inputs(args.inputs)
    if args.limit_shards:
        inputs = inputs[: args.limit_shards]
    print(f"Checking {len(inputs)} file(s)...")

    global_counts: Counter[tuple[str, str]] = Counter()
    key_shard_presence: Counter[tuple[str, str]] = Counter()
    total_rows = 0
    within_shard_dupe_rows = 0
    errors: list[ShardResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_shard, p, args.version): p for p in inputs}
        with tqdm(total=len(inputs), unit="file", desc="Checking shards") as pbar:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result.error:
                    errors.append(result)
                    tqdm.write(f"  [{i}/{len(inputs)}] {result.path}: ERROR - {result.error}")
                    pbar.update(1)
                    continue

                shard_dupe_rows = sum(c - 1 for c in result.counts.values() if c > 1)
                within_shard_dupe_rows += shard_dupe_rows
                total_rows += result.rows
                global_counts.update(result.counts)
                for key in result.counts:
                    key_shard_presence[key] += 1

                tqdm.write(
                    f"  [{i}/{len(inputs)}] {result.path}: {result.rows:,} rows, "
                    f"{len(result.counts):,} unique keys, {shard_dupe_rows:,} within-shard dupe rows"
                )
                pbar.set_postfix(rows=f"{total_rows:,}", dupes=f"{total_rows - len(global_counts):,}")
                pbar.update(1)

    total_unique = len(global_counts)
    total_dupe_rows = total_rows - total_unique
    keys_in_multiple_shards = sum(1 for v in key_shard_presence.values() if v > 1)

    print("\n=== Duplicate check complete ===")
    print(f"Files checked                 : {len(inputs):,}")
    print(f"Total rows                    : {total_rows:,}")
    print(f"Distinct (uri, version) keys  : {total_unique:,}")
    print(f"Duplicate rows (total - distinct): {total_dupe_rows:,}")
    print(f"  of which within a single shard: {within_shard_dupe_rows:,}")
    print(f"  of which only across shards    : {total_dupe_rows - within_shard_dupe_rows:,}")
    print(f"Keys appearing in >1 shard     : {keys_in_multiple_shards:,}")

    if args.top and global_counts:
        print(f"\nTop {args.top} most-repeated keys:")
        for (uri, version), count in global_counts.most_common(args.top):
            print(f"  {count:>8,}  uri={uri!r}  version={version!r}")

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:", file=sys.stderr)
        for r in errors:
            print(f"  {r.path}: {r.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
