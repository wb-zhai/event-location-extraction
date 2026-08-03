"""Ingest article relevance results from CSV files into the DB.

Target table:
    article_event_extraction.article_relevance (
        article_uri TEXT NOT NULL,
        relevance_version TEXT NOT NULL,
        is_relevant BOOLEAN NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (relevance_version, article_uri)
    )

CSV format:
    Each file has 2 or 3 columns and may or may not have a header row.
      - 2 columns, no header : article_uri, is_relevant
      - 3 columns, no header : article_uri, is_relevant, relevance_version
      - with header          : column names are matched (case-insensitively)
                                against known aliases for uri / label / version;
                                a version column is optional even with headers.
    If a file has no version column, pass --version on the command line.

Usage:
    # Local CSV(s), version taken from a column in the file
    python scripts/relevance/db_ingestion.py data/relevance_results.csv

    # CSV without a version column: provide it manually
    python scripts/relevance/db_ingestion.py data/relevance_results.csv --version gemini-2.5-flash-v1

    # Multiple files, some local some on GCS
    python scripts/relevance/db_ingestion.py data/a.csv gs://my-bucket/b.csv --version v3

    # Validate the files without touching the DB
    python scripts/relevance/db_ingestion.py data/a.csv --version v3 --dry-run

    # Ingest several large files concurrently (safe only if files don't share keys)
    python scripts/relevance/db_ingestion.py data/*.csv --version v3 --workers 4

Requires psycopg2:
    pip install psycopg2-binary
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, TextIO
from urllib.parse import urlparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    _env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    _env = {}


# ── Connection ────────────────────────────────────────────────────────────────

def _e(key: str, fallback: str | None = None) -> str | None:
    return _env.get(key) or os.environ.get(key) or fallback


def get_connection_params() -> dict:
    return {
        "host":     _e("SQL_HOST",     _e("PGHOST", "localhost")),
        "port":     _e("SQL_PORT",     _e("PGPORT", "5432")),
        "dbname":   _e("SQL_DATABASE", _e("PGDATABASE")),
        "user":     _e("SQL_USERNAME", _e("PGUSER")),
        "password": _e("SQL_PASSWORD", _e("PGPASSWORD")),
    }


# ── Input (local or GCS) ─────────────────────────────────────────────────────

def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid GCS URI: {uri}. Expected gs://bucket/path/to/file.csv")
    return parsed.netloc, parsed.path.lstrip("/")


def expand_inputs(paths: list[str]) -> list[str]:
    """Expand any directory (local) or prefix (gs://) input into the .csv
    files it contains. Plain file paths are passed through unchanged."""
    expanded: list[str] = []
    for path_or_uri in paths:
        if path_or_uri.startswith("gs://"):
            expanded.extend(_expand_gcs_input(path_or_uri))
        else:
            expanded.extend(_expand_local_input(path_or_uri))
    return expanded


def _expand_local_input(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.is_dir():
        return [path_str]
    files = sorted(str(f) for f in path.glob("*.csv"))
    if not files:
        print(f"Error: no .csv files found in directory {path_str}", file=sys.stderr)
        sys.exit(1)
    return files


def _expand_gcs_input(uri: str) -> list[str]:
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
        sys.exit(1)
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blob_name = blob_name.rstrip("/")
    if bucket.blob(blob_name).exists(client):
        return [uri]

    prefix = blob_name + "/"
    csv_names = sorted(
        blob.name for blob in client.list_blobs(bucket, prefix=prefix)
        if blob.name.endswith(".csv")
    )
    if not csv_names:
        print(
            f"Error: {uri} is not an object, and no .csv files were found "
            f"under prefix gs://{bucket_name}/{prefix}",
            file=sys.stderr,
        )
        sys.exit(1)
    return [f"gs://{bucket_name}/{name}" for name in csv_names]


@contextmanager
def open_input(path_or_uri: str) -> Generator[TextIO, None, None]:
    if path_or_uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
            sys.exit(1)
        bucket_name, blob_name = parse_gcs_uri(path_or_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        with blob.open("r", encoding="utf-8", newline="") as f:
            yield f
        return
    with open(path_or_uri, "r", encoding="utf-8", newline="") as f:
        yield f


class IngestionError(Exception):
    """A validation problem in an input file (bad label, missing version, ...)."""


# ── Header / column detection ────────────────────────────────────────────────

URI_ALIASES = {"uri", "article_uri", "url", "article_url"}
LABEL_ALIASES = {"label", "relevant", "is_relevant", "relevance", "relevance_label"}
VERSION_ALIASES = {"version", "relevance_version"}

TRUE_VALUES = {"true", "t", "1", "yes", "y", "relevant"}
FALSE_VALUES = {"false", "f", "0", "no", "n", "not relevant", "not_relevant", "irrelevant"}


def _normalize(cell: str) -> str:
    return cell.strip().lower().replace(" ", "_").replace("-", "_")


def sniff_header(row: list[str]) -> dict[str, int] | None:
    """Return {"uri": i, "label": i, "version": i|None} if `row` looks like a
    header row (i.e. it names the uri and label columns), else None."""
    normalized = [_normalize(c) for c in row]
    uri_idx = next((i for i, c in enumerate(normalized) if c in URI_ALIASES), None)
    label_idx = next((i for i, c in enumerate(normalized) if c in LABEL_ALIASES), None)
    if uri_idx is None or label_idx is None:
        return None
    version_idx = next((i for i, c in enumerate(normalized) if c in VERSION_ALIASES), None)
    recognized = {uri_idx, label_idx} | ({version_idx} if version_idx is not None else set())
    if len(recognized) != len(row):
        raise ValueError(f"Unrecognized column(s) in header {row!r}")
    return {"uri": uri_idx, "label": label_idx, "version": version_idx}


def columns_from_width(row: list[str]) -> dict[str, int]:
    if len(row) == 2:
        return {"uri": 0, "label": 1, "version": None}
    if len(row) == 3:
        return {"uri": 0, "label": 1, "version": 2}
    raise ValueError(f"Expected 2 or 3 columns (no header), got {len(row)}: {row!r}")


def parse_bool(cell: str, source: str, lineno: int) -> bool:
    value = cell.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise IngestionError(
        f"{source} line {lineno}: unparseable relevance label '{cell}'. "
        f"Expected one of {sorted(TRUE_VALUES | FALSE_VALUES)}."
    )


def _chain(*iterables):
    for it in iterables:
        yield from it


def read_rows(path_or_uri: str, cli_version: str | None) -> Iterator[tuple[str, bool, str]]:
    """Yield (article_uri, is_relevant, relevance_version) tuples for one file."""
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
                rows: Iterator[tuple[int, list[str]]] = _chain([(1, first_row)], enumerate(reader, start=2))
            else:
                rows = enumerate(reader, start=2)
        except ValueError as exc:
            raise IngestionError(f"{path_or_uri}: {exc}") from exc

        if columns["version"] is None and not cli_version:
            raise IngestionError(
                f"{path_or_uri} has no version column and no --version was "
                f"provided. Add a version column to the CSV, or pass --version <value>."
            )

        n_expected_cols = len(first_row)
        for lineno, row in rows:
            if not row:
                continue
            if len(row) != n_expected_cols:
                raise IngestionError(
                    f"{path_or_uri} line {lineno} has {len(row)} columns, "
                    f"expected {n_expected_cols}: {row!r}"
                )

            article_uri = row[columns["uri"]].strip()
            is_relevant = parse_bool(row[columns["label"]], path_or_uri, lineno)
            version_idx = columns["version"]
            row_version = row[version_idx].strip() if version_idx is not None else ""
            relevance_version = row_version or cli_version
            if not relevance_version:
                raise IngestionError(
                    f"{path_or_uri} line {lineno} has an empty version and no "
                    f"--version was provided. Add a version column to the CSV, or "
                    f"pass --version <value>."
                )

            yield article_uri, is_relevant, relevance_version


# ── DB upsert ─────────────────────────────────────────────────────────────────

_UPSERT_SQL_UPDATE = """
INSERT INTO article_event_extraction.article_relevance
    (article_uri, relevance_version, is_relevant)
VALUES %s
ON CONFLICT (relevance_version, article_uri)
DO UPDATE SET is_relevant = EXCLUDED.is_relevant
"""

_UPSERT_SQL_SKIP = """
INSERT INTO article_event_extraction.article_relevance
    (article_uri, relevance_version, is_relevant)
VALUES %s
ON CONFLICT (relevance_version, article_uri) DO NOTHING
"""


def flush_batch(cur, batch: dict[tuple[str, str], bool], on_conflict: str) -> tuple[int, int]:
    """Upsert `batch` (keyed by (article_uri, relevance_version)). Returns
    (attempted, affected) row counts."""
    if not batch:
        return 0, 0
    values = [(uri, version, is_relevant) for (uri, version), is_relevant in batch.items()]
    sql = _UPSERT_SQL_UPDATE if on_conflict == "update" else _UPSERT_SQL_SKIP
    psycopg2.extras.execute_values(cur, sql, values)
    return len(values), cur.rowcount


# ── Per-file processing ───────────────────────────────────────────────────────

@dataclass
class FileResult:
    path: str
    read: int = 0
    attempted: int = 0
    affected: int = 0
    error: str | None = None


def process_file(
    path_or_uri: str,
    conn_params: dict | None,
    version: str | None,
    batch_size: int,
    on_conflict: str,
    dry_run: bool,
    show_progress: bool,
) -> FileResult:
    """Read and upsert one file end-to-end on its own DB connection, so this
    can be called safely from a worker thread. Errors are captured on the
    result rather than raised/exited, since sys.exit() inside a worker thread
    would only kill that thread, not the whole run."""
    result = FileResult(path=path_or_uri)
    conn = None
    try:
        if not dry_run:
            assert conn_params is not None
            conn = psycopg2.connect(**conn_params)
        cur = conn.cursor() if conn else None
        batch: dict[tuple[str, str], bool] = {}
        for article_uri, is_relevant, relevance_version in read_rows(path_or_uri, version):
            result.read += 1
            batch[(article_uri, relevance_version)] = is_relevant
            if len(batch) >= batch_size:
                if cur is not None:
                    attempted, affected = flush_batch(cur, batch, on_conflict)
                    result.attempted += attempted
                    result.affected += affected
                batch.clear()
                if show_progress:
                    print(f"  read {result.read:,} rows\033[K", end="\r", file=sys.stderr, flush=True)

        if cur is not None:
            attempted, affected = flush_batch(cur, batch, on_conflict)
            result.attempted += attempted
            result.affected += affected
        if show_progress:
            print(file=sys.stderr)

        if conn:
            conn.commit()
    except IngestionError as exc:
        result.error = str(exc)
    except Exception as exc:
        result.error = f"unexpected error: {exc}"
    finally:
        if conn:
            conn.close()
    return result


def print_file_result(result: FileResult, dry_run: bool, on_conflict: str) -> None:
    print(f"\n{result.path}")
    if result.error:
        print(f"  Error: {result.error}")
        return
    print(f"  rows read     : {result.read:,}")
    if dry_run:
        print("  (dry run — no DB writes)")
    elif on_conflict == "skip":
        print(f"  upserted      : {result.affected:,}")
        print(f"  skipped (existing): {result.attempted - result.affected:,}")
    else:
        print(f"  upserted      : {result.affected:,}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest article relevance CSVs into article_event_extraction.article_relevance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="CSV file path(s) or folder(s), local or gs://bucket/path. A "
             "folder/prefix is expanded to the .csv files it contains.",
    )
    parser.add_argument(
        "--version", default=None,
        help="Relevance version to use for rows/files that don't specify one.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Rows per DB batch upsert (default 1000).",
    )
    parser.add_argument(
        "--on-conflict", choices=["update", "skip"], default="update",
        help="On (relevance_version, article_uri) conflict: overwrite is_relevant "
             "('update', default) or leave the existing row untouched ('skip').",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate the input files without connecting to the DB.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of files to ingest concurrently, each on its own DB "
             "connection (default 1 = sequential). Only safe when input files "
             "don't share (relevance_version, article_uri) keys.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    inputs = expand_inputs(args.inputs)
    if inputs != args.inputs:
        print(f"Resolved {len(args.inputs)} input path(s) to {len(inputs)} CSV file(s).")

    conn_params = None
    if not args.dry_run:
        conn_params = get_connection_params()
        print("Connecting to database...")
        try:
            test_conn = psycopg2.connect(**conn_params)
            test_conn.close()
        except Exception as exc:
            print(f"Connection failed: {exc}")
            sys.exit(1)
        print("Connected.")

    workers = min(args.workers, len(inputs))
    results: list[FileResult] = []

    if workers <= 1:
        for path_or_uri in inputs:
            result = process_file(
                path_or_uri, conn_params, args.version, args.batch_size,
                args.on_conflict, args.dry_run, show_progress=True,
            )
            results.append(result)
            print_file_result(result, args.dry_run, args.on_conflict)
    else:
        print(f"\nProcessing {len(inputs)} files with {workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_file, path_or_uri, conn_params, args.version,
                    args.batch_size, args.on_conflict, args.dry_run, False,
                ): path_or_uri
                for path_or_uri in inputs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print_file_result(result, args.dry_run, args.on_conflict)

    total_read = sum(r.read for r in results)
    total_attempted = sum(r.attempted for r in results)
    total_affected = sum(r.affected for r in results)
    errors = [r for r in results if r.error]

    print("\n=== Ingestion complete ===")
    print(f"Total rows read : {total_read:,}")
    if not args.dry_run:
        print(f"Total upserted  : {total_affected:,}")
        if args.on_conflict == "skip":
            print(f"Total skipped   : {total_attempted - total_affected:,}")

    if errors:
        print(f"\n{len(errors)} file(s) failed:", file=sys.stderr)
        for r in errors:
            print(f"  {r.path}: {r.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
