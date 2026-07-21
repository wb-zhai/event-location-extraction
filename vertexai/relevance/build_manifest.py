"""Build the pre-sharded manifest for the relevance-filter batch job.

Streams `SELECT uri, cloud_uri FROM article_downloads` (no body payload — a fast,
index-only-ish scan that barely touches the DB) and writes the rows round-robin into
`--num-shards` JSONL files under `--output-prefix`:

    {output_prefix}/manifest-000.jsonl
    {output_prefix}/manifest-001.jsonl
    ...

Each line is `{"id": <uri>, "gcs_path": <cloud_uri>}`. At ~140M rows the whole manifest is
~8 GB; pre-sharding means each Cloud Batch task later downloads only its own ~1/N slice and
never has to hold the full corpus in memory. Row `k` goes to file `k % num_shards`, so the
shards are evenly sized and load is balanced.

This script is deliberately self-contained — it defines its own DB connection, server-side
cursor, and GCS writers and imports nothing from the rest of the repo.

Usage:
    python vertexai/relevance/build_manifest.py \
        --output-prefix gs://my-bucket/relevance/manifests \
        --num-shards 100

    # local dry run
    python vertexai/relevance/build_manifest.py \
        --output-prefix /tmp/manifests --num-shards 4 --limit 1000

Requires: psycopg2-binary, google-cloud-storage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Config / DB connection (self-contained)
# ---------------------------------------------------------------------------


def _e(key: str, fallback: str | None = None) -> str | None:
    return _env.get(key) or os.environ.get(key) or fallback


def get_connection_params() -> dict:
    params = {
        "host":     _e("SQL_HOST",     _e("PGHOST", "localhost")),
        "port":     _e("SQL_PORT",     _e("PGPORT", "5432")),
        "dbname":   _e("SQL_DATABASE", _e("PGDATABASE")),
        "user":     _e("SQL_USERNAME", _e("PGUSER")),
        "password": _e("SQL_PASSWORD", _e("PGPASSWORD")),
    }
    missing = [k for k in ("dbname", "user", "password") if not params[k]]
    if missing:
        raise SystemExit(
            f"Missing DB config: {', '.join(missing)}. "
            f"Set SQL_* (or PG*) in the repo-root .env or the environment."
        )
    return params


# ---------------------------------------------------------------------------
# Output writers (local dir or gs:// prefix)
# ---------------------------------------------------------------------------


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URI: {uri}. Expected gs://bucket/path")
    return parsed.netloc, parsed.path.lstrip("/")


def open_shard_writers(stack: ExitStack, output_prefix: str, num_shards: int):
    """Open one text writer per shard under output_prefix, returned as a list."""
    names = [f"manifest-{i:03d}.jsonl" for i in range(num_shards)]

    if output_prefix.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
            sys.exit(1)
        bucket_name, prefix = parse_gcs_uri(output_prefix)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        writers = []
        for name in names:
            blob_name = f"{prefix.rstrip('/')}/{name}" if prefix else name
            blob = bucket.blob(blob_name)
            writers.append(stack.enter_context(blob.open("w", encoding="utf-8")))
        return writers

    out_dir = Path(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [stack.enter_context((out_dir / name).open("w", encoding="utf-8")) for name in names]


# ---------------------------------------------------------------------------
# DB streaming
# ---------------------------------------------------------------------------


def stream_rows(conn, language: str | None, limit: int | None):
    """Yield (uri, cloud_uri) via a server-side cursor so nothing is buffered."""
    where = "WHERE cloud_uri IS NOT NULL"
    params: list = []
    if language:
        where += " AND language = %s"
        params.append(language)
    sql = f"SELECT uri, cloud_uri FROM article_downloads {where}"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor(name="relevance_manifest", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.itersize = 5000
        cur.execute(sql, params)
        for row in cur:
            yield row["uri"], row["cloud_uri"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the pre-sharded relevance manifest.")
    p.add_argument("--output-prefix", required=True,
                   help="gs://bucket/path or local dir; writes {prefix}/manifest-NNN.jsonl")
    p.add_argument("--num-shards", type=int, required=True,
                   help="Number of manifest files (must equal the Cloud Batch --shards)")
    p.add_argument("--language", default=None, help="Optional article language filter (e.g. eng)")
    p.add_argument("--limit", type=int, default=None, help="Max rows (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    params = get_connection_params()
    print(f"Connecting to {params['host']}:{params['port']}/{params['dbname']} ...", file=sys.stderr)
    conn = psycopg2.connect(**params)

    written = [0] * args.num_shards
    skipped = 0
    try:
        with conn, ExitStack() as stack:
            writers = open_shard_writers(stack, args.output_prefix, args.num_shards)
            for i, (uri, cloud_uri) in enumerate(stream_rows(conn, args.language, args.limit)):
                if not uri or not cloud_uri:
                    skipped += 1
                    continue
                shard = i % args.num_shards
                writers[shard].write(
                    json.dumps({"id": str(uri), "gcs_path": str(cloud_uri)},
                               ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                written[shard] += 1
                total = sum(written)
                if total % 100000 == 0:
                    print(f"  {total} rows written\033[K", end="\r", file=sys.stderr, flush=True)
    finally:
        conn.close()

    total = sum(written)
    print(f"\nWrote {total} rows across {args.num_shards} shard(s) to {args.output_prefix}",
          file=sys.stderr)
    print(f"  per-shard min/max: {min(written)}/{max(written)}, skipped (null uri/cloud_uri): {skipped}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
