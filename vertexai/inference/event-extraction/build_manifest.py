"""Build the pre-sharded manifest for the event-extraction batch job.

Streams the relevant articles out of the DB (no body payload — a fast scan that barely
touches CloudSQL) and writes the rows round-robin into `--num-shards` JSONL files under
`--output-prefix`:

    {output_prefix}/manifest-000.jsonl
    {output_prefix}/manifest-001.jsonl
    ...

Each line carries only what the extraction prompt actually consumes:

    {"id": <uri>, "gcs_path": <cloud_uri>, "publish_date": <published_at>, "language": <lang>}

The article body is NOT in the manifest — it is fetched from GCS at inference time, one
object per article. That is a single GCS GET per article (the object holds title and body
together) and it keeps the manifest at ~200 B/row instead of ~5 KB/row; carrying bodies
here would instead mean exporting 33M `article_downloads.body` values out of production
CloudSQL, which is exactly the load this design avoids.

`publish_date` is `str(published_at)` ("YYYY-MM-DD HH:MM:SS"), matching how the training
data built `source.publish_date` — the student model saw that format, and vllm_infer.py
reads `source.publish_date` only. `language` selects the EN vs FR prompt pair.

At ~33M relevant rows the whole manifest is ~6.6 GB; pre-sharding means each Cloud Batch
task later downloads only its own ~1/N slice and never has to hold the full corpus in
memory. Row `k` goes to file `k % num_shards`, so the shards are evenly sized and load is
balanced.

This script is deliberately self-contained — it defines its own DB connection, server-side
cursor, and GCS writers and imports nothing from the rest of the repo.

Usage:
    python vertexai/inference/event-extraction/build_manifest.py \
        --output-prefix gs://my-bucket/extraction/manifests \
        --num-shards 200

    # local dry run
    python vertexai/inference/event-extraction/build_manifest.py \
        --output-prefix /tmp/manifests --num-shards 4 --limit 1000

    # if the planner picks a bad join plan (see --join-mode below)
    python vertexai/inference/event-extraction/build_manifest.py \
        --output-prefix gs://my-bucket/extraction/manifests \
        --num-shards 200 --join-mode memory

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
    # parents[3] is the repo root (this file is vertexai/inference/event-extraction/).
    # Note: the relevance builder uses parents[2], which points at vertexai/ -- no .env
    # lives there, so it silently falls back to os.environ.
    _env = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    _env = {}


# The title-only relevance run (~34.4M relevant articles). The older `mmbert-small-v1`
# classified on title+body and marks ~50.0M relevant.
DEFAULT_RELEVANCE_VERSION = "mmbert-small-title-only-v1"


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

# Driving side is article_relevance, whose PK is (relevance_version, article_uri) -- so the
# version + is_relevant filter is a PK-prefix scan. Check the plan with EXPLAIN before a full
# run: article_downloads is range-partitioned on published_at with ~300 partitions, so a
# nested-loop probe by uri alone can fan out across every one of them. We want a hash/merge
# join (a single sequential pass over article_downloads). If the planner insists otherwise,
# use --join-mode memory.
_SQL_JOIN = """
SELECT d.uri, d.cloud_uri, d.published_at, d.language
FROM article_event_extraction.article_relevance r
JOIN article_downloads d ON d.uri = r.article_uri
WHERE r.relevance_version = %(version)s
  AND r.is_relevant = true
  AND d.cloud_uri IS NOT NULL
"""

_SQL_RELEVANT_URIS = """
SELECT article_uri
FROM article_event_extraction.article_relevance
WHERE relevance_version = %(version)s AND is_relevant = true
"""

_SQL_ALL_DOWNLOADS = """
SELECT uri, cloud_uri, published_at, language
FROM article_downloads
WHERE cloud_uri IS NOT NULL
"""


def _server_side_cursor(conn, name: str):
    return conn.cursor(name=name, cursor_factory=psycopg2.extras.RealDictCursor)


def stream_rows_sql(conn, version: str, languages: list[str] | None, limit: int | None):
    """Yield (uri, cloud_uri, published_at, language) via the SQL join."""
    sql = _SQL_JOIN
    params: dict = {"version": version}
    if languages:
        sql += "  AND d.language = ANY(%(languages)s)\n"
        params["languages"] = languages
    if limit:
        sql += "LIMIT %(limit)s\n"
        params["limit"] = limit

    with _server_side_cursor(conn, "extraction_manifest") as cur:
        cur.itersize = 5000
        cur.execute(sql, params)
        for row in cur:
            yield row["uri"], row["cloud_uri"], row["published_at"], row["language"]


def load_relevant_uris(conn, version: str) -> set:
    """Load every relevant article_uri into a set, as int when the uris are numeric.

    ~33M numeric uris cost ~2 GB as ints (vs ~4 GB as str), which is the difference
    between this fitting on a laptop and not."""
    uris: set = set()
    non_numeric = 0
    with _server_side_cursor(conn, "extraction_relevant_uris") as cur:
        cur.itersize = 50000
        cur.execute(_SQL_RELEVANT_URIS, {"version": version})
        for row in cur:
            uri = row["article_uri"]
            if uri is None:
                continue
            try:
                uris.add(int(uri))
            except (TypeError, ValueError):
                non_numeric += 1
                uris.add(str(uri))
            if len(uris) % 1000000 == 0:
                print(f"  {len(uris)} relevant uris loaded\033[K", end="\r",
                      file=sys.stderr, flush=True)
    print(f"\nLoaded {len(uris)} relevant uris for version {version!r}"
          f"{f' ({non_numeric} non-numeric, kept as strings)' if non_numeric else ''}",
          file=sys.stderr)
    return uris


def _in_relevant(uri, relevant: set) -> bool:
    try:
        return int(uri) in relevant
    except (TypeError, ValueError):
        return str(uri) in relevant


def stream_rows_memory(conn, version: str, languages: list[str] | None, limit: int | None):
    """Yield the same tuples, but filter article_downloads in Python against a
    pre-loaded set of relevant uris. Avoids handing the planner a 33M-row join."""
    relevant = load_relevant_uris(conn, version)
    if not relevant:
        return

    sql = _SQL_ALL_DOWNLOADS
    params: dict = {}
    if languages:
        sql += "  AND language = ANY(%(languages)s)\n"
        params["languages"] = languages

    yielded = 0
    with _server_side_cursor(conn, "extraction_downloads") as cur:
        cur.itersize = 5000
        cur.execute(sql, params)
        for row in cur:
            if not _in_relevant(row["uri"], relevant):
                continue
            yield row["uri"], row["cloud_uri"], row["published_at"], row["language"]
            yielded += 1
            if limit and yielded >= limit:
                return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the pre-sharded event-extraction manifest.")
    p.add_argument("--output-prefix", required=True,
                   help="gs://bucket/path or local dir; writes {prefix}/manifest-NNN.jsonl")
    p.add_argument("--num-shards", type=int, required=True,
                   help="Number of manifest files (must equal the Cloud Batch --shards)")
    p.add_argument("--relevance-version", default=DEFAULT_RELEVANCE_VERSION,
                   help=f"article_relevance.relevance_version to filter on "
                        f"(default: {DEFAULT_RELEVANCE_VERSION}). Articles with no row for "
                        f"this version are excluded.")
    p.add_argument("--join-mode", choices=["sql", "memory"], default="sql",
                   help="sql: let Postgres join relevance to downloads (default). "
                        "memory: load relevant uris into a set first, then stream "
                        "article_downloads and filter in Python (~2 GB RAM at 33M rows).")
    p.add_argument("--language", choices=["eng", "fra", "both"], default=None,
                   help="Optional article language filter: eng, fra, or both (default: no filter)")
    p.add_argument("--limit", type=int, default=None, help="Max rows (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    languages = {"eng": ["eng"], "fra": ["fra"], "both": ["eng", "fra"]}.get(args.language)

    params = get_connection_params()
    print(f"Connecting to {params['host']}:{params['port']}/{params['dbname']} ...", file=sys.stderr)
    print(f"Relevance version: {args.relevance_version!r} (join-mode: {args.join_mode})",
          file=sys.stderr)
    conn = psycopg2.connect(**params)

    stream = stream_rows_sql if args.join_mode == "sql" else stream_rows_memory

    written = [0] * args.num_shards
    skipped = 0
    try:
        with conn, ExitStack() as stack:
            writers = open_shard_writers(stack, args.output_prefix, args.num_shards)
            rows = stream(conn, args.relevance_version, languages, args.limit)
            # Round-robin over *kept* rows, not over the raw stream index: counting
            # skipped rows would leave holes and unbalance the shards.
            kept = 0
            for uri, cloud_uri, published_at, language in rows:
                if not uri or not cloud_uri:
                    skipped += 1
                    continue
                shard = kept % args.num_shards
                kept += 1
                row = {"id": str(uri), "gcs_path": str(cloud_uri)}
                if published_at:
                    row["publish_date"] = str(published_at)
                if language:
                    row["language"] = str(language)
                writers[shard].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
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
    if total == 0:
        print(f"  WARNING: no rows. Is {args.relevance_version!r} the right relevance_version?",
              file=sys.stderr)


if __name__ == "__main__":
    main()
