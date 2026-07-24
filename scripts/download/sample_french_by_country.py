"""Sample N articles in a given language (default French), stratified by country.

Geo source is Event Registry's own concept tagging
(article_concept_association -> geo_taxonomy_concept_uris_direct_match -> geo_taxonomy),
not the custom string-matching pipeline — so this covers articles the string-matcher
never ran on or missed.

Candidate discovery is two-stage (a profiled rework of jerome's Slack recipe):
  1a. Fetch the URIs of every article in the language, one cheap query per month
      from --start-month (default: earliest article, i.e. full history) to
      --end-month — each hits a single article_downloads partition via its
      language index.
  1b. Geo-tag those URIs in chunks of --probe-chunk-size probed against
      article_concept_association's (article_uri, concept_uri) primary key,
      in parallel on --workers connections, collecting per-country pools.
Joining month windows directly against the concept table (the naive approach)
makes Postgres seq-scan the 2.3B-row / 470 GB table once per month; the chunked
PK probes measured ~1.25 ms per article instead (~25 min for the full French
history on 12 workers).

Ctrl+C cancels queued work and the in-flight queries server-side, then exits.

Stratification is round-robin: countries are visited in ascending candidate-count
order, one article per country per round, until N articles are selected or all pools
are exhausted. This keeps the per-country distribution as flat as the data allows —
rare countries contribute everything they have, surplus countries absorb the rest.
Articles tagged with several countries are assigned to the first country that picks
them; all their tagged countries are still recorded in the output record.

Selection is deterministic for a given --seed: each country's pool is ordered by
md5(uri + seed), independent of query arrival order.

Phase 3 (content fetch) writes each chunk to --output as soon as it's fetched,
retrying a failed chunk on a fresh connection a few times before giving up
(article_downloads content queries occasionally get canceled server-side).
If a run still dies partway through, rerun with --resume: it re-derives the same
candidates/selection (deterministic for a given --seed), skips articles already
in --output, and appends the rest.

Usage:
    # 500 French articles, stratified by country, full history -> today
    python scripts/download/sample_french_by_country.py \
        --n 500 --output dataset/french_sample.jsonl

    # Custom window, more parallel monthly queries, GCS output
    python scripts/download/sample_french_by_country.py \
        --n 2000 --start-month 2020-01 --end-month 2025-06 --workers 16 \
        --output gs://my-bucket/data/french_sample.jsonl

    # Small test run to inspect output format
    python scripts/download/sample_french_by_country.py \
        --n 20 --start-month 2025-05 --output /tmp/test_french.jsonl

    # Resume a Phase 3 run that died partway through
    python scripts/download/sample_french_by_country.py \
        --n 5000000 --output dataset/french_sample_5M.jsonl --start-month 2000-01 \
        --resume

Requires psycopg2:
    pip install psycopg2-binary
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Generator, TextIO
from urllib.parse import urlparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    _env = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_month(value: str) -> date:
    try:
        year, month = value.split("-")
        return date(int(year), int(month), 1)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"Invalid month {value!r}, expected YYYY-MM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N articles in one language, stratified by country.",
    )
    parser.add_argument("--n", type=int, required=True,
        help="Total number of articles to sample.")
    parser.add_argument("--output", "-o", required=True,
        help="Output JSONL path (local or gs://bucket/path).")
    parser.add_argument("--language", type=str, default="fra",
        help="article_downloads language filter (default fra).")
    parser.add_argument("--start-month", type=parse_month, default=None,
        help="First month to scan, YYYY-MM (default: earliest article for the language).")
    parser.add_argument("--end-month", type=parse_month, default=None,
        help="Last month to scan inclusive, YYYY-MM (default: current month).")
    parser.add_argument("--workers", type=int, default=12,
        help="Parallel DB connections for discovery queries (default 12).")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for deterministic sampling (default 42).")
    parser.add_argument("--probe-chunk-size", type=int, default=2_000,
        help="URIs per geo-probe query in phase 1b (default 2000; "
             "measured ~1.25 ms/article at 1200).")
    parser.add_argument("--chunk-size", type=int, default=5_000,
        help="Max URIs per article_downloads content fetch (default 5000).")
    parser.add_argument("--resume", action="store_true",
        help="Skip articles already present in --output (matched by 'id') and "
             "append the rest. Use after Phase 3 was interrupted or a chunk "
             "failed all its retries.")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.end_month is None:
        today = date.today()
        args.end_month = date(today.year, today.month, 1)
    if args.start_month is not None and args.end_month < args.start_month:
        parser.error("--end-month must not be before --start-month")
    return args


# ── GCS output ────────────────────────────────────────────────────────────────

def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid GCS URI: {uri}. Expected gs://bucket/path/to/file.jsonl")
    return parsed.netloc, parsed.path.lstrip("/")


@contextmanager
def open_output(path_or_uri: str, initial_text: str = "") -> Generator[TextIO, None, None]:
    """Open for writing. initial_text (the cleaned contents of a prior run,
    from load_resume_state) is written back first so --resume effectively
    appends without relying on true append-mode support (GCS blobs don't have
    one; a local 'a' open wouldn't let us drop a truncated trailing line)."""
    if path_or_uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
            sys.exit(1)
        bucket_name, blob_name = parse_gcs_uri(path_or_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        with blob.open("w", encoding="utf-8") as f:
            if initial_text:
                f.write(initial_text)
            yield f
        return
    output_path = Path(path_or_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if initial_text:
            f.write(initial_text)
        yield f


def load_resume_state(path_or_uri: str) -> tuple[set[str], str]:
    """Article URIs already written by a prior run, and the cleaned raw text
    to preserve. A trailing line that fails to parse (a partial write from a
    process that died mid-record) is dropped along with anything after it —
    given writes are append-only and flushed per chunk, corruption can only
    ever be at the tail."""
    if path_or_uri.startswith("gs://"):
        from google.cloud import storage
        bucket_name, blob_name = parse_gcs_uri(path_or_uri)
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        text = blob.download_as_text() if blob.exists() else ""
    else:
        p = Path(path_or_uri)
        text = p.read_text(encoding="utf-8") if p.exists() else ""
    uris: set[str] = set()
    clean_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            uris.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            break
        clean_lines.append(line)
    cleaned = "".join(line + "\n" for line in clean_lines)
    return uris, cleaned


# ── SQL ───────────────────────────────────────────────────────────────────────

# Phase 1a: URIs of one month of articles in the target language.
# article_downloads is range-partitioned on published_at, so the month window
# prunes to one partition and its language index serves the rest.
_MONTH_URIS_SQL = """
SELECT uri
FROM article_downloads
WHERE language = %(language)s
  AND published_at >= %(start)s
  AND published_at <  %(end)s
"""

# Phase 1b: geo concepts for a chunk of article URIs.
#
# The MATERIALIZED CTE is load-bearing. article_concept_association is 2.3B rows
# / 470 GB and its only index is the (article_uri, concept_uri) PK; the fence
# forces one index-only probe of that PK per chunk, then a hash join against the
# small geo mapping. Without it the planner flattens the join and either
# seq-scans the whole table (the original month-window join: >10 min per month,
# ×12 in parallel = I/O thrash) or re-scans the index once per geo concept
# (measured 68 s per 1200-URI chunk vs 1.5 s with the fence).
_GEO_PROBE_SQL = """
WITH assoc_rows AS MATERIALIZED (
    SELECT article_uri, concept_uri
    FROM article_concept_association
    WHERE article_uri = ANY(%(uris)s)
)
SELECT DISTINCT ar.article_uri, geo.adm0_code
FROM assoc_rows ar
JOIN geo_taxonomy_concept_uris_direct_match geo_concepts
    ON geo_concepts.uri = ar.concept_uri
JOIN concept_uris concepts
    ON concepts.concept_uri = ar.concept_uri
   AND concepts.concept_type = 'loc'
JOIN geo_taxonomy geo
    ON geo.adm_code = geo_concepts.code
"""

# Used by resolve_start_month's binary search instead of MIN(published_at).
# article_downloads has ~300 monthly partitions but no index on (language,
# published_at), only per-partition indexes on language and on published_at
# separately — so a plain MIN() aggregate can't stop at the first match and
# instead scans every matching row in every partition (measured: cost 16M+,
# tens of millions of rows for 'fra'). EXISTS+LIMIT 1 lets each pruned
# partition answer with a single index probe (measured ~30-50ms per call
# regardless of range size), so binary-searching the boundary is ~10 fast
# queries instead of one query that touches the whole table.
_EXISTS_BEFORE_SQL = """
SELECT EXISTS (
    SELECT 1 FROM article_downloads
    WHERE language = %(language)s AND published_at < %(boundary)s
    LIMIT 1
) AS any_before
"""

_COUNTRY_NAMES_SQL = """
SELECT adm0_code, adm_name AS country_name
FROM geo_taxonomy
WHERE adm_level = 0
"""

_CONTENT_SQL = """
SELECT uri, cloud_uri, title, body, published_at, article_type, source_uri
FROM article_downloads
WHERE uri = ANY(%(uris)s)
  AND language = %(language)s
"""


# ── Phase 1: candidate discovery ──────────────────────────────────────────────

class Cancelled(Exception):
    """Worker aborted because the run is being cancelled."""


_stop = threading.Event()
_tls = threading.local()
_conns_lock = threading.Lock()
_conns: list = []


def _thread_conn(params: dict):
    """Per-thread connection, registered so the main thread can cancel its query."""
    conn = getattr(_tls, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        _tls.conn = conn
        with _conns_lock:
            _conns.append(conn)
    return conn


def _drop_thread_conn() -> None:
    conn = getattr(_tls, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _tls.conn = None


def cancel_inflight_queries() -> None:
    """Ask the server to abort whatever the worker connections are running.

    conn.cancel() is safe to call from another thread — it opens a separate
    socket and sends a Postgres cancel request for that backend.
    """
    with _conns_lock:
        for conn in _conns:
            if not conn.closed:
                try:
                    conn.cancel()
                except Exception:
                    pass


def close_all_conns() -> None:
    with _conns_lock:
        for conn in _conns:
            try:
                conn.close()
            except Exception:
                pass
        _conns.clear()


def _run_query(params: dict, sql: str, query_params: dict, label: str) -> list:
    """Execute on this thread's connection; one retry on a fresh connection."""
    last_exc: Exception | None = None
    for _ in range(2):
        if _stop.is_set():
            raise Cancelled()
        try:
            conn = _thread_conn(params)
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                return cur.fetchall()
        except Exception as exc:
            if _stop.is_set():
                raise Cancelled()
            last_exc = exc
            _drop_thread_conn()
    raise RuntimeError(f"{label} failed after retry: {last_exc}") from last_exc


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Half-open [first-of-month, first-of-next-month) ranges, start..end inclusive."""
    ranges = []
    current = start
    while current <= end:
        nxt = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
        ranges.append((current, nxt))
        current = nxt
    return ranges


def fetch_month_uris(params: dict, language: str, start: date, end: date) -> list[str]:
    rows = _run_query(
        params, _MONTH_URIS_SQL,
        {"language": language, "start": start, "end": end},
        f"month {start:%Y-%m}",
    )
    return [str(r[0]) for r in rows]


def probe_uri_chunk(params: dict, uris: list[str]) -> list[tuple[str, str]]:
    rows = _run_query(params, _GEO_PROBE_SQL, {"uris": uris},
                      f"geo probe of {len(uris)} uris")
    return [(str(uri), str(adm0)) for uri, adm0 in rows]


def collect_candidates(
    params: dict,
    language: str,
    months: list[tuple[date, date]],
    workers: int,
    probe_chunk_size: int,
) -> dict[str, set[str]]:
    """Two-stage discovery; returns adm0_code -> set of article URIs."""
    pool = ThreadPoolExecutor(max_workers=workers)
    candidates: dict[str, set[str]] = {}
    try:
        # 1a: article URIs, one cheap partition-pruned query per month
        uris: list[str] = []
        futures = {
            pool.submit(fetch_month_uris, params, language, start, end): start
            for start, end in months
        }
        for done, future in enumerate(as_completed(futures), 1):
            uris.extend(future.result())  # propagate failures — a silent gap skews the sample
            print(f"  1a: {done}/{len(months)} months, {len(uris):,} article uris\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # 1b: geo-tag the URIs in chunked PK probes
        chunk_futures = {
            pool.submit(probe_uri_chunk, params, uris[i:i + probe_chunk_size]):
                len(uris[i:i + probe_chunk_size])
            for i in range(0, len(uris), probe_chunk_size)
        }
        t0 = time.time()
        processed = 0
        for future in as_completed(chunk_futures):
            for uri, adm0 in future.result():
                # intern: the same URI lands in several country sets
                candidates.setdefault(adm0, set()).add(sys.intern(uri))
            processed += chunk_futures[future]
            rate = processed / max(time.time() - t0, 1e-9)
            eta_min = (len(uris) - processed) / max(rate, 1e-9) / 60
            print(f"  1b: {processed:,}/{len(uris):,} articles geo-probed "
                  f"({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        pool.shutdown(wait=True)
    except BaseException:
        # Ctrl+C or a worker failure: drop queued chunks AND cancel the queries
        # the server is still running — otherwise shutdown() blocks on them.
        _stop.set()
        cancel_inflight_queries()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        close_all_conns()
    return candidates


# ── Phase 2: stratified selection ─────────────────────────────────────────────

def stratified_sample(
    candidates: dict[str, set[str]],
    n: int,
    seed: int,
) -> list[dict]:
    """Round-robin over countries, rarest first, until n articles or exhaustion.

    Each country's pool is ordered by md5(uri + seed) so selection is deterministic
    and independent of the order monthly queries completed in.
    """
    def sort_key(uri: str) -> str:
        return hashlib.md5(f"{uri}:{seed}".encode()).hexdigest()

    order = sorted(candidates, key=lambda c: (len(candidates[c]), c))
    pools = {c: iter(sorted(candidates[c], key=sort_key)) for c in order}

    seen: set[str] = set()
    selected: list[dict] = []
    active = deque(order)
    while active and len(selected) < n:
        adm0 = active.popleft()
        for uri in pools[adm0]:
            if uri in seen:
                continue
            seen.add(uri)
            selected.append({"article_uri": uri, "adm0_code": adm0})
            active.append(adm0)
            break
        # pool exhausted -> country drops out of the rotation
    return selected


# ── Phase 3: content fetch ────────────────────────────────────────────────────

def fetch_country_names(params: dict) -> dict[str, str]:
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_COUNTRY_NAMES_SQL)
            return {str(r["adm0_code"]): str(r["country_name"]) for r in cur.fetchall()}
    finally:
        conn.close()


def _fetch_content_chunk(conn, uris: list[str], language: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_CONTENT_SQL, {"uris": uris, "language": language})
        return [dict(r) for r in cur.fetchall()]


def fetch_and_write_content(
    params: dict,
    output: str,
    selected: list[dict],
    candidates: dict[str, set[str]],
    country_names: dict[str, str],
    language: str,
    chunk_size: int,
    already_written: set[str],
    existing_text: str,
    max_retries: int = 5,
) -> tuple[int, int, int, set[str]]:
    """Fetch content chunk by chunk and write matching records as they arrive.

    Content-fetch chunks against article_downloads occasionally get canceled
    server-side (statement_timeout, or the server killing a slow scan) —
    that's what stopped the original run at 165k/5M articles. Each chunk now
    gets its own retry-with-fresh-connection loop, and records are flushed to
    disk as soon as a chunk succeeds, so a chunk that exhausts its retries only
    loses that chunk (rerun with --resume) instead of the whole multi-hour run.

    Returns (written_this_run, missing, skipped, all_written_uris).
    """
    to_fetch = [m for m in selected if m["article_uri"] not in already_written]
    skipped = len(selected) - len(to_fetch)
    written = 0
    missing = 0
    total = len(to_fetch)
    written_uris: set[str] = set(already_written)

    conn = psycopg2.connect(**params)
    conn.autocommit = True
    try:
        with open_output(output, initial_text=existing_text) as f:
            for i in range(0, total, chunk_size):
                chunk = to_fetch[i:i + chunk_size]
                uris = [m["article_uri"] for m in chunk]
                rows: list[dict] | None = None
                last_exc: Exception | None = None
                for attempt in range(max_retries):
                    try:
                        rows = _fetch_content_chunk(conn, uris, language)
                        break
                    except Exception as exc:
                        last_exc = exc
                        try:
                            conn.close()
                        except Exception:
                            pass
                        wait = min(2 ** attempt, 30)
                        print(f"\n  chunk fetch failed ({exc}); "
                              f"reconnecting, retry {attempt + 1}/{max_retries} in {wait}s...",
                              file=sys.stderr)
                        time.sleep(wait)
                        conn = psycopg2.connect(**params)
                        conn.autocommit = True
                if rows is None:
                    raise RuntimeError(
                        f"content fetch failed after {max_retries} attempts "
                        f"(rerun with --resume to pick up where this left off): {last_exc}"
                    ) from last_exc

                content_map = {str(r["uri"]): r for r in rows}
                for meta in chunk:
                    content = content_map.get(meta["article_uri"])
                    if content is None:
                        missing += 1
                        continue
                    all_countries = sorted(
                        adm0 for adm0, member_uris in candidates.items()
                        if meta["article_uri"] in member_uris
                    )
                    record = build_record(meta, content, language, all_countries, country_names)
                    f.write(json.dumps(record, ensure_ascii=False, default=str,
                                        separators=(",", ":")) + "\n")
                    written += 1
                    written_uris.add(meta["article_uri"])
                f.flush()

                done = min(i + chunk_size, total)
                print(f"  fetched {done:,}/{total:,} articles "
                      f"({written:,} written, {missing:,} missing)\033[K",
                      end="\r", file=sys.stderr, flush=True)
            print(file=sys.stderr)
    finally:
        conn.close()
    return written, missing, skipped, written_uris


# ── Output ────────────────────────────────────────────────────────────────────

def build_record(
    meta: dict,
    content: dict,
    language: str,
    all_countries: list[str],
    country_names: dict[str, str],
) -> dict:
    return {
        "id": meta["article_uri"],
        "adm0_code": meta["adm0_code"],
        "country_name": country_names.get(meta["adm0_code"], meta["adm0_code"]),
        "adm0_codes_all": all_countries,
        "language": language,
        "source": {
            "title":        str(content.get("title") or ""),
            "text":         str(content.get("body") or ""),
            "published_at": content.get("published_at"),
            "article_type": content.get("article_type"),
            "source_url":   content.get("source_uri"),
            "cloud_uri":    content.get("cloud_uri"),
        },
    }


def print_summary(
    selected: list[dict],
    candidates: dict[str, set[str]],
    written_uris: set[str],
    country_names: dict[str, str],
    written: int,
    missing: int,
    skipped: int,
    output: str,
) -> None:
    per_country: dict[str, int] = {}
    for meta in selected:
        if meta["article_uri"] in written_uris:
            per_country[meta["adm0_code"]] = per_country.get(meta["adm0_code"], 0) + 1

    print("\n=== Stratified language sample complete ===")
    print(f"Articles in output : {len(written_uris):,}")
    if skipped:
        print(f"  ({written:,} written this run, {skipped:,} already present from --resume)")
    print(f"Countries covered  : {len(per_country):,} / {len(candidates):,} with candidates")
    if missing:
        print(f"Missing from DB    : {missing:,}")
    print(f"Output             : {output}")
    print("\nPer-country breakdown (selected / candidates):")
    for adm0 in sorted(per_country, key=lambda c: (-per_country[c], c)):
        name = country_names.get(adm0, adm0)
        print(f"  {name:<40} {per_country[adm0]:>6,} / {len(candidates[adm0]):,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _month_index(d: date) -> int:
    return d.year * 12 + (d.month - 1)


def _month_from_index(i: int) -> date:
    return date(i // 12, i % 12 + 1, 1)


def resolve_start_month(params: dict, language: str) -> date | None:
    """Earliest month with an article in this language, or None if there are none.

    Binary search using _EXISTS_BEFORE_SQL rather than MIN(published_at) — see
    the comment on that query for why the aggregate is much slower.
    """
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            def exists_before(boundary: date) -> bool:
                cur.execute(_EXISTS_BEFORE_SQL, {"language": language, "boundary": boundary})
                return bool(cur.fetchone()[0])

            today = date.today()
            hi_idx = _month_index(date(today.year, today.month, 1))
            if not exists_before(_month_from_index(hi_idx + 1)):
                return None

            lo_idx = _month_index(date(1970, 1, 1))
            while lo_idx < hi_idx:
                mid_idx = (lo_idx + hi_idx) // 2
                if exists_before(_month_from_index(mid_idx + 1)):
                    hi_idx = mid_idx
                else:
                    lo_idx = mid_idx + 1
            return _month_from_index(lo_idx)


def main() -> None:
    args = parse_args()
    params = get_connection_params()

    # Make main-thread queries (min-month lookup, content fetch) interruptible:
    # with this callback, Ctrl+C during execute() sends a server-side cancel
    # instead of blocking until the query completes.
    psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)

    if args.start_month is None:
        print(f"Resolving earliest month for language={args.language!r} ...")
        args.start_month = resolve_start_month(params, args.language)
        if args.start_month is None:
            print("No articles found for this language. Nothing to sample.")
            return
        if args.start_month > args.end_month:
            args.end_month = args.start_month
        print(f"  earliest month: {args.start_month:%Y-%m}")

    months = month_ranges(args.start_month, args.end_month)

    print(f"Phase 1: scanning {len(months)} months "
          f"({args.start_month:%Y-%m} .. {args.end_month:%Y-%m}) "
          f"for language={args.language!r} with {args.workers} workers...")
    candidates = collect_candidates(params, args.language, months, args.workers,
                                    args.probe_chunk_size)
    total_candidates = len(set().union(*candidates.values())) if candidates else 0
    print(f"  {total_candidates:,} distinct articles across {len(candidates):,} countries")
    if not candidates:
        print("No geo-tagged articles found for this language/window. Nothing to sample.")
        return

    print(f"\nPhase 2: stratified sampling of {args.n:,} articles...")
    selected = stratified_sample(candidates, args.n, args.seed)
    print(f"  selected {len(selected):,} articles across "
          f"{len({m['adm0_code'] for m in selected}):,} countries")

    try:
        country_names = fetch_country_names(params)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)

    already_written: set[str] = set()
    existing_text = ""
    if args.resume:
        already_written, existing_text = load_resume_state(args.output)
        if already_written:
            print(f"  --resume: {len(already_written):,} articles already in {args.output}")

    print(f"\nPhase 3: fetching content for {len(selected):,} articles "
          f"(writing to {args.output} as each chunk completes)...")
    written, missing, skipped, written_uris = fetch_and_write_content(
        params, args.output, selected, candidates, country_names, args.language,
        args.chunk_size, already_written, existing_text,
    )
    print_summary(selected, candidates, written_uris, country_names,
                  written, missing, skipped, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — queued work dropped, in-flight queries cancelled "
              "server-side.", file=sys.stderr)
        sys.exit(130)
    except psycopg2.extensions.QueryCanceledError:
        # wait_select turns Ctrl+C during a main-thread query into this
        print("\nInterrupted — query cancelled server-side.", file=sys.stderr)
        sys.exit(130)
