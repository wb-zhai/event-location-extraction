"""Download a country-stratified sample of English articles, split into positives
(risk-factor-tagged) and negatives (geo-tagged, no risk-factor tags).

This is a merge of two approaches developed in this repo:
  - from_db_matrix.py's positive/negative split by risk factor tags
    (article_risk_factor_tags, tag_method_id).
  - sample_french_by_country.py's fast candidate discovery: Event Registry's own
    concept tagging (article_concept_association -> geo_taxonomy_concept_uris_direct_match
    -> geo_taxonomy) probed in chunks against indexed primary keys, instead of joining
    month/country windows directly against the 2.3B-row / 470 GB association table
    (which forces a seq scan — measured >10 min per month before this rework).

Discovery is three stages (four with --relevance-version), all chunked, indexed
lookups — no table is ever scanned in full:
  1a. Fetch the URIs of every article in --language, one cheap query per month from
      --start-month (default: earliest article, i.e. full history) to --end-month —
      each hits a single article_downloads partition via its language index.
  1b. Geo-tag those URIs in chunks of --probe-chunk-size, probed against
      article_concept_association's (article_uri, concept_uri) primary key.
      Measured ~1.25 ms/article.
  2.  Risk-factor-tag the same URIs in chunks, probed against
      article_risk_factor_tags' article_uri index. Measured ~0.8 ms/article.
      Present in the result = positive candidate (with its risk factor ids);
      absent = negative candidate. This replaces from_db_matrix.py's per-risk-factor
      LATERAL oversampling — with real per-article RF data in hand there's no need
      to guess a sample-pool size, so --oversample is gone.
  3.  Optional, only with --relevance-version: relevance-tag the same URIs in
      chunks, probed against article_relevance's (relevance_version, article_uri)
      primary key. Present in the result = known relevant/not-relevant; absent =
      not yet classified for that version, and excluded from relevance-ratio
      stratification (see below) so unclassified articles never silently count
      as either label. Pass --relevance-version any to skip the version filter
      and take each article's most recent verdict regardless of which version
      produced it (that query can't use the PK the same way -- see the SQL
      comment on _RELEVANCE_PROBE_ANY_VERSION_SQL).
All stages run in parallel across --workers connections. Ctrl+C cancels queued
work and in-flight queries server-side, then exits.

Stratification (phase 3) is round-robin per label: positive and negative candidates
are each grouped by country and drawn rarest-country-first, one article per country
per round, until the pos/neg quota is hit or the pool is exhausted — so rare
countries contribute everything they have and surplus countries absorb the rest.
Within a country, articles are further grouped by publication year and drawn
rarest-year-first with the same round-robin, so a country's quota isn't dominated
by whichever years happen to have the most raw volume (news archives typically
skew recent). Within a (country, year) bucket, positives are ordered by
risk-factor-count descending (most diverse first, as in from_db_matrix.py's
balance_positives) then by md5(uri + seed) for determinism; negatives are ordered
by md5(uri + seed) alone. Publication year is known for free from phase 1a (one
query per month) — no extra DB round-trip. An article geo-tagged to several
countries is claimed by whichever country's round reaches it first; all of its
tagged countries are still recorded in the output.

With --relevance-version, each of the positive/negative pools is itself split
first by is_relevant using --relevant-ratio (same round()-based quota math as
--pos-ratio), *then* stratified by country/year as above within each of the four
(label, is_relevant) cells. Articles with no relevance row for that version are
dropped from consideration entirely once --relevance-version is set — they don't
count toward either side of --relevant-ratio.

Usage:
    # 50k English articles, 70% positive, full history -> today
    python scripts/download/from_db_matrix_v2.py \
        --n 50000 --output dataset/matrix_sample.jsonl

    # Custom split, bounded window, more workers, GCS output
    python scripts/download/from_db_matrix_v2.py \
        --n 100000 --pos-ratio 0.6 --start-month 2020-01 --end-month 2025-06 \
        --workers 16 --output gs://my-bucket/data/matrix_sample.jsonl

    # Small test run to inspect output format before a large download
    python scripts/download/from_db_matrix_v2.py \
        --n 200 --start-month 2025-05 --output /tmp/test_sample.jsonl

    # Also stratify by the relevance classifier's verdict: 80% relevant, 20% not,
    # within each of the positive/negative pools, pinned to one classifier version
    python scripts/download/from_db_matrix_v2.py \
        --n 50000 --relevance-version gemini-2.5-flash-v1 --relevant-ratio 0.8 \
        --output dataset/matrix_sample.jsonl

    # Same, but take each article's latest relevance verdict regardless of
    # which classifier version produced it
    python scripts/download/from_db_matrix_v2.py \
        --n 50000 --relevance-version any --relevant-ratio 0.8 \
        --output dataset/matrix_sample.jsonl

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
from typing import Callable, Generator, TextIO
from urllib.parse import urlparse

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    # This file lives at <repo_root>/scripts/download/from_db_matrix_v2.py, i.e.
    # two directories below the repo root -> parents[2].
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_month(value: str) -> date:
    try:
        year, month = value.split("-")
        return date(int(year), int(month), 1)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"Invalid month {value!r}, expected YYYY-MM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a country-stratified positive/negative sample from the DB.",
    )
    parser.add_argument("--n", type=int, required=True,
        help="Total number of articles to download.")
    parser.add_argument("--output", "-o", required=True,
        help="Output JSONL path (local or gs://bucket/path).")
    parser.add_argument("--language", type=str, default="eng",
        help="article_downloads language filter (default eng).")
    parser.add_argument("--pos-ratio", type=float, default=0.7,
        help="Fraction of articles with risk factor tags (default 0.7).")
    parser.add_argument("--tag-method-id", type=int, default=1,
        help="tag_method_id filter for the risk-factor tagger (default 1).")
    parser.add_argument("--relevance-version", type=str, default=None,
        help="article_relevance.relevance_version to stratify by (e.g. gemini-2.5-flash-v1), "
             "or the literal 'any' to use each article's most recent relevance verdict "
             "regardless of which version produced it. Omit to skip relevance "
             "stratification entirely (default).")
    parser.add_argument("--relevant-ratio", type=float, default=0.7,
        help="Fraction of each positive/negative pool that must be is_relevant=true "
             "(default 0.7). Only used when --relevance-version is set; articles with "
             "no relevance row for that version are excluded from consideration.")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for deterministic sampling (default 42).")
    parser.add_argument("--start-month", type=parse_month, default=None,
        help="First month to scan, YYYY-MM (default: earliest article for the language).")
    parser.add_argument("--end-month", type=parse_month, default=None,
        help="Last month to scan inclusive, YYYY-MM (default: current month).")
    parser.add_argument("--workers", type=int, default=12,
        help="Parallel DB connections for discovery queries (default 12).")
    parser.add_argument("--probe-chunk-size", type=int, default=2_000,
        help="URIs per geo/risk-factor probe query (default 2000).")
    parser.add_argument("--chunk-size", type=int, default=5_000,
        help="Max URIs per article_downloads content fetch (default 5000).")
    args = parser.parse_args()
    if not 0.0 < args.pos_ratio < 1.0:
        parser.error("--pos-ratio must be strictly between 0 and 1")
    if not 0.0 < args.relevant_ratio < 1.0:
        parser.error("--relevant-ratio must be strictly between 0 and 1")
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
def open_output(path_or_uri: str) -> Generator[TextIO, None, None]:
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
            yield f
        return
    output_path = Path(path_or_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yield f


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
# seq-scans the whole table or re-scans the index once per geo concept
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

# Phase 2: risk factor tags for a chunk of article URIs.
#
# article_risk_factor_tags is 103M rows / 17 GB, indexed on article_uri. A single
# query filtering the whole table by tag_method_id (as from_db_matrix.py's original
# per-risk-factor sampling effectively required) costs 10M+ and scans ~85M rows —
# almost the whole table matches tag_method_id=1. Chunking by article_uri instead
# turns it into an indexed ANY(...) lookup per chunk (measured ~0.8 ms/article),
# using the article_uri index directly rather than a full filtered scan.
_RF_PROBE_SQL = """
WITH rf_rows AS MATERIALIZED (
    SELECT article_uri, risk_factor
    FROM article_risk_factor_tags
    WHERE tag_method_id = %(tag_method_id)s
      AND article_uri = ANY(%(uris)s)
)
SELECT article_uri, array_agg(DISTINCT risk_factor ORDER BY risk_factor) AS risk_factor_ids
FROM rf_rows
GROUP BY article_uri
"""

# Phase 3 (optional, --relevance-version): relevance tags for a chunk of article URIs.
#
# article_relevance's primary key is (relevance_version, article_uri), so an
# equality filter on relevance_version plus a chunked ANY(...) on article_uri hits
# that PK directly — same chunked-probe shape as the RF query above, and for the
# same reason (avoid a full-table scan of every version's rows).
_RELEVANCE_PROBE_SQL = """
WITH rel_rows AS MATERIALIZED (
    SELECT article_uri, is_relevant, relevance_version
    FROM article_relevance
    WHERE relevance_version = %(relevance_version)s
      AND article_uri = ANY(%(uris)s)
)
SELECT article_uri, is_relevant, relevance_version
FROM rel_rows
"""

# Phase 3 variant for --relevance-version any: ignore relevance_version and take
# each article's most recently created verdict, whichever version produced it.
# NOTE: unlike the query above, this filters on article_uri alone, which is only
# the *second* column of article_relevance's (relevance_version, article_uri) PK
# -- so it can't reuse that index the same way and may cost a wider scan per
# chunk. Add an index on article_uri (or (article_uri, created_at)) if this is
# slow in practice.
_RELEVANCE_PROBE_ANY_VERSION_SQL = """
WITH rel_rows AS MATERIALIZED (
    SELECT article_uri, is_relevant, relevance_version, created_at
    FROM article_relevance
    WHERE article_uri = ANY(%(uris)s)
)
SELECT DISTINCT ON (article_uri) article_uri, is_relevant, relevance_version
FROM rel_rows
ORDER BY article_uri, created_at DESC
"""

# Used by resolve_start_month's binary search instead of MIN(published_at).
# article_downloads has ~300 monthly partitions but no index on (language,
# published_at) together, only per-partition indexes on language and on
# published_at separately — so a plain MIN() aggregate can't stop at the first
# match and instead scans every matching row in every partition (measured: cost
# 16M+, tens of millions of rows for 'fra'). EXISTS+LIMIT 1 lets each pruned
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

# risk_factors is a small (167-row) catalog table — its count stands in for
# "total possible risk factors" without touching article_risk_factor_tags.
# COUNT(DISTINCT risk_factor) FROM article_risk_factor_tags WHERE tag_method_id=X
# has the same problem as the old MIN(published_at) query: no index on
# (tag_method_id, risk_factor) together, so it scans ~85M matching rows
# (measured: still running after 2 minutes) instead of answering from an index.
_RF_NAMES_SQL = "SELECT id, name FROM risk_factors"

_CONTENT_SQL = """
SELECT uri, cloud_uri, title, body, published_at, article_type, source_uri
FROM article_downloads
WHERE uri = ANY(%(uris)s)
  AND language = %(language)s
"""


# ── Threaded, cancellable discovery ────────────────────────────────────────────

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


def probe_geo_chunk(params: dict, uris: list[str]) -> list[tuple[str, str]]:
    rows = _run_query(params, _GEO_PROBE_SQL, {"uris": uris},
                      f"geo probe of {len(uris)} uris")
    return [(str(uri), str(adm0)) for uri, adm0 in rows]


def probe_rf_chunk(params: dict, uris: list[str], tag_method_id: int) -> list[tuple[str, list[int]]]:
    rows = _run_query(params, _RF_PROBE_SQL, {"uris": uris, "tag_method_id": tag_method_id},
                      f"rf probe of {len(uris)} uris")
    return [(str(uri), [int(x) for x in rf_ids]) for uri, rf_ids in rows]


def probe_relevance_chunk(
    params: dict, uris: list[str], relevance_version: str,
) -> list[tuple[str, bool, str]]:
    """relevance_version="any" ignores version and takes each article's most
    recent verdict; any other value filters to that exact version."""
    if relevance_version == "any":
        rows = _run_query(params, _RELEVANCE_PROBE_ANY_VERSION_SQL, {"uris": uris},
                          f"relevance probe (any version) of {len(uris)} uris")
    else:
        rows = _run_query(params, _RELEVANCE_PROBE_SQL, {"uris": uris, "relevance_version": relevance_version},
                          f"relevance probe of {len(uris)} uris")
    return [(str(uri), bool(is_relevant), str(version)) for uri, is_relevant, version in rows]


def discover(
    params: dict,
    language: str,
    months: list[tuple[date, date]],
    workers: int,
    probe_chunk_size: int,
    tag_method_id: int,
    relevance_version: str | None = None,
) -> tuple[dict[str, set[str]], dict[str, list[int]], dict[str, int], dict[str, tuple[bool, str]]]:
    """Discovery: month URIs -> geo probe -> risk-factor probe -> (optional) relevance probe.

    Returns (candidates, rf_map, uri_year, relevance_map):
      candidates: adm0_code -> set of geo-tagged article URIs in `language`.
      rf_map: article_uri -> risk_factor_ids, present only for RF-tagged articles
              (i.e. positives). Absence from rf_map means negative candidate.
      uri_year: article_uri -> publication year, read off the month range each
                URI was fetched from in phase 1a (free — no extra query).
      relevance_map: article_uri -> (is_relevant, relevance_version), present only
                      when relevance_version is set and the URI has a row for it.
                      With relevance_version="any" the version in the tuple is
                      whichever version actually produced that article's most
                      recent verdict, not the literal "any". Empty dict if
                      relevance_version is None.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    candidates: dict[str, set[str]] = {}
    rf_map: dict[str, list[int]] = {}
    uri_year: dict[str, int] = {}
    relevance_map: dict[str, tuple[bool, str]] = {}
    try:
        # 1a: article URIs, one cheap partition-pruned query per month
        uris: list[str] = []
        futures = {
            pool.submit(fetch_month_uris, params, language, start, end): start
            for start, end in months
        }
        for done, future in enumerate(as_completed(futures), 1):
            month_start = futures[future]
            month_uris = future.result()  # propagate failures — a silent gap skews the sample
            uris.extend(month_uris)
            for uri in month_uris:
                uri_year[uri] = month_start.year
            print(f"  1a: {done}/{len(months)} months, {len(uris):,} article uris\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # 1b: geo-tag the URIs in chunked PK probes
        geo_futures = {
            pool.submit(probe_geo_chunk, params, uris[i:i + probe_chunk_size]):
                len(uris[i:i + probe_chunk_size])
            for i in range(0, len(uris), probe_chunk_size)
        }
        t0, processed = time.time(), 0
        for future in as_completed(geo_futures):
            for uri, adm0 in future.result():
                # intern: the same URI lands in several country sets
                candidates.setdefault(adm0, set()).add(sys.intern(uri))
            processed += geo_futures[future]
            rate = processed / max(time.time() - t0, 1e-9)
            eta_min = (len(uris) - processed) / max(rate, 1e-9) / 60
            print(f"  1b: {processed:,}/{len(uris):,} articles geo-probed "
                  f"({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # 2: risk-factor-tag the distinct geo-tagged URIs
        rf_uris = list({u for uris_ in candidates.values() for u in uris_})
        rf_futures = {
            pool.submit(probe_rf_chunk, params, rf_uris[i:i + probe_chunk_size], tag_method_id):
                len(rf_uris[i:i + probe_chunk_size])
            for i in range(0, len(rf_uris), probe_chunk_size)
        }
        t0, processed = time.time(), 0
        for future in as_completed(rf_futures):
            for uri, rf_ids in future.result():
                rf_map[uri] = rf_ids
            processed += rf_futures[future]
            rate = processed / max(time.time() - t0, 1e-9)
            eta_min = (len(rf_uris) - processed) / max(rate, 1e-9) / 60
            print(f"  2: {processed:,}/{len(rf_uris):,} articles risk-factor-probed "
                  f"({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # 3 (optional): relevance-tag the same distinct geo-tagged URIs
        if relevance_version is not None:
            rel_futures = {
                pool.submit(probe_relevance_chunk, params, rf_uris[i:i + probe_chunk_size], relevance_version):
                    len(rf_uris[i:i + probe_chunk_size])
                for i in range(0, len(rf_uris), probe_chunk_size)
            }
            t0, processed = time.time(), 0
            for future in as_completed(rel_futures):
                for uri, is_relevant, version_used in future.result():
                    relevance_map[uri] = (is_relevant, version_used)
                processed += rel_futures[future]
                rate = processed / max(time.time() - t0, 1e-9)
                eta_min = (len(rf_uris) - processed) / max(rate, 1e-9) / 60
                print(f"  3: {processed:,}/{len(rf_uris):,} articles relevance-probed "
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
    return candidates, rf_map, uri_year, relevance_map


# ── Phase 3: stratified selection ─────────────────────────────────────────────

def _round_robin_order(groups: dict[object, list[str]]) -> list[str]:
    """Interleave each group's (already-sorted) items, rarest group first.

    Generic across grouping level: used both for country pools (groups = years
    within one country) and can be reused wherever else "rarest bucket first"
    interleaving is needed.
    """
    order = sorted(groups, key=lambda g: (len(groups[g]), g))
    iters = {g: iter(groups[g]) for g in order}
    active = deque(order)
    result: list[str] = []
    while active:
        g = active.popleft()
        item = next(iters[g], None)
        if item is None:
            continue  # group exhausted -> drops out of the rotation
        result.append(item)
        active.append(g)
    return result


def stratified_sample(
    candidates: dict[str, set[str]],
    n: int,
    seed: int,
    priority: Callable[[str], object] | None = None,
    uri_year: dict[str, int] | None = None,
) -> list[dict]:
    """Round-robin over countries, rarest first, until n articles or exhaustion.

    Within each country, articles are further round-robinned by publication
    year (rarest year first) before the country-level draw sees them, so a
    country's quota isn't dominated by whichever years have the most raw
    volume (news archives typically skew recent). Articles with an unknown
    year (uri_year omitted, or missing an entry) fall into one shared
    "unknown" bucket per country.

    Within a (country, year) bucket, articles are ordered by
    (priority(uri), md5(uri + seed)) — priority lets callers favour e.g.
    risk-factor-diverse articles; ties (or no priority) fall back to a
    deterministic, query-order-independent hash.
    """
    priority_fn = priority or (lambda uri: 0)

    def sort_key(uri: str):
        return (priority_fn(uri), hashlib.md5(f"{uri}:{seed}".encode()).hexdigest())

    def country_pool(uris: set[str]) -> list[str]:
        by_year: dict[object, list[str]] = {}
        for uri in uris:
            year = (uri_year or {}).get(uri, "unknown")
            by_year.setdefault(year, []).append(uri)
        for year_uris in by_year.values():
            year_uris.sort(key=sort_key)
        return _round_robin_order(by_year)

    order = sorted(candidates, key=lambda c: (len(candidates[c]), c))
    pools = {c: iter(country_pool(candidates[c])) for c in order}

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


def _sample_label_pool(
    candidates: dict[str, set[str]],
    n: int,
    seed: int,
    uri_year: dict[str, int] | None,
    relevance_map: dict[str, tuple[bool, str]] | None,
    relevant_ratio: float | None,
    priority: Callable[[str], object] | None = None,
) -> list[dict]:
    """Draw n articles from one positive/negative pool.

    Without relevance_map this is just stratified_sample(). With it, the pool
    is first split into relevant/not-relevant sub-pools (same round()-based
    quota math as pos_ratio uses at the top level), each independently
    stratified by country/year. Articles with no entry in relevance_map (not
    yet classified for the requested version) are dropped entirely rather than
    landing in either sub-pool, so they can't silently skew --relevant-ratio.
    """
    if not relevance_map:
        return stratified_sample(candidates, n, seed, priority=priority, uri_year=uri_year)

    relevant_candidates: dict[str, set[str]] = {}
    not_relevant_candidates: dict[str, set[str]] = {}
    for adm0, uris in candidates.items():
        relevant = {u for u in uris if (relevance_map.get(u) or (None,))[0] is True}
        not_relevant = {u for u in uris if (relevance_map.get(u) or (None,))[0] is False}
        if relevant:
            relevant_candidates[adm0] = relevant
        if not_relevant:
            not_relevant_candidates[adm0] = not_relevant

    n_relevant = round(n * relevant_ratio)
    n_not_relevant = n - n_relevant
    selected = stratified_sample(
        relevant_candidates, n_relevant, seed, priority=priority, uri_year=uri_year,
    )
    selected += stratified_sample(
        not_relevant_candidates, n_not_relevant, seed, priority=priority, uri_year=uri_year,
    )
    return selected


def split_and_sample(
    candidates: dict[str, set[str]],
    rf_map: dict[str, list[int]],
    n: int,
    pos_ratio: float,
    seed: int,
    uri_year: dict[str, int] | None = None,
    relevance_map: dict[str, tuple[bool, str]] | None = None,
    relevant_ratio: float | None = None,
) -> list[dict]:
    """Split candidates into positive/negative pools by rf_map membership, then
    stratify each independently (each further split by is_relevant first when
    relevance_map is given). Positives favour risk-factor diversity."""
    positive_candidates: dict[str, set[str]] = {}
    negative_candidates: dict[str, set[str]] = {}
    for adm0, uris in candidates.items():
        pos = {u for u in uris if u in rf_map}
        neg = uris - pos
        if pos:
            positive_candidates[adm0] = pos
        if neg:
            negative_candidates[adm0] = neg

    n_pos = round(n * pos_ratio)
    n_neg = n - n_pos

    selected_pos = _sample_label_pool(
        positive_candidates, n_pos, seed, uri_year, relevance_map, relevant_ratio,
        priority=lambda uri: -len(rf_map[uri]),
    )
    for meta in selected_pos:
        meta["label"] = "positive"
        meta["risk_factor_ids"] = rf_map[meta["article_uri"]]

    selected_neg = _sample_label_pool(
        negative_candidates, n_neg, seed, uri_year, relevance_map, relevant_ratio,
    )
    for meta in selected_neg:
        meta["label"] = "negative"
        meta["risk_factor_ids"] = []

    selected = selected_pos + selected_neg
    if relevance_map:
        for meta in selected:
            entry = relevance_map.get(meta["article_uri"])
            if entry is not None:
                meta["is_relevant"], meta["relevance_version_used"] = entry
    return selected


# ── Phase 4: content fetch ─────────────────────────────────────────────────────

def fetch_content(
    cur: psycopg2.extras.RealDictCursor,
    uris: list[str],
    chunk_size: int,
    language: str,
) -> dict[str, dict]:
    content_map: dict[str, dict] = {}
    total = len(uris)
    for i in range(0, total, chunk_size):
        chunk = uris[i : i + chunk_size]
        cur.execute(_CONTENT_SQL, {"uris": chunk, "language": language})
        for row in cur.fetchall():
            content_map[str(row["uri"])] = dict(row)
        done = min(i + chunk_size, total)
        print(f"  fetched {done}/{total} articles\033[K", end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return content_map


# ── Output ────────────────────────────────────────────────────────────────────

def build_record(
    meta: dict,
    content: dict,
    language: str,
    all_countries: list[str],
    country_names: dict[str, str],
    rf_names: dict[int, str],
    relevance_version: str | None,
) -> dict:
    risk_factors = [rf_names.get(rf_id, str(rf_id)) for rf_id in meta["risk_factor_ids"]]
    record = {
        "id": meta["article_uri"],
        "label": meta["label"],
        "adm0_code": meta["adm0_code"],
        "country_name": country_names.get(meta["adm0_code"], meta["adm0_code"]),
        "adm0_codes_all": all_countries,
        "risk_factors": risk_factors,
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
    if relevance_version is not None:
        record["relevance"] = {
            # Per-article actual version (relevant with --relevance-version any,
            # where it varies per article); falls back to the requested version
            # for the ordinary single-version case.
            "version": meta.get("relevance_version_used", relevance_version),
            "is_relevant": meta.get("is_relevant"),
        }
    return record


def write_output(
    path_or_uri: str,
    selected: list[dict],
    candidates: dict[str, set[str]],
    content_map: dict[str, dict],
    country_names: dict[str, str],
    rf_names: dict[int, str],
    language: str,
    relevance_version: str | None,
) -> tuple[int, int]:
    written = missing = 0
    with open_output(path_or_uri) as f:
        for meta in selected:
            content = content_map.get(meta["article_uri"])
            if content is None:
                missing += 1
                continue
            all_countries = sorted(
                adm0 for adm0, uris in candidates.items() if meta["article_uri"] in uris
            )
            record = build_record(
                meta, content, language, all_countries, country_names, rf_names, relevance_version,
            )
            f.write(json.dumps(record, ensure_ascii=False, default=str,
                               separators=(",", ":")) + "\n")
            written += 1
    return written, missing


def print_summary(
    selected: list[dict],
    candidates: dict[str, set[str]],
    content_map: dict[str, dict],
    written: int,
    missing: int,
    n_rf: int,
    output: str,
    relevance_version: str | None,
) -> None:
    present = [m for m in selected if m["article_uri"] in content_map]
    n_pos = sum(1 for m in present if m["label"] == "positive")
    n_neg = sum(1 for m in present if m["label"] == "negative")
    covered_countries = len({m["adm0_code"] for m in present})
    covered_rf_ids: set[int] = set()
    for m in present:
        covered_rf_ids.update(m["risk_factor_ids"])
    covered_years = {
        content_map[m["article_uri"]]["published_at"].year
        for m in present
        if content_map[m["article_uri"]].get("published_at") is not None
    }

    print("\n=== Matrix sample complete ===")
    print(f"Total articles written : {written:,}")
    print(f"  Positives            : {n_pos:,}")
    print(f"  Negatives            : {n_neg:,}")
    if relevance_version is not None:
        n_relevant = sum(1 for m in present if m.get("is_relevant") is True)
        n_not_relevant = sum(1 for m in present if m.get("is_relevant") is False)
        print(f"  Relevant             : {n_relevant:,} (version={relevance_version})")
        print(f"  Not relevant         : {n_not_relevant:,}")
    print(f"Countries covered      : {covered_countries:,} / {len(candidates):,}")
    print(f"Risk factors covered   : {len(covered_rf_ids):,} / {n_rf:,}")
    if covered_years:
        print(f"Years covered          : {len(covered_years):,} ({min(covered_years)}-{max(covered_years)})")
    if missing:
        print(f"Missing from DB        : {missing:,}")
    print(f"Output                 : {output}")


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

    phase_label = "Phase 1-3" if args.relevance_version else "Phase 1-2"
    print(f"\n{phase_label}: scanning {len(months)} months "
          f"({args.start_month:%Y-%m} .. {args.end_month:%Y-%m}) "
          f"for language={args.language!r} with {args.workers} workers...")
    candidates, rf_map, uri_year, relevance_map = discover(
        params, args.language, months, args.workers, args.probe_chunk_size, args.tag_method_id,
        relevance_version=args.relevance_version,
    )
    total_candidates = len(set().union(*candidates.values())) if candidates else 0
    print(f"  {total_candidates:,} distinct articles across {len(candidates):,} countries, "
          f"{len(rf_map):,} risk-factor-tagged"
          + (f", {len(relevance_map):,} relevance-tagged" if args.relevance_version else ""))
    if not candidates:
        print("No geo-tagged articles found for this language/window. Nothing to sample.")
        return

    ratio_note = f", relevant-ratio={args.relevant_ratio}" if args.relevance_version else ""
    print(f"\nPhase 3: stratified pos/neg sampling of {args.n:,} articles "
          f"(pos-ratio={args.pos_ratio}{ratio_note})...")
    selected = split_and_sample(
        candidates, rf_map, args.n, args.pos_ratio, args.seed,
        uri_year=uri_year, relevance_map=relevance_map, relevant_ratio=args.relevant_ratio,
    )
    n_pos = sum(1 for m in selected if m["label"] == "positive")
    n_neg = len(selected) - n_pos
    print(f"  selected {len(selected):,} articles ({n_pos:,} positive, {n_neg:,} negative) "
          f"across {len({m['adm0_code'] for m in selected}):,} countries")

    print(f"\nPhase 4: fetching content for {len(selected):,} articles...")
    try:
        conn = psycopg2.connect(**params)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_COUNTRY_NAMES_SQL)
            country_names = {str(r["adm0_code"]): str(r["country_name"]) for r in cur.fetchall()}
            cur.execute(_RF_NAMES_SQL)
            rf_names = {int(r["id"]): str(r["name"]) for r in cur.fetchall()}
            n_rf = len(rf_names)
            content_map = fetch_content(
                cur, [m["article_uri"] for m in selected], args.chunk_size, args.language,
            )
    finally:
        conn.close()

    print(f"\nPhase 5: writing JSONL to {args.output} ...")
    written, missing = write_output(
        args.output, selected, candidates, content_map, country_names, rf_names, args.language,
        args.relevance_version,
    )
    print_summary(
        selected, candidates, content_map, written, missing, n_rf, args.output,
        args.relevance_version,
    )


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
