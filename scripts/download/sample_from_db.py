"""Download a country-stratified sample of articles from the DB, split into
relevant (is_relevant=true) and not-relevant according to the relevance
classifier in article_relevance.

Works for any --language article_relevance has verdicts for (default eng).
For --language eng specifically, the relevant pool also gets an optional
risk-factor-diversity signal from the older article_risk_factor_tags table
(see --tag-method-id below) — no other language currently has coverage in
that table, which is what makes this eng-only.

Built on sample_french_by_country.py's fast candidate discovery: Event Registry's
own concept tagging (article_concept_association -> geo_taxonomy_concept_uris_direct_match
-> geo_taxonomy) probed in chunks against indexed primary keys, instead of joining
month/country windows directly against the 2.3B-row / 470 GB association table
(which forces a seq scan — measured >10 min per month before this rework).

Discovery starts from article_relevance, not article_downloads — the classified
pool is a fraction of the full per-language article table, so pulling it
directly is cheaper than enumerating every article in --language first and
probing relevance per chunk. Four stages, all chunked, indexed lookups — no
table is ever scanned in full:
  1a. Fetch every (article_uri, is_relevant, relevance_version, created_at) row
      for the resolved relevance version(s) directly from article_relevance —
      an indexed range scan on the (relevance_version, article_uri) primary
      key's leading column, sized by how many articles have been classified,
      not by article_downloads' size. Default --relevance-version any resolves
      to every distinct version in the table (fetch_relevance_versions(), via
      the skip-scan-style _RELEVANCE_VERSIONS_SQL) and fetches each one,
      keeping each article's most recently created verdict across versions.
      Pass an explicit version to fetch just that one instead. No --language or
      date filtering happens at this stage — see 1b.
  1b. Confirm --language and (if --start-month/--end-month were given) the
      date window, via a chunked uri=ANY() lookup against article_downloads.
      This doubles as the source of publication year for stratification.
      Neither pool is selectable downstream without surviving this. Every
      is_relevant=true URI from 1a is confirmed (it feeds phase 3's
      country/year stratification, so the full pool matters), but the
      is_relevant=false side is downsampled BEFORE this query, not after:
      since that pool is never stratified (see below), there's no reason to
      pay language/date confirmation on all of it just to throw most of it
      away in phase 3's truncation. Instead we take a deterministic random
      subset sized at (needed not-relevant quota) x --not-relevant-oversample
      (see _deterministic_sample) and only confirm that. On a table where
      classified-but-not-relevant dwarfs classified-and-relevant (commonly
      100x+), this is what keeps 1b's cost proportional to what phase 3 will
      actually use instead of the full classified pool. NOTE: article_downloads
      is range-partitioned on published_at; without a date bound this lookup
      may not prune partitions as neatly as a per-month query would.
  1c. Geo-tag the confirmed is_relevant=true URIs from 1b, in chunks of
      --probe-chunk-size, probed against article_concept_association's
      (article_uri, concept_uri) primary key. Measured ~1.25 ms/article.
  1d. --language eng only: probe article_risk_factor_tags for the same
      confirmed is_relevant=true URIs (chunked uri=ANY(), filtered by
      --tag-method-id) to get each article's risk-factor tag count. Used as a
      priority signal in phase 3, not a hard filter or a fresh grouping —
      articles with more distinct risk factors are picked first within each
      (country, year) bucket, mirroring the old from_db_matrix.py's
      "maximise risk-factor diversity per country" positive-selection.
All stages run in parallel across --workers connections. Ctrl+C cancels queued
work and in-flight queries server-side, then exits.

Stratification (phase 3) only applies to the is_relevant=true pool: candidates
are grouped by country and drawn rarest-country-first, one article per country
per round, until its quota is hit or the pool is exhausted — so rare countries
contribute everything they have and surplus countries absorb the rest. Within
a country, articles are further grouped by publication year and drawn
rarest-year-first with the same round-robin, so a country's quota isn't
dominated by whichever years happen to have the most raw volume (news
archives typically skew recent). Within a (country, year) bucket, articles are
ordered by (risk-factor-diversity priority [eng only], md5(uri + seed)) for a
deterministic, query-order-independent pick. Publication year comes from phase
1b's language/date confirmation — the same lookup that verifies --language, so
no extra DB round-trip beyond that. An article geo-tagged to several countries
is claimed by whichever country's round reaches it first; all of its tagged
countries are still recorded in the output.

The is_relevant=false pool isn't stratified at all — it's just every confirmed
not-relevant candidate from 1b ordered by md5(uri + seed) and truncated to its
quota, since nothing downstream needs it balanced by country, year, or risk
factor (though it is, like the relevant pool, guaranteed to be --language and
in-window — see 1b).

--pos-ratio controls the split: round(n * pos_ratio) articles come from the
relevant (stratified) pool, the rest from the not-relevant (random) pool.

Content fetch writes each chunk to --output as soon as it's fetched, retrying
a failed chunk on a fresh connection a few times before giving up
(article_downloads content queries occasionally get canceled server-side). If
a run still dies partway through, rerun with --resume: it re-derives the same
candidates/selection (deterministic for a given --seed), skips articles
already in --output, and appends the rest.

Usage:
    # 50k English articles, 70% relevant (country/year-stratified, with
    # risk-factor-diversity priority within each bucket) and 30% not (uniform
    # random), trusting each article's latest is_relevant verdict regardless
    # of classifier version, full history -> today
    python scripts/download/sample_articles.py \
        --n 50000 --output dataset/matrix_sample.jsonl

    # French: same relevant/not-relevant split and country/year stratification,
    # just without the risk-factor-diversity signal (article_risk_factor_tags
    # has no French coverage)
    python scripts/download/sample_articles.py \
        --n 20000 --language fra --output dataset/fr_sample.jsonl

    # Custom split, bounded window, more workers, GCS output
    python scripts/download/sample_articles.py \
        --n 100000 --pos-ratio 0.6 --start-month 2020-01 --end-month 2025-06 \
        --workers 16 --output gs://my-bucket/data/matrix_sample.jsonl

    # Small test run to inspect output format before a large download
    python scripts/download/sample_articles.py \
        --n 200 --start-month 2025-05 --output /tmp/test_sample.jsonl

    # Pin relevance to one classifier version instead of trusting the latest
    # verdict regardless of version
    python scripts/download/sample_articles.py \
        --n 50000 --relevance-version gemini-2.5-flash-v1 \
        --output dataset/matrix_sample.jsonl

    # Resume a content-fetch run that died partway through
    python scripts/download/sample_articles.py \
        --n 5000000 --output dataset/matrix_sample_5M.jsonl --start-month 2000-01 \
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
import math
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
    # This file lives at <repo_root>/scripts/download/sample_articles.py, i.e.
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
        description="Download a country-stratified relevant/not-relevant sample from the DB.",
    )
    parser.add_argument("--n", type=int, required=True,
        help="Total number of articles to download.")
    parser.add_argument("--output", "-o", required=True,
        help="Output JSONL path (local or gs://bucket/path).")
    parser.add_argument("--language", type=str, default="eng",
        help="article_downloads language filter (default eng). Confirmed during "
             "discovery (phase 1b) for both the is_relevant=true and =false pools "
             "before either is selectable. Only --language eng gets the "
             "risk-factor-diversity signal (see --tag-method-id) — other "
             "languages get the same relevant/not-relevant, country/year- "
             "stratified sample otherwise.")
    parser.add_argument("--pos-ratio", type=float, default=0.7,
        help="Fraction of articles that are is_relevant=true (default 0.7). "
             "Those are geo-tagged and stratified by country/year; the rest "
             "are sampled uniformly at random from is_relevant=false candidates.")
    parser.add_argument("--relevance-version", type=str, default="any",
        help="article_relevance.relevance_version to sample from (e.g. "
             "gemini-2.5-flash-v1). Default 'any' fetches every distinct version in "
             "the table (fetch_relevance_versions()) and keeps each article's most "
             "recently created verdict across them — pin to a specific version to "
             "fetch just that one instead. Discovery starts from this table directly "
             "(phase 1a), not from article_downloads, which is what keeps the pool "
             "small from the start (see module docstring).")
    parser.add_argument("--tag-method-id", type=int, default=1,
        help="tag_method_id filter for article_risk_factor_tags (default 1). Only "
             "used when --language eng: articles in the relevant pool with more "
             "distinct risk-factor tags are prioritized within each (country, year) "
             "bucket (phase 1d), and a 'risk_factors' field is added to output "
             "records. Ignored for other languages, which have no coverage in that "
             "table yet.")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for deterministic sampling (default 42).")
    parser.add_argument("--not-relevant-oversample", type=float, default=5.0,
        help="Multiplier applied to the not-relevant quota (n * (1 - pos_ratio)) to size "
             "a random pre-filter of is_relevant=false candidates BEFORE they're confirmed "
             "against --language/date-window in phase 1b (default 5.0), instead of "
             "confirming the entire (often 100x+ larger) not-relevant pool just to discard "
             "most of it in phase 3. Raise this if the run ends short of its not-relevant "
             "quota (printed as a warning after phase 1b) — a narrow --language or "
             "--start-month/--end-month window can fail confirmation for most candidates.")
    parser.add_argument("--start-month", type=parse_month, default=None,
        help="Earliest publication month to allow, YYYY-MM (default: unbounded). "
             "Applied to both pools during discovery (phase 1b) — see --language.")
    parser.add_argument("--end-month", type=parse_month, default=None,
        help="Latest publication month to allow inclusive, YYYY-MM (default: current "
             "month). Applied to both pools during discovery (phase 1b) — see --language.")
    parser.add_argument("--workers", type=int, default=12,
        help="Parallel DB connections for discovery queries (default 12).")
    parser.add_argument("--probe-chunk-size", type=int, default=2_000,
        help="URIs per language/date, geo, or risk-factor probe query (default 2000).")
    parser.add_argument("--chunk-size", type=int, default=5_000,
        help="Max URIs per article_downloads content fetch (default 5000).")
    parser.add_argument("--resume", action="store_true",
        help="Skip articles already present in --output (matched by 'id') and "
             "append the rest. Use after a content-fetch run was interrupted or "
             "a chunk failed all its retries.")
    args = parser.parse_args()
    if not 0.0 < args.pos_ratio < 1.0:
        parser.error("--pos-ratio must be strictly between 0 and 1")
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.not_relevant_oversample <= 0:
        parser.error("--not-relevant-oversample must be positive")
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

# Phase 1a: every article classified under one relevance_version, no
# article_uri/language/date filter at all. article_event_extraction.article_relevance's
# primary key is (relevance_version, article_uri), so an equality filter on
# relevance_version (the PK's *leading* column) is a plain indexed range scan
# — cost is proportional to how many articles have been classified under this
# version, not to article_downloads' size. This is what lets discovery start
# here instead of enumerating every article in --language first (the earlier
# approach) and probing relevance per chunk.
_RELEVANCE_ALL_FOR_VERSION_SQL = """
SELECT article_uri, is_relevant, relevance_version, created_at
FROM article_event_extraction.article_relevance
WHERE relevance_version = %(relevance_version)s
"""

# Every distinct relevance_version in the table, for resolving
# --relevance-version any into a small list of versions to fetch (see
# _RELEVANCE_ALL_FOR_VERSION_SQL, one call per version). A plain SELECT
# DISTINCT relevance_version would still scan every row (or every index entry)
# once, touching the whole table — this recursive CTE instead walks the
# (relevance_version, article_uri) PK's leading column one distinct value at a
# time (each step is an indexed "next value after X" probe), the standard
# skip-scan-emulation trick for tables without native skip-scan support. Cost
# is O(distinct versions), not O(table size); this runs once per invocation.
_RELEVANCE_VERSIONS_SQL = """
WITH RECURSIVE versions AS (
    (
        SELECT relevance_version
        FROM article_event_extraction.article_relevance
        ORDER BY relevance_version
        LIMIT 1
    )
    UNION ALL
    (
        SELECT (
            SELECT relevance_version
            FROM article_event_extraction.article_relevance
            WHERE relevance_version > versions.relevance_version
            ORDER BY relevance_version
            LIMIT 1
        )
        FROM versions
        WHERE versions.relevance_version IS NOT NULL
    )
)
SELECT relevance_version FROM versions WHERE relevance_version IS NOT NULL
"""

# Phase 1b: confirm --language (and, if given, --start-month/--end-month) for a
# chunk of is_relevant=true article URIs from 1a, and read off published_at for
# year-stratification while at it. Built dynamically (see
# _build_language_date_probe_sql) since the date clauses are optional —
# psycopg2 needs static SQL text per query, and there's no clean way to make a
# WHERE clause conditionally no-op with plain parameters.
#
# NOTE: article_downloads is range-partitioned on published_at (see
# _CONTENT_SQL below, which already does a bare uri=ANY() lookup — but only
# ever against the small final-selected set, not a multi-million-URI discovery
# pool). Without a date bound this lookup may not prune partitions the way a
# per-month query would; it's only run against the relevant pool (usually much
# smaller than the full per-language table), but if it's slow in practice,
# passing --start-month/--end-month narrows it, or an index on uri would fix
# it structurally.
def _build_language_date_probe_sql(has_start: bool, has_end: bool) -> str:
    clauses = ["uri = ANY(%(uris)s)", "language = %(language)s"]
    if has_start:
        clauses.append("published_at >= %(start)s")
    if has_end:
        clauses.append("published_at < %(end)s")
    return "SELECT uri, published_at FROM article_downloads WHERE " + " AND ".join(clauses)


# Phase 1c: geo concepts for a chunk of article URIs. Runs only against the
# is_relevant=true subset of the relevance-filtered pool from 1b — the
# not-relevant pool is sampled uniformly at random and never reaches this
# query at all, so it never pays this stage's cost.
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

# Phase 1d (--language eng only): risk-factor tags for a chunk of article
# URIs, from the older tag_method_id-based tagging pipeline (see
# from_db_matrix.py, which this reprises just the diversity signal from).
# Used as a priority in phase 3, not a filter.
#
# article_risk_factor_tags is 103M rows / 17 GB, indexed on article_uri. A
# single query filtering the whole table by tag_method_id (as from_db_matrix.py's
# original per-risk-factor sampling effectively required) costs 10M+ and scans
# ~85M rows — almost the whole table matches tag_method_id=1. The MATERIALIZED
# CTE chunks by article_uri instead, turning it into an indexed ANY(...) lookup
# per chunk (measured ~0.8 ms/article), using the article_uri index directly
# rather than a full filtered scan — same rationale as the geo probe above.
_RISK_FACTOR_PROBE_SQL = """
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

# risk_factors is a small (167-row) catalog table — its row count stands in
# for "total possible risk factors" without touching article_risk_factor_tags.
# COUNT(DISTINCT risk_factor) FROM article_risk_factor_tags WHERE tag_method_id=X
# has the same problem as the probe above: no index on (tag_method_id, risk_factor)
# together, so it scans ~85M matching rows (measured: still running after 2
# minutes) instead of answering from an index — fetch_risk_factor_names() below
# is used for both the id->name mapping and this count (len(rf_names)).
_RISK_FACTOR_NAMES_SQL = "SELECT id, name FROM risk_factors"

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


def _deterministic_sample(uris: list[str], k: int, seed: int) -> list[str]:
    """Deterministic pseudo-random subset of `uris`, size min(k, len(uris)),
    ordered by md5(uri + seed) — the same reproducible, query-order-independent
    scheme used everywhere else in this module for sampling (see
    stratified_sample, sample_articles). Used to shrink the is_relevant=false
    pool BEFORE phase 1b's language/date confirmation (see discover())."""
    if k >= len(uris):
        return uris
    return sorted(uris, key=lambda uri: hashlib.md5(f"{uri}:{seed}".encode()).hexdigest())[:k]


def fetch_relevance_versions(params: dict, relevance_version: str) -> list[str]:
    """Resolve --relevance-version into the list of versions to fetch: just
    [relevance_version] as-is, or every distinct version in the table for the
    literal "any" (see _RELEVANCE_VERSIONS_SQL)."""
    if relevance_version != "any":
        return [relevance_version]
    with psycopg2.connect(**params) as conn:
        with conn.cursor() as cur:
            cur.execute(_RELEVANCE_VERSIONS_SQL)
            return [str(r[0]) for r in cur.fetchall()]


def fetch_relevance_for_version(
    params: dict, version: str,
) -> list[tuple[str, bool, str, object]]:
    """Every (article_uri, is_relevant, relevance_version, created_at) row for
    one relevance_version — see _RELEVANCE_ALL_FOR_VERSION_SQL."""
    rows = _run_query(params, _RELEVANCE_ALL_FOR_VERSION_SQL, {"relevance_version": version},
                      f"relevance fetch (version={version})")
    return [(str(uri), bool(is_relevant), str(row_version), created_at)
            for uri, is_relevant, row_version, created_at in rows]


def probe_language_date_chunk(
    params: dict, uris: list[str], language: str, sql: str, extra_params: dict,
) -> list[tuple[str, object]]:
    query_params = {"uris": uris, "language": language, **extra_params}
    rows = _run_query(params, sql, query_params, f"language/date probe of {len(uris)} uris")
    return [(str(uri), published_at) for uri, published_at in rows]


def probe_geo_chunk(params: dict, uris: list[str]) -> list[tuple[str, str]]:
    rows = _run_query(params, _GEO_PROBE_SQL, {"uris": uris},
                      f"geo probe of {len(uris)} uris")
    return [(str(uri), str(adm0)) for uri, adm0 in rows]


def probe_risk_factor_chunk(
    params: dict, tag_method_id: int, uris: list[str],
) -> list[tuple[str, list[int]]]:
    rows = _run_query(params, _RISK_FACTOR_PROBE_SQL,
                      {"tag_method_id": tag_method_id, "uris": uris},
                      f"risk-factor probe of {len(uris)} uris")
    return [(str(uri), [int(x) for x in risk_factor_ids]) for uri, risk_factor_ids in rows]


def discover(
    params: dict,
    language: str,
    workers: int,
    probe_chunk_size: int,
    relevance_versions: list[str],
    start_month: date | None,
    end_month: date | None,
    tag_method_id: int | None,
    seed: int,
    not_relevant_quota: int,
    not_relevant_oversample: float,
) -> tuple[dict[str, set[str]], dict[str, int], dict[str, tuple[bool, str]], dict[str, list[int]]]:
    """Discovery: relevance fetch -> language/date confirm (relevant pool in
    full, not-relevant pool pre-sampled) -> geo probe (relevant-only) ->
    risk-factor probe (relevant-only, --language eng only).

    relevance_versions is the already-resolved list from
    fetch_relevance_versions() — a single version to pin to, or every version
    in the table for --relevance-version any. tag_method_id is None to skip
    stage 1d entirely (any language other than eng). not_relevant_quota is
    the final number of not-relevant articles the run needs (n - n_relevant);
    not_relevant_oversample sizes the random pre-filter applied to the
    is_relevant=false pool before phase 1b spends a query on it (see module
    docstring's phase 1b section and _deterministic_sample) — the seed makes
    that pre-filter reproducible.

    Returns (candidates, uri_year, relevance_map, risk_factor_map):
      candidates: adm0_code -> set of geo-tagged, is_relevant=true article URIs
                  confirmed to be in `language` and (if bounded) the date window.
      uri_year: article_uri -> publication year, read off phase 1b's language/
                date confirmation. Only meaningful for candidates (the only
                ones stratified_sample needs it for), though it may also hold
                entries for confirmed not-relevant URIs.
      relevance_map: article_uri -> (is_relevant, relevance_version), restricted
                      to confirmed (--language, in-window) URIs — every
                      is_relevant=true one, and a random, quota-sized subset of
                      is_relevant=false ones (see above) — nothing outside that
                      is selectable by sample_articles(). With
                      relevance_version="any" the version in the tuple is
                      whichever version actually produced that article's most
                      recent verdict, not the literal "any".
      risk_factor_map: article_uri -> list of distinct risk_factor ids, for
                        the candidates subset only. Empty when tag_method_id
                        is None.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    candidates: dict[str, set[str]] = {}
    risk_factor_map: dict[str, list[int]] = {}
    try:
        # 1a: every classified article for the resolved version(s), no
        # language/date filter yet (see 1b).
        rel_futures = {
            pool.submit(fetch_relevance_for_version, params, version): version
            for version in relevance_versions
        }
        raw: dict[str, tuple[bool, str, object]] = {}
        for done, future in enumerate(as_completed(rel_futures), 1):
            for uri, is_relevant, row_version, created_at in future.result():
                prev = raw.get(uri)
                if prev is None or created_at > prev[2]:
                    raw[uri] = (is_relevant, row_version, created_at)
            print(f"  1a: {done}/{len(relevance_versions)} relevance version(s) fetched, "
                  f"{len(raw):,} distinct classified articles\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        relevance_map: dict[str, tuple[bool, str]] = {
            sys.intern(uri): (is_relevant, version) for uri, (is_relevant, version, _created_at) in raw.items()
        }
        del raw

        # 1b: confirm language/date for every is_relevant=true URI (it feeds
        # phase 3's stratification, so the full pool matters) plus a random,
        # quota-sized subset of is_relevant=false URIs (never stratified, so
        # there's no point confirming more of it than phase 3 will use — see
        # module docstring). Reads off publication year while at it (used for
        # stratification of the true pool only).
        relevant_uris = [uri for uri, (is_relevant, _v) in relevance_map.items() if is_relevant]
        not_relevant_uris = [uri for uri, (is_relevant, _v) in relevance_map.items() if not is_relevant]
        not_relevant_target = math.ceil(not_relevant_quota * not_relevant_oversample)
        sampled_not_relevant = _deterministic_sample(not_relevant_uris, not_relevant_target, seed)
        print(f"  1a: {len(relevant_uris):,} relevant / {len(not_relevant_uris):,} not-relevant "
              f"classified; sampling {len(sampled_not_relevant):,} not-relevant candidates "
              f"(quota {not_relevant_quota:,} x {not_relevant_oversample} oversample) before "
              f"confirming language/date", file=sys.stderr)
        all_uris = relevant_uris + sampled_not_relevant
        sql = _build_language_date_probe_sql(start_month is not None, end_month is not None)
        extra_params: dict = {}
        if start_month is not None:
            extra_params["start"] = start_month
        if end_month is not None:
            extra_params["end"] = end_month

        lang_futures = {
            pool.submit(probe_language_date_chunk, params, all_uris[i:i + probe_chunk_size],
                        language, sql, extra_params):
                len(all_uris[i:i + probe_chunk_size])
            for i in range(0, len(all_uris), probe_chunk_size)
        }
        confirmed: set[str] = set()
        uri_year: dict[str, int] = {}
        t0, processed = time.time(), 0
        for future in as_completed(lang_futures):
            for uri, published_at in future.result():
                uri = sys.intern(uri)
                confirmed.add(uri)
                if published_at is not None:
                    uri_year[uri] = published_at.year
            processed += lang_futures[future]
            rate = processed / max(time.time() - t0, 1e-9)
            eta_min = (len(all_uris) - processed) / max(rate, 1e-9) / 60
            print(f"  1b: {processed:,}/{len(all_uris):,} articles language/date-confirmed, "
                  f"{len(confirmed):,} matched ({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # Only confirmed (--language, and if bounded, in-window) articles are
        # selectable downstream, for either pool.
        relevance_map = {uri: relevance_map[uri] for uri in confirmed}
        n_confirmed_not_relevant = sum(1 for is_relevant, _v in relevance_map.values() if not is_relevant)
        if n_confirmed_not_relevant < not_relevant_quota:
            print(f"  warning: only {n_confirmed_not_relevant:,} not-relevant candidates confirmed "
                  f"(--language={language!r}/date-window match), short of the {not_relevant_quota:,} "
                  f"needed; raise --not-relevant-oversample (currently {not_relevant_oversample}) "
                  f"or rerun.", file=sys.stderr)

        # 1c: geo-tag the confirmed is_relevant=true URIs in chunked PK probes
        confirmed_list = [uri for uri, (is_relevant, _version) in relevance_map.items() if is_relevant]
        geo_futures = {
            pool.submit(probe_geo_chunk, params, confirmed_list[i:i + probe_chunk_size]):
                len(confirmed_list[i:i + probe_chunk_size])
            for i in range(0, len(confirmed_list), probe_chunk_size)
        }
        t0, processed = time.time(), 0
        for future in as_completed(geo_futures):
            for uri, adm0 in future.result():
                # intern: the same URI lands in several country sets
                candidates.setdefault(adm0, set()).add(sys.intern(uri))
            processed += geo_futures[future]
            rate = processed / max(time.time() - t0, 1e-9)
            eta_min = (len(confirmed_list) - processed) / max(rate, 1e-9) / 60
            print(f"  1c: {processed:,}/{len(confirmed_list):,} relevant articles geo-probed "
                  f"({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
                  end="\r", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        # 1d: risk-factor tags for the same confirmed relevant URIs, eng only
        if tag_method_id is not None:
            rf_futures = {
                pool.submit(probe_risk_factor_chunk, params, tag_method_id,
                            confirmed_list[i:i + probe_chunk_size]):
                    len(confirmed_list[i:i + probe_chunk_size])
                for i in range(0, len(confirmed_list), probe_chunk_size)
            }
            t0, processed = time.time(), 0
            for future in as_completed(rf_futures):
                for uri, risk_factor_ids in future.result():
                    risk_factor_map[sys.intern(uri)] = risk_factor_ids
                processed += rf_futures[future]
                rate = processed / max(time.time() - t0, 1e-9)
                eta_min = (len(confirmed_list) - processed) / max(rate, 1e-9) / 60
                print(f"  1d: {processed:,}/{len(confirmed_list):,} relevant articles risk-factor-probed, "
                      f"{len(risk_factor_map):,} tagged ({rate:,.0f}/s, ~{eta_min:.0f} min left)\033[K",
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
    return candidates, uri_year, relevance_map, risk_factor_map


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
    (priority(uri), md5(uri + seed)) — priority lets callers favour some
    dimension of interest (e.g. risk-factor-tag diversity — see
    discover()'s tag_method_id); ties (or no priority) fall back to a
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


def sample_articles(
    candidates: dict[str, set[str]],
    relevance_map: dict[str, tuple[bool, str]],
    n: int,
    pos_ratio: float,
    seed: int,
    uri_year: dict[str, int] | None,
    risk_factor_map: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Split n articles into a relevant pool (stratified by country/year,
    with risk-factor-tag-count as an in-bucket priority when risk_factor_map
    is given) and a not-relevant pool (uniform random — nothing downstream
    needs it balanced).

    round(n * pos_ratio) come from `candidates` (is_relevant=true, geo-tagged);
    the rest are drawn from every is_relevant=false URI in relevance_map
    (already confirmed --language/date-window by discover(), just never
    geo-tagged), ordered by md5(uri + seed) and truncated to quota — their
    adm0_code is None since they skip geo-tagging.

    When risk_factor_map is given (--language eng), every selected article
    (relevant and not-relevant alike) gets a "risk_factor_ids" key — the
    article's tags, or [] if it has none — so downstream output can carry a
    risk_factors field consistently. Left unset entirely for other languages.
    """
    n_relevant = round(n * pos_ratio)
    n_not_relevant = n - n_relevant

    priority = None
    if risk_factor_map:
        priority = lambda uri: -len(risk_factor_map.get(uri, ()))  # noqa: E731

    selected = stratified_sample(candidates, n_relevant, seed, priority=priority, uri_year=uri_year)

    not_relevant_uris = sorted(
        (uri for uri, (is_relevant, _) in relevance_map.items() if is_relevant is False),
        key=lambda uri: hashlib.md5(f"{uri}:{seed}".encode()).hexdigest(),
    )
    selected += [
        {"article_uri": uri, "adm0_code": None}
        for uri in not_relevant_uris[:n_not_relevant]
    ]

    for meta in selected:
        meta["is_relevant"], meta["relevance_version_used"] = relevance_map[meta["article_uri"]]
        if risk_factor_map is not None:
            meta["risk_factor_ids"] = risk_factor_map.get(meta["article_uri"], [])
    return selected


# ── Content fetch, resumable ──────────────────────────────────────────────────

def fetch_country_names(params: dict) -> dict[str, str]:
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_COUNTRY_NAMES_SQL)
            return {str(r["adm0_code"]): str(r["country_name"]) for r in cur.fetchall()}
    finally:
        conn.close()


def fetch_risk_factor_names(params: dict) -> dict[int, str]:
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_RISK_FACTOR_NAMES_SQL)
            return {int(r["id"]): str(r["name"]) for r in cur.fetchall()}
    finally:
        conn.close()


def _fetch_content_chunk(conn, uris: list[str], language: str) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_CONTENT_SQL, {"uris": uris, "language": language})
        return [dict(r) for r in cur.fetchall()]


def build_record(
    meta: dict,
    content: dict,
    language: str,
    all_countries: list[str],
    country_names: dict[str, str],
    rf_names: dict[int, str],
) -> dict:
    record = {
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
        "relevance": {
            # Per-article actual version (relevant with --relevance-version any,
            # where it varies per article).
            "version": meta.get("relevance_version_used"),
            "is_relevant": meta.get("is_relevant"),
        },
    }
    # Only present when discover() ran phase 1d (--language eng) — see
    # sample_articles()'s risk_factor_map handling.
    if "risk_factor_ids" in meta:
        record["risk_factors"] = sorted(
            rf_names.get(rf_id, str(rf_id)) for rf_id in meta["risk_factor_ids"]
        )
    return record


def fetch_and_write_content(
    params: dict,
    output: str,
    selected: list[dict],
    candidates: dict[str, set[str]],
    country_names: dict[str, str],
    rf_names: dict[int, str],
    language: str,
    chunk_size: int,
    already_written: set[str],
    existing_text: str,
    max_retries: int = 5,
) -> tuple[int, int, int, set[str]]:
    """Fetch content chunk by chunk and write matching records as they arrive.

    Content-fetch chunks against article_downloads occasionally get canceled
    server-side (statement_timeout, or the server killing a slow scan). Each
    chunk gets its own retry-with-fresh-connection loop, and records are
    flushed to disk as soon as a chunk succeeds, so a chunk that exhausts its
    retries only loses that chunk (rerun with --resume) instead of the whole
    multi-hour run.

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
                    record = build_record(meta, content, language, all_countries, country_names, rf_names)
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

def print_summary(
    selected: list[dict],
    candidates: dict[str, set[str]],
    written_uris: set[str],
    country_names: dict[str, str],
    written: int,
    missing: int,
    skipped: int,
    output: str,
    relevance_version: str,
    n_risk_factors: int,
) -> None:
    present = [m for m in selected if m["article_uri"] in written_uris]
    n_relevant = sum(1 for m in present if m["is_relevant"] is True)
    n_not_relevant = sum(1 for m in present if m["is_relevant"] is False)
    per_country: dict[str, int] = {}
    for meta in present:
        if meta["adm0_code"] is not None:
            per_country[meta["adm0_code"]] = per_country.get(meta["adm0_code"], 0) + 1

    print("\n=== Sample complete ===")
    print(f"Total articles written : {len(written_uris):,}")
    if skipped:
        print(f"  ({written:,} written this run, {skipped:,} already present from --resume)")
    print(f"  Relevant              : {n_relevant:,} (version={relevance_version}, country/year-stratified)")
    print(f"  Not relevant          : {n_not_relevant:,} (uniform random)")
    print(f"Countries covered       : {len(per_country):,} / {len(candidates):,}")
    if n_risk_factors:
        covered_rf_ids: set[int] = set()
        for m in present:
            if m["is_relevant"]:
                covered_rf_ids.update(m.get("risk_factor_ids") or [])
        print(f"Risk factors covered    : {len(covered_rf_ids):,} / {n_risk_factors:,}")
    if missing:
        print(f"Missing from DB         : {missing:,}")
    print(f"Output                  : {output}")
    print("\nPer-country breakdown (selected / candidates):")
    for adm0 in sorted(per_country, key=lambda c: (-per_country[c], c)):
        name = country_names.get(adm0, adm0)
        print(f"  {name:<40} {per_country[adm0]:>6,} / {len(candidates[adm0]):,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    params = get_connection_params()

    # Make main-thread queries (version resolution, content fetch) interruptible:
    # with this callback, Ctrl+C during execute() sends a server-side cancel
    # instead of blocking until the query completes.
    psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)

    add_risk_factors = args.language == "eng"
    tag_method_id = args.tag_method_id if add_risk_factors else None
    if not add_risk_factors and args.tag_method_id != 1:
        print(f"Note: --tag-method-id only applies to --language eng "
              f"(article_risk_factor_tags has no coverage for {args.language!r}); ignoring it.")

    relevance_versions = fetch_relevance_versions(params, args.relevance_version)
    if not relevance_versions:
        print(f"No relevance_version rows found at all "
              f"(relevance-version={args.relevance_version!r}). Nothing to sample.")
        return
    if args.relevance_version == "any":
        print(f"relevance-version 'any' resolved to {len(relevance_versions)} "
              f"classifier version(s): {relevance_versions}")
        if len(relevance_versions) > 10:
            print(f"  warning: phase 1a fetches all {len(relevance_versions)} of these "
                  f"separately — pin --relevance-version to a specific one if this is slow.")

    n_want_relevant = round(args.n * args.pos_ratio)
    n_want_not_relevant = args.n - n_want_relevant

    start_label = f"{args.start_month:%Y-%m}" if args.start_month is not None else "(unbounded)"
    print(f"\nPhase 1-2: discovering articles for language={args.language!r}, "
          f"relevance-version={args.relevance_version!r}, "
          f"relevant-pool window={start_label}..{args.end_month:%Y-%m}, "
          f"with {args.workers} workers"
          f"{f', tag-method-id={tag_method_id}' if tag_method_id is not None else ''}...")
    candidates, uri_year, relevance_map, risk_factor_map = discover(
        params, args.language, args.workers, args.probe_chunk_size,
        relevance_versions=relevance_versions,
        start_month=args.start_month, end_month=args.end_month,
        tag_method_id=tag_method_id,
        seed=args.seed, not_relevant_quota=n_want_not_relevant,
        not_relevant_oversample=args.not_relevant_oversample,
    )
    n_relevant_geo = len(set().union(*candidates.values())) if candidates else 0
    n_not_relevant_avail = sum(1 for is_relevant, _ in relevance_map.values() if is_relevant is False)
    print(f"  {n_relevant_geo:,} relevant articles geo-tagged across {len(candidates):,} countries; "
          f"{n_not_relevant_avail:,} not-relevant articles available")
    if tag_method_id is not None:
        print(f"  {len(risk_factor_map):,} relevant articles have >=1 risk-factor tag "
              f"(tag_method_id={tag_method_id})")
    if not relevance_map:
        print("No relevance-classified articles found for this language/window/version. "
              "Nothing to sample.")
        return

    n_risk_factors = 0
    rf_names: dict[int, str] = {}
    if tag_method_id is not None and risk_factor_map:
        rf_names = fetch_risk_factor_names(params)
        n_risk_factors = len(rf_names)

    print(f"\nPhase 3: sampling {args.n:,} articles (pos-ratio={args.pos_ratio} -> "
          f"{n_want_relevant:,} relevant [country/year-stratified"
          f"{', risk-factor-diversity priority' if risk_factor_map else ''}], "
          f"{args.n - n_want_relevant:,} not relevant [uniform random])...")
    selected = sample_articles(
        candidates, relevance_map, args.n, args.pos_ratio, args.seed, uri_year,
        risk_factor_map=risk_factor_map if tag_method_id is not None else None,
    )
    n_sel_relevant = sum(1 for m in selected if m["is_relevant"])
    n_sel_not_relevant = len(selected) - n_sel_relevant
    print(f"  selected {len(selected):,} articles ({n_sel_relevant:,} relevant, {n_sel_not_relevant:,} not relevant) "
          f"across {len({m['adm0_code'] for m in selected if m['adm0_code'] is not None}):,} countries")

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

    print(f"\nPhase 4: fetching content for {len(selected):,} articles "
          f"(writing to {args.output} as each chunk completes)...")
    written, missing, skipped, written_uris = fetch_and_write_content(
        params, args.output, selected, candidates, country_names, rf_names, args.language,
        args.chunk_size, already_written, existing_text,
    )
    print_summary(
        selected, candidates, written_uris, country_names, written, missing, skipped, args.output,
        args.relevance_version, n_risk_factors,
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
