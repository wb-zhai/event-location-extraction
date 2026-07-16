"""Download a balanced stratified sample of articles across all countries and risk factors.

Two sets of articles are sampled:
  - Positives: geo-tagged articles with at least one risk factor tag, selected to
    maximise risk-factor diversity per country.
  - Negatives: geo-tagged articles with no risk factor tags, sampled per country.

"Geo-tagged" here means Event Registry's own location-concept tagging
(article_concept_association -> geo_taxonomy_concept_uris_direct_match -> geo_taxonomy),
not the custom string-matching pipeline (article_location_tags, tag_method_id) used in
from_db_matrix.py. Event Registry tags every ingested article with concepts as a side
effect of ingestion, so this covers articles the string-matcher never ran on or missed.
Risk-factor tags are unrelated to location and still come from the custom tagger
(article_risk_factor_tags, tag_method_id) — see the module-level comment above each
query for details on why each source was picked.

Sampling is done entirely on the indexed tag/concept tables; article content is fetched
in a final batch lookup, so the large article_downloads table is never scanned.

Usage:
    # Minimal — 50k articles, 70% positive, default seed
    python scripts/data/download/from_db_matrix.py \
        --n 50000 --output dataset/matrix_sample.jsonl

    # Full options — custom split, seed, and GCS output
    python scripts/data/download/from_db_matrix.py \
        --n 100000 --pos-ratio 0.6 --seed 7 \
        --output gs://my-bucket/data/matrix_sample.jsonl

    # Increase oversample if many countries show 0 positives in the summary
    python scripts/data/download/from_db_matrix.py \
        --n 50000 --oversample 6 --output dataset/matrix_sample.jsonl

    # Small test run to inspect output format before a large download
    python scripts/data/download/from_db_matrix.py \
        --n 200 --output /tmp/test_sample.jsonl

Requires psycopg2:
    pip install psycopg2-binary
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
import sys
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a balanced stratified sample of articles from the DB.",
    )
    parser.add_argument("--n", type=int, required=True,
        help="Total number of articles to download.")
    parser.add_argument("--pos-ratio", type=float, default=0.7,
        help="Fraction of articles with risk factor tags (default 0.7).")
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for deterministic hash-based sampling (default 42).")
    parser.add_argument("--output", "-o", required=True,
        help="Output JSONL path (local or gs://bucket/path).")
    parser.add_argument("--tag-method-id", type=int, default=1,
        help="tag_method_id filter for the risk-factor tagger (default 1). "
             "Geo-tagging no longer uses this — see module docstring.")
    parser.add_argument("--chunk-size", type=int, default=5_000,
        help="Max URIs per article_downloads batch fetch (default 5000).")
    parser.add_argument("--language", type=str, default="eng",
        help="Language filter for article_downloads (default eng).")
    parser.add_argument(
        "--oversample", type=int, default=3,
        help=(
            "Per-risk-factor oversample multiplier (default 3). "
            "Fetches oversample×n_pos/n_rf candidate articles per risk factor "
            "so every country can fill its quota despite uneven RF coverage. "
            "Increase if the summary shows many countries with 0 positives."
        ),
    )
    args = parser.parse_args()
    if not 0.0 < args.pos_ratio < 1.0:
        parser.error("--pos-ratio must be strictly between 0 and 1")
    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.oversample < 1:
        parser.error("--oversample must be at least 1")
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

# Phase 1a: all countries that have geo-tagged articles, sorted by article count
#
# Geo source is Event Registry's own concept tagging, not the string-matching
# article_location_tags table. Mirrors the "performant" pattern from jerome's Slack
# query: geo_concepts (adm_code -> concept_uri) is built first and stays tiny, so it's
# the driving side of the join into article_concept_association (the huge fact table)
# instead of joining that table before filtering — and concept_uris.concept_type = 'loc'
# is applied via an ON clause (INNER JOIN), not a LEFT JOIN + WHERE, so Postgres can
# push the 'loc' filter down before touching article rows.
# Two-level GROUP BY (adm_code, then adm0_code) keeps the aggregate small before the
# final join to geo_taxonomy for country names.
_COUNTRIES_SQL = """
WITH geo_concepts AS (
    SELECT gc.uri AS concept_uri, geo.adm_code
    FROM geo_taxonomy_concept_uris_direct_match gc
    JOIN geo_taxonomy geo ON geo.adm_code = gc.code
),
by_adm_code AS (
    SELECT gcp.adm_code, assoc.article_uri
    FROM article_concept_association assoc
    JOIN concept_uris concepts
        ON concepts.concept_uri = assoc.concept_uri
       AND concepts.concept_type = 'loc'
    JOIN geo_concepts gcp
        ON gcp.concept_uri = assoc.concept_uri
    GROUP BY gcp.adm_code, assoc.article_uri
),
by_country AS (
    SELECT geo.adm0_code, b.article_uri
    FROM by_adm_code b
    JOIN geo_taxonomy geo ON geo.adm_code = b.adm_code
    GROUP BY geo.adm0_code, b.article_uri
)
SELECT
    g0.adm_name  AS country_name,
    bc.adm0_code,
    COUNT(*)     AS article_count
FROM by_country bc
JOIN geo_taxonomy g0
    ON g0.adm0_code = bc.adm0_code AND g0.adm_level = 0
GROUP BY g0.adm_name, bc.adm0_code
ORDER BY article_count DESC
"""

# Phase 1b: number of distinct risk factors
_RF_COUNT_SQL = """
SELECT COUNT(DISTINCT risk_factor) AS n_rf
FROM article_risk_factor_tags
WHERE tag_method_id = %(tag_method_id)s
"""

# Phase 2: sample positive article URIs, RF-stratified then joined to location
#
# Strategy (adapts zhai.py's CROSS JOIN LATERAL pattern):
#   1. For each risk factor, pick per_rf_limit articles pseudo-randomly
#      using hashtext() — fast, deterministic, avoids ORDER BY random().
#   2. Join to Event Registry's concept tags (article_concept_association ->
#      geo_taxonomy_concept_uris_direct_match -> geo_taxonomy) to get adm0_code,
#      instead of the string-matching article_location_tags table — an RF-tagged
#      article no longer needs to also have been geocoded by the string matcher.
#   3. Aggregate all risk factors per (article_uri, adm0_code).
#   4. Return sorted by (adm0_code, rf_count DESC) so the Python balancer
#      can fill each country's quota greedily without sorting.
_POSITIVE_SAMPLE_SQL = """
WITH distinct_risk_factors AS (
    SELECT DISTINCT risk_factor
    FROM article_risk_factor_tags
    WHERE tag_method_id = %(tag_method_id)s
),
rf_sampled AS (
    SELECT rf.risk_factor, a.article_uri
    FROM distinct_risk_factors rf
    CROSS JOIN LATERAL (
        SELECT article_uri
        FROM article_risk_factor_tags
        WHERE tag_method_id = %(tag_method_id)s
          AND risk_factor = rf.risk_factor
        ORDER BY hashtext(article_uri || %(seed_str)s)
        LIMIT %(per_rf_limit)s
    ) a
),
with_location AS (
    SELECT
        rs.article_uri,
        geo.adm0_code,
        array_agg(DISTINCT rs.risk_factor ORDER BY rs.risk_factor) AS risk_factor_ids
    FROM rf_sampled rs
    JOIN article_concept_association assoc
        ON assoc.article_uri = rs.article_uri
    JOIN concept_uris concepts
        ON concepts.concept_uri = assoc.concept_uri
       AND concepts.concept_type = 'loc'
    JOIN geo_taxonomy_concept_uris_direct_match geo_concepts
        ON geo_concepts.uri = assoc.concept_uri
    JOIN geo_taxonomy geo
        ON geo.adm_code = geo_concepts.code
    GROUP BY rs.article_uri, geo.adm0_code
)
SELECT
    adm0_code,
    article_uri,
    risk_factor_ids,
    array_length(risk_factor_ids, 1) AS rf_count
FROM with_location
ORDER BY adm0_code, rf_count DESC
"""

# Phase 3: sample negative article URIs (geo-tagged, no risk factor tags)
#
# Geo source is Event Registry's concept tags (see _COUNTRIES_SQL comment) rather than
# article_location_tags — this pulls in articles the string-matcher never processed,
# which is exactly the gap jerome's Slack query was meant to fill. The risk-factor
# anti-join is untouched since that tagger is unrelated to geolocation.
# Per-country LATERAL with hashtext ordering for pseudo-random selection.
# NOT EXISTS anti-join excludes articles that have any risk factor tag.
_NEGATIVE_SAMPLE_SQL = """
WITH countries AS (
    SELECT unnest(%(adm0_codes)s::text[]) AS adm0_code
)
SELECT c.adm0_code, a.article_uri
FROM countries c
CROSS JOIN LATERAL (
    SELECT article_uri
    FROM (
        SELECT DISTINCT assoc.article_uri
        FROM article_concept_association assoc
        JOIN concept_uris concepts
            ON concepts.concept_uri = assoc.concept_uri
           AND concepts.concept_type = 'loc'
        JOIN geo_taxonomy_concept_uris_direct_match geo_concepts
            ON geo_concepts.uri = assoc.concept_uri
        JOIN geo_taxonomy geo
            ON geo.adm_code = geo_concepts.code
        WHERE geo.adm0_code = c.adm0_code
          AND NOT EXISTS (
              SELECT 1
              FROM article_risk_factor_tags arft
              WHERE arft.article_uri = assoc.article_uri
                AND arft.tag_method_id = %(tag_method_id)s
          )
    ) deduped
    ORDER BY hashtext(article_uri || %(seed_str)s)
    LIMIT %(neg_per_country)s
) a
"""

# Phase 4: batch-fetch article content by URI
_CONTENT_SQL = """
SELECT uri, cloud_uri, title, body, published_at, article_type, source_uri
FROM article_downloads
WHERE uri = ANY(%(uris)s)
  AND language = %(language)s
"""


# ── Sampling logic ────────────────────────────────────────────────────────────

def balance_positives(
    pos_rows: list[dict],
    adm0_codes: list[str],
    pos_per_country: int,
    n_pos: int,
) -> list[dict]:
    """Assign positives to countries greedily, favouring most-RF-diverse articles.

    Rows arrive sorted by (adm0_code, rf_count DESC) from SQL, so each country's
    sub-list is already in the right order — no re-sort needed.
    An article appearing in multiple countries is assigned to whichever country
    is processed first (highest article_count, per the discovery query order).
    """
    by_country: dict[str, list[dict]] = defaultdict(list)
    for row in pos_rows:
        by_country[row["adm0_code"]].append(row)

    seen_uris: set[str] = set()
    selected: list[dict] = []
    for adm0_code in adm0_codes:
        taken = 0
        for row in by_country.get(adm0_code, []):
            if taken >= pos_per_country:
                break
            uri = str(row["article_uri"])
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            selected.append({
                "article_uri": uri,
                "adm0_code": adm0_code,
                "risk_factor_ids": list(row["risk_factor_ids"] or []),
                "label": "positive",
            })
            taken += 1
    return selected[:n_pos]


def select_negatives(
    neg_rows: list[dict],
    pos_uris: set[str],
    n_neg: int,
) -> list[dict]:
    """Deduplicate negatives against already-selected positives and return up to n_neg."""
    seen: set[str] = set(pos_uris)
    selected: list[dict] = []
    for row in neg_rows:
        if len(selected) >= n_neg:
            break
        uri = str(row["article_uri"])
        if uri in seen:
            continue
        seen.add(uri)
        selected.append({
            "article_uri": uri,
            "adm0_code": row["adm0_code"],
            "risk_factor_ids": [],
            "label": "negative",
        })
    return selected


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

def build_record(meta: dict, content: dict, rf_names: dict[int, str]) -> dict:
    risk_factors = [rf_names.get(int(rf_id), str(rf_id)) for rf_id in meta["risk_factor_ids"]]
    return {
        "id": str(meta["article_uri"]),
        "label": meta["label"],
        "adm0_code": meta["adm0_code"],
        "risk_factors": risk_factors,
        "source": {
            "title":        str(content.get("title") or ""),
            "text":         str(content.get("body") or ""),
            "published_at": content.get("published_at"),
            "article_type": content.get("article_type"),
            "source_url":   content.get("source_uri"),
            "cloud_uri":    content.get("cloud_uri"),
        },
    }


def write_output(
    path_or_uri: str,
    all_selected: list[dict],
    content_map: dict[str, dict],
    rf_names: dict[int, str],
) -> tuple[int, int]:
    written = missing = 0
    total = len(all_selected)
    with open_output(path_or_uri) as f:
        for meta in all_selected:
            content = content_map.get(meta["article_uri"])
            if content is None:
                missing += 1
                continue
            f.write(
                json.dumps(
                    build_record(meta, content, rf_names),
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                ) + "\n"
            )
            written += 1
            if written % 1000 == 0:
                print(f"  written {written}/{total}\033[K", end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return written, missing


def print_summary(
    written: int,
    missing: int,
    selected_pos: list[dict],
    selected_neg: list[dict],
    content_map: dict[str, dict],
    n_countries: int,
    n_rf: int,
    output: str,
) -> None:
    present = {m["article_uri"] for m in selected_pos + selected_neg if m["article_uri"] in content_map}
    n_pos_written = sum(1 for m in selected_pos if m["article_uri"] in present)
    n_neg_written = sum(1 for m in selected_neg if m["article_uri"] in present)
    covered_countries = len({m["adm0_code"] for m in selected_pos + selected_neg if m["article_uri"] in present})
    covered_rf_ids: set[int] = set()
    for m in selected_pos:
        if m["article_uri"] in present:
            covered_rf_ids.update(int(x) for x in m["risk_factor_ids"])

    print("\n=== Matrix sample complete ===")
    print(f"Total articles written : {written:,}")
    print(f"  Positives            : {n_pos_written:,}")
    print(f"  Negatives            : {n_neg_written:,}")
    print(f"Countries covered      : {covered_countries:,} / {n_countries:,}")
    print(f"Risk factors covered   : {len(covered_rf_ids):,} / {n_rf:,}")
    if missing:
        print(f"Missing from DB        : {missing:,}")
    print(f"Output                 : {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    params = get_connection_params()

    print("Connecting to database...")
    try:
        conn = psycopg2.connect(**params)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)
    print("Connected.")

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── Phase 1: Discovery ─────────────────────────────────────────
            print("\nPhase 1: discovering countries and risk factors...")
            cur.execute(_COUNTRIES_SQL)
            country_rows = cur.fetchall()
            if not country_rows:
                print("No geo-tagged articles found. Nothing to download.")
                return

            cur.execute(_RF_COUNT_SQL, {"tag_method_id": args.tag_method_id})
            n_rf = int(cur.fetchone()["n_rf"])
            if n_rf == 0:
                print("No risk factor tags found. Cannot sample positives.")
                return

            # Load risk factor id → name for output
            cur.execute("SELECT id, name FROM risk_factors")
            rf_names: dict[int, str] = {int(r["id"]): r["name"] for r in cur.fetchall()}

            n_countries = len(country_rows)
            adm0_codes = [r["adm0_code"] for r in country_rows]

            n_pos = round(args.n * args.pos_ratio)
            n_neg = args.n - n_pos
            pos_per_country = math.ceil(n_pos / n_countries)
            neg_per_country = math.ceil(n_neg / n_countries)
            # Oversample so every country has enough candidates after the balancing step.
            # per_rf_limit × n_rf gives the total candidate pool size for positives.
            per_rf_limit = math.ceil(n_pos * args.oversample / n_rf)
            seed_str = str(args.seed)

            print(f"  {n_countries} countries, {n_rf} risk factors")
            print(f"  target: {args.n:,} total  ({n_pos:,} positive + {n_neg:,} negative)")
            print(f"  pos_per_country={pos_per_country}, neg_per_country={neg_per_country}")
            print(f"  per_rf_limit={per_rf_limit}  (oversample={args.oversample}×, "
                  f"candidate pool ≤{per_rf_limit * n_rf:,} rows before dedup)")

            # ── Phase 2: Sample positive URIs ──────────────────────────────
            print("\nPhase 2: sampling positive article URIs (RF-stratified)...")
            cur.execute(_POSITIVE_SAMPLE_SQL, {
                "tag_method_id": args.tag_method_id,
                "seed_str": seed_str,
                "per_rf_limit": per_rf_limit,
            })
            pos_rows = cur.fetchall()
            print(f"  got {len(pos_rows):,} candidate (uri, country) pairs")

            # ── Phase 3: Balance positives by country in Python ────────────
            print("Phase 3: balancing positives by country / RF diversity...")
            selected_pos = balance_positives(pos_rows, adm0_codes, pos_per_country, n_pos)
            covered_pos_countries = len({r["adm0_code"] for r in selected_pos})
            print(f"  selected {len(selected_pos):,} positives across "
                  f"{covered_pos_countries} / {n_countries} countries")

            # ── Phase 4: Sample negative URIs ──────────────────────────────
            print("\nPhase 4: sampling negative article URIs (per-country)...")
            cur.execute(_NEGATIVE_SAMPLE_SQL, {
                "adm0_codes": adm0_codes,
                "tag_method_id": args.tag_method_id,
                "seed_str": seed_str,
                "neg_per_country": neg_per_country,
            })
            neg_rows = cur.fetchall()
            pos_uri_set = {r["article_uri"] for r in selected_pos}
            selected_neg = select_negatives(neg_rows, pos_uri_set, n_neg)
            covered_neg_countries = len({r["adm0_code"] for r in selected_neg})
            print(f"  selected {len(selected_neg):,} negatives across "
                  f"{covered_neg_countries} / {n_countries} countries")

            # ── Phase 5: Batch-fetch article content ───────────────────────
            all_selected = selected_pos + selected_neg
            all_uris = [r["article_uri"] for r in all_selected]
            print(f"\nPhase 5: fetching content for {len(all_uris):,} articles...")
            content_map = fetch_content(cur, all_uris, args.chunk_size, args.language)
            print(f"  resolved {len(content_map):,} articles")

        # ── Phase 6: Write JSONL ───────────────────────────────────────────
        print(f"\nPhase 6: writing JSONL to {args.output} ...")
        written, missing = write_output(args.output, all_selected, content_map, rf_names)
        print_summary(written, missing, selected_pos, selected_neg, content_map,
                      n_countries, n_rf, args.output)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
