#!/usr/bin/env python3
"""
Document-aligned stratified article sampling for the new Gemini Pro teacher dataset.

Hierarchy:
    broader event group -> event type -> geography -> article

Reuses Piri's scalable PySpark candidate retrieval, but replaces the fixed
event_type x country x publication_year quota with:
  1) approximately even group budgets, capped by availability
  2) event-type allocation within group, with rare types protected by caps
  3) geography allocation within event type
  4) strict candidates before fallback candidates
  5) global article-ID deduplication
  6) sparse-quota redistribution:
       same event type / other geographies
       -> same broader group / other event types
       -> other under-filled groups
  7) publication year retained for diagnostics only, NOT a hard stratum

IMPORTANT:
- For final document-compliant production, pass --relevant-uris-path if you have
  an exported relevant-article manifest.
- For final document-compliant geography, pass --geo-map-path if you have the
  authoritative pre-existing article->country mapping. Without it, this script
  falls back to Piri's geotaxonomy-name text matching and prints a warning.
"""

import argparse
import json
import math
import re
import unicodedata
import uuid
from collections import defaultdict
from urllib.parse import urlparse

from google.cloud import storage
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, countDistinct, from_unixtime, length, lit, row_number, when, xxhash64, year
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)
from flashtext2 import KeywordProcessor


DEFAULT_ARTICLES_GLOB = "gs://zhai-infra-db/article_downloads/2025-Dec-17/*.parquet"
DEFAULT_CONDITIONS_GCS = "gs://test-zhai-texts/scripts/retrieval_conditions.json"
DEFAULT_GEO_NAMES_GCS = "gs://test-zhai-texts/scripts/geotaxonomy_names.json"
DEFAULT_OUTPUT_ROOT = "gs://test-zhai-texts"

# Macro-group names used in the project's v4 taxonomy work.
EVENT_GROUP = {
    # Weather and natural hazards
    "drought": "weather and natural hazards",
    "flooding": "weather and natural hazards",
    "abnormal rainfall": "weather and natural hazards",
    "water shortage": "weather and natural hazards",
    "pest infestation": "weather and natural hazards",
    "cyclone": "weather and natural hazards",

    # Food insecurity
    "food scarcity": "food insecurity",
    "famine": "food insecurity",
    "crop failure": "food insecurity",

    # Public health
    "disease spread": "public health",

    # Economic stress
    "economic decline": "economic stress",
    "poverty": "economic stress",
    "food price inflation": "economic stress",
    "trade disruption": "economic stress",

    # Environmental degradation
    "environmental degradation": "environment",
    "land degradation": "environment",

    # Displacement and migration
    "internal displacement": "displacement and migration",
    "refugee movement": "displacement and migration",
    "migrant influx": "displacement and migration",
    "shelter loss": "displacement and migration",

    # Humanitarian disruption
    "aid obstruction": "humanitarian disruption",
    "funding shortfall": "humanitarian disruption",

    # Political instability
    "governance breakdown": "political instability",

    # Conflict and security
    "armed violence": "conflict and security",
    "militant activity": "conflict and security",
    "aerial bombardment": "conflict and security",
    "looting": "conflict and security",
    "human rights violation": "conflict and security",
    "supply route blockade": "conflict and security",
    "infrastructure destruction": "conflict and security",
}

_WS = re.compile(r"[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]")


def _clean(s):
    return _WS.sub(" ", unicodedata.normalize("NFKC", s))


def parse_gcs_uri(uri):
    p = urlparse(uri)
    if p.scheme != "gs" or not p.netloc:
        raise ValueError(f"Expected gs:// URI, got {uri}")
    return p.netloc, p.path.lstrip("/")


def read_gcs_json(uri):
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    return json.loads(client.bucket(bucket_name).blob(blob_name).download_as_text())


def allocate_even_with_caps(total, capacities):
    """
    Allocate approximately evenly across keys, never exceeding capacity.
    Unused quota is repeatedly redistributed among keys that can still absorb it.
    """
    capacities = {k: max(0, int(v)) for k, v in capacities.items()}
    q = {k: 0 for k in capacities}
    remaining = min(int(total), sum(capacities.values()))
    active = {k for k, cap in capacities.items() if cap > 0}

    while remaining > 0 and active:
        # Equal-share starting point. At least one slot per active key per round.
        share = max(1, remaining // len(active))
        progressed = False
        for k in sorted(active, key=lambda x: (capacities[x], str(x))):
            room = capacities[k] - q[k]
            if room <= 0:
                continue
            add = min(share, room, remaining)
            if add > 0:
                q[k] += add
                remaining -= add
                progressed = True
            if remaining == 0:
                break
        active = {k for k in active if q[k] < capacities[k]}
        if not progressed:
            break
    return q


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles-glob", default=DEFAULT_ARTICLES_GLOB)
    ap.add_argument("--conditions-gcs", default=DEFAULT_CONDITIONS_GCS)
    ap.add_argument("--geo-names-gcs", default=DEFAULT_GEO_NAMES_GCS)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--total-budget", type=int, default=50000)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--language", default="eng")
    ap.add_argument("--min-year", type=int, default=2013)
    ap.add_argument("--min-body-chars", type=int, default=200)

    # Optional upstream filters needed for full document compliance.
    ap.add_argument(
        "--relevant-uris-path", default=None,
        help="Optional Spark-readable path with a 'uri' column containing relevant articles."
    )
    ap.add_argument(
        "--relevant-format", default="parquet",
        choices=("parquet", "json", "csv")
    )
    ap.add_argument(
        "--geo-map-path", default=None,
        help="Optional Spark-readable article->country table with columns uri,country."
    )
    ap.add_argument(
        "--geo-map-format", default="parquet",
        choices=("parquet", "json", "csv")
    )
    return ap


def main():
    args = make_parser().parse_args()
    if args.total_budget < 1:
        raise SystemExit("--total-budget must be >= 1")
    if args.oversample < 1:
        raise SystemExit("--oversample must be >= 1")

    run_id = str(uuid.uuid4())
    pairs_path = f"{args.output_root.rstrip('/')}/teacher_pairs_{run_id}/"
    diagnostics_path = f"{args.output_root.rstrip('/')}/teacher_sampling_diagnostics_{run_id}/"
    manifest_path = f"{args.output_root.rstrip('/')}/teacher_manifest_{run_id}/"

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

    conditions = read_gcs_json(args.conditions_gcs)

    cond_by_type = {c["event_type"]: c for c in conditions["conditions"]}
    missing_groups = sorted(set(cond_by_type) - set(EVENT_GROUP))
    if missing_groups:
        raise RuntimeError(
            "No broader event-group mapping for: " + ", ".join(missing_groups)
        )

    # Map positive term -> one or more event types (do not silently overwrite collisions).
    term_to_types = defaultdict(list)
    for c in conditions["conditions"]:
        et = c["event_type"]
        for term in c["positive_terms"]:
            term_to_types[_clean(term).lower()].append(et)

    apply_excl = {
        c["event_type"]: bool(c["apply_conflict_exclusion"])
        for c in conditions["conditions"]
    }
    conflict_terms = [_clean(t).lower() for t in conditions["conflict_exclusion_terms"]]

    bc_term_to_types = spark.sparkContext.broadcast(dict(term_to_types))
    bc_apply_excl = spark.sparkContext.broadcast(apply_excl)
    bc_conflict = spark.sparkContext.broadcast(conflict_terms)
    bc_groups = spark.sparkContext.broadcast(EVENT_GROUP)

    if args.geo_map_path:
        use_text_geo = False
        print("[geo] using authoritative external uri->country map:", args.geo_map_path)
        geo_names = []
    else:
        use_text_geo = True
        print(
            "[geo] WARNING: no --geo-map-path supplied; using Piri's text-based "
            "geotaxonomy-name matching for this run."
        )
        geo_names = [_clean(x).lower() for x in read_gcs_json(args.geo_names_gcs)]
    bc_geo_names = spark.sparkContext.broadcast(geo_names)

    articles = spark.read.parquet(args.articles_glob)

    # Keep fields needed by the manifest and diagnostics.
    keep_cols = [
        "uri", "body", "published_at", "language", "title", "source_uri"
    ]
    available = set(articles.columns)
    keep_cols = [c for c in keep_cols if c in available]

    if "uri" not in keep_cols or "body" not in keep_cols or "published_at" not in keep_cols:
        raise RuntimeError("Article parquet must contain uri, body, and published_at.")

    df = articles.select(*keep_cols)

    if "language" in df.columns:
        df = df.filter(col("language") == lit(args.language))
    else:
        print("[language] WARNING: parquet has no language field; language filter not applied.")

    df = (
        df.withColumn("pub_year", year(from_unixtime(col("published_at") / 1_000_000_000)))
          .filter(col("pub_year") >= args.min_year)
          .filter(col("body").isNotNull() & (length(col("body")) >= args.min_body_chars))
    )

    # Document says to start from existing relevant articles.
    if args.relevant_uris_path:
        print("[relevance] filtering against:", args.relevant_uris_path)
        reader = spark.read.format(args.relevant_format)
        if args.relevant_format == "csv":
            reader = reader.option("header", "true")
        rel = reader.load(args.relevant_uris_path).select("uri").dropDuplicates(["uri"])
        df = df.join(rel, "uri", "inner")
    else:
        print(
            "[relevance] WARNING: no --relevant-uris-path supplied. "
            "This run validates stratification/retrieval, but is NOT the final "
            "document-compliant teacher manifest."
        )

    df = df.repartition(4000)

    if use_text_geo:
        schema = StructType([
            StructField("sampling_group", StringType(), False),
            StructField("event_type", StringType(), False),
            StructField("country", StringType(), False),
            StructField("pub_year", IntegerType(), True),
            StructField("pool", StringType(), False),
            StructField("uri", StringType(), False),
            StructField("matched_keyword", StringType(), False),
        ])

        def process_partition(rows):
            kp_event = KeywordProcessor(case_sensitive=False)
            for t in bc_term_to_types.value:
                kp_event.add_keyword(t)

            kp_geo = KeywordProcessor(case_sensitive=False)
            for s in bc_geo_names.value:
                kp_geo.add_keyword(s)

            kp_conflict = KeywordProcessor(case_sensitive=False)
            for s in bc_conflict.value:
                kp_conflict.add_keyword(s)

            for row in rows:
                text = row["body"]
                if not text:
                    continue
                text = _clean(text)
                try:
                    event_terms = sorted(set(kp_event.extract_keywords(text)))
                    if not event_terms:
                        continue
                    geos = sorted(set(kp_geo.extract_keywords(text)))
                    if not geos:
                        continue
                    has_conflict = bool(kp_conflict.extract_keywords(text))
                except BaseException:
                    continue

                by_event = defaultdict(list)
                for term in event_terms:
                    for et in bc_term_to_types.value.get(term, []):
                        by_event[et].append(term)

                for et, terms in by_event.items():
                    pool = (
                        "fallback"
                        if bc_apply_excl.value[et] and has_conflict
                        else "strict"
                    )
                    matched_keyword = sorted(terms)[0]
                    group = bc_groups.value[et]
                    for country in geos:
                        yield (
                            group, et, country, row["pub_year"],
                            pool, str(row["uri"]), matched_keyword
                        )

        pairs_df = spark.createDataFrame(
            df.select("body", "uri", "pub_year").rdd.mapPartitions(process_partition),
            schema=schema
        )
    else:
        event_schema = StructType([
            StructField("sampling_group", StringType(), False),
            StructField("event_type", StringType(), False),
            StructField("pub_year", IntegerType(), True),
            StructField("pool", StringType(), False),
            StructField("uri", StringType(), False),
            StructField("matched_keyword", StringType(), False),
        ])

        def process_partition_no_geo(rows):
            kp_event = KeywordProcessor(case_sensitive=False)
            for t in bc_term_to_types.value:
                kp_event.add_keyword(t)
            kp_conflict = KeywordProcessor(case_sensitive=False)
            for s in bc_conflict.value:
                kp_conflict.add_keyword(s)

            for row in rows:
                text = row["body"]
                if not text:
                    continue
                text = _clean(text)
                try:
                    event_terms = sorted(set(kp_event.extract_keywords(text)))
                    if not event_terms:
                        continue
                    has_conflict = bool(kp_conflict.extract_keywords(text))
                except BaseException:
                    continue

                by_event = defaultdict(list)
                for term in event_terms:
                    for et in bc_term_to_types.value.get(term, []):
                        by_event[et].append(term)

                for et, terms in by_event.items():
                    pool = (
                        "fallback"
                        if bc_apply_excl.value[et] and has_conflict
                        else "strict"
                    )
                    yield (
                        bc_groups.value[et], et, row["pub_year"], pool,
                        str(row["uri"]), sorted(terms)[0]
                    )

        event_df = spark.createDataFrame(
            df.select("body", "uri", "pub_year").rdd.mapPartitions(process_partition_no_geo),
            schema=event_schema
        )

        reader = spark.read.format(args.geo_map_format)
        if args.geo_map_format == "csv":
            reader = reader.option("header", "true")
        geo = (
            reader.load(args.geo_map_path)
            .select("uri", "country")
            .filter(col("country").isNotNull())
            .dropDuplicates(["uri", "country"])
        )
        pairs_df = event_df.join(geo, "uri", "inner").select(
            "sampling_group", "event_type", "country", "pub_year",
            "pool", "uri", "matched_keyword"
        )

    # One row per article/stratum; preserve strict if duplicate evidence yields both.
    dedup_w = Window.partitionBy(
        "sampling_group", "event_type", "country", "uri"
    ).orderBy(
        when(col("pool") == "strict", 0).otherwise(1),
        col("matched_keyword")
    )
    pairs_df = (
        pairs_df.withColumn("_d", row_number().over(dedup_w))
                .filter(col("_d") == 1)
                .drop("_d")
                .cache()
    )

    print("[pairs] writing candidate pairs:", pairs_path)
    pairs_df.write.mode("overwrite").parquet(pairs_path)

    # Availability at each level. These are small enough to collect.
    group_caps_rows = (
        pairs_df.groupBy("sampling_group")
                .agg(countDistinct("uri").alias("n"))
                .collect()
    )
    group_caps = {r["sampling_group"]: int(r["n"]) for r in group_caps_rows}
    group_quotas = allocate_even_with_caps(args.total_budget, group_caps)

    type_caps_rows = (
        pairs_df.groupBy("sampling_group", "event_type")
                .agg(countDistinct("uri").alias("n"))
                .collect()
    )
    type_caps = defaultdict(dict)
    for r in type_caps_rows:
        type_caps[r["sampling_group"]][r["event_type"]] = int(r["n"])

    type_quotas = {}
    for group, gq in group_quotas.items():
        q = allocate_even_with_caps(gq, type_caps.get(group, {}))
        for et, n in q.items():
            type_quotas[(group, et)] = n

    geo_caps_rows = (
        pairs_df.groupBy("sampling_group", "event_type", "country")
                .agg(countDistinct("uri").alias("n"))
                .collect()
    )
    geo_caps = defaultdict(dict)
    for r in geo_caps_rows:
        geo_caps[(r["sampling_group"], r["event_type"])][r["country"]] = int(r["n"])

    geo_quotas = {}
    for key, tq in type_quotas.items():
        q = allocate_even_with_caps(tq, geo_caps.get(key, {}))
        for country, n in q.items():
            geo_quotas[(key[0], key[1], country)] = n

    quota_rows = [
        (g, et, c, int(q), int(geo_caps[(g, et)][c]))
        for (g, et, c), q in geo_quotas.items()
        if q > 0
    ]
    quota_schema = StructType([
        StructField("sampling_group", StringType(), False),
        StructField("event_type", StringType(), False),
        StructField("country", StringType(), False),
        StructField("quota", IntegerType(), False),
        StructField("candidate_count", IntegerType(), False),
    ])
    quota_df = spark.createDataFrame(quota_rows, schema=quota_schema)

    # Build only an oversized candidate pool around the final quotas.
    ranked_w = Window.partitionBy(
        "sampling_group", "event_type", "country"
    ).orderBy(
        when(col("pool") == "strict", 0).otherwise(1),
        xxhash64(col("uri"), lit(args.seed))
    )

    oversized = (
        pairs_df.join(
            quota_df,
            ["sampling_group", "event_type", "country"],
            "inner"
        )
        .withColumn("_rank", row_number().over(ranked_w))
        .filter(col("_rank") <= col("quota") * lit(args.oversample))
        .select(
            "sampling_group", "event_type", "country", "pub_year",
            "pool", "uri", "matched_keyword", "quota",
            "candidate_count", "_rank"
        )
    )

    # <= total_budget * oversample rows in the normal case, safe for driver-side
    # exact global dedup + redistribution.
    rows = oversized.collect()

    by_stratum = defaultdict(list)
    for r in rows:
        key = (r["sampling_group"], r["event_type"], r["country"])
        by_stratum[key].append(r)

    # Already ranked strict-first in Spark, but make ordering explicit.
    for key in by_stratum:
        by_stratum[key].sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["_rank"])
            )
        )

    selected_ids = set()
    selected = []
    selected_by_group = defaultdict(int)
    selected_by_type = defaultdict(int)
    selected_by_stratum = defaultdict(int)

    def take_from_rows(candidate_rows, n):
        taken = 0
        for r in candidate_rows:
            if taken >= n:
                break
            uri = r["uri"]
            if uri in selected_ids:
                continue
            selected_ids.add(uri)
            selected.append(r)
            k3 = (r["sampling_group"], r["event_type"], r["country"])
            k2 = (r["sampling_group"], r["event_type"])
            selected_by_stratum[k3] += 1
            selected_by_type[k2] += 1
            selected_by_group[r["sampling_group"]] += 1
            taken += 1
        return taken

    # First pass: exact geography quotas. Rarest strata claim articles first.
    strata_order = sorted(
        geo_quotas,
        key=lambda k: (geo_caps[k[:2]].get(k[2], 0), k[0], k[1], k[2])
    )
    for key in strata_order:
        need = geo_quotas[key] - selected_by_stratum[key]
        if need > 0:
            take_from_rows(by_stratum.get(key, []), need)

    # Redistribution 1: same event type / other geographies.
    for key2, target in sorted(type_quotas.items()):
        deficit = target - selected_by_type[key2]
        if deficit <= 0:
            continue
        candidates = []
        for key3, rr in by_stratum.items():
            if key3[:2] == key2:
                candidates.extend(rr)
        candidates.sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["candidate_count"]),
                int(r["_rank"])
            )
        )
        take_from_rows(candidates, deficit)

    # Redistribution 2: same broader group / other event types.
    for group, target in sorted(group_quotas.items()):
        deficit = target - selected_by_group[group]
        if deficit <= 0:
            continue
        candidates = []
        for key3, rr in by_stratum.items():
            if key3[0] == group:
                candidates.extend(rr)
        candidates.sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["candidate_count"]),
                int(r["_rank"])
            )
        )
        take_from_rows(candidates, deficit)

    # Redistribution 3: other under-filled groups / global remainder.
    global_deficit = min(args.total_budget, sum(group_caps.values())) - len(selected)
    if global_deficit > 0:
        all_candidates = list(rows)
        all_candidates.sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["candidate_count"]),
                int(r["_rank"]),
                r["sampling_group"],
                r["event_type"],
                r["country"]
            )
        )
        take_from_rows(all_candidates, global_deficit)

    selected_records = [
        (
            r["sampling_group"], r["event_type"], r["country"],
            int(r["pub_year"]) if r["pub_year"] is not None else None,
            r["pool"], r["uri"], r["matched_keyword"]
        )
        for r in selected
    ]

    selected_schema = StructType([
        StructField("sampling_group", StringType(), False),
        StructField("sampling_event_type", StringType(), False),
        StructField("geography_field", StringType(), False),
        StructField("pub_year", IntegerType(), True),
        StructField("pool", StringType(), False),
        StructField("article_id", StringType(), False),
        StructField("matched_keyword", StringType(), False),
    ])
    selected_df = spark.createDataFrame(selected_records, schema=selected_schema)

    # Join article metadata/body for Gemini handoff.
    meta_cols = ["uri", "body", "published_at", "language", "title", "source_uri"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    bodies = df.select(*meta_cols).dropDuplicates(["uri"])

    manifest = (
        selected_df.join(
            bodies,
            selected_df.article_id == bodies.uri,
            "inner"
        )
        .drop("uri")
    )

    print("[manifest] writing:", manifest_path)
    manifest.write.mode("overwrite").parquet(manifest_path)

    # Diagnostics required by the method document.
    diag_group = (
        manifest.groupBy("sampling_group")
                .agg(countDistinct("article_id").alias("selected_articles"))
                .withColumn("diagnostic_level", lit("group"))
    )
    diag_type = (
        manifest.groupBy("sampling_group", "sampling_event_type")
                .agg(countDistinct("article_id").alias("selected_articles"))
                .withColumn("diagnostic_level", lit("event_type"))
    )
    diag_geo = (
        manifest.groupBy("sampling_group", "sampling_event_type", "geography_field")
                .agg(countDistinct("article_id").alias("selected_articles"))
                .withColumn("diagnostic_level", lit("event_type_x_geography"))
    )

    # Write separate subfolders to avoid schema-union ambiguity.
    diag_group.write.mode("overwrite").parquet(diagnostics_path + "by_group/")
    diag_type.write.mode("overwrite").parquet(diagnostics_path + "by_event_type/")
    diag_geo.write.mode("overwrite").parquet(diagnostics_path + "by_event_type_geography/")

    print()
    print("DONE")
    print(f"candidate pairs: {pairs_path}")
    print(f"manifest:        {manifest_path}")
    print(f"diagnostics:     {diagnostics_path}")
    print(f"selected unique articles: {len(selected_records):,}")
    print(f"requested budget:         {args.total_budget:,}")
    if not args.relevant_uris_path:
        print("WARNING: relevance filter was not applied.")
    if not args.geo_map_path:
        print("WARNING: geography used text matching, not authoritative pre-existing metadata.")


if __name__ == "__main__":
    main()
