#!/usr/bin/env python3
"""
Final group-first stratified teacher selection from the already-built
authoritative candidate-pair table.

This script does NOT rescan the corpus for retrieval and does NOT rebuild
relevance or geography. It consumes the authoritative candidate pairs.

Allocation hierarchy:
    broader event group -> event type -> country -> article

Budget is data-derived rather than hand-picked:
    per_group_target = reference_per_country * observed_number_of_countries

With the current census:
    63 * 81 = 5,103 articles per broader group
    5,103 * 9 groups = 45,927 total target

Within each level:
  - allocate approximately evenly, capped by true availability
  - strict candidates before fallback
  - globally deduplicate article IDs
  - rarest geography strata claim articles first
  - redistribute shortfalls:
        same event type / other countries
        -> same broader group / other event types
        -> other under-filled groups
  - publication year is diagnostic only, not a hard stratum
"""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    broadcast,
    col,
    countDistinct,
    desc,
    lit,
    row_number,
    when,
    xxhash64,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
)

DEFAULT_ARTICLES_GLOB = "gs://zhai-infra-db/article_downloads/2025-Dec-17/*.parquet"
DEFAULT_OUTPUT_ROOT = "gs://test-zhai-texts"


def allocate_even_with_caps(total, capacities):
    capacities = {k: max(0, int(v)) for k, v in capacities.items()}
    q = {k: 0 for k in capacities}
    remaining = min(int(total), sum(capacities.values()))
    active = {k for k, cap in capacities.items() if cap > 0}

    while remaining > 0 and active:
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-pairs-path", required=True)
    ap.add_argument("--articles-glob", default=DEFAULT_ARTICLES_GLOB)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument(
        "--reference-per-country",
        type=int,
        default=63,
        help="Data-derived reference count used to derive each broader-group target."
    )
    ap.add_argument(
        "--oversample",
        type=int,
        default=20,
        help="Candidate reserve multiplier retained before exact global dedup."
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.reference_per_country < 1:
        raise SystemExit("--reference-per-country must be >= 1")
    if args.oversample < 1:
        raise SystemExit("--oversample must be >= 1")

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

    run_id = str(uuid.uuid4())
    root = args.output_root.rstrip("/")
    assignments_path = f"{root}/teacher_group_balanced_assignments_{run_id}/"
    manifest_path = f"{root}/teacher_group_balanced_manifest_{run_id}/"
    diagnostics_path = f"{root}/teacher_group_balanced_diagnostics_{run_id}/"

    pairs = (
        spark.read.parquet(args.candidate_pairs_path)
        .select(
            "sampling_group",
            "event_type",
            "country",
            "pub_year",
            "pool",
            "uri",
            "matched_keyword",
        )
    )

    # Preserve strict when duplicate evidence puts the same article in the same stratum.
    dedup_w = Window.partitionBy(
        "sampling_group", "event_type", "country", "uri"
    ).orderBy(
        when(col("pool") == "strict", lit(0)).otherwise(lit(1)),
        col("matched_keyword"),
    )

    pairs = (
        pairs.withColumn("_d", row_number().over(dedup_w))
             .filter(col("_d") == 1)
             .drop("_d")
             .cache()
    )

    # -------------------------------
    # 1. Data-derived group target
    # -------------------------------
    n_countries = pairs.select("country").distinct().count()
    n_groups = pairs.select("sampling_group").distinct().count()

    per_group_target = args.reference_per_country * n_countries
    requested_total = per_group_target * n_groups

    print(f"[allocation] observed countries: {n_countries}")
    print(f"[allocation] observed groups:    {n_groups}")
    print(f"[allocation] reference/country: {args.reference_per_country}")
    print(f"[allocation] per-group target:  {per_group_target:,}")
    print(f"[allocation] total target:      {requested_total:,}")

    group_caps_rows = (
        pairs.groupBy("sampling_group")
             .agg(countDistinct("uri").alias("n"))
             .collect()
    )
    group_caps = {r["sampling_group"]: int(r["n"]) for r in group_caps_rows}

    # Equal group starting point, capped by availability.
    initial_group_quotas = {
        g: min(per_group_target, cap) for g, cap in group_caps.items()
    }

    # If a whole group cannot fill its data-derived target, redistribute the
    # remainder across other groups that still have capacity.
    allocated_initial = sum(initial_group_quotas.values())
    remainder = min(requested_total, sum(group_caps.values())) - allocated_initial

    group_quotas = dict(initial_group_quotas)
    if remainder > 0:
        spare_caps = {
            g: max(0, group_caps[g] - group_quotas[g])
            for g in group_caps
        }
        extra = allocate_even_with_caps(remainder, spare_caps)
        for g, q in extra.items():
            group_quotas[g] += q

    final_total_target = sum(group_quotas.values())

    # -------------------------------
    # 2. Event-type quotas in groups
    # -------------------------------
    type_caps_rows = (
        pairs.groupBy("sampling_group", "event_type")
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

    # -------------------------------
    # 3. Geography quotas in types
    # -------------------------------
    geo_caps_rows = (
        pairs.groupBy("sampling_group", "event_type", "country")
             .agg(countDistinct("uri").alias("n"))
             .collect()
    )
    geo_caps = defaultdict(dict)
    for r in geo_caps_rows:
        geo_caps[(r["sampling_group"], r["event_type"])][r["country"]] = int(r["n"])

    geo_quotas = {}
    for key2, tq in type_quotas.items():
        q = allocate_even_with_caps(tq, geo_caps.get(key2, {}))
        for country, n in q.items():
            if n > 0:
                geo_quotas[(key2[0], key2[1], country)] = n

    quota_rows = [
        (
            g, et, c,
            int(q),
            int(geo_caps[(g, et)][c]),
        )
        for (g, et, c), q in geo_quotas.items()
        if q > 0
    ]

    quota_schema = StructType([
        StructField("sampling_group", StringType(), False),
        StructField("event_type", StringType(), False),
        StructField("country", StringType(), False),
        StructField("quota", IntegerType(), False),
        StructField("candidate_count", LongType(), False),
    ])

    quota_df = spark.createDataFrame(quota_rows, schema=quota_schema)

    # -------------------------------
    # 4. Oversized strict-first pools
    # -------------------------------
    ranked_w = Window.partitionBy(
        "sampling_group", "event_type", "country"
    ).orderBy(
        when(col("pool") == "strict", lit(0)).otherwise(lit(1)),
        xxhash64(col("uri"), lit(args.seed)),
        col("matched_keyword"),
    )

    oversized = (
        pairs.join(
            quota_df,
            ["sampling_group", "event_type", "country"],
            "inner"
        )
        .withColumn("_rank", row_number().over(ranked_w))
        .filter(
            col("_rank")
            <= when(
                col("candidate_count") < col("quota") * lit(args.oversample),
                col("candidate_count")
            ).otherwise(col("quota") * lit(args.oversample))
        )
        .select(
            "sampling_group", "event_type", "country",
            "pub_year", "pool", "uri", "matched_keyword",
            "quota", "candidate_count", "_rank"
        )
    )

    print("[selection] collecting oversized candidate reserve...")
    rows = oversized.collect()
    print(f"[selection] candidate reserve rows: {len(rows):,}")

    by_stratum = defaultdict(list)
    for r in rows:
        key3 = (r["sampling_group"], r["event_type"], r["country"])
        by_stratum[key3].append(r)

    for key3 in by_stratum:
        by_stratum[key3].sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["_rank"]),
            )
        )

    # -------------------------------
    # 5. Exact global dedup selection
    # -------------------------------
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
            uri = str(r["uri"])
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

    # Geography pass: sparse strata claim articles first.
    strata_order = sorted(
        geo_quotas,
        key=lambda k: (
            geo_caps[k[:2]].get(k[2], 0),
            k[0], k[1], k[2],
        )
    )

    for key3 in strata_order:
        need = geo_quotas[key3] - selected_by_stratum[key3]
        if need > 0:
            take_from_rows(by_stratum.get(key3, []), need)

    # Redistribution 1: same event type, other countries.
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
                int(r["_rank"]),
            )
        )
        take_from_rows(candidates, deficit)

    # Redistribution 2: same broader group, other event types.
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
                int(r["_rank"]),
                r["event_type"],
                r["country"],
            )
        )
        take_from_rows(candidates, deficit)

    # Redistribution 3: other under-filled groups.
    global_deficit = final_total_target - len(selected)
    if global_deficit > 0:
        all_candidates = list(rows)
        all_candidates.sort(
            key=lambda r: (
                0 if r["pool"] == "strict" else 1,
                int(r["candidate_count"]),
                int(r["_rank"]),
                r["sampling_group"],
                r["event_type"],
                r["country"],
            )
        )
        take_from_rows(all_candidates, global_deficit)

    selected_records = [
        (
            r["sampling_group"],
            r["event_type"],
            r["country"],
            int(r["pub_year"]) if r["pub_year"] is not None else None,
            r["pool"],
            str(r["uri"]),
            r["matched_keyword"],
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

    selected_df = spark.createDataFrame(selected_records, schema=selected_schema).cache()

    print("[assignments] writing:", assignments_path)
    selected_df.write.mode("overwrite").parquet(assignments_path)

    # -------------------------------
    # 6. Gemini-ready manifest
    # -------------------------------
    articles = spark.read.parquet(args.articles_glob)
    wanted = [
        "uri", "title", "body", "published_at",
        "source_uri", "cloud_uri", "article_type", "language",
    ]
    wanted = [c for c in wanted if c in articles.columns]

    bodies = articles.select(*wanted).dropDuplicates(["uri"])

    manifest = (
        bodies.join(
            broadcast(selected_df),
            bodies.uri == selected_df.article_id,
            "inner",
        )
        .drop(bodies.uri)
    )

    print("[manifest] writing:", manifest_path)
    manifest.write.mode("overwrite").parquet(manifest_path)

    # -------------------------------
    # 7. Diagnostics
    # -------------------------------
    group_targets_df = spark.createDataFrame(
        [
            (g, int(group_caps[g]), int(group_quotas[g]), int(selected_by_group[g]))
            for g in sorted(group_caps)
        ],
        schema="""
            sampling_group string,
            candidate_articles long,
            target_articles long,
            selected_articles long
        """
    )
    (
        group_targets_df.orderBy("sampling_group")
                        .coalesce(1)
                        .write.mode("overwrite").option("header", "true")
                        .csv(diagnostics_path + "by_group/")
    )

    type_targets_df = spark.createDataFrame(
        [
            (
                g, et,
                int(type_caps[g][et]),
                int(type_quotas[(g, et)]),
                int(selected_by_type[(g, et)]),
            )
            for (g, et) in sorted(type_quotas)
        ],
        schema="""
            sampling_group string,
            sampling_event_type string,
            candidate_articles long,
            target_articles long,
            selected_articles long
        """
    )
    (
        type_targets_df.orderBy("sampling_group", "sampling_event_type")
                       .coalesce(1)
                       .write.mode("overwrite").option("header", "true")
                       .csv(diagnostics_path + "by_event_type/")
    )

    geo_diag_rows = [
        (
            g, et, c,
            int(geo_caps[(g, et)][c]),
            int(geo_quotas[(g, et, c)]),
            int(selected_by_stratum[(g, et, c)]),
            int(max(0, geo_quotas[(g, et, c)] - selected_by_stratum[(g, et, c)])),
        )
        for (g, et, c) in sorted(geo_quotas)
    ]
    geo_diag_df = spark.createDataFrame(
        geo_diag_rows,
        schema="""
            sampling_group string,
            sampling_event_type string,
            geography_field string,
            candidate_articles long,
            target_articles long,
            selected_articles long,
            initial_geo_shortfall long
        """
    )
    (
        geo_diag_df.orderBy(
            "sampling_group", "sampling_event_type", "geography_field"
        )
        .coalesce(1)
        .write.mode("overwrite").option("header", "true")
        .csv(diagnostics_path + "by_event_type_geography/")
    )

    (
        selected_df.groupBy("sampling_group", "sampling_event_type", "pool")
                   .agg(countDistinct("article_id").alias("selected_articles"))
                   .orderBy("sampling_group", "sampling_event_type", "pool")
                   .coalesce(1)
                   .write.mode("overwrite").option("header", "true")
                   .csv(diagnostics_path + "strict_fallback/")
    )

    print()
    print("DONE: final group-first teacher selection")
    print(f"observed countries:             {n_countries}")
    print(f"broader groups:                 {n_groups}")
    print(f"reference per country:          {args.reference_per_country}")
    print(f"derived target per group:       {per_group_target:,}")
    print(f"derived total target:           {final_total_target:,}")
    print(f"selected unique articles:       {len(selected):,}")
    print(f"remaining global shortfall:     {max(0, final_total_target - len(selected)):,}")
    print(f"assignments:                    {assignments_path}")
    print(f"manifest:                       {manifest_path}")
    print(f"diagnostics:                    {diagnostics_path}")


if __name__ == "__main__":
    main()
