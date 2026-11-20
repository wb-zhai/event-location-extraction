#!/usr/bin/env python3
"""
Join pre-geo event candidates with an authoritative URI -> country map.

Inputs
------
--event-candidates-path
    Parquet written by prepare_geo_candidates_dataproc.py. Expected columns:
    sampling_group, event_type, pub_year, pool, uri, matched_keyword

--geo-map-path
    CSV or Parquet with columns: uri, country

Outputs
-------
Complete event_type x country candidate pairs and diagnostics.
No sampling budget is applied.
"""

import argparse
import uuid

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, countDistinct, row_number, when


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-candidates-path", required=True)
    ap.add_argument("--geo-map-path", required=True)
    ap.add_argument("--geo-map-format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--output-root", default="gs://test-zhai-texts")
    args = ap.parse_args()

    spark = SparkSession.builder.getOrCreate()

    run_id = str(uuid.uuid4())
    root = args.output_root.rstrip("/")
    pairs_path = f"{root}/authoritative_geo_candidate_pairs_{run_id}/"
    diagnostics_path = f"{root}/authoritative_geo_diagnostics_{run_id}/"

    events = (
        spark.read.parquet(args.event_candidates_path)
        .select(
            "sampling_group", "event_type", "pub_year",
            "pool", "uri", "matched_keyword"
        )
    )

    reader = spark.read.format(args.geo_map_format)
    if args.geo_map_format == "csv":
        reader = reader.option("header", "true")

    geo = (
        reader.load(args.geo_map_path)
        .select(
            col("uri").cast("string").alias("uri"),
            col("country").cast("string").alias("country"),
        )
        .filter(col("uri").isNotNull() & col("country").isNotNull())
        .dropDuplicates(["uri", "country"])
    )

    pairs = events.join(geo, "uri", "inner")

    # An article may have matched several positive terms for the same
    # event_type/country. Keep one row for that stratum, preferring STRICT.
    w = Window.partitionBy(
        "sampling_group", "event_type", "country", "uri"
    ).orderBy(
        when(col("pool") == "strict", 0).otherwise(1),
        col("matched_keyword"),
    )

    pairs = (
        pairs.withColumn("_rn", row_number().over(w))
             .filter(col("_rn") == 1)
             .drop("_rn")
             .cache()
    )

    print("[pairs] writing:", pairs_path)
    pairs.write.mode("overwrite").parquet(pairs_path)

    print("[diagnostics] writing:", diagnostics_path)

    (
        pairs.groupBy("sampling_group")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .write.mode("overwrite")
             .parquet(diagnostics_path + "candidate_by_group/")
    )

    (
        pairs.groupBy("sampling_group", "event_type")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .write.mode("overwrite")
             .parquet(diagnostics_path + "candidate_by_event_type/")
    )

    (
        pairs.groupBy("sampling_group", "event_type", "country")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .write.mode("overwrite")
             .parquet(diagnostics_path + "candidate_by_event_type_geography/")
    )

    (
        pairs.groupBy("sampling_group", "event_type", "country", "pool")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .write.mode("overwrite")
             .parquet(diagnostics_path + "candidate_by_stratum_pool/")
    )

    (
        pairs.groupBy("pub_year")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .write.mode("overwrite")
             .parquet(diagnostics_path + "candidate_by_year/")
    )

    # Useful coverage diagnostic: how many pre-geo candidate URIs found at least
    # one authoritative country in the database.
    event_uri_count = events.select("uri").dropDuplicates(["uri"]).count()
    mapped_uri_count = pairs.select("uri").dropDuplicates(["uri"]).count()

    print()
    print("DONE: authoritative geography candidate census")
    print(f"candidate pairs: {pairs_path}")
    print(f"diagnostics:     {diagnostics_path}")
    print(f"pre-geo unique candidate URIs: {event_uri_count:,}")
    print(f"geo-mapped unique candidate URIs: {mapped_uri_count:,}")
    if event_uri_count:
        print(f"geo coverage: {mapped_uri_count / event_uri_count:.2%}")


if __name__ == "__main__":
    main()
