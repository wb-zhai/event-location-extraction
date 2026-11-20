#!/usr/bin/env python3
"""
Convert the final stratified teacher Parquet manifest into the JSONL schema
expected by scripts/event_extraction/generation/generate.py.

Important:
- keeps article_id as both top-level "id" and "article_id"
- keeps sampling metadata for post-Gemini diagnostics
- preserves top-level language so downstream SFT can select English/French prompts
- nests article content under "source"
- converts published_at nanoseconds to YYYY-MM-DD
- intentionally does NOT create a "candidates" field, because the sampling
  event type is a retrieval stratum, not a ground-truth restriction
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format, from_unixtime, lit, struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Final teacher manifest Parquet path")
    ap.add_argument("--output", required=True, help="Output GCS/local directory for JSONL parts")
    ap.add_argument("--num-parts", type=int, default=8)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("export-teacher-manifest-for-gemini").getOrCreate()
    spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

    df = spark.read.parquet(args.input)

    required = {
        "article_id",
        "title",
        "body",
        "published_at",
        "sampling_group",
        "sampling_event_type",
        "geography_field",
        "pool",
        "matched_keyword",
        "pub_year",
        "language",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError("Missing required columns: " + ", ".join(missing))

    # article_downloads.published_at is nanoseconds since Unix epoch in this snapshot.
    publish_date = date_format(
        from_unixtime(col("published_at").cast("double") / lit(1_000_000_000.0)),
        "yyyy-MM-dd",
    )

    out = (
        df.select(
            col("article_id").cast("string").alias("id"),
            col("article_id").cast("string").alias("article_id"),
            col("sampling_group"),
            col("sampling_event_type"),
            col("geography_field"),
            col("pool"),
            col("matched_keyword"),
            col("pub_year"),
            col("language").cast("string").alias("language"),
            struct(
                col("title").cast("string").alias("title"),
                col("body").cast("string").alias("text"),
                lit("").alias("source_url"),
                publish_date.alias("publish_date"),
            ).alias("source"),
        )
        .dropDuplicates(["id"])
    )

    n = out.count()
    print(f"records to export: {n:,}")

    (
        out.repartition(args.num_parts)
        .write.mode("overwrite")
        .json(args.output)
    )

    print("DONE: Gemini input export")
    print(f"records: {n:,}")
    print(f"output:  {args.output}")
    print("NOTE: no 'candidates' field was written; Gemini will see the full teacher ontology.")


if __name__ == "__main__":
    main()
