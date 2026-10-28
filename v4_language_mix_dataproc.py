#!/usr/bin/env python3
import argparse, json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, from_unixtime, length, year

DEFAULT_ARTICLES_GLOB = "gs://zhai-infra-db/article_downloads/2025-Dec-17/*.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--articles-glob", default=DEFAULT_ARTICLES_GLOB)
ap.add_argument("--min-year", type=int, default=2013)
ap.add_argument("--min-body-chars", type=int, default=200)
ap.add_argument("--relevant-uris-path", default=None)
ap.add_argument("--relevant-format", choices=["parquet","json","csv"], default="parquet")
args = ap.parse_args()

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")
df = spark.read.parquet(args.articles_glob)

df = (
    df.select("uri","body","published_at","language")
      .withColumn("pub_year", year(from_unixtime(col("published_at") / 1_000_000_000)))
      .filter(col("pub_year") >= args.min_year)
      .filter(col("body").isNotNull() & (length(col("body")) >= args.min_body_chars))
      .filter(col("language").isin("eng","fra"))
)

if args.relevant_uris_path:
    reader = spark.read.format(args.relevant_format)
    if args.relevant_format == "csv":
        reader = reader.option("header","true")
    rel = reader.load(args.relevant_uris_path).select("uri").dropDuplicates(["uri"])
    df = df.join(rel, "uri", "inner")

rows = df.groupBy("language").agg(countDistinct("uri").alias("articles")).collect()
counts = {r["language"]: int(r["articles"]) for r in rows}
eng, fra = counts.get("eng",0), counts.get("fra",0)
total = eng + fra
print(json.dumps({
    "eng": eng,
    "fra": fra,
    "eng_share": eng/total if total else 0,
    "fra_share": fra/total if total else 0,
    "total_eng_fra": total,
    "relevance_filter_applied": bool(args.relevant_uris_path)
}, indent=2))
