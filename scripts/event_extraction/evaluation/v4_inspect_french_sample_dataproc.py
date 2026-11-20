#!/usr/bin/env python3
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--pairs", required=True)
args = ap.parse_args()

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.legacy.parquet.nanosAsLong", "true")

m = spark.read.parquet(args.manifest).cache()
p = spark.read.parquet(args.pairs).cache()

print("\n=== MANIFEST SUMMARY ===")
print("selected_unique_articles:", m.select("article_id").distinct().count())

if "language" in m.columns:
    print("language_counts:")
    m.groupBy("language").agg(
        countDistinct("article_id").alias("articles")
    ).orderBy("language").show(100, False)

print("groups:")
m.groupBy("sampling_group").agg(
    countDistinct("article_id").alias("articles")
).orderBy("sampling_group").show(100, False)

print("event_types:")
m.groupBy("sampling_group", "sampling_event_type").agg(
    countDistinct("article_id").alias("articles")
).orderBy("sampling_group", "sampling_event_type").show(200, False)

print("pool:")
m.groupBy("pool").agg(
    countDistinct("article_id").alias("articles")
).orderBy("pool").show(20, False)

print("\n=== CANDIDATE AVAILABILITY ===")
print("candidate_unique_articles:", p.select("uri").distinct().count())

print("candidate_groups:")
p.groupBy("sampling_group").agg(
    countDistinct("uri").alias("candidate_articles")
).orderBy("sampling_group").show(100, False)

print("candidate_event_types:")
p.groupBy("sampling_group", "event_type").agg(
    countDistinct("uri").alias("candidate_articles")
).orderBy("sampling_group", "event_type").show(200, False)

print("candidate_pool:")
p.groupBy("pool").agg(
    countDistinct("uri").alias("candidate_articles")
).orderBy("pool").show(20, False)

print("observed_event_types:", p.select("event_type").distinct().count())
print("observed_groups:", p.select("sampling_group").distinct().count())
print("observed_countries:", p.select("country").distinct().count())
