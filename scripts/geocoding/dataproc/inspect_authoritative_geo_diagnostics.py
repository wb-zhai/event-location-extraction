#!/usr/bin/env python3
import argparse
import uuid
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, countDistinct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics-path", required=True)
    ap.add_argument("--candidate-pairs-path", required=True)
    ap.add_argument("--output-root", default="gs://test-zhai-texts")
    args = ap.parse_args()

    spark = SparkSession.builder.getOrCreate()
    root = args.output_root.rstrip("/")
    run_id = str(uuid.uuid4())
    out = f"{root}/authoritative_geo_inspection_{run_id}/"
    droot = args.diagnostics_path.rstrip("/") + "/"

    by_group = spark.read.parquet(droot + "candidate_by_group/")
    by_type = spark.read.parquet(droot + "candidate_by_event_type/")
    by_geo = spark.read.parquet(droot + "candidate_by_event_type_geography/")
    by_pool = spark.read.parquet(droot + "candidate_by_stratum_pool/")
    by_year = spark.read.parquet(droot + "candidate_by_year/")
    pairs = spark.read.parquet(args.candidate_pairs_path)

    by_group.orderBy(desc("candidate_articles")).coalesce(1).write.mode("overwrite").option("header","true").csv(out + "by_group/")
    by_type.orderBy(desc("candidate_articles")).coalesce(1).write.mode("overwrite").option("header","true").csv(out + "by_event_type/")
    by_geo.orderBy("sampling_group","event_type",desc("candidate_articles")).coalesce(1).write.mode("overwrite").option("header","true").csv(out + "by_event_type_geography/")
    by_pool.orderBy("sampling_group","event_type","country","pool").coalesce(1).write.mode("overwrite").option("header","true").csv(out + "by_stratum_pool/")
    by_year.orderBy("pub_year").coalesce(1).write.mode("overwrite").option("header","true").csv(out + "by_year/")

    type_country_cov = (
        pairs.groupBy("sampling_group","event_type")
             .agg(
                 countDistinct("country").alias("n_countries"),
                 countDistinct("uri").alias("candidate_articles")
             )
             .orderBy("sampling_group","event_type")
    )
    type_country_cov.coalesce(1).write.mode("overwrite").option("header","true").csv(out + "event_type_country_coverage/")

    pool_type = (
        pairs.groupBy("sampling_group","event_type","pool")
             .agg(countDistinct("uri").alias("candidate_articles"))
             .orderBy("sampling_group","event_type","pool")
    )
    pool_type.coalesce(1).write.mode("overwrite").option("header","true").csv(out + "strict_fallback_by_event_type/")

    total_unique_articles = pairs.select("uri").dropDuplicates(["uri"]).count()
    total_pairs = pairs.count()
    n_event_types = pairs.select("event_type").distinct().count()
    n_countries = pairs.select("country").distinct().count()
    n_groups = pairs.select("sampling_group").distinct().count()

    print()
    print("DONE: authoritative candidate diagnostics inspection")
    print(f"unique geo-mapped candidate articles: {total_unique_articles:,}")
    print(f"event-country candidate rows:         {total_pairs:,}")
    print(f"broader groups:                       {n_groups:,}")
    print(f"event types:                          {n_event_types:,}")
    print(f"countries:                            {n_countries:,}")
    print(f"inspection output:                    {out}")

if __name__ == "__main__":
    main()
