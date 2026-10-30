#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import duckdb
from pathlib import Path


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"

def build(
    input_glob: Path,
    output_dir: Path,
    database: Path,
    gap_days: int = 3,
    threads: int = 6,
    memory_limit: str = "8GB",
) -> dict:
    shards = sorted(input_glob.parent.glob(input_glob.name))
    if not shards:
        raise FileNotFoundError(f"No input shards found matching {input_glob}")
    if gap_days < 0:
        raise ValueError("gap_days must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if database.exists():
        database.unlink()

    clusters_path = output_dir / "table3_wb_3d.parquet"
    assignments_path = output_dir / "table3_wb_3d_assignments.parquet"
    summary_path = output_dir / "table3_wb_3d_summary.json"
    con = duckdb.connect(str(database))
    try:
        con.execute(f"PRAGMA threads={int(threads)}")
        con.execute(f"PRAGMA memory_limit={sql_quote(memory_limit)}")
        con.execute(f"PRAGMA temp_directory={sql_quote(str(temp_dir))}")
        con.execute("PRAGMA preserve_insertion_order=false")
        source = sql_quote(str(input_glob))

        print("Reading Table 2 and preserving all mentions...", flush=True)
        con.execute(
            f"""
            CREATE TABLE source AS
            SELECT
                row_number() OVER () AS source_row_id,
                CAST(shard AS VARCHAR) AS shard,
                CAST(article_id AS VARCHAR) AS article_id,
                CAST(event_index AS VARCHAR) AS event_index,
                CAST(component_index AS VARCHAR) AS component_index,
                CAST(adm_code AS VARCHAR) AS adm_code_raw,
                upper(trim(CAST(adm_code AS VARCHAR))) AS adm_code,
                TRY_CAST(adm_level AS INTEGER) AS adm_level,
                lower(trim(CAST(event_type AS VARCHAR))) AS event_type,
                CAST(severity AS VARCHAR) AS severity_raw,
                lower(trim(CAST(severity AS VARCHAR))) AS severity,
                CAST(start_time AS VARCHAR) AS start_time_raw,
                lower(trim(CAST(start_precision AS VARCHAR))) AS start_precision,
                CAST(end_time AS VARCHAR) AS end_time_raw,
                lower(trim(CAST(end_precision AS VARCHAR))) AS end_precision,
                CAST(NULL AS VARCHAR) AS time_status_raw
            FROM read_json_auto({source}, format='newline_delimited',
                                union_by_name=true)
            """
        )
        source_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]

        print("Normalizing date precision and eligibility...", flush=True)
        con.execute(
            """
            CREATE TABLE prepared AS
            WITH parsed AS (
                SELECT *,
                    CASE WHEN regexp_matches(coalesce(trim(start_time_raw), ''),
                        '^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
                        THEN TRY_CAST(start_time_raw AS DATE) END AS start_date,
                    CASE WHEN regexp_matches(coalesce(trim(end_time_raw), ''),
                        '^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
                        THEN TRY_CAST(end_time_raw AS DATE) END AS end_date,
                    CASE
                        WHEN regexp_matches(coalesce(trim(start_time_raw), ''),
                            '^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
                            AND TRY_CAST(start_time_raw AS DATE) IS NOT NULL
                            THEN 'exact_day'
                        WHEN regexp_matches(coalesce(trim(start_time_raw), ''),
                            '^[0-9]{4}-[0-9]{2}$')
                            AND TRY_CAST(trim(start_time_raw) || '-01' AS DATE)
                                IS NOT NULL THEN 'month'
                        WHEN regexp_matches(coalesce(trim(start_time_raw), ''),
                            '^[0-9]{4}$')
                            AND TRY_CAST(trim(start_time_raw) || '-01-01' AS DATE)
                                IS NOT NULL THEN 'year'
                        WHEN start_time_raw IS NULL OR trim(start_time_raw) = ''
                            THEN 'missing'
                        ELSE 'unresolved_or_invalid'
                    END AS start_value_precision,
                    CASE
                        WHEN regexp_matches(coalesce(trim(end_time_raw), ''),
                            '^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
                            AND TRY_CAST(end_time_raw AS DATE) IS NOT NULL
                            THEN 'exact_day'
                        WHEN regexp_matches(coalesce(trim(end_time_raw), ''),
                            '^[0-9]{4}-[0-9]{2}$')
                            AND TRY_CAST(trim(end_time_raw) || '-01' AS DATE)
                                IS NOT NULL THEN 'month'
                        WHEN regexp_matches(coalesce(trim(end_time_raw), ''),
                            '^[0-9]{4}$')
                            AND TRY_CAST(trim(end_time_raw) || '-01-01' AS DATE)
                                IS NOT NULL THEN 'year'
                        WHEN end_time_raw IS NULL OR trim(end_time_raw) = ''
                            OR upper(trim(end_time_raw)) IN ('UNRESOLVED', 'OPEN_END')
                            THEN 'missing_or_open'
                        ELSE 'unresolved_or_invalid'
                    END AS end_value_precision
                FROM source
            ),
            classified AS (
                SELECT *,
                    CASE
                        WHEN start_date IS NULL THEN NULL
                        WHEN end_date IS NOT NULL AND end_date >= start_date
                            THEN end_date
                        ELSE start_date
                    END AS interval_end_date
                FROM parsed
            )
            SELECT *,
                CASE
                    WHEN start_value_precision <> 'exact_day'
                        THEN 'start_not_exact_day'
                    WHEN start_precision IS NOT NULL
                        AND start_precision NOT IN ('', 'day')
                        THEN 'start_precision_label_mismatch'
                    WHEN end_date IS NOT NULL AND end_date < start_date
                        THEN 'end_before_start'
                    WHEN adm_code IS NULL OR trim(adm_code) = ''
                        THEN 'missing_adm_code'
                    WHEN adm_level IS NULL THEN 'missing_adm_level'
                    WHEN event_type IS NULL OR trim(event_type) = ''
                        THEN 'missing_event_type'
                    ELSE 'eligible_exact_day'
                END AS assignment_status,
                (
                    start_value_precision = 'exact_day'
                    AND (start_precision IS NULL OR start_precision IN ('', 'day'))
                    AND (end_date IS NULL OR end_date >= start_date)
                    AND adm_code IS NOT NULL AND trim(adm_code) <> ''
                    AND adm_level IS NOT NULL
                    AND event_type IS NOT NULL AND trim(event_type) <> ''
                ) AS is_cluster_eligible
            FROM classified
            """
        )
        eligible_rows = con.execute(
            "SELECT COUNT(*) FROM prepared WHERE is_cluster_eligible"
        ).fetchone()[0]

        print(f"Eligible exact-day mentions: {eligible_rows:,}", flush=True)
        print("Assigning interval-connected components...", flush=True)
        con.execute(
            f"""
            CREATE TABLE assigned AS
            WITH ordered AS (
                SELECT *,
                    max(interval_end_date) OVER (
                        PARTITION BY adm_code, adm_level, event_type
                        ORDER BY start_date, interval_end_date, article_id,
                                 event_index, component_index, source_row_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    ) AS prior_max_end
                FROM prepared
                WHERE is_cluster_eligible
            ),
            marked AS (
                SELECT *,
                    CASE
                        WHEN prior_max_end IS NULL THEN 1
                        WHEN date_diff('day', prior_max_end, start_date)
                             > {int(gap_days)} THEN 1
                        ELSE 0
                    END AS starts_new_cluster
                FROM ordered
            )
            SELECT *,
                sum(starts_new_cluster) OVER (
                    PARTITION BY adm_code, adm_level, event_type
                    ORDER BY start_date, interval_end_date, article_id,
                             event_index, component_index, source_row_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS temporal_cluster_number
            FROM marked
            """
        )
        con.execute(
            """
            CREATE TABLE clusters AS
            WITH grouped AS (
                SELECT
                    adm_code, adm_level, event_type, temporal_cluster_number,
                    min(start_date) AS cluster_start_date,
                    max(interval_end_date) AS cluster_end_date,
                    count(*) AS n_mentions,
                    count(DISTINCT article_id) AS n_articles,
                    count(*) FILTER (WHERE start_date = interval_end_date)
                        AS n_single_day_mentions,
                    count(*) FILTER (WHERE interval_end_date > start_date)
                        AS n_multi_day_mentions,
                    count(*) FILTER (
                        WHERE severity IS NULL OR severity = ''
                    ) AS n_mentions_missing_severity,
                    string_agg(DISTINCT severity_raw, ' | ' ORDER BY severity_raw)
                        FILTER (WHERE severity_raw IS NOT NULL
                            AND trim(severity_raw) <> '') AS severity_labels,
                    string_agg(DISTINCT article_id, ' | ' ORDER BY article_id)
                        FILTER (WHERE article_id IS NOT NULL) AS article_ids,
                    string_agg(DISTINCT concat(coalesce(article_id, ''), 'E',
                        coalesce(event_index, ''), 'C',
                        coalesce(component_index, '')), ' | ') AS mention_ids
                FROM assigned
                GROUP BY adm_code, adm_level, event_type, temporal_cluster_number
            )
            SELECT
                md5(adm_code || '|' || CAST(adm_level AS VARCHAR) || '|' ||
                    event_type || '|' ||
                    CAST(temporal_cluster_number AS VARCHAR) || '|' ||
                    CAST(cluster_start_date AS VARCHAR)) AS cluster_uid,
                *,
                date_diff('day', cluster_start_date, cluster_end_date) + 1
                    AS cluster_span_days
            FROM grouped WHERE cluster_span_days > 5
            """
        )
        con.execute(
            f"""
            CREATE TABLE assignments AS
            SELECT
                p.*, a.start_date AS event_date, a.prior_max_end,
                a.starts_new_cluster, a.temporal_cluster_number,
                c.cluster_uid, c.cluster_start_date, c.cluster_end_date,
                c.cluster_span_days,
                CASE WHEN c.cluster_uid IS NULL THEN 'not_clustered'
                     ELSE 'clustered' END AS clustering_status,
                'interval_3d_v2' AS clustering_version
            FROM prepared p
            LEFT JOIN assigned a ON p.source_row_id = a.source_row_id
            LEFT JOIN clusters c
              ON a.adm_code = c.adm_code
             AND a.adm_level = c.adm_level
             AND a.event_type = c.event_type
             AND a.temporal_cluster_number = c.temporal_cluster_number
            """
        )
        con.execute(
            f"""
            COPY (SELECT * FROM clusters ORDER BY adm_code, event_type,
                  cluster_start_date)
            TO {sql_quote(str(clusters_path))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.execute(
            f"""
            COPY (SELECT * FROM assignments ORDER BY source_row_id)
            TO {sql_quote(str(assignments_path))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        assignment_rows = con.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        cluster_rows = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        clustered_mentions = con.execute(
            "SELECT coalesce(sum(n_mentions), 0) FROM clusters"
        ).fetchone()[0]
        duplicate_uids = con.execute(
            "SELECT count(*) - count(DISTINCT cluster_uid) FROM clusters"
        ).fetchone()[0]
        unassigned = con.execute(
            "SELECT count(*) FROM assignments "
            "WHERE is_cluster_eligible AND cluster_uid IS NULL"
        ).fetchone()[0]
        boundary_violations = con.execute(
            f"""
            WITH prior AS (
                SELECT *,
                    lag(cluster_end_date) OVER (
                        PARTITION BY adm_code, adm_level, event_type
                        ORDER BY cluster_start_date, temporal_cluster_number
                    ) AS prior_end
                FROM clusters
            )
            SELECT count(*) FROM prior
            WHERE prior_end IS NOT NULL
              AND date_diff('day', prior_end, cluster_start_date)
                    <= {int(gap_days)}
            """
        ).fetchone()[0]
        precision_counts = con.execute(
            """
            SELECT start_value_precision, assignment_status, count(*) AS mentions
            FROM prepared GROUP BY 1, 2 ORDER BY mentions DESC
            """
        ).fetchall()
    finally:
        con.close()

    checks = {
        "source_to_assignment_conservation": source_rows == assignment_rows,
        "eligible_to_cluster_conservation": eligible_rows == clustered_mentions,
        "all_eligible_mentions_assigned": unassigned == 0,
        "cluster_uid_unique": duplicate_uids == 0,
        "boundary_violations": boundary_violations == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"3-day validation failed: {checks}")

    summary = {
        "clustering_method": "interval_connected_components_running_max_end",
        "clustering_version": "interval_3d_v2",
        "gap_threshold_days": gap_days,
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "cluster_rows": cluster_rows,
        "assignment_rows": assignment_rows,
        "checks": checks,
        "precision_counts": [
            {"start_value_precision": row[0], "assignment_status": row[1],
             "mentions": row[2]}
            for row in precision_counts
        ],
        "outputs": {
            "clusters": str(clusters_path),
            "assignments": str(assignments_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Clusters: {clusters_path}")
    print(f"Assignments: {assignments_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/table2_wb_raw_mentions/shard-*.jsonl.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/handoff"))
    parser.add_argument("--db", type=Path, default=Path("data/_3d.duckdb"))
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()
    build(args.input, args.output_dir, args.db, args.gap_days,
          args.threads, args.memory_limit)


if __name__ == "__main__":
    main()
