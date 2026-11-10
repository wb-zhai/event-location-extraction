import argparse
import json
import duckdb
from pathlib import Path



#TIME:  
#       (1)miss year to days
#       (2)miss end date being article date when ongoing (extraction postprocess)

#LOCATION:
#       (3)miss adm to city linking



def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"

def build(
    input_dir: Path,
    output_dir: Path,
    database: Path,
    gap_days: int = 3,
    min_mentions: int = 3,
    threads: int = 6,
    memory_limit: str = "8GB",
) -> dict:
    
    shards = sorted(input_dir.parent.glob(input_dir.name))
    if not shards:
        raise FileNotFoundError(f"No input shards found in {input_dir}")
    if gap_days < 0:
        raise ValueError("gap_days must be non-negative")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    clusters_path = output_dir / Path("clusters_"+str(gap_days)+"days_"+str(min_mentions)+"min_mentions.parquet")
    con = duckdb.connect(str(database))
    
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"PRAGMA memory_limit={sql_quote(memory_limit)}")
    con.execute(f"PRAGMA temp_directory={(sql_quote(str(temp_dir)))}")
    con.execute("PRAGMA preserve_insertion_order=false")
    source = (sql_quote(str(input_dir)))
    
    
    print("Reading Table at " + str(input_dir), flush=True)
    
    con.execute(
            f"""
            CREATE TABLE source AS
            SELECT
                shard,
                article_id,
                event_index,
                component_index,
                
                event_type,
                
                CAST(adm_level AS INTEGER) AS adm_level,
                adm_code,
                adm_name,
                
                TRY_CAST(start_time AS DATE) start_date,
                start_precision,
                TRY_CAST(end_time AS DATE) end_date,
                end_precision,
                time_status,
                
                severity
                
            FROM read_json_auto({source}, format='newline_delimited',
                                union_by_name=true)
            """
        )
    
    
    source_rows = con.execute("SELECT COUNT(*) FROM source").fetchone()[0]
    
    print("Filtering uneligible...", flush=True)
    #end date when ongoing is bugged, we need article date not null.. now taking null becuase data is and not good..
    
    con.execute(
            """
            CREATE TABLE filtered AS
                SELECT *, 
                    CASE
                        WHEN end_date IS NOT NULL AND end_date >= start_date THEN end_date
                        ELSE start_date
                    END AS interval_end_date
                FROM source
                WHERE start_precision = 'day'
                AND start_date IS NOT NULL
                AND adm_code IS NOT NULL 
                AND adm_level IS NOT NULL
                AND event_type IS NOT NULL
                AND start_date >= DATE '2017-01-01'
                AND start_date < DATE '2027-01-01'
            """
        )
    
    filtered_rows = con.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
    
    
    print("Ordering by adm_code...", flush=True)
    con.execute(f"""
    CREATE TABLE ordered AS
    SELECT
        *,
        MAX(interval_end_date) OVER (
                    PARTITION BY adm_code, event_type
                    ORDER BY start_date, interval_end_date, article_id,
                             event_index, component_index
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_running_max_end
    FROM filtered
    """)
        
    con.execute(f"""
    CREATE TABLE assigned AS
    WITH marked AS (
        SELECT
            *,
            CASE
                WHEN prior_running_max_end IS NULL THEN 1
                WHEN date_diff(
                    'day',
                    prior_running_max_end,
                    start_date
                ) > {gap_days}
                    THEN 1
                ELSE 0
            END AS starts_new_cluster
        FROM ordered
    ),
    numbered AS (
        SELECT
            *,
            SUM(starts_new_cluster) OVER (
                PARTITION BY adm_code, event_type
                ORDER BY
                    start_date,
                    interval_end_date,
                    article_id,
                    event_index,
                    component_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS temporal_cluster_number
        FROM marked
    )
    SELECT *
    FROM numbered
    """)
    
    #remember to filter our small ones
    assigned_rows = con.execute("SELECT COUNT(*) FROM assigned").fetchone()[0]
    
    print("Aggregating clusters ...", flush=True)

    con.execute(
                """
                CREATE TABLE clusters AS
                WITH grouped AS (
                    SELECT
                        adm_code, adm_level, event_type, temporal_cluster_number,
                        
                        MIN(start_date) AS cluster_start_date,
                        MAX(interval_end_date) AS cluster_end_date,
                        
                        COUNT(*) AS n_mentions,
                        COUNT(DISTINCT article_id) AS n_articles,
                        
                        COUNT(*) FILTER (WHERE start_date = interval_end_date)
                            AS n_single_day_mentions,
                        count(*) FILTER (WHERE interval_end_date > start_date)
                            AS n_multi_day_mentions,
                            
                        string_agg(DISTINCT severity, ' | ' ORDER BY severity)
                            FILTER (WHERE severity IS NOT NULL
                                AND trim(severity) <> '') AS severity_labels,
                                
                        string_agg(DISTINCT article_id, ' | ' ORDER BY article_id)
                            FILTER (WHERE article_id IS NOT NULL) AS article_ids,
                            
                        string_agg(
                        DISTINCT concat(
                            coalesce(CAST(article_id AS VARCHAR), ''), 'E',
                            coalesce(CAST(event_index AS VARCHAR), ''), 'C',
                            coalesce(CAST(component_index AS VARCHAR), '')
                        ),
                        ' | '
                            ) AS mention_ids
                            
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
                FROM grouped 
                """
            )
    
    cluster_rows = con.execute("SELECT COUNT(*) FROM assigned").fetchone()[0]
    
    con.execute(
            f"""
            CREATE OR REPLACE TABLE clustersss AS
            SELECT * FROM clusters
            WHERE n_mentions >= {int(min_mentions)}
            """
        )
            
    con.execute(
            f"""
            COPY (
                SELECT * FROM clustersss
                ORDER BY adm_code, adm_level, event_type, cluster_start_date
            )
            TO {sql_quote(clusters_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    
    con.close()
    
    # print(len(source_rows), len(filtered_rows), len(cluster_rows))
            
    
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=str,
                        default="data/table2_wb_raw_mentions")
    parser.add_argument("--output-dir", type=Path, default=Path("data/handoff"))
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--min_mentions", type=int, default=1)
    parser.add_argument("--db", type=Path, default=Path("data/_3d.duckdb"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit", default="10GB")
    args = parser.parse_args()
    
    build(Path(args.input_dir + "/shard-*.jsonl.gz"), args.output_dir, args.db, args.days, args.min_mentions, args.threads, args.memory_limit)


if __name__ == "__main__":
    main()
