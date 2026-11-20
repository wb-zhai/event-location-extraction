#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/geography/global_v4_audit/db_ingestion_inspection.txt"

mkdir -p "$(dirname "$OUT")"

echo "Writing inspection to:"
echo "$OUT"
echo

{
  echo "================================================================================"
  echo "GLOBAL V4 DATABASE INGESTION INSPECTION"
  echo "generated: $(date -Iseconds)"
  echo "repo: $ROOT"
  echo "================================================================================"

  echo
  echo "================================================================================"
  echo "1. SQL / MIGRATIONS / DB / INGESTION FILES"
  echo "================================================================================"

  find . \
    -type f \
    \( -iname '*.sql' \
       -o -iname '*migration*' \
       -o -iname '*schema*' \
       -o -iname '*database*' \
       -o -iname '*db*.py' \
       -o -iname '*ingest*.py' \) \
    -not -path './.git/*' \
    | sort

  echo
  echo "================================================================================"
  echo "2. TABLE / INSERT / COPY REFERENCES"
  echo "================================================================================"

  grep -RIn \
    --exclude-dir=.git \
    --include='*.py' \
    --include='*.sql' \
    -E 'CREATE TABLE|ALTER TABLE|INSERT INTO|COPY |article_event_extraction|risk_matches|locations|article_relevance|article_downloads|event_type|adm_code' \
    . \
    2>/dev/null \
    | head -500 \
    || true

  echo
  echo "================================================================================"
  echo "3. DATABASE CONNECTION CODE"
  echo "================================================================================"

  grep -RIn \
    --exclude-dir=.git \
    --include='*.py' \
    --include='*.sql' \
    -E 'psycopg|psycopg2|sqlalchemy|DB_HOST|DB_PORT|DB_NAME|DB_USER|DATABASE_URL|POSTGRES|MYSQL' \
    . \
    2>/dev/null \
    | head -400 \
    || true

  echo
  echo "================================================================================"
  echo "4. EXISTING GEO CSV INGESTION SCRIPT"
  echo "================================================================================"

  if [ -f scripts/geocoding/to_csv_ingest.py ]; then
    sed -n '1,260p' scripts/geocoding/to_csv_ingest.py
  else
    echo "NOT FOUND: scripts/geocoding/to_csv_ingest.py"
  fi

  echo
  echo "================================================================================"
  echo "5. EXISTING RELEVANCE DB INGESTION SCRIPT"
  echo "================================================================================"

  if [ -f scripts/relevance/db_ingestion.py ]; then
    sed -n '1,380p' scripts/relevance/db_ingestion.py
  else
    echo "NOT FOUND: scripts/relevance/db_ingestion.py"
  fi

  echo
  echo "================================================================================"
  echo "6. FINAL V4 ARTIFACTS"
  echo "================================================================================"

  echo "Final geocoded shards:"
  find outputs/geography/global_v4_audit/final_geocoded \
    -maxdepth 1 \
    -name 'shard-*.geo.jsonl' \
    | wc -l

  echo
  echo "Compact article parquet files:"
  find /home/bonadossou/global_v4_work/compact_event_metadata_v1/articles \
    -name 'shard-*.parquet' 2>/dev/null \
    | wc -l

  echo
  echo "Compact event parquet files:"
  find /home/bonadossou/global_v4_work/compact_event_metadata_v1/events \
    -name 'shard-*.parquet' 2>/dev/null \
    | wc -l

  echo
  echo "Relevant sizes:"
  du -sh \
    outputs/geography/global_v4_audit/final_geocoded \
    /home/bonadossou/global_v4_work/compact_event_metadata_v1 \
    2>/dev/null \
    || true

  echo
  echo "================================================================================"
  echo "7. SAMPLE FINAL EVENT SCHEMA"
  echo "================================================================================"

  python3 - <<'PY'
import json
from pathlib import Path

p = Path("outputs/geography/global_v4_audit/final_geocoded/shard-000.geo.jsonl")

if not p.exists():
    print("MISSING:", p)
else:
    with p.open(encoding="utf-8") as f:
        obj = json.loads(next(f))

    print("ARTICLE KEYS:")
    print(list(obj.keys()))

    preds = obj.get("predictions") or []

    if preds:
        print("\nEVENT KEYS:")
        print(list(preds[0].keys()))

        print("\nFIRST EVENT:")
        print(json.dumps(preds[0], ensure_ascii=False, indent=2)[:12000])
    else:
        print("No predictions in first article.")
PY

  echo
  echo "================================================================================"
  echo "8. COMPACT PARQUET SCHEMAS"
  echo "================================================================================"

  python3 - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path(
    "/home/bonadossou/global_v4_work/"
    "compact_event_metadata_v1"
)

for name in ("articles", "events"):
    files = sorted((root / name).glob("shard-*.parquet"))

    print("\n---", name.upper(), "---")

    if not files:
        print("NO FILES")
        continue

    pf = pq.ParquetFile(files[0])

    print("sample file:", files[0])
    print("rows:", pf.metadata.num_rows)
    print(pf.schema_arrow)
PY

  echo
  echo "================================================================================"
  echo "9. LIVE POSTGRESQL DATABASE"
  echo "================================================================================"

  if ! command -v psql >/dev/null 2>&1; then
    echo "psql is not installed; skipping live DB inspection."

  elif ! command -v pg_isready >/dev/null 2>&1; then
    echo "pg_isready is not installed; skipping live DB inspection."

  elif ! pg_isready -h localhost -p 5439 >/dev/null 2>&1; then
    echo "PostgreSQL tunnel at localhost:5439 is not reachable."
    echo "Skipping live DB inspection."

  else
    echo "PostgreSQL reachable at localhost:5439"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9A. DATABASE / SERVER INFO"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c "
SELECT
    current_database() AS database,
    current_user AS db_user,
    version() AS postgres_version;
" || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9B. SCHEMAS"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c '\dn+' \
      || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9C. TABLES"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c '\dt *.*' \
      || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9D. USER TABLE COLUMNS"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c "
SELECT
    table_schema,
    table_name,
    ordinal_position,
    column_name,
    data_type,
    udt_name,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema NOT IN (
    'pg_catalog',
    'information_schema'
)
ORDER BY
    table_schema,
    table_name,
    ordinal_position;
" || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9E. PRIMARY / UNIQUE / FOREIGN KEY CONSTRAINTS"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c "
SELECT
    tc.table_schema,
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name,
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
LEFT JOIN information_schema.constraint_column_usage ccu
  ON tc.constraint_name = ccu.constraint_name
 AND tc.table_schema = ccu.table_schema
WHERE tc.table_schema NOT IN (
    'pg_catalog',
    'information_schema'
)
ORDER BY
    tc.table_schema,
    tc.table_name,
    tc.constraint_type,
    tc.constraint_name,
    kcu.ordinal_position;
" || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9F. INDEXES"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c "
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname NOT IN (
    'pg_catalog',
    'information_schema'
)
ORDER BY
    schemaname,
    tablename,
    indexname;
" || true

    echo
    echo "--------------------------------------------------------------------------------"
    echo "9G. TABLE SIZES / ESTIMATED ROW COUNTS"
    echo "--------------------------------------------------------------------------------"

    psql \
      -h localhost \
      -p 5439 \
      -U zhai \
      -d zhai_db \
      -P pager=off \
      -c "
SELECT
    schemaname,
    relname AS table_name,
    n_live_tup AS estimated_rows,
    pg_size_pretty(
      pg_total_relation_size(
        quote_ident(schemaname) || '.' || quote_ident(relname)
      )
    ) AS total_size
FROM pg_stat_user_tables
ORDER BY
    pg_total_relation_size(
      quote_ident(schemaname) || '.' || quote_ident(relname)
    ) DESC;
" || true

  fi

  echo
  echo "================================================================================"
  echo "INSPECTION COMPLETE"
  echo "================================================================================"

} 2>&1 | tee "$OUT"

echo
echo "Saved full inspection to:"
echo "$OUT"
