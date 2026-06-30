"""Quick DB diagnostics: active queries + article_location_tags index info."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

try:
    from dotenv import dotenv_values
    _env = dotenv_values(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    _env = {}


def _e(key: str, fallback: str | None = None) -> str | None:
    return _env.get(key) or os.environ.get(key) or fallback


def get_connection_params() -> dict:
    return {
        "host":     _e("SQL_HOST",     _e("PGHOST", "localhost")),
        "port":     _e("SQL_PORT",     _e("PGPORT", "5432")),
        "dbname":   _e("SQL_DATABASE", _e("PGDATABASE")),
        "user":     _e("SQL_USERNAME", _e("PGUSER")),
        "password": _e("SQL_PASSWORD", _e("PGPASSWORD")),
    }


ACTIVE_QUERIES_SQL = """
SELECT
    pid,
    state,
    wait_event_type,
    wait_event,
    now() - query_start AS duration,
    left(query, 120) AS query_snippet
FROM pg_stat_activity
WHERE state = 'active'
  AND pid <> pg_backend_pid()
ORDER BY query_start;
"""

INDEXES_SQL = """
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'article_location_tags'
ORDER BY indexname;
"""

ROW_COUNT_SQL = """
SELECT reltuples::bigint AS approx_rows
FROM pg_class
WHERE relname = 'article_location_tags';
"""


def main() -> None:
    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

        print("=== Active queries ===")
        cur.execute(ACTIVE_QUERIES_SQL)
        rows = cur.fetchall()
        if not rows:
            print("  (none)")
        for r in rows:
            print(f"  pid={r['pid']} state={r['state']} "
                  f"wait={r['wait_event_type']}/{r['wait_event']} "
                  f"duration={r['duration']}")
            print(f"    {r['query_snippet']}")

        print("\n=== article_location_tags indexes ===")
        cur.execute(INDEXES_SQL)
        for r in cur.fetchall():
            print(f"  {r['indexname']}")
            print(f"    {r['indexdef']}")

        print("\n=== article_location_tags approximate row count ===")
        cur.execute(ROW_COUNT_SQL)
        row = cur.fetchone()
        print(f"  {row['approx_rows']:,}" if row else "  (no result)")

    conn.close()


if __name__ == "__main__":
    main()
