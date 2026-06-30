"""Cancel your own running article_countries query without touching other users' queries."""

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


FIND_SQL = """
SELECT pid, now() - query_start AS duration, left(query, 80) AS query_snippet
FROM pg_stat_activity
WHERE state = 'active'
  AND usename = current_user
  AND pid <> pg_backend_pid()
ORDER BY query_start;
"""

CANCEL_SQL = """
SELECT pid, pg_cancel_backend(pid) AS cancelled
FROM pg_stat_activity
WHERE state = 'active'
  AND usename = current_user
  AND pid <> pg_backend_pid();
"""


def main() -> None:
    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT current_user AS u")
        print(f"Connected as: {cur.fetchone()['u']}")

        cur.execute(FIND_SQL)
        rows = cur.fetchall()
        if not rows:
            print("No matching queries found.")
            conn.close()
            return

        print("Queries to cancel:")
        for r in rows:
            print(f"  pid={r['pid']}  duration={r['duration']}  {r['query_snippet']}")

        confirm = input("\nCancel these? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            conn.close()
            return

        cur.execute(CANCEL_SQL)
        for r in cur.fetchall():
            status = "cancelled" if r["cancelled"] else "failed (may have finished)"
            print(f"  pid={r['pid']}: {status}")

    conn.close()


if __name__ == "__main__":
    main()
