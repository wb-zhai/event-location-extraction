"""List countries by article count.

Usage:
    python scripts/data/download/list_countries.py [--limit N]

Requires psycopg2:
    pip install psycopg2-binary
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="List countries by article count.")
    parser.add_argument("--limit", type=int, default=0, help="Show only top N rows (0 = all).")
    parser.add_argument("--asc", action="store_true", help="Sort ascending (fewest first).")
    args = parser.parse_args()

    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    order = "ASC" if args.asc else "DESC"
    limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""

    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                WITH article_countries AS (
                    SELECT DISTINCT a.article_uri, geo.adm0_code
                    FROM public.article_location_tags AS a
                    JOIN public.geo_taxonomy AS geo ON a.adm_code = geo.adm_code
                    WHERE a.tag_method_id = 1
                )
                SELECT
                    g0.adm_name AS country,
                    ac.adm0_code,
                    COUNT(*) AS article_count
                FROM article_countries ac
                JOIN public.geo_taxonomy AS g0
                    ON g0.adm0_code = ac.adm0_code
                    AND g0.adm_level = 0
                GROUP BY g0.adm_name, ac.adm0_code
                ORDER BY article_count {order}
                {limit_clause}
                """
            )
            rows = cur.fetchall()

    if not rows:
        print("No results.")
        return

    col_w = max(len(r["country"]) for r in rows)
    print(f"{'Country':<{col_w}}  {'Code':<8}  {'Articles':>10}")
    print("-" * (col_w + 22))
    for r in rows:
        print(f"{r['country']:<{col_w}}  {r['adm0_code']:<8}  {r['article_count']:>10,}")

    conn.close()


if __name__ == "__main__":
    main()
