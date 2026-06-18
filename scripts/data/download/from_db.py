"""Download articles from the database for a given country.

Usage:
    python scripts/data/download/from_db.py

Requires psycopg2:
    pip install psycopg2-binary
"""

import json
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


def prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt user for input; skip entirely if default is already set."""
    if default:
        print(f"  {label}: (from .env)")
        return default
    import getpass
    while True:
        value = (getpass.getpass(f"  {label}: ") if secret else input(f"  {label}: ").strip())
        if value:
            return value
        print(f"  {label} is required.")


def get_connection_params() -> dict:
    print("\n--- Database connection ---")
    return {
        "host":     prompt("Host",     _e("SQL_HOST",     _e("PGHOST", "localhost"))),
        "port":     prompt("Port",     _e("SQL_PORT",     _e("PGPORT", "5432"))),
        "dbname":   prompt("Database", _e("SQL_DATABASE", _e("PGDATABASE"))),
        "user":     prompt("User",     _e("SQL_USERNAME", _e("PGUSER"))),
        "password": prompt("Password", _e("SQL_PASSWORD", _e("PGPASSWORD")), secret=True),
    }


def find_country(cur, country_name: str) -> list[dict]:
    cur.execute(
        """
        SELECT DISTINCT adm_name, adm0_code
        FROM public.geo_taxonomy
        WHERE adm0_name ILIKE %s
          AND adm_level = 0
        ORDER BY adm_name
        """,
        (f"%{country_name}%",),
    )
    return cur.fetchall()


def fetch_articles(cur, adm0_code: str) -> list[dict]:
    """Fetch article URIs and their adm_codes for a country (string-match method 1)."""
    cur.execute(
        """
        SELECT DISTINCT
            a.article_uri,
            a.adm_code
        FROM public.article_location_tags AS a
        JOIN public.geo_taxonomy AS geo ON a.adm_code = geo.adm_code
        WHERE geo.adm0_code = %s
          AND a.tag_method_id = 1
        ORDER BY a.article_uri
        """,
        (adm0_code,),
    )
    return cur.fetchall()


def main() -> None:
    params = get_connection_params()

    print("\nConnecting to database...")
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    print("Connected.")

    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Country selection
            while True:
                print("\n--- Country selection ---")
                country_input = prompt("Country name (or part of it)")
                matches = find_country(cur, country_input)

                if not matches:
                    print(f"  No countries found matching '{country_input}'. Try again.")
                    continue

                if len(matches) == 1:
                    chosen = matches[0]
                    print(f"  Found: {chosen['adm_name']} ({chosen['adm0_code']})")
                else:
                    print("  Multiple matches:")
                    for i, m in enumerate(matches):
                        print(f"    [{i}] {m['adm_name']} ({m['adm0_code']})")
                    idx = prompt("Select index")
                    try:
                        chosen = matches[int(idx)]
                    except (ValueError, IndexError):
                        print("  Invalid selection. Try again.")
                        continue

                confirm = input(f"  Use '{chosen['adm_name']}' ({chosen['adm0_code']})? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    break

            adm0_code = chosen["adm0_code"]

            # Output path
            print("\n--- Output ---")
            default_output = f"dataset/{adm0_code.lower()}_articles.jsonl"
            output_path = Path(prompt("Output file", default_output))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Fetch
            print(f"\nFetching articles for {chosen['adm_name']} ({adm0_code})...")
            rows = fetch_articles(cur, adm0_code)
            print(f"Found {len(rows)} rows.")

            if not rows:
                print("Nothing to write.")
                return

            # Group adm_codes per article_uri
            articles: dict[str, list[str]] = {}
            for row in rows:
                articles.setdefault(row["article_uri"], []).append(row["adm_code"])

            with output_path.open("w") as f:
                for uri, adm_codes in articles.items():
                    f.write(json.dumps({"article_uri": uri, "adm_codes": adm_codes}) + "\n")

            print(f"Wrote {len(articles)} articles to {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
