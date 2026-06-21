"""Download articles from the database for a given country.

Usage:
    python scripts/data/download/from_db.py
    python scripts/data/download/from_db.py --output path/to/articles.jsonl
    python scripts/data/download/from_db.py --output gs://bucket/path/articles.jsonl

Requires psycopg2:
    pip install psycopg2-binary
"""

import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

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


def count_articles(cur, adm0_code: str) -> int:
    cur.execute(
        """
        SELECT COUNT(DISTINCT a.article_uri)
        FROM public.article_location_tags AS a
        JOIN public.geo_taxonomy AS geo ON a.adm_code = geo.adm_code
        WHERE geo.adm0_code = %s
          AND a.tag_method_id = 1
        """,
        (adm0_code,),
    )
    return cur.fetchone()["count"]


def stream_articles(conn, adm0_code: str):
    """Stream one row per article with aggregated adm_codes."""
    with conn.cursor(name="article_stream", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.itersize = 500
        cur.execute(
            """
            SELECT DISTINCT ON (tags.article_uri)
                tags.article_uri AS uri,
                tags.adm_codes,
                ad.cloud_uri,
                ad.title,
                ad.body,
                ad.published_at,
                ad.article_type,
                ad.source_uri
            FROM (
                SELECT
                    a.article_uri,
                    array_agg(DISTINCT a.adm_code ORDER BY a.adm_code) AS adm_codes
                FROM public.article_location_tags AS a
                JOIN public.geo_taxonomy AS geo ON a.adm_code = geo.adm_code
                WHERE geo.adm0_code = %s
                  AND a.tag_method_id = 1
                GROUP BY a.article_uri
            ) tags
            JOIN public.article_downloads AS ad ON ad.uri = tags.article_uri
            ORDER BY tags.article_uri
            """,
            (adm0_code,),
        )
        yield from cur


def build_record(row: dict) -> dict:
    return {
        "id": str(row["uri"]),
        "status": "ok",
        "source": {
            "title": str(row["title"] or ""),
            "text": str(row["body"] or ""),
            "published_at": row["published_at"],
            "article_type": row["article_type"],
            "source_uri": row["source_uri"],
            "cloud_uri": row["cloud_uri"],
        },
        "adm_codes": list(row["adm_codes"] or []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download articles from the database for a selected country.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Path to write the JSONL output file. Supports local paths and gs://bucket/object. "
            "Defaults to dataset/country/{country_code}_articles.jsonl."
        ),
    )
    parser.add_argument(
        "--outoput",
        dest="output",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid GCS URI: {uri}. Expected gs://bucket/path/to/file.jsonl")
    return parsed.netloc, parsed.path.lstrip("/")


@contextmanager
def open_output(path_or_uri: str) -> TextIO:
    if path_or_uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print("google-cloud-storage is not installed. Run: pip install google-cloud-storage")
            sys.exit(1)

        bucket_name, blob_name = parse_gcs_uri(path_or_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        with blob.open("w", encoding="utf-8") as f:
            yield f
        return

    output_path = Path(path_or_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yield f


def main() -> None:
    args = parse_args()
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

            print(f"\nCounting articles for {chosen['adm_name']}...")
            total = count_articles(cur, adm0_code)
            print(f"Found {total} unique articles.")

            if total == 0:
                print("Nothing to write.")
                return

        # Output path
        print("\n--- Output ---")
        default_output = f"dataset/country/{adm0_code.lower()}_articles.jsonl"
        output_path = args.output or prompt("Output file", default_output)

        # Stream and write
        print("\nDownloading...")
        written = 0
        with open_output(output_path) as f:
            for row in stream_articles(conn, adm0_code):
                f.write(json.dumps(build_record(row), ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
                written += 1
                if written % 500 == 0:
                    print(f"  {written}/{total}\033[K", end="\r", file=sys.stderr, flush=True)

        print(f"\nWrote {written} articles to {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
