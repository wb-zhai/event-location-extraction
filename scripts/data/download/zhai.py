from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import dotenv
from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "dataset/zhai/raw/zhai_with_tags_raw.jsonl"
DEFAULT_ARTICLE_LIMIT = 100


def load_sql_password() -> str:
    dotenv.load_dotenv(REPO_ROOT / ".env")
    password = dotenv.get_key(REPO_ROOT / ".env", "SQL_PASSWORD")
    if not password:
        raise ValueError("SQL_PASSWORD not found in .env file")
    return password


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tagged ZHAI article rows and save them as JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSONL output path. Default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_ARTICLE_LIMIT,
        help="Number of distinct tagged articles to download. Use 0 for all articles.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Rows to fetch per batch while streaming results.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_query() -> str:
    return """
WITH selected_articles AS (
    SELECT DISTINCT article_uri
    FROM article_risk_factor_tags
    WHERE tag_method_id = 1
    LIMIT :article_limit
)
SELECT
    ad.uri,
    ad.title,
    ad.body,
    rf.name AS risk_factor_tag,
    gt.adm_name AS location_tag,
    arft.article_position_start AS risk_factor_position_start,
    arft.article_position_end AS risk_factor_position_end,
    alt.article_position_start AS location_position_start,
    alt.article_position_end AS location_position_end
FROM selected_articles sa
JOIN article_downloads ad
    ON ad.uri = sa.article_uri
JOIN article_location_tags alt
    ON ad.uri = alt.article_uri
    AND alt.tag_method_id = 1
JOIN article_risk_factor_tags arft
    ON ad.uri = arft.article_uri
    AND arft.tag_method_id = 1
JOIN risk_factors rf
    ON arft.risk_factor = rf.id
JOIN geo_taxonomy gt
    ON alt.adm_code = gt.adm_code
"""


def iter_rows(engine: Any, limit: int, chunk_size: int) -> Iterator[Mapping[str, Any]]:
    article_limit = None if limit == 0 else limit
    statement = text(build_query())
    with engine.connect().execution_options(
        stream_results=True,
        yield_per=chunk_size,
    ) as connection:
        result = connection.execute(statement, {"article_limit": article_limit})
        for row in result.mappings():
            yield row


def write_jsonl(rows: Iterator[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for count, row in enumerate(rows, start=1):
            output_file.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            )
            output_file.write("\n")

    return count


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    password = load_sql_password()
    db_conn_string = f"postgresql://zhai:{password}@localhost:5439/zhai_db"

    # To open the tunnel:
    # gcloud compute ssh zhai-bastion --project=zerohungerai --zone=us-east4-a --tunnel-through-iap --ssh-flag="-L 5439:localhost:5432"
    engine = create_engine(db_conn_string)
    output_path = resolve_path(args.output)
    rows = iter_rows(engine, limit=args.limit, chunk_size=args.chunk_size)
    count = write_jsonl(rows, output_path)
    print(f"Wrote {count} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
