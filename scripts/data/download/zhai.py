from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "dataset/zhai/raw/zhai_with_tags_raw.jsonl"
DEFAULT_ARTICLE_LIMIT = 100


def render_progress(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    current = min(max(current, 0), total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_progress(prefix: str, current: int, total: int) -> None:
    print(
        f"\r{prefix} {render_progress(current, total)}",
        end="",
        file=sys.stderr,
        flush=True,
    )


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
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Sample articles approximately evenly across risk factors.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_selected_articles_cte(
    limit: int, excluded_uris: set[str] | None = None
) -> str:
    limit_clause = "" if limit == 0 else f"\n    LIMIT {limit}"
    exclude_clause = ""
    if excluded_uris:
        uris_str = ", ".join(f"'{u}'" for u in excluded_uris)
        exclude_clause = f" AND article_uri NOT IN ({uris_str})"
    return f"""
WITH selected_articles AS (
    SELECT DISTINCT article_uri
    FROM article_risk_factor_tags
    WHERE tag_method_id = 1{exclude_clause}
    {limit_clause}
)"""


def build_stratified_selected_articles_cte(
    limit: int, per_factor_limit: int, excluded_uris: set[str] | None = None
) -> str:
    final_limit_clause = f"\n    LIMIT {limit}" if limit > 0 else ""
    exclude_clause = ""
    if excluded_uris:
        uris_str = ", ".join(f"'{u}'" for u in excluded_uris)
        exclude_clause = f" AND t.article_uri NOT IN ({uris_str})"
    return f"""
WITH distinct_risk_factors AS (
    SELECT DISTINCT risk_factor
    FROM article_risk_factor_tags
    WHERE tag_method_id = 1
),
selected_articles_raw AS (
    SELECT a.article_uri
    FROM distinct_risk_factors r
    CROSS JOIN LATERAL (
        SELECT article_uri
        FROM article_risk_factor_tags t
        WHERE t.tag_method_id = 1 AND t.risk_factor = r.risk_factor{exclude_clause}
        LIMIT {per_factor_limit}
    ) a
),
selected_articles AS (
    SELECT DISTINCT article_uri
    FROM selected_articles_raw
    {final_limit_clause}
)"""


def build_query(limit: int, excluded_uris: set[str] | None = None) -> str:
    return f"""
{build_selected_articles_cte(limit, excluded_uris)}
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
ORDER BY
    ad.uri,
    arft.article_position_start,
    arft.article_position_end,
    alt.article_position_start,
    alt.article_position_end
"""


def build_stratified_query(
    limit: int, per_factor_limit: int, excluded_uris: set[str] | None = None
) -> str:
    return f"""
{build_stratified_selected_articles_cte(limit, per_factor_limit, excluded_uris)}
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
ORDER BY
    ad.uri,
    arft.article_position_start,
    arft.article_position_end,
    alt.article_position_start,
    alt.article_position_end
"""


def build_article_query(
    limit: int,
    stratified: bool,
    per_factor_limit: int = 0,
    excluded_uris: set[str] | None = None,
) -> str:
    if stratified:
        return build_stratified_query(limit, per_factor_limit, excluded_uris)
    return build_query(limit, excluded_uris)


def build_selected_articles_count_query(
    limit: int,
    stratified: bool,
    per_factor_limit: int = 0,
    excluded_uris: set[str] | None = None,
) -> str:
    selected_articles_cte = (
        build_stratified_selected_articles_cte(limit, per_factor_limit, excluded_uris)
        if stratified
        else build_selected_articles_cte(limit, excluded_uris)
    )
    return f"""
{selected_articles_cte}
SELECT COUNT(*) FROM selected_articles
"""


def iter_rows(
    engine: Any,
    limit: int,
    chunk_size: int,
    stratified: bool = False,
    per_factor_limit: int = 0,
    excluded_uris: set[str] | None = None,
) -> Iterator[Mapping[str, Any]]:
    statement = text(
        build_article_query(
            limit,
            stratified=stratified,
            per_factor_limit=per_factor_limit,
            excluded_uris=excluded_uris,
        )
    )
    with engine.connect().execution_options(
        stream_results=True,
        yield_per=chunk_size,
    ) as connection:
        result = connection.execute(statement)
        for row in result.mappings():
            yield row


def count_selected_articles(
    engine: Any,
    limit: int,
    stratified: bool = False,
    per_factor_limit: int = 0,
    excluded_uris: set[str] | None = None,
) -> int:
    statement = text(
        build_selected_articles_count_query(
            limit,
            stratified=stratified,
            per_factor_limit=per_factor_limit,
            excluded_uris=excluded_uris,
        )
    )
    with engine.connect() as connection:
        return int(connection.execute(statement).scalar_one())


def slice_text(text: str, start: Any, end: Any) -> str:
    start_index = int(start)
    end_index = int(end)
    return text[start_index:end_index]


def build_article_record(article_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    first_row = article_rows[0]
    body = str(first_row["body"] or "")
    events: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()

    for row in article_rows:
        event_start = int(row["risk_factor_position_start"])
        event_end = int(row["risk_factor_position_end"])
        event_type = str(row["risk_factor_tag"])
        event_key = (event_type, event_start, event_end)
        event = events.setdefault(
            event_key,
            {
                "event_type": event_type,
                "trigger_text": slice_text(body, event_start, event_end),
                "start_char": event_start,
                "end_char": event_end,
                "arguments": [],
            },
        )

        argument_start = int(row["location_position_start"])
        argument_end = int(row["location_position_end"])
        argument = {
            "role": "location",
            "text": slice_text(body, argument_start, argument_end),
            "start_char": argument_start,
            "end_char": argument_end,
        }
        argument_key = (
            argument["role"],
            argument["text"],
            argument_start,
            argument_end,
        )
        existing_argument_keys = {
            (
                item["role"],
                item["text"],
                item["start_char"],
                item["end_char"],
            )
            for item in event["arguments"]
        }
        if argument_key not in existing_argument_keys:
            event["arguments"].append(argument)

    return {
        "id": str(first_row["uri"]),
        "status": "ok",
        "source": {
            "title": str(first_row["title"] or ""),
            "text": body,
            "source_url": "",
            "publish_date": "",
        },
        "events": list(events.values()),
    }


def iter_articles(rows: Iterator[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    article_rows: list[Mapping[str, Any]] = []
    last_uri: str | None = None

    for row in rows:
        uri = str(row["uri"])
        if last_uri is None:
            last_uri = uri
        if uri != last_uri:
            yield build_article_record(article_rows)
            article_rows = []
            last_uri = uri
        article_rows.append(row)

    if article_rows:
        yield build_article_record(article_rows)


def write_jsonl(
    articles: Iterator[dict[str, Any]],
    output_path: Path,
    article_total: int,
    offset: int = 0,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if offset > 0 else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as output_file:
        for count, article in enumerate(articles, start=offset + 1):
            print_progress("[ARTICLES]", count, article_total)
            output_file.write(
                json.dumps(
                    article,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                )
            )
            output_file.write("\n")

    if article_total > 0:
        print(file=sys.stderr, flush=True)

    return count


def get_existing_uris(path: Path) -> set[str]:
    uris = set()
    if not path.exists():
        return uris
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "id" in data:
                    uris.add(data["id"])
            except json.JSONDecodeError:
                pass
    return uris


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

    existing_uris = get_existing_uris(output_path)
    offset = len(existing_uris)

    if args.limit > 0 and offset >= args.limit:
        print(f"Already downloaded {offset} articles. Skipping.")
        return 0

    remaining_limit = args.limit - offset if args.limit > 0 else 0

    per_factor_limit = 0
    if args.stratified and remaining_limit > 0:
        with engine.connect() as conn:
            risk_factor_count = conn.execute(
                text(
                    "SELECT COUNT(DISTINCT risk_factor) FROM article_risk_factor_tags WHERE tag_method_id = 1"
                )
            ).scalar_one()
            per_factor_limit = int(
                math.ceil(remaining_limit / max(risk_factor_count, 1))
            )

    article_total = count_selected_articles(
        engine,
        limit=remaining_limit,
        stratified=args.stratified,
        per_factor_limit=per_factor_limit,
        excluded_uris=existing_uris,
    )
    rows = iter_rows(
        engine,
        limit=remaining_limit,
        chunk_size=args.chunk_size,
        stratified=args.stratified,
        per_factor_limit=per_factor_limit,
        excluded_uris=existing_uris,
    )
    articles = iter_articles(rows)
    count = write_jsonl(articles, output_path, article_total, offset=offset)
    total_written = offset + count if count else offset
    print(
        f"Total appended: {count}. Total entries in {output_path}: {total_written}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
