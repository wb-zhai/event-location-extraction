import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import trafilatura
except ImportError:  # pragma: no cover - exercised in environments without the dep.
    trafilatura = None


GDELT_GKG_GEOJSON_URL = "https://api.gdeltproject.org/api/v1/gkg_geojson"
DEFAULT_QUERY = "FOOD_SECURITY"
DEFAULT_OUTPUT = Path("dataset/gdelt/food_security_articles.jsonl")
DEFAULT_USER_AGENT = (
    "event-location-extraction/1.0 "
    "(https://github.com/openai/codex; contact=local-script)"
)
TIMESPAN_STEP_MINUTES = 15


def render_progress(current: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    current = min(max(current, 0), total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def print_progress(prefix: str, current: int, total: int) -> None:
    print(f"{prefix} {render_progress(current, total)}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the most recent GDELT GKG article matches for a query and "
            "save cleaned article text plus metadata as JSONL."
        )
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        required=True,
        help="Number of successfully extracted articles to save.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path to the JSONL output file. Defaults to "
            f"{DEFAULT_OUTPUT.as_posix()}."
        ),
    )
    parser.add_argument(
        "--timespan-minutes",
        type=int,
        default=60,
        help="Initial GDELT recency window in minutes. Rounded to 15-minute steps.",
    )
    parser.add_argument(
        "--max-timespan-minutes",
        type=int,
        default=1440,
        help="Maximum GDELT recency window in minutes. GDELT caps this at 1440.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="GDELT query token or phrase. Defaults to FOOD_SECURITY.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Optional language filter, for example 'English' or 'eng'.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for both GDELT and article fetches.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent header to send on outgoing HTTP requests.",
    )
    parser.add_argument(
        "--max-articles-to-scan",
        type=int,
        default=None,
        help=(
            "Maximum number of unique article URLs to attempt fetching. "
            "Defaults to max(50, 3 * limit)."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of HTTP fetch attempts per request.",
    )

    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.timespan_minutes < TIMESPAN_STEP_MINUTES:
        parser.error("--timespan-minutes must be at least 15")
    if args.max_timespan_minutes < args.timespan_minutes:
        parser.error("--max-timespan-minutes must be >= --timespan-minutes")
    if args.max_timespan_minutes > 1440:
        parser.error("--max-timespan-minutes cannot exceed 1440")
    if args.max_articles_to_scan is None:
        args.max_articles_to_scan = max(50, args.limit * 3)
    if args.max_articles_to_scan < args.limit:
        parser.error("--max-articles-to-scan must be >= --limit")
    return args


def round_timespan_minutes(value: int) -> int:
    rounded = int(round(value / TIMESPAN_STEP_MINUTES) * TIMESPAN_STEP_MINUTES)
    return min(1440, max(TIMESPAN_STEP_MINUTES, rounded))


def build_gdelt_query(query: str, lang: str | None = None) -> str:
    terms = []
    if lang:
        terms.append(f"lang:{lang}")
    if query:
        terms.append(query)
    return ",".join(terms)


def build_gdelt_url(query: str, timespan_minutes: int) -> str:
    params = {
        "QUERY": query,
        "TIMESPAN": str(timespan_minutes),
        "OUTPUTTYPE": "1",
        "OUTPUTFIELDS": "url,name,tone,lang,seendate,sourcecountry",
    }
    return f"{GDELT_GKG_GEOJSON_URL}?{urlencode(params)}"


def canonicalize_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def http_get(url: str, timeout: float, user_agent: str, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_json(url: str, timeout: float, user_agent: str, retries: int) -> dict[str, Any]:
    body = http_get(url, timeout=timeout, user_agent=user_agent, retries=retries)
    return json.loads(body)


def get_feature_properties(feature: dict[str, Any]) -> dict[str, Any]:
    properties = feature.get("properties")
    if isinstance(properties, dict):
        return properties
    return {}


def get_feature_url(feature: dict[str, Any]) -> str:
    properties = get_feature_properties(feature)
    for key in ("url", "URL", "sourceurl", "SourceURL"):
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_geometry_coordinates(feature: dict[str, Any]) -> list[float] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    try:
        return [float(coordinates[0]), float(coordinates[1])]
    except (TypeError, ValueError):
        return None


def discover_article_candidates(
    *,
    query: str,
    lang: str | None,
    initial_timespan_minutes: int,
    max_timespan_minutes: int,
    target_candidate_count: int,
    timeout: float,
    user_agent: str,
    retries: int,
) -> tuple[list[dict[str, Any]], int]:
    gdelt_query = build_gdelt_query(query, lang)
    timespan_minutes = round_timespan_minutes(initial_timespan_minutes)
    max_timespan_minutes = round_timespan_minutes(max_timespan_minutes)
    seen_urls: set[str] = set()
    candidates: list[dict[str, Any]] = []
    requested_total = target_candidate_count

    while True:
        print(
            f"[DISCOVER] querying GDELT with timespan={timespan_minutes}m "
            f"for up to {requested_total} unique URLs",
            flush=True,
        )
        payload = fetch_json(
            build_gdelt_url(gdelt_query, timespan_minutes),
            timeout=timeout,
            user_agent=user_agent,
            retries=retries,
        )
        features = payload.get("features") or []
        for feature in features:
            if not isinstance(feature, dict):
                continue
            source_url = canonicalize_url(get_feature_url(feature))
            if not source_url or source_url in seen_urls:
                continue
            seen_urls.add(source_url)
            candidates.append(
                {
                    "source_url": source_url,
                    "feature": feature,
                    "timespan_minutes": timespan_minutes,
                }
            )
            print_progress("[DISCOVER]", len(candidates), requested_total)
            if len(candidates) >= target_candidate_count:
                return candidates, timespan_minutes

        if timespan_minutes >= max_timespan_minutes:
            return candidates, timespan_minutes
        next_timespan = min(max_timespan_minutes, timespan_minutes + TIMESPAN_STEP_MINUTES)
        if next_timespan == timespan_minutes:
            return candidates, timespan_minutes
        timespan_minutes = next_timespan


def _metadata_value(metadata: Any, *keys: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value
        return None
    for key in keys:
        value = getattr(metadata, key, None)
        if value not in (None, ""):
            return value
    return None


def extract_article_html(html: str, url: str) -> dict[str, Any] | None:
    if trafilatura is None:
        raise RuntimeError(
            "trafilatura is required for article extraction. Add it to the environment first."
        )

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text:
        return None

    cleaned_text = text.strip()
    if len(cleaned_text.split()) < 30:
        return None

    metadata = trafilatura.bare_extraction(
        html,
        url=url,
        favor_precision=True,
        with_metadata=True,
    )
    return {
        "title": _metadata_value(metadata, "title"),
        "text": cleaned_text,
        "publish_date": _metadata_value(metadata, "date", "publish_date"),
        "language": _metadata_value(metadata, "language", "lang"),
    }


def build_output_record(
    *,
    article_id: str,
    query: str,
    source_url: str,
    extracted: dict[str, Any],
    feature: dict[str, Any],
) -> dict[str, Any]:
    properties = get_feature_properties(feature)
    return {
        "id": article_id,
        "query": query,
        "gdelt_query_theme": query,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "title": extracted.get("title"),
        "text": extracted["text"],
        "language": extracted.get("language") or properties.get("lang"),
        "publish_date": extracted.get("publish_date"),
        "gdelt_location_name": properties.get("name"),
        "gdelt_coordinates": get_geometry_coordinates(feature),
        "gdelt_tone": properties.get("tone"),
        "gdelt_raw_properties": properties,
    }


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def append_jsonl_record(record: dict[str, Any], output_handle) -> None:
    output_handle.write(json.dumps(record, ensure_ascii=False))
    output_handle.write("\n")
    output_handle.flush()


def build_shortfall_reasons(
    *,
    requested_limit: int,
    discovered_url_count: int,
    fetched_article_count: int,
    skipped_or_failed_count: int,
    final_timespan_minutes: int,
    max_timespan_minutes: int,
    max_articles_to_scan: int,
) -> list[str]:
    reasons: list[str] = []
    if discovered_url_count < requested_limit:
        if final_timespan_minutes >= max_timespan_minutes:
            reasons.append(
                "discovery exhausted the maximum GDELT timespan before finding enough unique URLs"
            )
        else:
            reasons.append("GDELT returned fewer unique URLs than requested in the scanned window")
    if discovered_url_count >= max_articles_to_scan and fetched_article_count < requested_limit:
        reasons.append(
            "article fetching stopped at --max-articles-to-scan before enough successful extractions"
        )
    if fetched_article_count < min(discovered_url_count, requested_limit) and skipped_or_failed_count:
        reasons.append(
            "some discovered URLs could not be downloaded or did not yield usable article text"
        )
    if not reasons and fetched_article_count < requested_limit:
        reasons.append("the requested number of articles could not be reached with the available candidates")
    return reasons


def collect_articles(
    args: argparse.Namespace,
    *,
    output_handle=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, final_timespan = discover_article_candidates(
        query=args.query,
        lang=args.lang,
        initial_timespan_minutes=args.timespan_minutes,
        max_timespan_minutes=args.max_timespan_minutes,
        target_candidate_count=args.max_articles_to_scan,
        timeout=args.timeout,
        user_agent=args.user_agent,
        retries=args.retries,
    )

    records: list[dict[str, Any]] = []
    skipped = 0

    scan_total = min(len(candidates), args.max_articles_to_scan)
    print(
        f"[FETCH] attempting extraction from {scan_total} discovered URLs "
        f"to collect {args.limit} articles",
        flush=True,
    )

    for index, candidate in enumerate(candidates[: args.max_articles_to_scan], start=1):
        if len(records) >= args.limit:
            break

        source_url = candidate["source_url"]
        print_progress("[FETCH]", index, scan_total)
        try:
            html = http_get(
                source_url,
                timeout=args.timeout,
                user_agent=args.user_agent,
                retries=args.retries,
            )
            extracted = extract_article_html(html, source_url)
        except Exception as exc:
            skipped += 1
            print(f"[SKIP] {source_url} ({exc})", file=sys.stderr)
            continue

        if not extracted:
            skipped += 1
            print(f"[SKIP] {source_url} (no usable article text)", file=sys.stderr)
            continue

        article_id = f"gdelt_{len(records):05d}"
        records.append(
            build_output_record(
                article_id=article_id,
                query=args.query,
                source_url=source_url,
                extracted=extracted,
                feature=candidate["feature"],
            )
        )
        if output_handle is not None:
            append_jsonl_record(records[-1], output_handle)

    summary = {
        "discovered_url_count": len(candidates),
        "fetched_article_count": len(records),
        "skipped_or_failed_count": skipped,
        "final_timespan_minutes": final_timespan,
        "output_path": str(args.output),
    }
    return records, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_handle:
        records, summary = collect_articles(args, output_handle=output_handle)

    print(f"Discovered URLs: {summary['discovered_url_count']}")
    print(f"Fetched articles: {summary['fetched_article_count']}")
    print(f"Skipped/failed: {summary['skipped_or_failed_count']}")
    print(f"Final timespan (minutes): {summary['final_timespan_minutes']}")
    print(f"Output: {summary['output_path']}")
    if summary["fetched_article_count"] < args.limit:
        reasons = build_shortfall_reasons(
            requested_limit=args.limit,
            discovered_url_count=summary["discovered_url_count"],
            fetched_article_count=summary["fetched_article_count"],
            skipped_or_failed_count=summary["skipped_or_failed_count"],
            final_timespan_minutes=summary["final_timespan_minutes"],
            max_timespan_minutes=args.max_timespan_minutes,
            max_articles_to_scan=args.max_articles_to_scan,
        )
        print(
            (
                f"Warning: requested {args.limit} articles but only "
                f"{summary['fetched_article_count']} were extracted."
            ),
            file=sys.stderr,
        )
        if summary["discovered_url_count"] < args.limit:
            print(
                (
                    f"Reason: only {summary['discovered_url_count']} unique URLs were discovered "
                    f"for a target of {args.limit}."
                ),
                file=sys.stderr,
            )
        for reason in reasons:
            print(f"Reason: {reason}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
