import argparse
import csv
import html
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import trafilatura
except ImportError:  # pragma: no cover - exercised in environments without the dep.
    trafilatura = None


MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DEFAULT_QUERY = "FOOD_SECURITY"
DEFAULT_OUTPUT = Path("dataset/gdelt/food_security_historical_articles.jsonl")
DEFAULT_USER_AGENT = (
    "event-location-extraction/1.0 "
    "(https://github.com/openai/codex; contact=local-script)"
)
GKG_FILENAME_RE = re.compile(r"/(\d{14})\.(translation\.)?gkg\.csv\.zip$", re.IGNORECASE)
TRANSLATION_LANG_RE = re.compile(r"srclc:([^;]+);", re.IGNORECASE)
PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)

GKG_COLUMNS = [
    "gkg_record_id",
    "date",
    "source_collection_identifier",
    "source_common_name",
    "document_identifier",
    "counts",
    "v2_counts",
    "themes",
    "v2_themes",
    "locations",
    "v2_locations",
    "persons",
    "v2_persons",
    "organizations",
    "v2_organizations",
    "tone",
    "dates",
    "gcam",
    "sharing_image",
    "related_images",
    "social_image_embeds",
    "social_video_embeds",
    "quotations",
    "all_names",
    "amounts",
    "translation_info",
    "extras",
]


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
            "Download historical GDELT GKG matches from the raw archive, then "
            "fetch full article text and save the results as JSONL."
        )
    )
    parser.add_argument("-n", "--limit", type=int, required=True)
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Inclusive UTC date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Inclusive UTC date in YYYY-MM-DD format.",
    )
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Optional exact language code filter from GKG translation metadata, e.g. eng.",
    )
    parser.add_argument(
        "--keep-missing-lang",
        action="store_true",
        help="When --lang is set, keep rows whose GKG translation language is missing instead of filtering them out.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on archive files scanned after date filtering.",
    )
    parser.add_argument(
        "--max-articles-to-scan",
        type=int,
        default=None,
        help="Maximum number of unique article URLs to attempt fetching.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent article fetch/extraction workers.",
    )
    parser.add_argument(
        "--scan-log-every",
        type=int,
        default=5000,
        help="Print a row-scan progress update every N rows within each archive.",
    )
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    args.start_dt = parse_iso_date(args.start_date, "start-date")
    args.end_dt = parse_iso_date(args.end_date, "end-date")
    if args.end_dt < args.start_dt:
        parser.error("--end-date must be on or after --start-date")
    if args.max_articles_to_scan is None:
        args.max_articles_to_scan = max(100, args.limit * 5)
    if args.max_articles_to_scan < args.limit:
        parser.error("--max-articles-to-scan must be >= --limit")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scan_log_every < 1:
        parser.error("--scan-log-every must be at least 1")
    return args


def parse_iso_date(value: str, field_name: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:  # pragma: no cover - exercised by argparse failure paths.
        raise SystemExit(f"--{field_name} must use YYYY-MM-DD format") from exc


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


def http_get_bytes(url: str, timeout: float, user_agent: str, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/octet-stream,application/zip,text/plain,text/html,*/*",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def http_get_text(url: str, timeout: float, user_agent: str, retries: int) -> str:
    body = http_get_bytes(url, timeout=timeout, user_agent=user_agent, retries=retries)
    return body.decode("utf-8", errors="replace")


def parse_masterfilelist(
    text: str,
    *,
    start_dt: datetime,
    end_dt: datetime,
    max_files: int | None = None,
) -> list[str]:
    urls: list[tuple[datetime, str]] = []
    end_exclusive = end_dt.replace(hour=23, minute=59, second=59)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        url = parts[-1]
        match = GKG_FILENAME_RE.search(url)
        if not match:
            continue
        file_dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if not (start_dt <= file_dt <= end_exclusive):
            continue
        urls.append((file_dt, url))

    urls.sort(key=lambda item: item[0])
    selected = [url for _, url in urls]
    if max_files is not None:
        selected = selected[:max_files]
    return selected


def list_archive_files(
    *,
    start_dt: datetime,
    end_dt: datetime,
    timeout: float,
    user_agent: str,
    retries: int,
    max_files: int | None = None,
) -> list[str]:
    masterfile_text = http_get_text(
        MASTERFILELIST_URL,
        timeout=timeout,
        user_agent=user_agent,
        retries=retries,
    )
    return parse_masterfilelist(
        masterfile_text,
        start_dt=start_dt,
        end_dt=end_dt,
        max_files=max_files,
    )


def parse_gkg_row(columns: list[str]) -> dict[str, str]:
    row = {}
    for index, name in enumerate(GKG_COLUMNS):
        row[name] = columns[index] if index < len(columns) else ""
    return row


def split_semicolon_field(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(";") if part]


def theme_matches(row: dict[str, str], query: str) -> bool:
    query_upper = query.upper()
    for field_name in ("themes", "v2_themes"):
        for part in split_semicolon_field(row.get(field_name, "")):
            theme_name = part.split(",", 1)[0].upper()
            if theme_name == query_upper:
                return True
    return False


def extract_language(row: dict[str, str]) -> str | None:
    match = TRANSLATION_LANG_RE.search(row.get("translation_info", ""))
    if match:
        return match.group(1).strip()
    return None


def extract_title_from_extras(extras: str) -> str | None:
    match = PAGE_TITLE_RE.search(extras or "")
    if not match:
        return None
    value = html.unescape(match.group(1)).strip()
    return value or None


def parse_v2_locations(value: str) -> list[dict[str, Any]]:
    locations = []
    for entry in split_semicolon_field(value):
        parts = entry.split("#")
        if len(parts) < 8:
            continue
        try:
            latitude = float(parts[5]) if parts[5] else None
            longitude = float(parts[6]) if parts[6] else None
        except ValueError:
            latitude = None
            longitude = None
        locations.append(
            {
                "location_type": parts[0],
                "full_name": parts[1],
                "country_code": parts[2],
                "adm1_code": parts[3],
                "latitude": latitude,
                "longitude": longitude,
                "feature_id": parts[7],
            }
        )
    return locations


def parse_tone(value: str) -> float | None:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def iter_matching_archive_records(
    archive_bytes: bytes,
    *,
    archive_url: str,
    query: str,
    lang: str | None,
    keep_missing_lang: bool = False,
    scan_log_every: int = 5000,
) -> Iterable[dict[str, Any]]:
    rows_scanned = 0
    theme_hits = 0
    yielded = 0
    lang_filtered = 0
    skipped_lang_counts: Counter[str] = Counter()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for member_name in zf.namelist():
            if not member_name.endswith(".gkg.csv"):
                continue
            print(f"[SCAN] opening {member_name} from {archive_url}", flush=True)
            with zf.open(member_name) as handle:
                with io.TextIOWrapper(handle, encoding="utf-8", errors="replace") as text_handle:
                    reader = csv.reader(text_handle, delimiter="\t")
                    for columns in reader:
                        rows_scanned += 1
                        if rows_scanned % scan_log_every == 0:
                            print(
                                f"[SCAN] rows={rows_scanned} theme_hits={theme_hits} yielded={yielded} archive={archive_url}",
                                flush=True,
                            )
                        row = parse_gkg_row(columns)
                        source_url = canonicalize_url(row.get("document_identifier", ""))
                        if not source_url:
                            continue
                        if not theme_matches(row, query):
                            continue
                        theme_hits += 1
                        row_lang = extract_language(row)
                        if lang:
                            if row_lang is None and keep_missing_lang:
                                pass
                            elif (row_lang or "").lower() != lang.lower():
                                lang_filtered += 1
                                skipped_lang_counts[row_lang or "missing"] += 1
                                continue
                        yielded += 1
                        print(
                            f"[YIELD] #{yielded} lang={row_lang or 'unknown'} url={source_url}",
                            flush=True,
                        )
                        yield {
                            "archive_url": archive_url,
                            "gkg_record_id": row.get("gkg_record_id"),
                            "gdelt_date": row.get("date"),
                            "source_url": source_url,
                            "source_common_name": row.get("source_common_name"),
                            "title_hint": extract_title_from_extras(row.get("extras", "")),
                            "language_hint": row_lang,
                            "v2_locations": parse_v2_locations(row.get("v2_locations", "")),
                            "tone": parse_tone(row.get("tone", "")),
                            "raw_row": row,
                        }
    if skipped_lang_counts:
        summary = ", ".join(
            f"{language}={count}"
            for language, count in skipped_lang_counts.most_common()
        )
        print(
            f"[SCAN] language-skips total={lang_filtered} breakdown={summary} archive={archive_url}",
            flush=True,
        )
    print(
        f"[SCAN] completed rows={rows_scanned} theme_hits={theme_hits} yielded={yielded} lang_filtered={lang_filtered} archive={archive_url}",
        flush=True,
    )


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


def extract_article_html(html_text: str, url: str) -> dict[str, Any] | None:
    if trafilatura is None:
        raise RuntimeError(
            "trafilatura is required for article extraction. Add it to the environment first."
        )

    text = trafilatura.extract(
        html_text,
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
        html_text,
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
    candidate: dict[str, Any],
    extracted: dict[str, Any],
) -> dict[str, Any]:
    first_location = candidate["v2_locations"][0] if candidate["v2_locations"] else None
    coordinates = None
    if first_location and first_location["longitude"] is not None and first_location["latitude"] is not None:
        coordinates = [first_location["longitude"], first_location["latitude"]]

    return {
        "id": article_id,
        "query": query,
        "gdelt_query_theme": query,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_url": candidate["source_url"],
        "title": extracted.get("title") or candidate.get("title_hint"),
        "text": extracted["text"],
        "language": extracted.get("language") or candidate.get("language_hint"),
        "publish_date": extracted.get("publish_date"),
        "gdelt_date": candidate.get("gdelt_date"),
        "gdelt_archive_url": candidate.get("archive_url"),
        "gdelt_source_name": candidate.get("source_common_name"),
        "gdelt_location_name": first_location["full_name"] if first_location else None,
        "gdelt_coordinates": coordinates,
        "gdelt_tone": candidate.get("tone"),
        "gdelt_raw_properties": candidate["raw_row"],
    }


def fetch_candidate_article(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    html_text = http_get_text(
        candidate["source_url"],
        timeout=args.timeout,
        user_agent=args.user_agent,
        retries=args.retries,
    )
    extracted = extract_article_html(html_text, candidate["source_url"])
    if not extracted:
        return None
    return extracted


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
    archive_file_count: int,
    discovered_url_count: int,
    fetched_article_count: int,
    skipped_or_failed_count: int,
    max_files: int | None,
    max_articles_to_scan: int,
) -> list[str]:
    reasons: list[str] = []
    if discovered_url_count < requested_limit:
        if archive_file_count == 0:
            reasons.append("no archive files matched the requested date range")
        else:
            reasons.append("the historical GKG files contained fewer unique matching URLs than requested")
    if max_files is not None and archive_file_count >= max_files and discovered_url_count < requested_limit:
        reasons.append("archive scanning stopped at --max-files before the full date range was exhausted")
    if discovered_url_count >= max_articles_to_scan and fetched_article_count < requested_limit:
        reasons.append(
            "article fetching stopped at --max-articles-to-scan before enough successful extractions"
        )
    if fetched_article_count < min(discovered_url_count, requested_limit) and skipped_or_failed_count:
        reasons.append(
            "some historical URLs could not be downloaded or no longer yield usable article text"
        )
    if not reasons and fetched_article_count < requested_limit:
        reasons.append("the requested number of articles could not be reached with the available candidates")
    return reasons


def collect_articles(
    args: argparse.Namespace,
    *,
    output_handle=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_urls = list_archive_files(
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        timeout=args.timeout,
        user_agent=args.user_agent,
        retries=args.retries,
        max_files=args.max_files,
    )
    print(f"[ARCHIVES] selected {len(archive_urls)} archive files", flush=True)

    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    discovered_url_count = 0
    skipped = 0
    archive_file_count = len(archive_urls)
    archives_scanned = 0
    print(
        f"[FETCH] streaming historical matches until {args.limit} articles are collected "
        f"or {args.max_articles_to_scan} URLs are attempted using {args.workers} workers",
        flush=True,
    )

    max_pending = max(args.workers * 2, args.workers)
    pending: dict[Future, dict[str, Any]] = {}

    def drain_completed(*, block: bool) -> None:
        nonlocal skipped
        if not pending:
            return
        if block:
            done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending if future.done()}
            if not done:
                return

        for future in done:
            candidate = pending.pop(future)
            source_url = candidate["source_url"]
            try:
                extracted = future.result()
            except Exception as exc:
                skipped += 1
                print(f"[SKIP] {source_url} ({exc})", file=sys.stderr)
                continue

            if not extracted:
                skipped += 1
                print(f"[SKIP] {source_url} (no usable article text)", file=sys.stderr)
                continue

            records.append(
                build_output_record(
                    article_id=f"gdelt_hist_{len(records):05d}",
                    query=args.query,
                    candidate=candidate,
                    extracted=extracted,
                )
            )
            if output_handle is not None:
                append_jsonl_record(records[-1], output_handle)
            print_progress("[SUCCESS]", len(records), args.limit)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for archive_index, archive_url in enumerate(archive_urls, start=1):
            if len(records) >= args.limit or discovered_url_count >= args.max_articles_to_scan:
                break

            print_progress("[ARCHIVES]", archive_index, archive_file_count)
            archives_scanned = archive_index
            archive_bytes = http_get_bytes(
                archive_url,
                timeout=args.timeout,
                user_agent=args.user_agent,
                retries=args.retries,
            )
            for candidate in iter_matching_archive_records(
                archive_bytes,
                archive_url=archive_url,
                query=args.query,
                lang=args.lang,
                keep_missing_lang=args.keep_missing_lang,
                scan_log_every=args.scan_log_every,
            ):
                if len(records) >= args.limit or discovered_url_count >= args.max_articles_to_scan:
                    break

                source_url = candidate["source_url"]
                if source_url in seen_urls:
                    continue
                seen_urls.add(source_url)
                discovered_url_count += 1
                print_progress("[MATCHES]", discovered_url_count, args.max_articles_to_scan)
                future = executor.submit(fetch_candidate_article, candidate, args)
                pending[future] = candidate
                print_progress("[FETCH]", discovered_url_count, args.max_articles_to_scan)

                if len(pending) >= max_pending:
                    drain_completed(block=True)
                else:
                    drain_completed(block=False)

            drain_completed(block=False)

        while pending and len(records) < args.limit:
            drain_completed(block=True)

        for future in pending:
            future.cancel()

    summary = {
        "archive_file_count": archives_scanned,
        "discovered_url_count": discovered_url_count,
        "fetched_article_count": len(records),
        "skipped_or_failed_count": skipped,
        "output_path": str(args.output),
    }
    return records, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_handle:
        records, summary = collect_articles(args, output_handle=output_handle)
    print(f"Archive files scanned: {summary['archive_file_count']}")
    print(f"Discovered URLs: {summary['discovered_url_count']}")
    print(f"Fetched articles: {summary['fetched_article_count']}")
    print(f"Skipped/failed: {summary['skipped_or_failed_count']}")
    print(f"Output: {summary['output_path']}")
    if summary["fetched_article_count"] < args.limit:
        reasons = build_shortfall_reasons(
            requested_limit=args.limit,
            archive_file_count=summary["archive_file_count"],
            discovered_url_count=summary["discovered_url_count"],
            fetched_article_count=summary["fetched_article_count"],
            skipped_or_failed_count=summary["skipped_or_failed_count"],
            max_files=args.max_files,
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
