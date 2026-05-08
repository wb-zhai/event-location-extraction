from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data"
    / "download"
    / "gdelt_historical.py"
)
SPEC = importlib.util.spec_from_file_location("gdelt_historical_downloader", MODULE_PATH)
gdelt_historical = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gdelt_historical)


def build_archive_bytes(rows: list[list[str]]) -> bytes:
    csv_bytes = "".join("\t".join(row) + "\n" for row in rows).encode("utf-8")

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20250427000000.gkg.csv", csv_bytes)
    return zip_bytes.getvalue()


def make_row(url: str, *, themes: str, v2_themes: str = "", translation_info: str = "") -> list[str]:
    row = [""] * len(gdelt_historical.GKG_COLUMNS)
    row[0] = "20250427000000-1"
    row[1] = "20250427000000"
    row[2] = "1"
    row[3] = "example.com"
    row[4] = url
    row[7] = themes
    row[8] = v2_themes
    row[10] = "1#Kenya#KE#KE00#0#-1.2921#36.8219#K123"
    row[15] = "1.25,0,0,0,0,0"
    row[25] = translation_info
    row[26] = "<PAGE_TITLE>Sample Title</PAGE_TITLE>"
    return row


def test_parse_masterfilelist_filters_date_range() -> None:
    masterfile_text = "\n".join(
        [
            "1 abc http://data.gdeltproject.org/gdeltv2/20240427000000.gkg.csv.zip",
            "1 abc http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
            "1 abc http://data.gdeltproject.org/gdeltv2/20250428000000.translation.gkg.csv.zip",
        ]
    )

    urls = gdelt_historical.parse_masterfilelist(
        masterfile_text,
        start_dt=gdelt_historical.parse_iso_date("2025-04-27", "start-date"),
        end_dt=gdelt_historical.parse_iso_date("2025-04-27", "end-date"),
    )

    assert urls == [
        "http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
    ]


def test_iter_matching_archive_records_filters_theme_and_lang() -> None:
    archive_bytes = build_archive_bytes(
        [
            make_row(
                "https://example.com/a",
                themes="FOOD_SECURITY;OTHER",
                translation_info="srclc:eng;",
            ),
            make_row(
                "https://example.com/b",
                themes="OTHER",
                v2_themes="FOOD_SECURITY,12;DROUGHT,20",
                translation_info="srclc:fra;",
            ),
            make_row("https://example.com/c", themes="OTHER"),
        ]
    )

    records = list(
        gdelt_historical.iter_matching_archive_records(
            archive_bytes,
            archive_url="http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
            query="FOOD_SECURITY",
            lang="eng",
            scan_log_every=1000,
        )
    )

    assert [record["source_url"] for record in records] == ["https://example.com/a"]
    assert records[0]["title_hint"] == "Sample Title"
    assert records[0]["gdelt_date"] == "20250427000000"


def test_iter_matching_archive_records_keeps_missing_lang_when_enabled() -> None:
    archive_bytes = build_archive_bytes(
        [
            make_row(
                "https://example.com/a",
                themes="FOOD_SECURITY;OTHER",
                translation_info="",
            ),
            make_row(
                "https://example.com/b",
                themes="FOOD_SECURITY",
                translation_info="srclc:fra;",
            ),
        ]
    )

    records = list(
        gdelt_historical.iter_matching_archive_records(
            archive_bytes,
            archive_url="http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
            query="FOOD_SECURITY",
            lang="eng",
            keep_missing_lang=True,
            scan_log_every=1000,
        )
    )

    assert [record["source_url"] for record in records] == ["https://example.com/a"]
    assert records[0]["language_hint"] is None


def test_collect_articles_scans_archives_and_writes_results(monkeypatch, tmp_path: Path) -> None:
    opened_archives: list[str] = []

    monkeypatch.setattr(
        gdelt_historical,
        "list_archive_files",
        lambda **_: [
            "http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
            "http://data.gdeltproject.org/gdeltv2/20250427150000.gkg.csv.zip",
        ],
    )

    def fake_http_get_bytes(url: str, timeout: float, user_agent: str, retries: int) -> bytes:
        opened_archives.append(url)
        return b"archive-bytes"

    def fake_iter_matching_archive_records(
        archive_bytes: bytes,
        *,
        archive_url: str,
        query: str,
        lang: str | None,
        keep_missing_lang: bool,
        scan_log_every: int,
    ):
        assert archive_bytes == b"archive-bytes"
        assert keep_missing_lang is False
        assert scan_log_every == 5000
        yield {
            "archive_url": archive_url,
            "gkg_record_id": "20250427000000-1",
            "gdelt_date": "20250427000000",
            "source_url": "https://example.com/a",
            "source_common_name": "example.com",
            "title_hint": "Archive title",
            "language_hint": "eng",
            "v2_locations": [
                {
                    "full_name": "Kenya",
                    "latitude": -1.2921,
                    "longitude": 36.8219,
                }
            ],
            "tone": 1.25,
            "raw_row": {"document_identifier": "https://example.com/a"},
        }

    monkeypatch.setattr(gdelt_historical, "http_get_bytes", fake_http_get_bytes)
    monkeypatch.setattr(
        gdelt_historical,
        "iter_matching_archive_records",
        fake_iter_matching_archive_records,
    )
    monkeypatch.setattr(
        gdelt_historical,
        "http_get_text",
        lambda url, timeout, user_agent, retries: "<html>ok</html>",
    )
    monkeypatch.setattr(
        gdelt_historical,
        "extract_article_html",
        lambda html_text, url: {
            "title": "Fetched title",
            "text": "word " * 40,
            "publish_date": "2025-04-27",
            "language": "eng",
        },
    )

    args = gdelt_historical.parse_args(
        [
            "--limit",
            "1",
            "--start-date",
            "2025-04-27",
            "--end-date",
            "2025-04-27",
            "--output",
            str(tmp_path / "historical.jsonl"),
        ]
    )
    records, summary = gdelt_historical.collect_articles(args)
    gdelt_historical.write_jsonl(records, args.output)

    assert len(records) == 1
    assert summary["archive_file_count"] == 1
    assert summary["discovered_url_count"] == 1
    assert summary["fetched_article_count"] == 1
    assert opened_archives == ["http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip"]
    parsed = json.loads(args.output.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["source_url"] == "https://example.com/a"
    assert parsed["gdelt_archive_url"].endswith(".gkg.csv.zip")


def test_main_prints_shortfall_reason(monkeypatch, tmp_path: Path, capsys) -> None:
    record = {
        "id": "gdelt_hist_00000",
        "query": "FOOD_SECURITY",
        "gdelt_query_theme": "FOOD_SECURITY",
        "retrieved_at": "2026-04-27T10:00:00+00:00",
        "source_url": "https://example.com/article",
        "title": "Example",
        "text": "word " * 40,
        "language": "eng",
        "publish_date": "2025-04-27",
        "gdelt_date": "20250427000000",
        "gdelt_archive_url": "http://data.gdeltproject.org/gdeltv2/20250427000000.gkg.csv.zip",
        "gdelt_source_name": "example.com",
        "gdelt_location_name": "Kenya",
        "gdelt_coordinates": [36.8219, -1.2921],
        "gdelt_tone": 1.25,
        "gdelt_raw_properties": {"document_identifier": "https://example.com/article"},
    }

    monkeypatch.setattr(
        gdelt_historical,
        "collect_articles",
        lambda args, output_handle=None: (
            [record],
            {
                "archive_file_count": 1,
                "discovered_url_count": 1,
                "fetched_article_count": 1,
                "skipped_or_failed_count": 0,
                "output_path": str(args.output),
            },
        ),
    )

    output = tmp_path / "historical_shortfall.jsonl"
    exit_code = gdelt_historical.main(
        [
            "--limit",
            "3",
            "--start-date",
            "2025-04-27",
            "--end-date",
            "2025-04-27",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    stderr = capsys.readouterr().err
    assert "only 1 unique URLs were discovered for a target of 3" in stderr
