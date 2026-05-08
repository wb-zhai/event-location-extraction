from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "download" / "gdelt.py"
)
SPEC = importlib.util.spec_from_file_location("gdelt_downloader", MODULE_PATH)
gdelt = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gdelt)


def make_feature(url: str, *, name: str = "Nairobi", tone: str = "1.5") -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [36.8219, -1.2921]},
        "properties": {
            "url": url,
            "name": name,
            "tone": tone,
            "lang": "English",
            "seendate": "20260427T120000Z",
        },
    }


def test_discover_article_candidates_dedupes_and_expands_timespan(monkeypatch) -> None:
    payloads = {
        60: {
            "features": [
                make_feature("https://example.com/a"),
                make_feature("https://example.com/a/"),
            ]
        },
        75: {
            "features": [
                make_feature("https://example.com/a"),
                make_feature("https://example.com/b"),
            ]
        },
    }

    def fake_fetch_json(url: str, timeout: float, user_agent: str, retries: int) -> dict:
        assert timeout == 5
        assert user_agent == "ua"
        assert retries == 2
        timespan = int(url.split("TIMESPAN=")[1].split("&", 1)[0])
        return payloads[timespan]

    monkeypatch.setattr(gdelt, "fetch_json", fake_fetch_json)

    candidates, final_timespan = gdelt.discover_article_candidates(
        query="FOOD_SECURITY",
        lang=None,
        initial_timespan_minutes=60,
        max_timespan_minutes=90,
        target_candidate_count=2,
        timeout=5,
        user_agent="ua",
        retries=2,
    )

    assert [candidate["source_url"] for candidate in candidates] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert final_timespan == 75


def test_collect_articles_skips_failed_extractions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        gdelt,
        "discover_article_candidates",
        lambda **_: (
            [
                {
                    "source_url": "https://example.com/a",
                    "feature": make_feature("https://example.com/a"),
                    "timespan_minutes": 60,
                },
                {
                    "source_url": "https://example.com/b",
                    "feature": make_feature("https://example.com/b"),
                    "timespan_minutes": 60,
                },
                {
                    "source_url": "https://example.com/c",
                    "feature": make_feature("https://example.com/c"),
                    "timespan_minutes": 60,
                },
            ],
            60,
        ),
    )

    def fake_http_get(url: str, timeout: float, user_agent: str, retries: int) -> str:
        return f"<html>{url}</html>"

    def fake_extract_article_html(html: str, url: str) -> dict | None:
        if url.endswith("/a"):
            return None
        if url.endswith("/b"):
            raise RuntimeError("paywalled")
        return {
            "title": "Recovered article",
            "text": "word " * 40,
            "publish_date": "2026-04-27",
            "language": "English",
        }

    monkeypatch.setattr(gdelt, "http_get", fake_http_get)
    monkeypatch.setattr(gdelt, "extract_article_html", fake_extract_article_html)

    args = gdelt.parse_args(
        [
            "--limit",
            "1",
            "--output",
            str(tmp_path / "articles.jsonl"),
        ]
    )
    records, summary = gdelt.collect_articles(args)

    assert len(records) == 1
    assert records[0]["source_url"] == "https://example.com/c"
    assert summary["discovered_url_count"] == 3
    assert summary["fetched_article_count"] == 1
    assert summary["skipped_or_failed_count"] == 2


def test_write_jsonl_outputs_expected_schema(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    records = [
        {
            "id": "gdelt_00000",
            "query": "FOOD_SECURITY",
            "gdelt_query_theme": "FOOD_SECURITY",
            "retrieved_at": "2026-04-27T10:00:00+00:00",
            "source_url": "https://example.com/article",
            "title": "Example",
            "text": "word " * 40,
            "language": "English",
            "publish_date": "2026-04-27",
            "gdelt_location_name": "Nairobi",
            "gdelt_coordinates": [36.8219, -1.2921],
            "gdelt_tone": "1.5",
            "gdelt_raw_properties": {"url": "https://example.com/article"},
        }
    ]

    gdelt.write_jsonl(records, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == "gdelt_00000"
    assert parsed["text"].startswith("word")
    assert parsed["gdelt_coordinates"] == [36.8219, -1.2921]


def test_main_writes_fewer_than_requested_when_discovery_exhausted(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    record = {
        "id": "gdelt_00000",
        "query": "FOOD_SECURITY",
        "gdelt_query_theme": "FOOD_SECURITY",
        "retrieved_at": "2026-04-27T10:00:00+00:00",
        "source_url": "https://example.com/article",
        "title": "Example",
        "text": "word " * 40,
        "language": "English",
        "publish_date": "2026-04-27",
        "gdelt_location_name": "Nairobi",
        "gdelt_coordinates": [36.8219, -1.2921],
        "gdelt_tone": "1.5",
        "gdelt_raw_properties": {"url": "https://example.com/article"},
    }

    monkeypatch.setattr(
        gdelt,
        "collect_articles",
        lambda args, output_handle=None: (
            [record],
            {
                "discovered_url_count": 1,
                "fetched_article_count": 1,
                "skipped_or_failed_count": 0,
                "final_timespan_minutes": 1440,
                "output_path": str(args.output),
            },
        ),
    )

    output = tmp_path / "boundary.jsonl"
    exit_code = gdelt.main(["--limit", "2", "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    stderr = capsys.readouterr().err
    assert "requested 2 articles but only 1 were extracted" in stderr
    assert "only 1 unique URLs were discovered for a target of 2" in stderr
