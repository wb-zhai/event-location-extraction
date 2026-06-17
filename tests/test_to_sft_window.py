"""Tests for the paragraph windowing logic in to_sft.py."""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v3.to_sft import (
    split_paragraphs,
    split_oversized_paragraphs,
    locate_event_span,
    build_paragraph_windows,
    coalesce_short_windows,
    _build_windowed_records,
    _print_window_stats,
    filter_annotation,
)


# ---------------------------------------------------------------------------
# split_paragraphs
# ---------------------------------------------------------------------------

def test_split_paragraphs_basic():
    text = "Hello world.\n\nSecond paragraph.\n\nThird."
    paras = split_paragraphs(text)
    assert len(paras) == 3
    for start, end in paras:
        assert text[start:end].strip()

def test_split_paragraphs_offsets_roundtrip():
    text = "Para A.\n\nPara B.\n\nPara C."
    paras = split_paragraphs(text)
    reconstructed = [text[s:e] for s, e in paras]
    assert reconstructed[0] == "Para A."
    assert reconstructed[1] == "Para B."
    assert reconstructed[2] == "Para C."

def test_split_paragraphs_single():
    text = "Just one paragraph."
    paras = split_paragraphs(text)
    assert len(paras) == 1
    assert paras[0] == (0, len(text))

def test_split_paragraphs_empty():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n  ") == []

def test_split_paragraphs_multiple_blank_lines():
    text = "First.\n\n\n\nSecond."
    paras = split_paragraphs(text)
    assert len(paras) == 2

def test_split_oversized_paragraphs_keeps_oversized_sentence_whole():
    text = "word " * 300
    paras = split_paragraphs(text)
    split_paras = split_oversized_paragraphs(text, paras, max_chars=80)
    assert split_paras == [(0, len(text.strip()))]

def test_split_oversized_paragraphs_prefers_sentences():
    sentences = [
        "Flooding damaged homes in the valley.",
        "Officials opened shelters for displaced families.",
        "Road access remained limited after the storm.",
    ]
    text = " ".join(sentences)
    paras = split_paragraphs(text)
    split_paras = split_oversized_paragraphs(text, paras, max_chars=70)
    chunks = [text[start:end] for start, end in split_paras]
    assert chunks == sentences

def test_split_oversized_paragraphs_never_splits_inside_sentence():
    text = (
        "This sentence has no useful punctuation and is intentionally long enough "
        "that it must remain a complete sentence."
    )
    paras = split_paragraphs(text)
    split_paras = split_oversized_paragraphs(text, paras, max_chars=55)
    chunks = [text[start:end] for start, end in split_paras]
    assert chunks == [text]


# ---------------------------------------------------------------------------
# locate_event_span
# ---------------------------------------------------------------------------

def test_locate_event_span_simple():
    text = "The flood hit the village of Springfield in June 2024."
    event = {"grounding_quote": "flood hit the village of Springfield"}
    span = locate_event_span(text, event)
    assert span is not None
    s, e = span
    assert text[s:e] == "flood hit the village of Springfield"

def test_locate_event_span_includes_location_text():
    text = "In July 2023, a drought struck the northern region. Crops failed everywhere."
    event = {
        "grounding_quote": "drought struck the northern region",
        "event_location_text": "northern region",
        "event_time_text": "July 2023",
    }
    span = locate_event_span(text, event)
    assert span is not None
    s, e = span
    window = text[s:e]
    assert "drought struck the northern region" in window
    assert "northern region" in window
    assert "July 2023" in window

def test_locate_event_span_picks_nearest_occurrence():
    # "northern" appears twice; the one near the grounding quote should be chosen
    text = (
        "In the northern highlands, farmers struggled. "
        "A flood hit the northern valley in March.\n\n"
        "Elsewhere, unrelated content."
    )
    event = {
        "grounding_quote": "flood hit the northern valley",
        "event_location_text": "northern",
    }
    span = locate_event_span(text, event)
    assert span is not None
    s, e = span
    window = text[s:e]
    # The span should cover the grounding quote
    assert "flood hit the northern valley" in window

def test_locate_event_span_not_stated_skipped():
    text = "Flooding in May 2023 caused damage."
    event = {
        "grounding_quote": "Flooding in May 2023",
        "event_location_text": "not_stated",
        "event_time_text": "",
    }
    span = locate_event_span(text, event)
    assert span is not None

def test_locate_event_span_missing_quote_returns_none():
    text = "Nothing relevant here."
    event = {"grounding_quote": "this text does not exist in source"}
    span = locate_event_span(text, event)
    assert span is None


# ---------------------------------------------------------------------------
# build_paragraph_windows
# ---------------------------------------------------------------------------

def _make_paras(lengths: list[int]) -> list[tuple[int, int]]:
    """Construct fake paragraph offset tuples from a list of char lengths."""
    paras = []
    cursor = 0
    for length in lengths:
        paras.append((cursor, cursor + length))
        cursor += length + 10  # simulate gap between paragraphs
    return paras

def test_build_paragraph_windows_basic():
    paras = _make_paras([100, 100, 100, 100])
    windows = build_paragraph_windows(paras, max_chars=250, max_paras=10, overlap=0)
    # First window should cover paras 0+1 (200 chars), second paras 2+3
    assert len(windows) == 2
    assert windows[0] == (0, 2)
    assert windows[1] == (2, 4)

def test_build_paragraph_windows_overlap():
    paras = _make_paras([100, 100, 100, 100])
    windows = build_paragraph_windows(paras, max_chars=250, max_paras=10, overlap=1)
    # With overlap=1: window 0 = [0,2), window 1 starts at 2-1+1=2 → [1,3), window 2 = [2,4)
    assert len(windows) == 3
    # Each consecutive pair shares one paragraph
    for i in range(len(windows) - 1):
        assert windows[i][1] - 1 == windows[i + 1][0]

def test_build_paragraph_windows_single_span():
    paras = _make_paras([5000])
    windows = build_paragraph_windows(paras, max_chars=3000, max_paras=10, overlap=1)
    assert len(windows) == 1
    assert windows[0] == (0, 1)

def test_build_paragraph_windows_max_paras_cap():
    paras = _make_paras([50] * 20)  # 20 tiny paragraphs
    windows = build_paragraph_windows(paras, max_chars=10000, max_paras=5, overlap=0)
    for lo, hi in windows:
        assert hi - lo <= 5

def test_build_paragraph_windows_empty():
    assert build_paragraph_windows([], max_chars=3000, max_paras=10, overlap=1) == []

def test_build_paragraph_windows_single():
    paras = _make_paras([200])
    windows = build_paragraph_windows(paras, max_chars=3000, max_paras=10, overlap=1)
    assert windows == [(0, 1)]

def test_coalesce_short_windows_merges_tail_when_it_fits():
    paras = _make_paras([100, 100, 20])
    windows = [(0, 2), (2, 3)]
    coalesced = coalesce_short_windows(
        paras, windows, min_chars=80, max_chars=250, max_paras=10
    )
    assert coalesced == [(0, 3)]

def test_coalesce_short_windows_keeps_short_window_when_merge_exceeds_target():
    paras = _make_paras([100, 100, 20])
    windows = [(0, 2), (2, 3)]
    coalesced = coalesce_short_windows(
        paras, windows, min_chars=80, max_chars=210, max_paras=10
    )
    assert coalesced == windows


# ---------------------------------------------------------------------------
# Full windowed record building
# ---------------------------------------------------------------------------

_SYSTEM = "You are an extractor."
_TEMPLATE = "<article>{{ARTICLE_TEXT}}</article><date>{{PUBLISH_DATE}}</date>"


def _make_row(text: str, events: list[dict]) -> dict:
    return {
        "source": {"text": text, "publish_date": "2024-01-01"},
        "annotation": {"document_relevance": "relevant", "events": events},
    }


def test_windowed_records_event_in_correct_window():
    para1 = "The drought struck southern farmlands badly."
    para2 = "Thousands of families lost their harvest this season."
    text = para1 + "\n\n" + para2
    event = {
        "event_type": "drought",
        "grounding_quote": "drought struck southern farmlands",
        "event_location_text": "southern farmlands",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=200, max_paras=1, overlap=0
    )
    # With max_paras=1, each paragraph is its own window
    assert len(records) == 2
    # Event should appear only in the first window (contains para1)
    events_in_windows = [r["_window_events"] for r in records]
    assert events_in_windows[0] == 1
    assert events_in_windows[1] == 0

def test_windowed_records_event_in_overlap_appears_in_both():
    para1 = "Flooding hit the valley last week."
    para2 = "More rain is expected as flooding continues."  # overlap paragraph
    para3 = "Local authorities issued warnings."
    text = para1 + "\n\n" + para2 + "\n\n" + para3

    # Event grounded in para2 (the overlap paragraph)
    event = {
        "event_type": "flooding",
        "grounding_quote": "flooding continues",
        "event_location_text": "not_stated",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    # max_paras=2, overlap=1 → windows [0,2) and [1,3)
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=10000, max_paras=2, overlap=1
    )
    assert len(records) == 2
    # para2 is in both windows, so the event should appear in both
    assert records[0]["_window_events"] == 1
    assert records[1]["_window_events"] == 1

def test_windowed_records_empty_window_kept():
    para1 = "The conflict caused food insecurity."
    para2 = "This paragraph has nothing relevant."
    text = para1 + "\n\n" + para2
    event = {
        "event_type": "conflict",
        "grounding_quote": "conflict caused food insecurity",
        "event_location_text": "not_stated",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=100, max_paras=1, overlap=0
    )
    assert len(records) == 2
    # One window has the event, one is empty — both should be present
    assert any(r["_window_events"] == 0 for r in records)
    assert any(r["_window_events"] == 1 for r in records)

def test_windowed_records_grounding_text_in_window():
    """Every event's grounding_quote must be a substring of its window's input text."""
    para1 = "Drought conditions worsened in March 2023 across eastern Ethiopia."
    para2 = "Aid organizations responded quickly."
    para3 = "Floods struck the southern lowlands in April, displacing thousands."
    text = para1 + "\n\n" + para2 + "\n\n" + para3

    events = [
        {
            "event_type": "drought",
            "grounding_quote": "Drought conditions worsened",
            "event_location_text": "eastern Ethiopia",
            "event_time_text": "March 2023",
        },
        {
            "event_type": "flooding",
            "grounding_quote": "Floods struck the southern lowlands",
            "event_location_text": "southern lowlands",
            "event_time_text": "April",
        },
    ]
    row = _make_row(text, events)
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=10000, max_paras=15, overlap=1
    )
    for rec in records:
        output = json.loads(rec["output"])
        window_input = rec["input"]
        for ev in output.get("events", []):
            gq = ev.get("grounding_quote", "")
            if gq:
                assert gq in window_input, f"grounding_quote {gq!r} not in window"

def test_windowed_records_ignore_strips_fields():
    text = "Flood hit the region in June."
    event = {
        "event_type": "flooding",
        "grounding_quote": "Flood hit the region",
        "event_location_text": "the region",
        "event_time_text": "June",
    }
    row = _make_row(text, [event])
    ignore = {"event_location_text", "event_time_text", "document_relevance"}
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, ignore, max_chars=10000, max_paras=15, overlap=1
    )
    for rec in records:
        output = json.loads(rec["output"])
        for ev in output.get("events", []):
            assert "event_location_text" not in ev
            assert "event_time_text" not in ev

def test_windowed_records_splits_long_paragraph():
    text = " ".join(
        [
            "Flooding damaged homes in the valley.",
            "Officials opened shelters for displaced families.",
            "Road access remained limited after the storm.",
            "Relief teams delivered water and food.",
        ]
    )
    row = _make_row(text, [])
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=70, max_paras=15, overlap=0
    )
    assert len(records) > 1
    assert max(r["_window_chars"] for r in records) <= 70

def test_windowed_records_refuses_oversized_event_expansion():
    paras = [
        "Grounding flood event starts here.",
        *[f"Filler paragraph {idx}." for idx in range(20)],
        "The distant location marker is here.",
    ]
    text = "\n\n".join(paras)
    event = {
        "event_type": "flooding",
        "grounding_quote": "Grounding flood event starts",
        "event_location_text": "distant location marker",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    records = _build_windowed_records(
        row, _SYSTEM, _TEMPLATE, set(), max_chars=80, max_paras=3, overlap=0
    )
    assert max(r["_window_chars"] for r in records) <= 80
    assert max(r["_window_paras"] for r in records) <= 3
    assert sum(r["_window_events"] for r in records) == 0

def test_windowed_records_drops_short_no_event_window():
    text = "Relevant flooding damaged homes.\n\nTiny."
    event = {
        "event_type": "flooding",
        "grounding_quote": "flooding damaged homes",
        "event_location_text": "not_stated",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    records = _build_windowed_records(
        row,
        _SYSTEM,
        _TEMPLATE,
        set(),
        max_chars=40,
        max_paras=1,
        overlap=0,
        min_chars=10,
    )
    assert len(records) == 1
    assert records[0]["_window_events"] == 1

def test_windowed_records_keeps_short_event_window():
    text = "Flood."
    event = {
        "event_type": "flooding",
        "grounding_quote": "Flood",
        "event_location_text": "not_stated",
        "event_time_text": "not_stated",
    }
    row = _make_row(text, [event])
    records = _build_windowed_records(
        row,
        _SYSTEM,
        _TEMPLATE,
        set(),
        max_chars=40,
        max_paras=1,
        overlap=0,
        min_chars=10,
    )
    assert len(records) == 1
    assert records[0]["_window_chars"] < 10
    assert records[0]["_window_events"] == 1

def test_print_window_stats_includes_windows_without_events(capsys):
    records = [
        {"_window_chars": 100, "_window_paras": 1, "_window_events": 0},
        {"_window_chars": 200, "_window_paras": 2, "_window_events": 1},
        {"_window_chars": 300, "_window_paras": 3, "_window_events": 0},
    ]
    _print_window_stats(records)
    captured = capsys.readouterr()
    assert "windows without events: 2 (66.7%)" in captured.out
