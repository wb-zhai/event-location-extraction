from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.train.eval_sft import evaluate, write_doc_event_report


def _row(text: str, events: list[dict]) -> dict:
    return {"question": text, "answer": {"events": events}}


def _window_row(
    text: str,
    events: list[dict],
    start: int,
    end: int,
    doc_id: str | None = None,
) -> dict:
    row = _row(text, events)
    row["metadata"] = {
        "document_char_start": start,
        "document_char_end": end,
    }
    if doc_id is not None:
        row["metadata"]["doc_id"] = doc_id
    return row


def _pred(events: list[dict]) -> dict:
    return {"prediction": {"events": events}}


def _event(event_type: str, trigger_text: str) -> dict:
    return {"event_type": event_type, "trigger": {"text": trigger_text}}


def _event_at(event_type: str, text: str, start: int, end: int) -> dict:
    return {
        "event_type": event_type,
        "trigger": {"text": text, "start": start, "end": end},
    }


def test_argument_span_payload_is_scored() -> None:
    event = {
        "event_type": "flooding",
        "trigger": {"text": "flooding"},
        "arguments": [
            {
                "role": "location",
                "span": {"text": "Bangladesh"},
                "location_type": "country",
            }
        ],
    }

    metrics = evaluate(
        [_row("flooding affected Bangladesh", [event])],
        [_pred([event])],
    )

    assert metrics["argument"]["strict"]["precision"] == pytest.approx(1.0)
    assert metrics["argument"]["strict"]["recall"] == pytest.approx(1.0)
    assert metrics["argument"]["strict"]["f1"] == pytest.approx(1.0)
    assert metrics["argument"]["trigger_span_only"]["f1"] == pytest.approx(1.0)


def test_argument_document_text_does_not_require_exact_trigger_span() -> None:
    gold_event = {
        "event_type": "flooding",
        "trigger": {"text": "flooding"},
        "arguments": [
            {
                "role": "location",
                "span": {"text": "Bangladesh"},
                "location_type": "country",
            }
        ],
    }
    pred_event = {
        "event_type": "flooding",
        "trigger": {"text": "severe flooding"},
        "arguments": [
            {
                "role": "location",
                "span": {"text": "Bangladesh"},
                "location_type": "country",
            }
        ],
    }

    metrics = evaluate(
        [_row("severe flooding affected Bangladesh", [gold_event])],
        [_pred([pred_event])],
    )

    assert metrics["argument"]["strict"]["f1"] == pytest.approx(0.0)
    assert metrics["argument"]["event_role_span"]["f1"] == pytest.approx(1.0)
    assert metrics["argument"]["document_text"]["f1"] == pytest.approx(1.0)


def test_event_level_collapses_duplicate_gold_mentions() -> None:
    metrics = evaluate(
        [
            _row(
                "extreme weather events and climate crisis",
                [
                    _event("climate event", "extreme weather events"),
                    _event("climate event", "extreme weather events"),
                    _event("climate event", "climate crisis"),
                ],
            )
        ],
        [_pred([_event("climate event", "extreme weather events")])],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["exact_match"] == pytest.approx(1.0)


def test_event_level_collapses_duplicate_predictions() -> None:
    metrics = evaluate(
        [_row("drought and heat", [_event("drought conditions", "drought")])],
        [
            _pred(
                [
                    _event("drought conditions", "drought"),
                    _event("drought conditions", "drought"),
                    _event("drought conditions", "missing trigger"),
                ]
            )
        ],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["exact_match"] == pytest.approx(1.0)


def test_event_level_ignores_wrong_or_ungrounded_trigger() -> None:
    metrics = evaluate(
        [_row("flooding damaged homes", [_event("flooding", "flooding")])],
        [_pred([_event("flooding", "not in the document")])],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(1.0)
    assert metrics["event"]["strict"]["f1"] == pytest.approx(0.0)


def test_event_level_wrong_type_counts_false_positive_and_false_negative() -> None:
    metrics = evaluate(
        [_row("flooding damaged homes", [_event("flooding", "flooding")])],
        [_pred([_event("drought conditions", "flooding")])],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(0.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(0.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(0.0)
    assert metrics["event_type"]["row"]["exact_match"] == pytest.approx(0.0)


def test_doc_event_span_text_collapses_duplicate_prediction_mentions() -> None:
    metrics = evaluate(
        [_row("rain and rain", [_event_at("weather", "rain", 0, 4)])],
        [
            _pred(
                [
                    _event_at("weather", "rain", 0, 4),
                    _event_at("weather", "rain", 9, 13),
                ]
            )
        ],
    )

    assert metrics["event"]["strict"]["precision"] == pytest.approx(0.5)
    assert metrics["event"]["document_text"]["precision"] == pytest.approx(1.0)
    assert metrics["event"]["document_text"]["recall"] == pytest.approx(1.0)
    assert metrics["event"]["document_text"]["f1"] == pytest.approx(1.0)


def test_doc_event_span_text_weak_matches_broader_trigger_string() -> None:
    metrics = evaluate(
        [_row("heavy rain damaged crops", [_event_at("weather", "rain", 6, 10)])],
        [_pred([_event_at("weather", "heavy rain", 0, 10)])],
    )

    assert metrics["event"]["document_text"]["f1"] == pytest.approx(0.0)
    assert metrics["event"]["document_text_weak"]["precision"] == pytest.approx(1.0)
    assert metrics["event"]["document_text_weak"]["recall"] == pytest.approx(1.0)
    assert metrics["event"]["document_text_weak"]["f1"] == pytest.approx(1.0)


def test_event_level_exact_match_requires_equal_row_event_type_sets() -> None:
    metrics = evaluate(
        [
            _row("flooding damaged homes", [_event("flooding", "flooding")]),
            _row("drought and heat", [_event("drought conditions", "drought")]),
        ],
        [
            _pred([_event("flooding", "flooding")]),
            _pred(
                [
                    _event("drought conditions", "drought"),
                    _event("increased heat", "heat"),
                ]
            ),
        ],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(2 / 3)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(0.8)
    assert metrics["event_type"]["row"]["exact_match"] == pytest.approx(0.5)


def test_evaluate_ignores_malformed_prediction_event_items() -> None:
    metrics = evaluate(
        [_row("military conflict", [_event("military conflict", "military conflict")])],
        [
            _pred(
                [
                    _event("military conflict", "military conflict"),
                    "military conflict",
                    {
                        "event_type": "military conflict",
                        "trigger": {"text": "military conflict"},
                        "arguments": ["bad argument"],
                    },
                ]
            )
        ],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["row"]["f1"] == pytest.approx(1.0)
    assert metrics["event"]["strict"]["precision"] == pytest.approx(1.0)
    assert metrics["event"]["strict"]["recall"] == pytest.approx(1.0)
    assert metrics["event"]["strict"]["f1"] == pytest.approx(1.0)


def test_doc_event_level_merges_windows_before_scoring() -> None:
    metrics = evaluate(
        [
            _window_row("flooding damaged homes", [_event("flooding", "flooding")], 0, 23),
            _window_row("homes after flooding", [], 10, 29),
        ],
        [
            _pred([]),
            _pred([_event("flooding", "flooding")]),
        ],
    )

    assert metrics["event_type"]["row"]["precision"] == pytest.approx(0.0)
    assert metrics["event_type"]["row"]["recall"] == pytest.approx(0.0)
    assert metrics["event_type"]["document"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["document"]["recall"] == pytest.approx(1.0)
    assert metrics["event_type"]["document"]["f1"] == pytest.approx(1.0)
    assert metrics["event_type"]["document"]["exact_match"] == pytest.approx(1.0)


def test_doc_event_level_offset_reset_starts_new_document() -> None:
    metrics = evaluate(
        [
            _window_row("flooding damaged homes", [_event("flooding", "flooding")], 0, 23),
            _window_row("drought damaged crops", [_event("drought conditions", "drought")], 0, 21),
        ],
        [
            _pred([_event("flooding", "flooding")]),
            _pred([]),
        ],
    )

    assert metrics["event_type"]["document"]["precision"] == pytest.approx(1.0)
    assert metrics["event_type"]["document"]["recall"] == pytest.approx(0.5)
    assert metrics["event_type"]["document"]["f1"] == pytest.approx(2 / 3)
    assert metrics["event_type"]["document"]["exact_match"] == pytest.approx(0.5)


def test_write_doc_event_report_compares_event_types(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    write_doc_event_report(
        [
            _window_row(
                "flooding and drought",
                [
                    _event("flooding", "flooding"),
                    _event("drought conditions", "drought"),
                ],
                0,
                20,
            ),
            _window_row("more flooding", [_event("flooding", "flooding")], 10, 24),
        ],
        [
            _pred(
                [
                    _event("flooding", "flooding"),
                    _event("increased heat", "heat"),
                ]
            ),
            _pred([_event("flooding", "flooding")]),
        ],
        report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report == {
        "documents": [
            {
                "doc_key": "offset-doc-0",
                "correct": ["flooding"],
                "missed": ["drought conditions"],
                "wrong": ["increased heat"],
            }
        ]
    }


def test_write_doc_event_report_uses_metadata_doc_id(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    write_doc_event_report(
        [
            _window_row(
                "flooding damaged homes",
                [_event("flooding", "flooding")],
                0,
                23,
                doc_id="article-123",
            ),
            _window_row("homes after flooding", [], 10, 29, doc_id="article-123"),
        ],
        [
            _pred([]),
            _pred([_event("flooding", "flooding")]),
        ],
        report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["documents"][0]["doc_key"] == "article-123"
