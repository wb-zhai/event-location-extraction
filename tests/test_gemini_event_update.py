from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = (
    REPO_ROOT / "scripts" / "data" / "generation" / "gemini_event_update.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_event_update", MODULE_PATH)
gemini_event_update = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gemini_event_update)

gemini_event_gen = gemini_event_update.gemini_event_gen


def argument(
    role: str,
    text: str,
    start: int,
    end: int,
    location_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "role": role,
        "text": text,
        "start_char": start,
        "end_char": end,
    }
    if location_type is not None:
        payload["location_type"] = location_type
    return payload


def event(
    trigger_text: str,
    event_type: str,
    start: int,
    end: int,
    arguments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "trigger_text": trigger_text,
        "start_char": start,
        "end_char": end,
        "arguments": arguments or [],
    }


def test_events_target_preserves_original_arguments_on_exact_match() -> None:
    text = "Drought hit Somalia."
    record = {
        "events": [
            event(
                "Drought",
                "weather shocks",
                0,
                7,
                [argument("location", "Somalia", 12, 19, "country")],
            )
        ]
    }

    cleaned = gemini_event_update._clean_updated_payload(
        {
            "events": [
                event("Drought", "weather shocks", 0, 7),
                event("hit", "weather shocks", 8, 11),
            ]
        },
        record,
        text,
        {"weather shocks"},
        gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
        gemini_event_update.TARGET_MODE_EVENTS,
        True,
        {"location"},
        {"weather shocks": ["location"]},
        {"country", "other"},
    )

    assert cleaned == [
        {
            "event_type": "weather shocks",
            "trigger_text": "Drought",
            "start_char": 0,
            "end_char": 7,
            "arguments": [
                {
                    "role": "location",
                    "text": "Somalia",
                    "location_type": "country",
                    "start_char": 12,
                    "end_char": 19,
                }
            ],
            "rationale": "",
        },
        {
            "event_type": "weather shocks",
            "trigger_text": "hit",
            "start_char": 8,
            "end_char": 11,
            "arguments": [],
            "rationale": "",
        },
    ]


def test_arguments_target_preserves_event_fields_and_updates_arguments_only() -> None:
    text = "Drought hit Somalia and Ethiopia."
    record = {
        "events": [
            event(
                "Drought",
                "weather shocks",
                0,
                7,
                [argument("location", "Somalia", 12, 19, "country")],
            )
        ]
    }

    cleaned = gemini_event_update._clean_updated_payload(
        {
            "events": [
                {
                    "event_type": "conflicts and violence",
                    "trigger_text": "hit",
                    "start_char": 8,
                    "end_char": 11,
                    "arguments": [
                        argument("location", "Ethiopia", 24, 32, "country"),
                    ],
                }
            ]
        },
        record,
        text,
        {"weather shocks", "conflicts and violence"},
        gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
        gemini_event_update.TARGET_MODE_ARGUMENTS,
        True,
        {"location"},
        {"weather shocks": ["location"]},
        {"country", "other"},
    )

    assert cleaned == [
        {
            "event_type": "weather shocks",
            "trigger_text": "Drought",
            "start_char": 0,
            "end_char": 7,
            "arguments": [
                {
                    "role": "location",
                    "text": "Ethiopia",
                    "location_type": "country",
                    "start_char": 24,
                    "end_char": 32,
                }
            ],
        }
    ]


def test_arguments_target_keeps_original_arguments_when_event_row_is_missing() -> None:
    text = "Drought hit Somalia."
    record = {
        "events": [
            event(
                "Drought",
                "weather shocks",
                0,
                7,
                [argument("location", "Somalia", 12, 19, "country")],
            )
        ]
    }

    cleaned = gemini_event_update._clean_updated_payload(
        {"events": []},
        record,
        text,
        {"weather shocks"},
        gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
        gemini_event_update.TARGET_MODE_ARGUMENTS,
        True,
        {"location"},
        {"weather shocks": ["location"]},
        {"country", "other"},
    )

    assert cleaned == record["events"]


def test_spans_target_events_uses_normal_span_cleaning() -> None:
    text = "Drought crisis hit Somalia."
    cleaned = gemini_event_update._clean_updated_payload(
        {
            "spans": [
                {
                    "span_text": "Drought crisis",
                    "label": "weather shocks",
                    "start_char": 0,
                    "end_char": 14,
                }
            ]
        },
        {"spans": []},
        text,
        {"weather shocks"},
        gemini_event_gen.OUTPUT_MODE_SPANS,
        gemini_event_update.TARGET_MODE_EVENTS,
        True,
        None,
        None,
        None,
    )

    assert cleaned == [
        {
            "span_text": "Drought crisis",
            "label": "weather shocks",
            "start_char": 0,
            "end_char": 14,
            "rationale": "",
        }
    ]


def test_validate_args_rejects_arguments_target_for_spans() -> None:
    with pytest.raises(SystemExit):
        gemini_event_update.parse_args(
            [
                "input.jsonl",
                "output.jsonl",
                "--ontology",
                "ontology.json",
                "--output-mode",
                gemini_event_gen.OUTPUT_MODE_SPANS,
                "--target-mode",
                gemini_event_update.TARGET_MODE_ARGUMENTS,
            ]
        )
