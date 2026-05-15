from __future__ import annotations

import argparse
import importlib.util
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "scripts" / "data" / "generation" / "gemini_event_gen.py"
SPEC = importlib.util.spec_from_file_location("gemini_event_gen", MODULE_PATH)
gemini_event_gen = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gemini_event_gen)


class FakeResponse:
    def __init__(self, parsed: dict[str, Any], metadata: dict[str, Any] | None = None):
        self.parsed = parsed
        self.text = json.dumps(parsed)
        self.metadata = metadata or {}


class FakeGeminiClient:
    model_name = "fake-gemini"

    def __init__(self, outcomes: list[dict[str, Any] | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        yield FakeResponse(outcome, {"prompt_tokens": 10, "completion_tokens": 2})


def span(
    text: str,
    label: str,
    start: int,
    end: int,
    rationale: str = "",
) -> dict[str, Any]:
    return {
        "span_text": text,
        "label": label,
        "start_char": start,
        "end_char": end,
        "rationale": rationale,
    }


def event(
    trigger_text: str,
    event_type: str,
    start: int,
    end: int,
    arguments: list[dict[str, Any]] | None = None,
    rationale: str = "",
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "trigger_text": trigger_text,
        "start_char": start,
        "end_char": end,
        "arguments": arguments or [],
        "rationale": rationale,
    }


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


def formatted_example_payloads(formatted: str) -> list[dict[str, Any]]:
    payloads = []
    for chunk in formatted.split("\n\n"):
        if not chunk.startswith("Example "):
            continue
        _, payload = chunk.split(":\n", maxsplit=1)
        payloads.append(json.loads(payload))
    return payloads


def make_worker_args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "model": "extract-model",
        "temperature": 0.0,
        "max_tokens": 1024,
        "reasoning_effort": "disable",
        "max_retries": 0,
        "initial_backoff": 0,
        "max_backoff": 0,
        "strict_offsets": True,
        "self_consistency": False,
        "self_consistency_samples": 1,
        "self_consistency_temperature": 0.0,
        "self_consistency_min_successful_samples": 1,
        "output_mode": gemini_event_gen.OUTPUT_MODE_SPANS,
        "long_document_mode": False,
        "long_document_threshold_chars": 1000,
        "window_target_chars": 1000,
        "window_max_chars": 1500,
        "window_overlap_sentences": 1,
        "enable_verifier": False,
        "enable_synthetic_gaps": False,
        "enable_relevance_filter": False,
        "relevance_model": None,
        "relevance_max_chars": 3000,
        "relevance_confidence_threshold": 0.8,
        "example_sample_size": None,
        "verbose": False,
        "mode": "quality_first",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_format_examples_uses_only_annotated_window() -> None:
    prefix = "Unannotated background. " * 20
    trigger = "drought crisis"
    suffix = " More unrelated background." * 20
    text = prefix + trigger + suffix
    start = len(prefix)
    record = {
        "title": "Example title is omitted",
        "text": text,
        "events": [event(trigger, "weather shocks", start, start + len(trigger))],
    }

    payloads = formatted_example_payloads(
        gemini_event_gen.format_examples([record], gemini_event_gen.OUTPUT_MODE_SPANS)
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert "title" not in payload
    assert "Unannotated background. Unannotated background." not in payload["text"]
    assert trigger in payload["text"]
    assert len(payload["text"]) <= 1000


def test_format_examples_shifts_span_offsets_to_window() -> None:
    text = "Intro. Drought crisis damaged crops. Outro."
    start = text.index("Drought crisis")
    record = {
        "text": text,
        "events": [event("Drought crisis", "weather shocks", start, start + 14)],
    }

    payload = formatted_example_payloads(
        gemini_event_gen.format_examples([record], gemini_event_gen.OUTPUT_MODE_SPANS)
    )[0]
    span = payload["expected_output"]["spans"][0]

    assert payload["text"][span["start_char"] : span["end_char"]] == "Drought crisis"
    assert span["start_char"] == payload["text"].index("Drought crisis")


def test_format_examples_retains_arguments_inside_event_window() -> None:
    text = "Somalia faced a drought crisis after poor rains."
    location_start = text.index("Somalia")
    trigger_start = text.index("drought crisis")
    record = {
        "text": text,
        "events": [
            event(
                "drought crisis",
                "weather shocks",
                trigger_start,
                trigger_start + len("drought crisis"),
                arguments=[
                    argument(
                        "location",
                        "Somalia",
                        location_start,
                        location_start + len("Somalia"),
                    )
                ],
            )
        ],
    }

    payload = formatted_example_payloads(
        gemini_event_gen.format_examples(
            [record], gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        )
    )[0]
    output_event = payload["expected_output"]["events"][0]
    output_argument = output_event["arguments"][0]

    assert payload["text"][
        output_event["start_char"] : output_event["end_char"]
    ] == "drought crisis"
    assert payload["text"][
        output_argument["start_char"] : output_argument["end_char"]
    ] == "Somalia"


def test_format_examples_splits_distant_annotations() -> None:
    first = "drought crisis"
    second = "locust outbreak"
    text = first + (" filler." * 180) + " " + second
    second_start = text.index(second)
    record = {
        "text": text,
        "events": [
            event(first, "weather shocks", 0, len(first)),
            event(
                second,
                "pests and diseases",
                second_start,
                second_start + len(second),
            ),
        ],
    }

    payloads = formatted_example_payloads(
        gemini_event_gen.format_examples([record], gemini_event_gen.OUTPUT_MODE_SPANS)
    )

    assert len(payloads) == 2
    assert all(len(payload["text"]) <= 1000 for payload in payloads)
    assert first in payloads[0]["text"]
    assert second in payloads[1]["text"]


def test_sample_examples_samples_exact_windows_from_pool(monkeypatch: Any) -> None:
    first = "drought crisis"
    second = "locust outbreak"
    article_a = first + (" filler." * 180) + " " + second
    second_start = article_a.index(second)
    article_b = "Flooding damaged crops."
    record_a = {
        "text": article_a,
        "events": [
            event(first, "weather shocks", 0, len(first)),
            event(
                second,
                "pests and diseases",
                second_start,
                second_start + len(second),
            ),
        ],
    }
    record_b = {
        "text": article_b,
        "events": [event("Flooding", "weather shocks", 0, len("Flooding"))],
    }
    sampled_population: list[dict[str, Any]] = []

    def fake_random_sample(
        population: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        nonlocal sampled_population
        sampled_population = population
        assert k == 2
        return [population[2], population[0]]

    monkeypatch.setattr(gemini_event_gen.random, "sample", fake_random_sample)

    sampled = gemini_event_gen.sample_examples(
        [record_a, record_b],
        2,
        gemini_event_gen.OUTPUT_MODE_SPANS,
    )

    assert len(sampled_population) == 3
    assert len(sampled) == 2
    assert all(record["_compact_example_window"] for record in sampled)
    assert "Flooding" in sampled[0]["text"]
    assert first in sampled[1]["text"]


def test_sample_examples_discards_windows_without_labels() -> None:
    labeled = {
        "text": "Drought crisis.",
        "events": [event("Drought", "weather shocks", 0, len("Drought"))],
    }
    empty_label = {
        "text": "Flooding.",
        "events": [event("Flooding", "", 0, len("Flooding"))],
    }
    bad_offsets = {
        "text": "Locust outbreak.",
        "events": [event("Locust", "pests and diseases", 50, 56)],
    }

    sampled = gemini_event_gen.sample_examples(
        [labeled, empty_label, bad_offsets],
        10,
        gemini_event_gen.OUTPUT_MODE_SPANS,
    )

    assert len(sampled) == 1
    assert sampled[0]["expected_output"]["spans"][0]["label"] == "weather shocks"


def test_merge_self_consistency_spans_keeps_majority_only() -> None:
    text = "drought crisis and locust outbreak"
    drought = span("drought crisis", "weather shocks", 0, 14, "weather")
    locust = span("locust outbreak", "pests and diseases", 19, 34, "pest")

    merged, support, threshold = gemini_event_gen.merge_self_consistency_spans(
        [[drought, locust], [drought], [drought], [locust], []],
        text,
    )

    assert threshold == 3
    assert merged == [{**drought, "span_text": "drought crisis", "support": 3}]
    assert support[(0, 14, "weather shocks")] == 3
    assert support[(19, 34, "pests and diseases")] == 2


def test_merge_self_consistency_spans_counts_duplicate_once_per_sample() -> None:
    text = "drought crisis"
    drought = span("drought crisis", "weather shocks", 0, 14)

    merged, support, threshold = gemini_event_gen.merge_self_consistency_spans(
        [[drought, drought], [], []],
        text,
    )

    assert threshold == 2
    assert merged == []
    assert support[(0, 14, "weather shocks")] == 1


def test_merge_self_consistency_spans_uses_most_common_rationale_first_seen_tie() -> None:
    text = "drought crisis"

    merged, _, _ = gemini_event_gen.merge_self_consistency_spans(
        [
            [span("drought crisis", "weather shocks", 0, 14, "first")],
            [span("drought crisis", "weather shocks", 0, 14, "second")],
            [span("drought crisis", "weather shocks", 0, 14, "second")],
        ],
        text,
    )

    assert merged[0]["rationale"] == "second"
    assert merged[0]["span_text"] == text[0:14]


def test_merge_self_consistency_events_keeps_majority_events_and_arguments() -> None:
    text = "missiles from Gaza again raining down on Israel"
    attack = event(
        "raining down",
        "conflicts and violence",
        25,
        37,
        [
            argument("source_location", "Gaza", 14, 18, "city"),
            argument("target_location", "Israel", 41, 47),
        ],
        "attack",
    )
    minority_event = event("missiles", "conflicts and violence", 0, 8)

    merged, event_support, argument_support, threshold = (
        gemini_event_gen.merge_self_consistency_events(
            [
                [attack, minority_event],
                [
                    event(
                        "raining down",
                        "conflicts and violence",
                        25,
                        37,
                        [argument("source_location", "Gaza", 14, 18, "country")],
                        "conflict",
                    )
                ],
                [
                    event(
                        "raining down",
                        "conflicts and violence",
                        25,
                        37,
                        [
                            argument("source_location", "Gaza", 14, 18),
                            argument("location", "Israel", 41, 47),
                        ],
                        "conflict",
                    )
                ],
            ],
            text,
        )
    )

    assert threshold == 2
    assert merged == [
        {
            "event_type": "conflicts and violence",
            "trigger_text": "raining down",
            "start_char": 25,
            "end_char": 37,
            "arguments": [
                {
                    "role": "source_location",
                    "text": "Gaza",
                    "location_type": "city",
                    "start_char": 14,
                    "end_char": 18,
                    "support": 3,
                }
            ],
            "rationale": "conflict",
            "support": 3,
        }
    ]
    event_key = (25, 37, "conflicts and violence")
    assert event_support[event_key] == 3
    assert event_support[(0, 8, "conflicts and violence")] == 1
    assert argument_support[event_key][(14, 18, "source_location")] == 3
    assert argument_support[event_key][(41, 47, "target_location")] == 1


def test_format_examples_preserves_location_type() -> None:
    text = "Somalia faced a drought crisis after poor rains."
    location_start = text.index("Somalia")
    trigger_start = text.index("drought crisis")
    record = {
        "text": text,
        "events": [
            event(
                "drought crisis",
                "weather shocks",
                trigger_start,
                trigger_start + len("drought crisis"),
                arguments=[
                    argument(
                        "location",
                        "Somalia",
                        location_start,
                        location_start + len("Somalia"),
                        "country",
                    )
                ],
            )
        ],
    }

    payload = formatted_example_payloads(
        gemini_event_gen.format_examples(
            [record], gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        )
    )[0]

    assert (
        payload["expected_output"]["events"][0]["arguments"][0]["location_type"]
        == "country"
    )


def test_merge_self_consistency_events_counts_duplicates_once_per_sample() -> None:
    text = "Drought hit Somalia."
    drought = event(
        "Drought",
        "weather shocks",
        0,
        7,
        [
            argument("location", "Somalia", 12, 19),
            argument("location", "Somalia", 12, 19),
        ],
    )

    merged, event_support, argument_support, threshold = (
        gemini_event_gen.merge_self_consistency_events(
            [[drought, drought], [], []],
            text,
        )
    )

    event_key = (0, 7, "weather shocks")
    assert threshold == 2
    assert merged == []
    assert event_support[event_key] == 1
    assert argument_support[event_key][(12, 19, "location")] == 1


def test_clean_spans_filters_invalid_labels_and_offsets() -> None:
    text = "drought crisis"
    cleaned = gemini_event_gen.clean_spans(
        {
            "spans": [
                span("drought crisis", "weather shocks", 0, 14),
                span("drought crisis", "unknown", 0, 14),
                span("missing text", "weather shocks", 0, 12),
            ]
        },
        text,
        {"weather shocks"},
        strict_offsets=True,
    )

    assert cleaned == [span("drought crisis", "weather shocks", 0, 14)]


def test_clean_relevance_decision_coerces_and_clamps() -> None:
    cleaned = gemini_event_gen.clean_relevance_decision(
        {"is_relevant": False, "confidence": "1.7", "reason": 123}
    )

    assert cleaned == {
        "is_relevant": False,
        "confidence": 1.0,
        "reason": "123",
    }


def test_clean_relevance_decision_defaults_to_relevant_on_malformed_input() -> None:
    cleaned = gemini_event_gen.clean_relevance_decision(["bad"])

    assert cleaned == {
        "is_relevant": True,
        "confidence": 0.0,
        "reason": "",
    }


def test_spans_prompt_treats_source_and_target_as_context() -> None:
    assert (
        '"missiles from Gaza again raining down on Israel" -> "raining down"'
        in gemini_event_gen.DEFAULT_USER_PROMPT
    )
    assert "sources, targets" in gemini_event_gen.DEFAULT_USER_PROMPT


def test_clean_events_with_args_repairs_offsets_filters_and_dedupes() -> None:
    text = "missiles from Gaza again raining down on Israel"
    cleaned = gemini_event_gen.clean_events_with_args(
        {
            "events": [
                {
                    "event_type": "conflicts and violence",
                    "trigger_text": "raining down",
                    "start_char": 0,
                    "end_char": 12,
                    "arguments": [
                        {
                            "role": "source_location",
                            "text": "Gaza",
                            "start_char": 0,
                            "end_char": 4,
                        },
                        {
                            "role": "source_location",
                            "text": "Gaza",
                            "start_char": 14,
                            "end_char": 18,
                        },
                        {
                            "role": "target_location",
                            "text": "Israel",
                            "start_char": 41,
                            "end_char": 47,
                        },
                        {
                            "role": "invalid_role",
                            "text": "missiles",
                            "start_char": 0,
                            "end_char": 8,
                        },
                        {
                            "role": "actor",
                            "text": "missing",
                            "start_char": 0,
                            "end_char": 7,
                        },
                    ],
                    "rationale": "attack",
                },
                {
                    "event_type": "unknown",
                    "trigger_text": "raining down",
                    "start_char": 25,
                    "end_char": 37,
                    "arguments": [],
                },
            ]
        },
        text,
        {"conflicts and violence"},
        strict_offsets=True,
        argument_roles={"location", "source_location", "target_location"},
        event_argument_roles={
            "conflicts and violence": [
                "location",
                "source_location",
                "target_location",
            ]
        },
    )

    assert cleaned == [
        {
            "event_type": "conflicts and violence",
            "trigger_text": "raining down",
            "start_char": 25,
            "end_char": 37,
            "arguments": [
                {
                    "role": "source_location",
                    "text": "Gaza",
                    "start_char": 14,
                    "end_char": 18,
                },
                {
                    "role": "target_location",
                    "text": "Israel",
                    "start_char": 41,
                    "end_char": 47,
                },
            ],
            "rationale": "attack",
        }
    ]


def test_clean_events_with_args_enforces_roles_by_event_type() -> None:
    text = "Drought hit Somalia after supplies arrived from Kenya."
    cleaned = gemini_event_gen.clean_events_with_args(
        {
            "events": [
                {
                    "event_type": "weather shocks",
                    "trigger_text": "Drought",
                    "start_char": 0,
                    "end_char": 7,
                    "arguments": [
                        {
                            "role": "location",
                            "text": "Somalia",
                            "start_char": 12,
                            "end_char": 19,
                        },
                        {
                            "role": "source_location",
                            "text": "Kenya",
                            "start_char": 50,
                            "end_char": 55,
                        },
                    ],
                }
            ]
        },
        text,
        {"weather shocks"},
        strict_offsets=True,
        argument_roles={"location", "source_location", "target_location"},
        event_argument_roles={"weather shocks": ["location"]},
    )

    assert cleaned[0]["arguments"] == [
        {
            "role": "location",
            "text": "Somalia",
            "start_char": 12,
            "end_char": 19,
        }
    ]


def test_clean_events_with_args_defaults_missing_location_type_to_other() -> None:
    text = "Drought hit Somalia."
    cleaned = gemini_event_gen.clean_events_with_args(
        {
            "events": [
                {
                    "event_type": "weather shocks",
                    "trigger_text": "Drought",
                    "start_char": 0,
                    "end_char": 7,
                    "arguments": [
                        {
                            "role": "location",
                            "text": "Somalia",
                            "start_char": 12,
                            "end_char": 19,
                        }
                    ],
                }
            ]
        },
        text,
        {"weather shocks"},
        strict_offsets=True,
        argument_roles={"location"},
        event_argument_roles={"weather shocks": ["location"]},
        location_types={"country", "province", "district", "city", "other"},
    )

    assert cleaned[0]["arguments"] == [
        {
            "role": "location",
            "text": "Somalia",
            "location_type": "other",
            "start_char": 12,
            "end_char": 19,
        }
    ]


def test_clean_events_with_args_drops_invalid_location_type() -> None:
    text = "Drought hit Somalia."
    cleaned = gemini_event_gen.clean_events_with_args(
        {
            "events": [
                {
                    "event_type": "weather shocks",
                    "trigger_text": "Drought",
                    "start_char": 0,
                    "end_char": 7,
                    "arguments": [
                        {
                            "role": "location",
                            "text": "Somalia",
                            "location_type": "region",
                            "start_char": 12,
                            "end_char": 19,
                        }
                    ],
                }
            ]
        },
        text,
        {"weather shocks"},
        strict_offsets=True,
        argument_roles={"location"},
        event_argument_roles={"weather shocks": ["location"]},
        location_types={"country", "province", "district", "city", "other"},
    )

    assert cleaned[0]["arguments"] == []


def test_normalizes_argument_roles_from_risk_ontology() -> None:
    raw = gemini_event_gen.load_json_tolerant(
        REPO_ROOT / "ontologies" / "risk-factors" / "risk.cluster.description.json"
    )
    events = gemini_event_gen.normalize_ontology(raw)
    roles = gemini_event_gen.normalize_argument_roles(raw)
    roles_by_event = gemini_event_gen.normalize_event_argument_roles(
        raw, set(events), set(roles)
    )
    location_types = gemini_event_gen.normalize_location_types(raw)

    assert set(roles) == {"location", "source_location", "target_location"}
    assert set(location_types) == {
        "country",
        "province",
        "district",
        "city",
        "other",
    }
    assert roles_by_event["weather shocks"] == ["location"]
    assert roles_by_event["conflicts and violence"] == [
        "location",
        "source_location",
        "target_location",
    ]


def test_argument_role_normalization_requires_ontology_fields() -> None:
    raw = {"events": {"weather shocks": "Weather events."}}
    events = gemini_event_gen.normalize_ontology(raw)

    try:
        gemini_event_gen.normalize_argument_roles(raw)
    except ValueError as exc:
        assert "argument_roles" in str(exc)
    else:
        raise AssertionError("Expected missing argument_roles to fail")

    try:
        gemini_event_gen.normalize_event_argument_roles(
            {"events": raw["events"], "argument_roles": {"location": "Place."}},
            set(events),
            {"location"},
        )
    except ValueError as exc:
        assert "event_argument_roles" in str(exc)
    else:
        raise AssertionError("Expected missing event_argument_roles to fail")


def test_load_records_accepts_json_array_with_title_and_body(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "uri": "1000000018",
                    "title": "Food prices rise",
                    "body": "Drought reduced the harvest.",
                    "url": "https://example.test/article",
                    "published_at": "2018-11-21T14:18:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = gemini_event_gen.load_records(path)

    assert records == [
        {
            "uri": "1000000018",
            "title": "Food prices rise",
            "body": "Drought reduced the harvest.",
            "url": "https://example.test/article",
            "published_at": "2018-11-21T14:18:00Z",
            "id": "1000000018",
            "text": "Drought reduced the harvest.",
            "source_url": "https://example.test/article",
            "publish_date": "2018-11-21T14:18:00Z",
        }
    ]


def test_load_records_preserves_jsonl_text_inputs(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"id": "r1", "title": "Title", "text": "Body text."}) + "\n",
        encoding="utf-8",
    )

    records = gemini_event_gen.load_records(path)

    assert records[0]["id"] == "r1"
    assert records[0]["title"] == "Title"
    assert records[0]["text"] == "Body text."


def test_load_records_reads_nested_source_fields(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "r1",
                "source": {
                    "title": "Nested title",
                    "text": "Nested body.",
                    "source_url": "https://example.test",
                    "publish_date": "2020-01-01",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = gemini_event_gen.load_records(path)

    assert records[0]["title"] == "Nested title"
    assert records[0]["text"] == "Nested body."
    assert records[0]["source_url"] == "https://example.test"
    assert records[0]["publish_date"] == "2020-01-01"


def test_dedupe_records_removes_duplicate_title_text_before_limit() -> None:
    records = [
        {"id": "1", "title": "A", "text": "same"},
        {"id": "2", "title": "A", "text": "same"},
        {"id": "3", "title": "B", "text": "other"},
    ]

    selected = gemini_event_gen.select_limited_records(
        gemini_event_gen.dedupe_records(records),
        limit=2,
        random_sample=False,
    )

    assert [record["id"] for record in selected] == ["1", "3"]


def test_select_pending_records_applies_limit_after_completed_exclusions() -> None:
    records = [
        {"id": "1", "title": "A", "text": "one"},
        {"id": "2", "title": "B", "text": "two"},
        {"id": "3", "title": "C", "text": "three"},
        {"id": "4", "title": "D", "text": "four"},
    ]

    selected = gemini_event_gen.select_pending_records(
        records,
        completed_record_ids={"1", "2"},
        limit=2,
        random_sample=False,
    )

    assert [record["id"] for record in selected] == ["3", "4"]


def test_build_article_windows_packs_sentence_units_by_budget() -> None:
    text = "Drought hit Somalia.\n\nMarkets reopened.\n\nLocusts spread quickly."

    windows = gemini_event_gen.build_article_windows(
        text,
        target_chars=20,
        max_chars=40,
        overlap_sentences=0,
    )

    assert [window.text for window in windows] == [
        "Drought hit Somalia.",
        "Markets reopened.",
        "Locusts spread quickly.",
    ]
    assert [(window.start_char, window.end_char) for window in windows] == [
        (0, 20),
        (22, 39),
        (41, 64),
    ]
    assert windows[0].overlap_prev is False
    assert windows[1].overlap_prev is False
    assert windows[2].overlap_prev is False
    assert windows[0].core_start_char == 0
    assert windows[0].core_end_char == 20


def test_build_article_windows_splits_long_text_without_blank_lines() -> None:
    text = "Drought hit Somalia. Markets reopened. Locusts spread quickly."

    windows = gemini_event_gen.build_article_windows(
        text,
        target_chars=25,
        max_chars=60,
        overlap_sentences=0,
    )

    assert [window.text for window in windows] == [
        "Drought hit Somalia.",
        "Markets reopened.",
        "Locusts spread quickly.",
    ]


def test_project_window_spans_does_not_search_whole_article() -> None:
    text = "drought crisis\n\nother paragraph\n\ndrought crisis"
    windows = gemini_event_gen.build_article_windows(
        text,
        target_chars=14,
        max_chars=20,
        overlap_sentences=0,
    )

    projected = gemini_event_gen.project_window_spans(
        [span("drought crisis", "weather shocks", 0, 14)],
        windows[-1],
        text,
    )

    assert projected == [
        {
            "span_text": "drought crisis",
            "label": "weather shocks",
            "start_char": 33,
            "end_char": 47,
            "rationale": "",
            "window_indices": [2],
            "trigger_in_core": True,
        }
    ]


def test_merge_window_events_deduplicates_and_preserves_window_indices() -> None:
    text = "missiles from Gaza again raining down on Israel"
    events = [
        {
            **event(
                "raining down",
                "conflicts and violence",
                25,
                37,
                [argument("source_location", "Gaza", 14, 18)],
            ),
            "window_indices": [0],
        },
        {
            **event(
                "raining down",
                "conflicts and violence",
                25,
                37,
                [
                    argument("source_location", "Gaza", 14, 18),
                    argument("target_location", "Israel", 41, 47),
                ],
            ),
            "window_indices": [1],
        },
    ]
    events[0]["arguments"][0]["window_indices"] = [0]
    events[1]["arguments"][0]["window_indices"] = [1]
    events[1]["arguments"][1]["window_indices"] = [1]

    merged = gemini_event_gen.merge_window_events(events, text)

    assert merged == [
        {
            "event_type": "conflicts and violence",
            "trigger_text": "raining down",
            "start_char": 25,
            "end_char": 37,
            "arguments": [
                {
                    "role": "source_location",
                    "text": "Gaza",
                    "start_char": 14,
                    "end_char": 18,
                    "window_indices": [0],
                },
                {
                    "role": "target_location",
                    "text": "Israel",
                    "start_char": 41,
                    "end_char": 47,
                    "window_indices": [1],
                },
            ],
            "rationale": "",
            "window_indices": [0, 1],
            "core_window_indices": [],
        }
    ]


def test_generate_one_long_document_mode_projects_and_merges_windows() -> None:
    text = "drought crisis\n\nother paragraph\n\ndrought crisis"
    client = FakeGeminiClient(
        [
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
            {"spans": []},
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            long_document_mode=True,
            long_document_threshold_chars=1,
            window_target_chars=14,
            window_max_chars=20,
            window_overlap_sentences=0,
        )
    )

    assert result["status"] == "ok"
    assert result["spans"] == [
        {
            "span_text": "drought crisis",
            "label": "weather shocks",
            "start_char": 0,
            "end_char": 14,
            "rationale": "",
            "window_indices": [0],
            "core_window_indices": [0],
        },
        {
            "span_text": "drought crisis",
            "label": "weather shocks",
            "start_char": 33,
            "end_char": 47,
            "rationale": "",
            "window_indices": [2],
            "core_window_indices": [2],
        },
    ]
    long_document = result["llm"]["metadata"]["long_document"]
    assert long_document["enabled"] is True
    assert long_document["window_count"] == 3
    assert long_document["window_boundary"] == "sentence_adaptive_overlap"
    assert len(client.calls) == 3
    assert client.calls[0]["prompt"] == (
        "- weather shocks: weather\ntitle\ndrought crisis"
    )
    assert "Offsets must index the window text" not in client.calls[0]["prompt"]
    assert "Window:" not in client.calls[0]["prompt"]
    assert "Original article character range" not in client.calls[0]["prompt"]


def test_generate_one_long_document_mode_supports_self_consistency() -> None:
    text = "drought crisis\n\nother paragraph"
    client = FakeGeminiClient(
        [
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
            {"spans": []},
            {"spans": []},
            {"spans": []},
            {"spans": []},
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            self_consistency=True,
            self_consistency_samples=3,
            self_consistency_temperature=0.7,
            self_consistency_min_successful_samples=2,
            long_document_mode=True,
            long_document_threshold_chars=1,
            window_target_chars=14,
            window_max_chars=20,
            window_overlap_sentences=0,
        )
    )

    assert result["status"] == "ok"
    assert result["spans"] == [
        {
            "span_text": "drought crisis",
            "label": "weather shocks",
            "start_char": 0,
            "end_char": 14,
            "rationale": "",
            "support": 2,
            "window_indices": [0],
            "core_window_indices": [0],
        }
    ]
    long_document = result["llm"]["metadata"]["long_document"]
    assert long_document["self_consistency"] == {
        "enabled": True,
        "samples_requested": 3,
        "temperature": 0.7,
        "min_successful_samples": 2,
    }
    assert long_document["window_errors"] == []
    assert len(client.calls) == 6
    assert all(
        call["override_settings"] == {"temperature": 0.7} for call in client.calls
    )


def test_generate_one_self_consistency_partial_failures_succeeds() -> None:
    text = "drought crisis and locust outbreak"
    client = FakeGeminiClient(
        [
            {"spans": [span("drought crisis", "weather shocks", 0, 14, "a")]},
            RuntimeError("sample failed"),
            {"spans": [span("drought crisis", "weather shocks", 0, 14, "a")]},
            {"spans": [span("locust outbreak", "pests and diseases", 19, 34)]},
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- weather shocks: weather\n- pests and diseases: pests",
            labels={"weather shocks", "pests and diseases"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            self_consistency=True,
            self_consistency_samples=4,
            self_consistency_temperature=0.7,
            self_consistency_min_successful_samples=3,
        )
    )

    assert result["status"] == "ok"
    assert result["spans"] == [
        {
            "span_text": "drought crisis",
            "label": "weather shocks",
            "start_char": 0,
            "end_char": 14,
            "rationale": "a",
            "support": 2,
        }
    ]
    self_consistency = result["llm"]["self_consistency"]
    assert self_consistency["samples_requested"] == 4
    assert self_consistency["samples_succeeded"] == 3
    assert self_consistency["threshold"] == 2
    assert self_consistency["temperature"] == 0.7
    assert self_consistency["sample_errors"] == [
        {"sample": 1, "error": "sample failed"}
    ]
    assert result["llm"]["metadata"] == {
        "completion_tokens": 6,
        "prompt_tokens": 30,
    }
    assert all(
        call["override_settings"] == {"temperature": 0.7} for call in client.calls
    )


def test_generate_one_self_consistency_samples_examples_per_call(
    monkeypatch: Any,
) -> None:
    text = "drought crisis"
    client = FakeGeminiClient(
        [
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
        ]
    )
    example_a = {
        "title": "Example A",
        "text": "Locusts spread.",
        "events": [event("Locusts", "pests and diseases", 0, 7)],
    }
    example_b = {
        "title": "Example B",
        "text": "Flooding damaged crops.",
        "events": [event("Flooding", "weather shocks", 0, 8)],
    }
    samples = [[example_a], [example_b]]

    def fake_sample_examples(
        examples: list[dict[str, Any]],
        sample_size: int | None,
        output_mode: str,
    ) -> list[dict[str, Any]]:
        assert sample_size == 1
        assert output_mode == gemini_event_gen.OUTPUT_MODE_SPANS
        return samples.pop(0)

    monkeypatch.setattr(gemini_event_gen, "sample_examples", fake_sample_examples)

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            self_consistency=True,
            self_consistency_samples=2,
            self_consistency_temperature=0.7,
            self_consistency_min_successful_samples=2,
            examples=[example_a, example_b],
            example_sample_size=1,
        )
    )

    assert result["status"] == "ok"
    assert "Locusts" in client.calls[0]["prompt"]
    assert "Flooding" in client.calls[1]["prompt"]
    assert "Example A" not in client.calls[0]["prompt"]
    assert "Example B" not in client.calls[1]["prompt"]
    assert samples == []


def test_generate_one_self_consistency_errors_when_too_few_samples_succeed() -> None:
    client = FakeGeminiClient(
        [
            {"spans": [span("drought crisis", "weather shocks", 0, 14)]},
            RuntimeError("failed one"),
            RuntimeError("failed two"),
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": "drought crisis"},
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            self_consistency=True,
            self_consistency_samples=3,
            self_consistency_temperature=0.7,
            self_consistency_min_successful_samples=2,
        )
    )

    assert result["status"] == "error"
    assert "1/3 samples succeeded" in result["error"]
    assert result["llm"]["self_consistency"]["samples_succeeded"] == 1
    assert len(result["llm"]["self_consistency"]["sample_errors"]) == 2


def test_generate_one_events_with_args_outputs_events_schema() -> None:
    text = "missiles from Gaza again raining down on Israel"
    client = FakeGeminiClient(
        [
            {
                "events": [
                    {
                        "event_type": "conflicts and violence",
                        "trigger_text": "raining down",
                        "start_char": 25,
                        "end_char": 37,
                        "arguments": [
                            {
                                "role": "source_location",
                                "text": "Gaza",
                                "start_char": 14,
                                "end_char": 18,
                            },
                            {
                                "role": "target_location",
                                "text": "Israel",
                                "start_char": 41,
                                "end_char": 47,
                            },
                        ],
                    }
                ]
            }
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- conflicts and violence: conflict",
            labels={"conflicts and violence"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            output_mode=gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
            argument_roles={"location", "source_location", "target_location"},
            event_argument_roles={
                "conflicts and violence": [
                    "location",
                    "source_location",
                    "target_location",
                ]
            },
        )
    )

    assert result["status"] == "ok"
    assert "spans" not in result
    assert result["events"] == [
        {
            "event_type": "conflicts and violence",
            "trigger_text": "raining down",
            "start_char": 25,
            "end_char": 37,
            "arguments": [
                {
                    "role": "source_location",
                    "text": "Gaza",
                    "start_char": 14,
                    "end_char": 18,
                },
                {
                    "role": "target_location",
                    "text": "Israel",
                    "start_char": 41,
                    "end_char": 47,
                },
            ],
            "rationale": "",
        }
    ]
    assert client.calls[0]["response_format"] == {
        "events": list[gemini_event_gen.ExtractedEvent]
    }


def test_generate_one_verbose_logs_prompt_and_answer(caplog: Any) -> None:
    text = "missiles from Gaza again raining down on Israel"
    client = FakeGeminiClient(
        [
            {
                "events": [
                    event("raining down", "conflicts and violence", 25, 37)
                ]
            }
        ]
    )
    caplog.set_level(logging.INFO, logger="gemini_event_gen")

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- conflicts and violence: conflict",
            labels={"conflicts and violence"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            output_mode=gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
            argument_roles={"location", "source_location", "target_location"},
            event_argument_roles={
                "conflicts and violence": [
                    "location",
                    "source_location",
                    "target_location",
                ]
            },
            verbose=True,
        )
    )

    assert result["status"] == "ok"
    assert "LLM CALL extract attempt=1 id=r1 PROMPT" in caplog.text
    assert "LLM CALL extract attempt=1 id=r1 ANSWER" in caplog.text
    assert "Step: extraction" in caplog.text
    assert "User prompt:" in caplog.text
    assert "raining down" in caplog.text


def test_generate_one_events_with_args_verifier_filters_rejected_events() -> None:
    text = "Drought hit Somalia. Markets reopened."
    client = FakeGeminiClient(
        [
            {
                "events": [
                    event(
                        "Drought",
                        "weather shocks",
                        0,
                        7,
                        [argument("location", "Somalia", 12, 19)],
                    ),
                    event("Markets reopened", "market disruption", 21, 37),
                ]
            },
            {
                "decisions": [
                    {
                        "event_type": "weather shocks",
                        "start_char": 0,
                        "end_char": 7,
                        "decision": "accept",
                        "reason": "explicit",
                        "confidence": 0.9,
                    },
                    {
                        "event_type": "market disruption",
                        "start_char": 21,
                        "end_char": 37,
                        "decision": "reject",
                        "reason": "not food security",
                        "confidence": 0.8,
                    },
                ]
            },
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- weather shocks: weather\n- market disruption: market",
            labels={"weather shocks", "market disruption"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            output_mode=gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
            argument_roles={"location"},
            event_argument_roles={
                "weather shocks": ["location"],
                "market disruption": ["location"],
            },
            enable_verifier=True,
        )
    )

    assert result["status"] == "ok"
    assert [event["event_type"] for event in result["events"]] == ["weather shocks"]
    verifier = result["llm"]["metadata"]["verifier"]
    assert verifier["enabled"] is True
    assert len(verifier["decisions"]) == 2
    assert client.calls[1]["response_format"] == {
        "decisions": list[gemini_event_gen.VerifierDecision]
    }


def test_generate_one_events_with_args_self_consistency_succeeds() -> None:
    text = "missiles from Gaza again raining down on Israel"
    client = FakeGeminiClient(
        [
            {
                "events": [
                    event(
                        "raining down",
                        "conflicts and violence",
                        25,
                        37,
                        [
                            argument("source_location", "Gaza", 14, 18),
                            argument("target_location", "Israel", 41, 47),
                        ],
                        "attack",
                    )
                ]
            },
            RuntimeError("sample failed"),
            {
                "events": [
                    event(
                        "raining down",
                        "conflicts and violence",
                        25,
                        37,
                        [argument("source_location", "Gaza", 14, 18)],
                        "attack",
                    )
                ]
            },
            {
                "events": [
                    event(
                        "missiles",
                        "conflicts and violence",
                        0,
                        8,
                        [argument("source_location", "Gaza", 14, 18)],
                    )
                ]
            },
        ]
    )

    result = asyncio.run(
        gemini_event_gen.generate_one(
            client=client,
            record={"id": "r1", "title": "title", "text": text},
            ontology_text="- conflicts and violence: conflict",
            labels={"conflicts and violence"},
            system_prompt="system",
            user_prompt_template="{ontology}\n{title}\n{text}",
            max_retries=0,
            initial_backoff=0,
            max_backoff=0,
            strict_offsets=True,
            self_consistency=True,
            self_consistency_samples=4,
            self_consistency_temperature=0.7,
            self_consistency_min_successful_samples=3,
            output_mode=gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
            argument_roles={"location", "source_location", "target_location"},
            event_argument_roles={
                "conflicts and violence": [
                    "location",
                    "source_location",
                    "target_location",
                ]
            },
        )
    )

    assert result["status"] == "ok"
    assert "spans" not in result
    assert result["events"] == [
        {
            "event_type": "conflicts and violence",
            "trigger_text": "raining down",
            "start_char": 25,
            "end_char": 37,
            "arguments": [
                {
                    "role": "source_location",
                    "text": "Gaza",
                    "start_char": 14,
                    "end_char": 18,
                    "support": 2,
                }
            ],
            "rationale": "attack",
            "support": 2,
        }
    ]
    self_consistency = result["llm"]["self_consistency"]
    assert self_consistency["samples_requested"] == 4
    assert self_consistency["samples_succeeded"] == 3
    assert self_consistency["threshold"] == 2
    assert self_consistency["sample_errors"] == [
        {"sample": 1, "error": "sample failed"}
    ]
    assert self_consistency["event_support"] == [
        {
            "start_char": 25,
            "end_char": 37,
            "event_type": "conflicts and violence",
            "support": 2,
        }
    ]
    assert self_consistency["argument_support"] == [
        {
            "event_start_char": 25,
            "event_end_char": 37,
            "event_type": "conflicts and violence",
            "start_char": 14,
            "end_char": 18,
            "role": "source_location",
            "support": 2,
            "threshold": 2,
        }
    ]
    assert all(
        call["override_settings"] == {"temperature": 0.7} for call in client.calls
    )


def test_worker_filters_irrelevant_record_to_empty_output(monkeypatch: Any) -> None:
    constructors: list[str] = []

    class DummyClient:
        def __init__(self, model_name: str, **_: Any) -> None:
            constructors.append(model_name)
            self.model_name = model_name

    async def fake_classify_article_relevance(**_: Any) -> dict[str, Any]:
        return {
            "decision": "irrelevant",
            "is_relevant": False,
            "confidence": 0.95,
            "reason": "sports recap",
            "filtered": True,
            "threshold": 0.8,
            "model": "relevance-model",
            "max_chars": 3000,
            "text_chars_used": 120,
            "metadata": {"prompt_tokens": 4, "completion_tokens": 1},
        }

    async def fake_generate_one(**_: Any) -> dict[str, Any]:
        raise AssertionError("generate_one should not be called for filtered records")

    monkeypatch.setattr(gemini_event_gen, "GeminiLLMClient", DummyClient)
    monkeypatch.setattr(
        gemini_event_gen, "classify_article_relevance", fake_classify_article_relevance
    )
    monkeypatch.setattr(gemini_event_gen, "generate_one", fake_generate_one)

    input_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    input_queue.put_nowait({"id": "r1", "title": "Title", "text": "Body"})
    input_queue.put_nowait(None)

    asyncio.run(
        gemini_event_gen.worker(
            worker_id=0,
            input_queue=input_queue,
            output_queue=output_queue,
            args=make_worker_args(
                enable_relevance_filter=True,
                relevance_model="relevance-model",
            ),
            system_prompt="system",
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            argument_roles=set(),
            event_argument_roles={},
            argument_roles_text="",
            event_argument_roles_text="",
            user_prompt_template="{ontology}\n{title}\n{text}",
            examples=[],
        )
    )

    result = output_queue.get_nowait()
    assert result["status"] == "ok"
    assert result["spans"] == []
    assert result["llm"]["model"] == "relevance-model"
    assert result["llm"]["relevance"]["filtered"] is True
    assert result["llm"]["pipeline"]["enable_relevance_filter"] is True
    assert constructors == ["relevance-model"]


def test_classify_article_relevance_uses_supported_dict_schema() -> None:
    client = FakeGeminiClient(
        [{"is_relevant": True, "confidence": 0.91, "reason": "mentions drought"}]
    )

    result = asyncio.run(
        gemini_event_gen.classify_article_relevance(
            client=client,
            title="Drought pressures crops",
            text="A drought is reducing harvests and market supply.",
            record_id="r1",
            max_chars=500,
            confidence_threshold=0.8,
        )
    )

    assert result["is_relevant"] is True
    assert result["filtered"] is False
    assert client.calls[0]["response_format"] == {
        "is_relevant": bool,
        "confidence": float,
        "reason": str,
    }


def test_worker_relevance_failure_falls_through_to_extraction(monkeypatch: Any) -> None:
    constructors: list[str] = []

    class DummyClient:
        def __init__(self, model_name: str, **_: Any) -> None:
            constructors.append(model_name)
            self.model_name = model_name

    async def fake_classify_article_relevance(**_: Any) -> dict[str, Any]:
        raise RuntimeError("relevance failed")

    async def fake_generate_one(**_: Any) -> dict[str, Any]:
        return {
            "id": "r1",
            "status": "ok",
            "source": {"title": "Title", "text": "Body"},
            "spans": [],
            "llm": {"model": "extract-model", "metadata": {}, "output_mode": "spans"},
        }

    monkeypatch.setattr(gemini_event_gen, "GeminiLLMClient", DummyClient)
    monkeypatch.setattr(
        gemini_event_gen, "classify_article_relevance", fake_classify_article_relevance
    )
    monkeypatch.setattr(gemini_event_gen, "generate_one", fake_generate_one)

    input_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    input_queue.put_nowait({"id": "r1", "title": "Title", "text": "Body"})
    input_queue.put_nowait(None)

    asyncio.run(
        gemini_event_gen.worker(
            worker_id=0,
            input_queue=input_queue,
            output_queue=output_queue,
            args=make_worker_args(enable_relevance_filter=True),
            system_prompt="system",
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            argument_roles=set(),
            event_argument_roles={},
            argument_roles_text="",
            event_argument_roles_text="",
            user_prompt_template="{ontology}\n{title}\n{text}",
            examples=[],
        )
    )

    result = output_queue.get_nowait()
    assert result["status"] == "ok"
    assert result["llm"]["relevance"]["decision"] == "error"
    assert result["llm"]["relevance"]["filtered"] is False
    assert "relevance failed" in result["llm"]["relevance"]["error"]
    assert constructors == ["extract-model", "extract-model"]


def test_worker_uses_separate_relevance_model_before_extraction(
    monkeypatch: Any,
) -> None:
    constructors: list[str] = []

    class DummyClient:
        def __init__(self, model_name: str, **_: Any) -> None:
            constructors.append(model_name)
            self.model_name = model_name

    async def fake_classify_article_relevance(**_: Any) -> dict[str, Any]:
        return {
            "decision": "irrelevant",
            "is_relevant": False,
            "confidence": 0.3,
            "reason": "uncertain, keep",
            "filtered": False,
            "threshold": 0.8,
            "model": "relevance-model",
            "max_chars": 3000,
            "text_chars_used": 50,
            "metadata": {"prompt_tokens": 4, "completion_tokens": 1},
        }

    async def fake_generate_one(**_: Any) -> dict[str, Any]:
        return {
            "id": "r1",
            "status": "ok",
            "source": {"title": "Title", "text": "Body"},
            "spans": [],
            "llm": {"model": "extract-model", "metadata": {}, "output_mode": "spans"},
        }

    monkeypatch.setattr(gemini_event_gen, "GeminiLLMClient", DummyClient)
    monkeypatch.setattr(
        gemini_event_gen, "classify_article_relevance", fake_classify_article_relevance
    )
    monkeypatch.setattr(gemini_event_gen, "generate_one", fake_generate_one)

    input_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    input_queue.put_nowait({"id": "r1", "title": "Title", "text": "Body"})
    input_queue.put_nowait(None)

    asyncio.run(
        gemini_event_gen.worker(
            worker_id=0,
            input_queue=input_queue,
            output_queue=output_queue,
            args=make_worker_args(
                enable_relevance_filter=True,
                relevance_model="relevance-model",
            ),
            system_prompt="system",
            ontology_text="- weather shocks: weather",
            labels={"weather shocks"},
            argument_roles=set(),
            event_argument_roles={},
            argument_roles_text="",
            event_argument_roles_text="",
            user_prompt_template="{ontology}\n{title}\n{text}",
            examples=[],
        )
    )

    result = output_queue.get_nowait()
    assert result["status"] == "ok"
    assert result["llm"]["relevance"]["model"] == "relevance-model"
    assert result["llm"]["pipeline"]["enable_relevance_filter"] is True
    assert constructors == ["relevance-model", "extract-model"]


def test_apply_pipeline_defaults_applies_relevance_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "enable_relevance_filter: true",
                "relevance_model: gemini-2.5-flash-lite",
                "relevance_max_chars: 2222",
                "relevance_confidence_threshold: 0.9",
            ]
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        input=Path("in.jsonl"),
        output=Path("out.jsonl"),
        config=config_path,
        mode="quality_first",
        ontology=Path("ontology.json"),
        model="gemini-2.5-flash",
        workers=1,
        temperature=0.0,
        max_tokens=4096,
        reasoning_effort="disable",
        output_mode=None,
        max_retries=0,
        initial_backoff=0.0,
        max_backoff=0.0,
        limit=None,
        random_sample=False,
        examples=Path("examples.jsonl"),
        example_sample_size=None,
        progress=False,
        retry_failed=False,
        overwrite=False,
        strict_offsets=None,
        self_consistency=None,
        self_consistency_samples=None,
        self_consistency_temperature=None,
        self_consistency_min_successful_samples=3,
        long_document_mode=None,
        long_document_threshold_chars=1000,
        window_target_chars=None,
        window_max_chars=None,
        window_overlap_sentences=None,
        enable_verifier=None,
        enable_synthetic_gaps=None,
        enable_relevance_filter=None,
        relevance_model=None,
        relevance_max_chars=3000,
        relevance_confidence_threshold=0.8,
        report=None,
        system_prompt_file=None,
        user_prompt_file=None,
        env_file=Path(".env"),
        verbose=False,
        log_level="INFO",
    )

    gemini_event_gen.apply_pipeline_defaults(args)

    assert args.enable_relevance_filter is True
    assert args.relevance_model == "gemini-2.5-flash-lite"
    assert args.relevance_max_chars == 2222
    assert args.relevance_confidence_threshold == 0.9
