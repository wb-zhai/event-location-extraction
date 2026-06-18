from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2 import gemini_common
from scripts.data.generation_v2.annotate_gemini import SYSTEM_PROMPT, build_prompt, clean_payload
from scripts.data.generation_v2.audit_report import build_report
from scripts.data.generation_v2.gemini_common import (
    GeneratedPayload,
    GeminiTask,
    VerifierDecision,
    batch_generation_config,
    build_batch_request,
    extract_batch_response,
    interactive_response_format,
    run_interactive_tasks,
)
from scripts.data.generation_v3.io_utils import completed_task_keys, has_completed_task_keys, iter_jsonl, prune_retryable_error_rows
from scripts.data.generation_v2.recover_annotations import recover_file, recover_offsets, recover_record
from scripts.data.generation_v2.sample_articles import (
    SamplingSummary,
    append_sample_records_to_limit,
    keyword_quality_score,
    keyword_terms,
    sample_records,
)
from scripts.data.generation_v2.usage import normalize_usage
from scripts.data.generation_v2.verify_gemini import (
    build_prompt as build_verifier_prompt,
    output_row_for_result as verifier_output_row_for_result,
    verification_excerpts,
)
from scripts.data.generation_v2.window_articles import main as window_articles_main
from src.llms.llm_client import LLMClient


ONTOLOGY = {
    "events": {"artillery bombing": "Shelling or artillery attack."},
    "argument_roles": {"location": "Place where event occurs."},
    "location_types": {"country": "Country", "other": "Other place"},
    "event_argument_roles": {"artillery bombing": ["location"]},
}


def test_annotation_prompt_uses_structured_gemini_template() -> None:
    prompt = build_prompt({"title": "Context", "text": "Shelling damaged homes in Gaza."}, ONTOLOGY)

    assert "<role>" in SYSTEM_PROMPT
    assert "<instructions>" in SYSTEM_PROMPT
    assert "<constraints>" in SYSTEM_PROMPT
    assert "<output_format>" in SYSTEM_PROMPT
    assert "Return only the final structured JSON object." in SYSTEM_PROMPT
    assert "<context>" in prompt
    assert "<task>" in prompt
    assert "<final_instruction>" in prompt
    assert '<article_text offset_source="true">' in prompt
    assert "Think step by step internally" in prompt


def test_recover_offsets_exact_and_unique_repair() -> None:
    text = "Shelling damaged homes in Gaza."
    assert recover_offsets("Shelling", text, 0, 8) == (0, 8, "exact")
    assert recover_offsets("Gaza", text, -1, -1) == (26, 30, "repaired_nearby")


def test_recover_offsets_rejects_ambiguous_repair() -> None:
    text = "Gaza saw shelling. Gaza residents fled."
    assert recover_offsets("Gaza", text, -1, -1) is None


def test_recover_record_keeps_linked_event_arguments_and_locations() -> None:
    record = {
        "id": "a1",
        "text": "Shelling damaged homes in Gaza.",
        "events": [
            {
                "event_type": "artillery bombing",
                "start": 0,
                "end": 8,
                "text": "Shelling",
                "arguments": [
                    {
                        "role": "location",
                        "start": 27,
                        "end": 31,
                        "text": "Gaza",
                        "location_type": "other",
                    }
                ],
            }
        ],
        "locations": [{"start": 27, "end": 31, "text": "Gaza", "location_type": "other"}],
    }
    recovered = recover_record(record)
    assert [span["span_kind"] for span in recovered] == ["event", "argument", "location"]
    assert recovered[1]["linked_event_id"] == "a1::event::0"


def test_recover_file_preserves_source_error_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "verified.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    input_path.write_text(
        '{"id":"article-1","status":"error","error":"bad json",'
        '"text":"Shelling damaged homes in Gaza.",'
        '"metadata":{"task_key":"article-1::w0","verifier_decision":"error"}}\n',
        encoding="utf-8",
    )

    rows = recover_file(input_path, output_path, repair_window_chars=200, overwrite=True)

    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "bad json"
    assert rows[0]["events"] == []
    assert rows[0]["locations"] == []
    assert rows[0]["llm"]["metadata"]["generation_v2"]["recovery_status"] == "source_error"


def test_recover_file_merges_window_rows_per_article(tmp_path: Path) -> None:
    input_path = tmp_path / "verified.jsonl"
    output_path = tmp_path / "recovered.jsonl"
    text = "Heavy shelling damaged homes in Gaza. More shelling hit Rafah."
    rows = [
        {
            "id": "article-1",
            "text": text,
            "events": [
                {
                    "event_type": "artillery bombing",
                    "start": 6,
                    "end": 14,
                    "text": "shelling",
                    "arguments": [
                        {
                            "role": "location",
                            "start": 32,
                            "end": 36,
                            "text": "Gaza",
                            "location_type": "other",
                        }
                    ],
                }
            ],
            "locations": [{"start": 32, "end": 36, "text": "Gaza", "location_type": "other"}],
            "metadata": {
                "task_key": "article-1::w0",
                "window_indices": [0],
                "usage": {"extractor": {"input_tokens": 3}},
            },
        },
        {
            "id": "article-1",
            "text": text,
            "events": [
                {
                    "event_type": "artillery bombing",
                    "start": 12,
                    "end": 14,
                    "text": "ng",
                    "arguments": [],
                },
                {
                    "event_type": "artillery bombing",
                    "start": 43,
                    "end": 51,
                    "text": "shelling",
                    "arguments": [
                        {
                            "role": "location",
                            "start": 56,
                            "end": 61,
                            "text": "Rafah",
                            "location_type": "other",
                        }
                    ],
                },
            ],
            "locations": [
                {"start": 32, "end": 36, "text": "Gaza", "location_type": "other"},
                {"start": 56, "end": 61, "text": "Rafah", "location_type": "other"},
            ],
            "metadata": {
                "task_key": "article-1::w1",
                "window_indices": [1],
                "usage": {"extractor": {"input_tokens": 5}},
            },
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    recovered = recover_file(input_path, output_path, repair_window_chars=200, overwrite=True)

    assert len(recovered) == 1
    assert recovered[0]["id"] == "article-1"
    assert recovered[0]["status"] == "ok"
    assert recovered[0]["source"]["text"] == text
    assert [event["trigger_text"] for event in recovered[0]["events"]] == ["shelling", "shelling"]
    assert [location["text"] for location in recovered[0]["locations"]] == ["Gaza", "Rafah"]
    generation_v2 = recovered[0]["llm"]["metadata"]["generation_v2"]
    assert generation_v2["merged_task_keys"] == ["article-1::w0", "article-1::w1"]
    assert generation_v2["window_indices"] == [0, 1]
    assert generation_v2["usage"]["extractor"]["input_tokens"] == 8
    assert len(list(iter_jsonl(output_path))) == 1


def test_append_sample_records_to_limit_preserves_existing_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "sampled.jsonl"
    output_path.write_text(
        "\n".join(json.dumps(row) for row in [{"id": "a1"}, {"id": "a2"}]) + "\n",
        encoding="utf-8",
    )
    selected = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}, {"id": "a4"}]

    appended = append_sample_records_to_limit(output_path, selected, 3)

    assert appended == 1
    assert [row["id"] for row in iter_jsonl(output_path)] == ["a1", "a2", "a3"]


def test_append_sample_records_to_limit_skips_article_identity_duplicates(tmp_path: Path) -> None:
    output_path = tmp_path / "sampled.jsonl"
    text = " ".join(["Food insecurity worsened after drought."] * 30)
    output_path.write_text(
        json.dumps(
            {
                "id": "existing",
                "title": "Food crisis deepens in the region",
                "text": text,
                "source_url": "https://www.example.test/news/story?utm_source=x&b=2&a=1",
                "publish_date": "2024-01-02T03:04:05Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selected = [
        {
            "id": "duplicate-source",
            "title": "Different title",
            "text": text.replace("Food", "Severe food"),
            "source_url": "https://example.test/news/story?a=1&b=2",
            "publish_date": "2024-01-02",
        },
        {
            "id": "new",
            "title": "A distinct food crisis story",
            "text": text.replace("drought", "flooding"),
            "source_url": "https://example.test/news/other",
            "publish_date": "2024-01-03",
        },
    ]

    appended = append_sample_records_to_limit(output_path, selected, 2)

    assert appended == 1
    assert [row["id"] for row in iter_jsonl(output_path)] == ["existing", "new"]


def test_window_articles_append_missing_only_windows_new_articles(tmp_path: Path, monkeypatch: Any) -> None:
    input_path = tmp_path / "sampled.jsonl"
    output_path = tmp_path / "windows.jsonl"
    text = "Shelling damaged homes in Gaza. " * 30
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a1", "title": "one", "text": text},
                {"id": "a2", "title": "two", "text": text},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path.write_text(
        json.dumps({"id": "a1::w0", "article_id": "a1", "text": text}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "window_articles.py",
            str(input_path),
            str(output_path),
            "--append-missing",
            "--target-chars",
            "2000",
        ],
    )

    assert window_articles_main() == 0

    rows = list(iter_jsonl(output_path))
    assert [row["article_id"] for row in rows] == ["a1", "a2"]


def test_clean_payload_projects_window_offsets_to_article_offsets() -> None:
    article_text = "Intro. Shelling damaged homes in Gaza."
    source_record = {
        "id": "a1::w0",
        "article_id": "a1",
        "text": "Shelling damaged homes in Gaza.",
        "article_text": article_text,
        "window": {
            "window_index": 0,
            "start": 7,
            "end": len(article_text),
            "core_start": 7,
            "core_end": len(article_text),
        },
    }
    payload = {
        "events": [
            {
                "event_type": "artillery bombing",
                "start": 0,
                "end": 8,
                "text": "Shelling",
                "arguments": [
                    {
                        "role": "location",
                        "start": 27,
                        "end": 31,
                        "text": "Gaza",
                        "location_type": "other",
                    }
                ],
            }
        ],
        "locations": [{"start": 27, "end": 31, "text": "Gaza", "location_type": "other"}],
    }
    cleaned = clean_payload(
        payload=payload,
        source_record=source_record,
        ontology=ONTOLOGY,
        model="gemini-3.1-pro-preview",
        api_mode="interactive",
        settings={"temperature": 0.0, "reasoning_effort": "high", "max_tokens": 1024},
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert cleaned["events"][0]["start"] == 7
    assert cleaned["events"][0]["arguments"][0]["start"] == 33
    assert cleaned["metadata"]["task_key"] == "a1::w0"
    assert cleaned["metadata"]["usage"]["extractor"]["input_tokens"] == 10
    assert "text" not in cleaned["source"]


def test_verifier_prompt_uses_candidate_excerpts_not_full_article() -> None:
    text = "outside prefix " * 300 + "Shelling damaged homes in Gaza." + " outside suffix" * 300
    start = text.index("Shelling")
    location_start = text.index("Gaza")
    record = {
        "id": "a1",
        "text": text,
        "events": [
            {
                "event_type": "artillery bombing",
                "start": start,
                "end": start + len("Shelling"),
                "text": "Shelling",
                "arguments": [
                    {
                        "role": "location",
                        "start": location_start,
                        "end": location_start + len("Gaza"),
                        "text": "Gaza",
                        "location_type": "other",
                    }
                ],
            }
        ],
        "locations": [
            {
                "start": location_start,
                "end": location_start + len("Gaza"),
                "text": "Gaza",
                "location_type": "other",
            }
        ],
    }

    excerpts = verification_excerpts(record, context_chars=20)
    prompt = build_verifier_prompt(record, ONTOLOGY)

    assert len(excerpts) == 1
    assert excerpts[0]["start_char"] > 0
    assert "Shelling damaged homes in Gaza." in prompt
    assert text not in prompt
    assert "article-level offsets" in prompt


def test_completed_task_keys_prefers_metadata_task_key(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    output.write_text(
        '\n'.join(
            [
                '{"id":"article-1","metadata":{"task_key":"article-1::w0"}}',
                '{"id":"article-1","status":"error","metadata":{"task_key":"article-1::w1"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert completed_task_keys(output, retry_failed=True) == {"article-1::w0"}
    assert completed_task_keys(output, retry_failed=False) == {"article-1::w0", "article-1::w1"}
    assert has_completed_task_keys(output, retry_failed=True)


def test_prune_retryable_error_rows_removes_stale_errors(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    output.write_text(
        '\n'.join(
            [
                '{"id":"article-1","metadata":{"task_key":"article-1::w0"}}',
                '{"id":"article-1","status":"error","metadata":{"task_key":"article-1::w1"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert prune_retryable_error_rows(output, retry_failed=True) == 1
    rows = list(iter_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["metadata"]["task_key"] == "article-1::w0"


def test_verifier_error_rows_keep_task_key_and_verifier_metadata() -> None:
    class Args:
        verifier_model = "gemini-3.1-pro-preview"

    row = verifier_output_row_for_result(
        task=GeminiTask(
            key="article-1::w0",
            record={
                "id": "article-1",
                "text": "Shelling damaged homes in Gaza.",
                "metadata": {"usage": {"extractor": {"input_tokens": 3}}},
            },
            prompt="prompt",
        ),
        result={"key": "article-1::w0", "error": "bad response", "usage": {"prompt_tokens": 5}},
        ontology=ONTOLOGY,
        args=Args(),
        settings={"temperature": 0.0, "reasoning_effort": "high", "max_tokens": 1024},
        api_mode="interactive",
        run_id="run-1",
    )

    assert row["status"] == "error"
    assert row["metadata"]["task_key"] == "article-1::w0"
    assert row["metadata"]["verifier_decision"] == "error"
    assert row["metadata"]["usage"]["verifier"]["input_tokens"] == 5


def test_verifier_fix_keeps_article_level_offsets() -> None:
    class Args:
        model = "gemini-3.1-pro-preview"
        verifier_model = "gemini-3.1-pro-preview"

    text = "Intro. Shelling damaged homes in Gaza."
    row = verifier_output_row_for_result(
        task=GeminiTask(
            key="article-1::w0",
            record={
                "id": "article-1",
                "text": text,
                "events": [],
                "locations": [],
                "metadata": {
                    "annotation_model": "gemini-3.1-pro-preview",
                    "task_key": "article-1::w0",
                    "usage": {"extractor": {"input_tokens": 3}},
                },
            },
            prompt="prompt",
        ),
        result={
            "key": "article-1::w0",
            "payload": {
                "decision": "fix",
                "events": [
                    {
                        "event_type": "artillery bombing",
                        "start": 7,
                        "end": 15,
                        "text": "Shelling",
                        "arguments": [
                            {
                                "role": "location",
                                "start": 33,
                                "end": 37,
                                "text": "Gaza",
                                "location_type": "other",
                            }
                        ],
                    }
                ],
                "locations": [{"start": 33, "end": 37, "text": "Gaza", "location_type": "other"}],
            },
            "usage": {"prompt_tokens": 5},
        },
        ontology=ONTOLOGY,
        args=Args(),
        settings={"temperature": 0.0, "reasoning_effort": "high", "max_tokens": 1024},
        api_mode="interactive",
        run_id="run-1",
    )

    assert row["events"][0]["start"] == 7
    assert row["events"][0]["arguments"][0]["start"] == 33
    assert row["metadata"]["verifier_decision"] == "fix"


def test_normalize_usage_supports_aliases_and_batch_usage_metadata() -> None:
    usage = normalize_usage(
        {
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 3,
                "cachedContentTokenCount": 2,
                "thoughtsTokenCount": 5,
            }
        }
    )
    assert usage == {
        "input_tokens": 11,
        "output_tokens": 3,
        "cached_input_tokens": 2,
        "thoughts_tokens": 5,
    }
    assert normalize_usage({"prompt_tokens": 7, "completion_tokens": 2})["input_tokens"] == 7


def test_audit_report_includes_model_based_cost_estimate() -> None:
    records = [
        {
            "id": "article-1",
            "events": [],
            "locations": [],
            "metadata": {
                "annotation_model": "gemini-2.5-flash",
                "verifier_model": "gemini-2.5-flash-lite",
                "usage": {
                    "extractor": {
                        "input_tokens": 1_000,
                        "output_tokens": 100,
                        "cached_input_tokens": 100,
                    },
                    "verifier": {
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "cached_input_tokens": 0,
                    },
                    "total": {
                        "input_tokens": 1_500,
                        "output_tokens": 150,
                        "cached_input_tokens": 100,
                    },
                },
            },
        }
    ]

    report = build_report(records, pricing_mode="standard")

    assert report["cost_estimate"]["components"]["extractor"]["total_cost_usd"] == "0.00052300"
    assert report["cost_estimate"]["components"]["verifier"]["total_cost_usd"] == "0.00007000"
    assert report["cost_estimate"]["total_cost_usd"] == "0.00059300"
    assert report["cost_estimate"]["skipped"] == []


def test_batch_generation_config_includes_controls_and_schema() -> None:
    config = batch_generation_config(
        model="gemini-3.1-pro-preview",
        settings={"temperature": 0.2, "reasoning_effort": "high", "max_tokens": 123},
        response_schema=GeneratedPayload,
    )
    assert config["temperature"] == 0.2
    assert config["maxOutputTokens"] == 123
    assert config["responseMimeType"] == "application/json"
    assert config["thinkingConfig"] == {"thinkingLevel": "high"}
    assert "properties" in config["responseJsonSchema"]


def test_interactive_response_format_returns_pydantic_class() -> None:
    generated_response_format = interactive_response_format(GeneratedPayload)
    verifier_response_format = interactive_response_format(VerifierDecision)

    assert generated_response_format is GeneratedPayload
    assert verifier_response_format is VerifierDecision


def test_llm_client_parse_response_format_accepts_pydantic_classes_and_instances() -> None:
    assert LLMClient.parse_response_format(VerifierDecision) is VerifierDecision
    assert LLMClient.parse_response_format(VerifierDecision.model_construct()) is VerifierDecision
    assert LLMClient.parse_response_format({"decision": str}, add_cot_field=False).__name__ == "ResponseFormat"


def test_build_batch_request_uses_top_level_batch_fields() -> None:
    config = batch_generation_config(
        model="gemini-3.1-pro-preview",
        settings={"temperature": 0.2, "reasoning_effort": "high", "max_tokens": 123},
        response_schema=GeneratedPayload,
    )
    request = build_batch_request(
        GeminiTask(key="r1", record={}, prompt="prompt"),
        system_prompt="system",
        generation_config=config,
    )["request"]
    assert "config" not in request
    assert request["systemInstruction"]["parts"][0]["text"] == "system"
    assert request["generationConfig"]["temperature"] == 0.2
    assert request["contents"][0]["parts"][0]["text"] == "prompt"


def test_extract_batch_response_parses_payload_and_usage() -> None:
    line = {
        "key": "r1",
        "response": {
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"events":[],"locations":[],'
                                    '"has_target_event":false,"negative_reason":"none"}'
                                )
                            }
                        ]
                    }
                }
            ],
        },
    }
    key, payload, usage, error, thought_summaries = extract_batch_response(line)
    assert key == "r1"
    assert payload["events"] == []
    assert usage["input_tokens"] == 4
    assert error is None
    assert thought_summaries == []


def test_batch_generation_config_can_request_thought_summaries() -> None:
    config = batch_generation_config(
        model="gemini-3.1-pro-preview",
        settings={
            "temperature": 0.0,
            "reasoning_effort": "low",
            "max_tokens": 32,
            "save_thought_summaries": True,
        },
        response_schema=GeneratedPayload,
    )
    assert config["thinkingConfig"]["includeThoughts"] is True


def test_extract_batch_response_skips_thought_parts_when_parsing_payload() -> None:
    line = {
        "key": "r1",
        "response": {
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "checked spans", "thought": True},
                            {
                                "text": (
                                    '{"events":[],"locations":[],'
                                    '"has_target_event":false,"negative_reason":"none"}'
                                )
                            },
                        ]
                    }
                }
            ],
        },
    }
    key, payload, usage, error, thought_summaries = extract_batch_response(line)
    assert key == "r1"
    assert payload["locations"] == []
    assert usage["input_tokens"] == 4
    assert error is None
    assert thought_summaries == ["checked spans"]


def test_run_interactive_tasks_respects_worker_limit(monkeypatch: Any) -> None:
    state: dict[str, Any] = {
        "in_flight": 0,
        "max_in_flight": 0,
        "response_format": None,
        "include_thoughts": None,
    }

    class FakeResponse:
        text = '{"events":[],"locations":[],"has_target_event":false,"negative_reason":"none"}'
        metadata = {"prompt_tokens": 1, "completion_tokens": 1, "thought_summaries": ["summary"]}
        parsed = None

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def generate(self, *args: Any, **kwargs: Any):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            state["response_format"] = kwargs["response_format"]
            state["include_thoughts"] = kwargs["include_thoughts"]
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            yield FakeResponse()

    monkeypatch.setattr(gemini_common, "GeminiLLMClient", FakeClient)
    tasks = [GeminiTask(key=str(index), record={}, prompt="prompt") for index in range(5)]
    results = asyncio.run(
        run_interactive_tasks(
            tasks=tasks,
            model="gemini",
            system_prompt="system",
            settings={
                "temperature": 0.0,
                "reasoning_effort": "disable",
                "max_tokens": 32,
                "save_thought_summaries": True,
            },
            response_schema=GeneratedPayload,
            workers=2,
            desc="test",
        )
    )
    assert len(results) == 5
    assert state["max_in_flight"] == 2
    assert state["response_format"] is GeneratedPayload
    assert state["include_thoughts"] is True
    assert results[0]["thought_summaries"] == ["summary"]


def test_run_interactive_tasks_retries_malformed_json_at_zero_temperature(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"calls": 0, "temperatures": []}

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text
            self.metadata = {"prompt_tokens": 1, "completion_tokens": 1}
            self.parsed = None

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def generate(self, *args: Any, **kwargs: Any):
            state["calls"] += 1
            state["temperatures"].append(kwargs["override_settings"]["temperature"])
            if state["calls"] == 1:
                yield FakeResponse('{"events":[')
            else:
                yield FakeResponse(
                    '{"events":[],"locations":[],'
                    '"has_target_event":false,"negative_reason":"none"}'
                )

    monkeypatch.setattr(gemini_common, "GeminiLLMClient", FakeClient)
    result = asyncio.run(
        run_interactive_tasks(
            tasks=[GeminiTask(key="w0", record={}, prompt="prompt")],
            model="gemini",
            system_prompt="system",
            settings={"temperature": 1.0, "reasoning_effort": "disable", "max_tokens": 32},
            response_schema=GeneratedPayload,
            workers=1,
            desc="test",
        )
    )[0]

    assert not result.get("error")
    assert result["payload"]["events"] == []
    assert state["temperatures"] == [1.0, 0.0]


def test_generation_v2_has_no_token_label_export_script() -> None:
    assert not Path("scripts/data/generation_v2/to_token_labels.py").exists()


def test_keyword_quality_score_prioritizes_food_insecurity_terms(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )

    terms = keyword_terms(ontology)
    score = keyword_quality_score(
        "Food insecurity worsens",
        "Aid groups warned of food shortages, malnutrition, and drought.",
        terms,
    )

    assert score >= 10.0


def test_keyword_sampling_adds_risk_factor_bucket_and_score(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    long_text = " ".join(["Food insecurity and food shortages are worsening after drought."] * 20)
    records = [{"id": "food-1", "title": "Food crisis", "text": long_text}]

    selected = sample_records(records, ontology_path=ontology, limit=1, seed=1, keyword=True)

    assert selected[0]["source_bucket"] == "keyword_risk_factor"
    assert selected[0]["keyword_quality_score"] >= 4.0


def test_sampling_outputs_compact_rows_without_nested_source(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    text = " ".join(["Food insecurity and food shortages worsened after drought."] * 40)
    records = [
        {
            "id": "article-1",
            "status": "ok",
            "source": {
                "title": "Nested title",
                "text": text,
                "source_url": "https://example.test/article",
                "publish_date": "2024-01-01",
            },
            "title": "Nested title",
            "text": text,
            "source_url": "https://example.test/article",
            "publish_date": "2024-01-01",
            "events": [{"event_type": "food scarcity"}],
        }
    ]

    selected = sample_records(records, ontology_path=ontology, limit=1, seed=1, keyword=True)

    assert set(selected[0]) == {
        "id",
        "title",
        "text",
        "source_url",
        "publish_date",
        "source_bucket",
        "quality_score",
        "overall_score",
        "keyword_quality_score",
    }


def test_sampling_deduplicates_repeated_titles_and_keeps_best_candidate(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    weak_text = " ".join(["Conflict affected markets."] * 40)
    strong_text = " ".join(["Food insecurity and food shortages worsened after drought."] * 40)
    records = [
        {"id": "weak", "title": "Repeated food crisis title", "text": weak_text},
        {"id": "strong", "title": "Repeated food crisis title", "text": strong_text},
    ]

    selected = sample_records(records, ontology_path=ontology, limit=None, seed=1, keyword=True)

    assert [row["id"] for row in selected] == ["strong"]


def test_sampling_deduplicates_identical_normalized_text(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    text = " ".join(["Food insecurity and food shortages worsened after drought."] * 40)
    records = [
        {"id": "first", "title": "First title", "text": text},
        {"id": "second", "title": "Second title", "text": text.replace(" ", "  ")},
    ]

    selected = sample_records(records, ontology_path=ontology, limit=None, seed=1, keyword=True)

    assert len(selected) == 1
    assert selected[0]["id"] == "first"


def test_sampling_deduplicates_article_specific_canonical_source_url(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    text = " ".join(["Food insecurity and food shortages worsened after drought."] * 40)
    records = [
        {
            "id": "first",
            "title": "First food crisis article",
            "text": text,
            "source_url": "https://www.example.test/news/story?utm_medium=social&b=2&a=1",
            "publish_date": "2024-02-03T10:11:12Z",
        },
        {
            "id": "second",
            "title": "Second food crisis article",
            "text": text.replace("drought", "conflict"),
            "source_url": "https://example.test/news/story?a=1&b=2",
            "publish_date": "2024-02-03",
        },
    ]

    selected = sample_records(records, ontology_path=ontology, limit=None, seed=1, keyword=True)

    assert [row["id"] for row in selected] == ["first"]


def test_sampling_does_not_deduplicate_bare_source_domain(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    first_text = " ".join(["Food scarcity affected households after drought."] * 40)
    second_text = " ".join(["Market disruption increased food prices after conflict."] * 40)
    records = [
        {
            "id": "first",
            "title": "Food scarcity article from one country",
            "text": first_text,
            "source_url": "https://example.test",
            "publish_date": "2024-02-03",
        },
        {
            "id": "second",
            "title": "Market disruption article from another country",
            "text": second_text,
            "source_url": "https://www.example.test/",
            "publish_date": "2024-02-03",
        },
    ]

    selected = sample_records(records, ontology_path=ontology, limit=None, seed=1, keyword=True)

    assert {row["id"] for row in selected} == {"first", "second"}


def test_sampling_deduplicates_near_duplicate_text_with_parsing_variation(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    base_sentence = "Food insecurity and food shortages worsened after drought in the region."
    text = " ".join([base_sentence] * 45)
    varied_text = " ".join(["ADVERTISEMENT", base_sentence] + [base_sentence] * 43 + ["Related coverage"])
    records = [
        {"id": "first", "title": "First food scarcity report", "text": text},
        {"id": "second", "title": "Second food scarcity report", "text": varied_text},
    ]

    selected = sample_records(records, ontology_path=ontology, limit=None, seed=1, keyword=True)

    assert len(selected) == 1


def test_sampling_can_order_output_by_score(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    low_text = " ".join(["Market disruption increased staple prices after conflict."] * 20)
    high_text = " ".join(["Food scarcity affected households."] * 40)
    records = [
        {"id": "low", "title": "Low scoring food scarcity article", "text": low_text},
        {"id": "high", "title": "High scoring food scarcity article", "text": high_text},
    ]

    selected = sample_records(
        records,
        ontology_path=ontology,
        limit=None,
        seed=1,
        keyword=False,
        order_by_score=True,
    )

    assert [row["id"] for row in selected] == ["high", "low"]
    assert selected[0]["overall_score"] > selected[1]["overall_score"]


def test_sampling_summary_counts_filtered_duplicates_and_output(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"events":{"food scarcity":"Food is insufficiently available, causing hunger risk."}}',
        encoding="utf-8",
    )
    long_text = " ".join(["Food scarcity affected households."] * 40)
    short_text = "Too short."
    records = [
        {"id": "a-kept", "title": "Food scarcity article one", "text": long_text},
        {"id": "b-duplicate", "title": "Food scarcity article two", "text": long_text.replace(" ", "  ")},
        {"id": "filtered", "title": "Short article", "text": short_text},
    ]
    summary = SamplingSummary()

    selected = sample_records(
        records,
        ontology_path=ontology,
        limit=None,
        seed=1,
        keyword=False,
        summary=summary,
    )

    assert [row["id"] for row in selected] == ["a-kept"]
    assert summary.input_count == 3
    assert summary.quality_filtered_count == 1
    assert summary.scored_count == 2
    assert summary.duplicate_removed_count == 1
    assert summary.unique_count == 1
    assert summary.selected_count == 1
