from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "scripts" / "data" / "generation_v3" / "generate.py"
SPEC = importlib.util.spec_from_file_location("generation_v3_generate", MODULE_PATH)
generate = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(generate)


def test_render_prompt_fills_publish_date_separately() -> None:
    template = (
        "<context>\n"
        "<publish_date>{{PUBLISH_DATE}}</publish_date>\n"
        "<article_text>\n"
        "{{ARTICLE_TEXT}}\n"
        "</article_text>\n"
        "</context>"
    )
    record = {
        "title": "Harvest warning",
        "publish_date": "2025-05-12",
        "text": "Crop losses were reported in Sindh.",
    }

    prompt = generate.render_prompt(template, record)

    assert "<publish_date>2025-05-12</publish_date>" in prompt
    assert "Title: Harvest warning\n\nCrop losses were reported in Sindh." in prompt
    assert "publish_date: 2025-05-12" not in prompt


def test_render_prompt_uses_not_stated_for_missing_publish_date() -> None:
    prompt = generate.render_prompt(
        "<publish_date>{{PUBLISH_DATE}}</publish_date>\n{{ARTICLE_TEXT}}",
        {"title": "", "text": "Flooding affected the district."},
    )

    assert "<publish_date>not_stated</publish_date>" in prompt


def test_candidate_labels_dedupes_in_order() -> None:
    labels = generate.candidate_labels(
        {"candidates": [" food crisis ", "drought", "food crisis"]}
    )

    assert labels == ["food crisis", "drought"]


@pytest.mark.parametrize(
    "record",
    [
        {"candidates": "drought"},
        {"candidates": []},
        {"candidates": ["drought", ""]},
        {"candidates": ["drought", 3]},
    ],
)
def test_candidate_labels_rejects_malformed_values(record: dict) -> None:
    with pytest.raises(ValueError):
        generate.candidate_labels(record)


def test_annotation_schema_adds_candidate_event_type_enum() -> None:
    schema = generate.annotation_schema(["food crisis", "drought"])
    event_type_schema = schema["properties"]["events"]["items"]["properties"][
        "event_type"
    ]
    default_event_type_schema = generate.annotation_schema()["properties"]["events"][
        "items"
    ]["properties"]["event_type"]

    assert event_type_schema["enum"] == ["food crisis", "drought"]
    assert "enum" not in default_event_type_schema


def test_render_system_prompt_strips_output_schema_and_inserts_candidates() -> None:
    system_prompt = (
        "<role>extract</role>\n"
        "<output_schema>\n"
        '{"events": []}\n'
        "</output_schema>\n"
        "<allowed_event_types>\n"
        "old label\n"
        "</allowed_event_types>\n"
        "<extraction_rules>keep me</extraction_rules>"
    )

    rendered, labels = generate.render_system_prompt(
        system_prompt, {"candidates": ["food crisis", "drought"]}
    )

    assert labels == ["food crisis", "drought"]
    assert "<output_schema>" not in rendered
    assert '{"events": []}' not in rendered
    assert (
        "<allowed_event_types>\nfood crisis\ndrought\n</allowed_event_types>"
        in rendered
    )
    assert "<extraction_rules>keep me</extraction_rules>" in rendered


def test_validate_candidate_event_types_rejects_off_candidate_event() -> None:
    annotation = {
        "document_relevance": "relevant",
        "events": [
            {
                "event_type": "floods",
                "event_location_text": "not_stated",
                "event_location": "not_stated",
                "event_time_text": "not_stated",
                "event_time": "not_stated",
                "time_status": "not_stated",
                "affected_entity": "not_stated",
                "affected_group": "not_stated",
                "severity": "not_stated",
                "modality": "asserted",
                "grounding_quote": "Flooding affected the district.",
            }
        ],
    }

    with pytest.raises(ValueError, match="not in candidates"):
        generate.validate_candidate_event_types(annotation, ["drought"])


def test_batch_request_line_includes_per_record_system_prompt_and_schema() -> None:
    system_prompt = (
        "<output_schema>{}</output_schema>\n"
        "<allowed_event_types>\nold label\n</allowed_event_types>"
    )
    task = generate.build_batch_tasks(
        [
            {
                "id": "doc-1",
                "title": "Flooding",
                "text": "Flooding affected the district.",
                "candidates": ["floods"],
            }
        ],
        "{{ARTICLE_TEXT}}",
        system_prompt,
    )[0]
    request_config = {
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 1024,
        }
    }

    line = generate.batch_request_line(task, request_config)
    request = line["request"]

    assert request["systemInstruction"]["parts"][0]["text"] == task.system_prompt
    assert "<output_schema>" not in task.system_prompt
    event_type_schema = request["generationConfig"]["responseSchema"]["properties"][
        "events"
    ]["items"]["properties"]["event_type"]
    assert event_type_schema["enum"] == ["floods"]
    assert "responseSchema" not in request_config["generationConfig"]


def test_batch_request_config_requests_thoughts_when_reasoning_enabled() -> None:
    args = argparse.Namespace(
        model="gemini-3.1-pro-preview",
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort="high",
        include_thoughts=False,
    )

    config = generate.batch_request_config(args)

    assert config["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high",
        "includeThoughts": True,
    }


def test_batch_request_config_does_not_request_thoughts_when_reasoning_disabled() -> None:
    args = argparse.Namespace(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort="disable",
        include_thoughts=False,
    )

    config = generate.batch_request_config(args)

    assert config["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_run_sync_writes_record_level_error_for_malformed_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyClient:
        def __init__(self, **kwargs) -> None:
            self.reasoning_effort = kwargs.get("reasoning_effort")

    monkeypatch.setattr(generate, "GeminiLLMClient", DummyClient)

    output = tmp_path / "out.jsonl"
    args = argparse.Namespace(
        output=output,
        model="gemini-test",
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort=None,
        workers=1,
        include_thoughts=False,
    )
    system_prompt = "<allowed_event_types>\nfloods\n</allowed_event_types>"
    records = [{"id": "bad", "title": "", "text": "Flooding.", "candidates": []}]

    asyncio.run(generate.run_sync(args, records, "{{ARTICLE_TEXT}}", system_prompt))

    result = json.loads(output.read_text(encoding="utf-8").strip())
    assert result["id"] == "bad"
    assert result["status"] == "error"
    assert "candidates must contain at least one label" in result["error"]


def test_run_sync_requests_thoughts_when_reasoning_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, object] = {}

    class DummyResponse:
        text = json.dumps({"document_relevance": "not_relevant", "events": []})
        metadata = {
            "thoughts_token_count": 4,
            "thought_summaries": ["checked relevance"],
        }

    class DummyClient:
        def __init__(self, **kwargs) -> None:
            self.reasoning_effort = kwargs.get("reasoning_effort")

        async def generate(self, *args, **kwargs):
            state["include_thoughts"] = kwargs["include_thoughts"]
            yield DummyResponse()

    monkeypatch.setattr(generate, "GeminiLLMClient", DummyClient)

    output = tmp_path / "out.jsonl"
    args = argparse.Namespace(
        output=output,
        model="gemini-test",
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort="high",
        workers=1,
        include_thoughts=False,
    )
    system_prompt = "<allowed_event_types>\nfloods\n</allowed_event_types>"
    records = [{"id": "ok", "title": "", "text": "No relevant event."}]

    asyncio.run(generate.run_sync(args, records, "{{ARTICLE_TEXT}}", system_prompt))

    result = json.loads(output.read_text(encoding="utf-8").strip())
    assert state["include_thoughts"] is True
    assert result["llm"]["metadata"]["thought_summaries"] == ["checked relevance"]
