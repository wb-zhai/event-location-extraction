from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "scripts" / "data" / "generation_v3" / "fix_events.py"
SPEC = importlib.util.spec_from_file_location("fix_events", MODULE_PATH)
fix_events = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(fix_events)

EventRow = fix_events.EventRow
FixTask = fix_events.FixTask
EventDecision = fix_events.EventDecision
FixedEvent = fix_events.FixedEvent
ArticleFixResult = fix_events.ArticleFixResult

ARTICLE_TEXT = "A severe drought hit the Sahel region in 2022, affecting millions of people."
PUBLISH_DATE = "2023-01-01"


def make_row(row: int = 0, event_type: str = "drought", doc_id: str = "doc1") -> EventRow:
    return EventRow(
        row=row,
        doc_id=doc_id,
        source_text=ARTICLE_TEXT,
        publish_date=PUBLISH_DATE,
        event={
            "event_type": event_type,
            "event_location_text": "the Sahel region",
            "event_location": "Sahel",
            "event_time_text": "2022",
            "event_time": "2022",
            "time_status": "past",
            "affected_entity": "not_stated",
            "affected_group": "millions of people",
            "severity": "high",
            "modality": "asserted",
            "grounding_quote": "A severe drought hit the Sahel region in 2022",
        },
        errors=["event_type 'drought' not in ontology"],
    )


ONTOLOGY = fix_events.load_ontology_labels(
    REPO_ROOT / "ontologies" / "zhai" / "ontology.events.json"
)


# ---------------------------------------------------------------------------
# render_user_prompt
# ---------------------------------------------------------------------------


def test_render_user_prompt_fills_all_placeholders() -> None:
    template = (
        "<publish_date>{{PUBLISH_DATE}}</publish_date>\n"
        "{{ARTICLE_TEXT}}\n"
        "{{INVALID_EVENTS}}"
    )
    events = [{"event": {"event_type": "drought"}, "errors": ["off ontology"]}]
    result = fix_events.render_user_prompt(template, PUBLISH_DATE, ARTICLE_TEXT, events)
    assert PUBLISH_DATE in result
    assert ARTICLE_TEXT in result
    assert "drought" in result


def test_render_user_prompt_uses_not_stated_for_missing_date() -> None:
    template = "{{PUBLISH_DATE}}"
    result = fix_events.render_user_prompt(template, "", ARTICLE_TEXT, [])
    assert result == "not_stated"


# ---------------------------------------------------------------------------
# build_tasks — per-event grouping
# ---------------------------------------------------------------------------


def test_build_tasks_per_event_one_task_per_row() -> None:
    rows = [make_row(row=0, doc_id="d1"), make_row(row=1, doc_id="d1"), make_row(row=2, doc_id="d2")]
    template = "{{PUBLISH_DATE}} {{ARTICLE_TEXT}} {{INVALID_EVENTS}}"
    tasks = fix_events.build_tasks(rows, template, mode="per-event")
    assert len(tasks) == 3
    for task, row in zip(tasks, rows):
        assert len(task.rows) == 1
        assert task.rows[0].row == row.row


# ---------------------------------------------------------------------------
# build_tasks — per-article grouping
# ---------------------------------------------------------------------------


def test_build_tasks_per_article_groups_by_doc_id() -> None:
    rows = [make_row(row=0, doc_id="d1"), make_row(row=1, doc_id="d1"), make_row(row=2, doc_id="d2")]
    template = "{{PUBLISH_DATE}} {{ARTICLE_TEXT}} {{INVALID_EVENTS}}"
    tasks = fix_events.build_tasks(rows, template, mode="per-article")
    assert len(tasks) == 2
    assert len(tasks[0].rows) == 2
    assert len(tasks[1].rows) == 1


def test_build_tasks_per_article_prompt_includes_both_events() -> None:
    rows = [make_row(row=0, doc_id="d1"), make_row(row=1, doc_id="d1")]
    template = "{{INVALID_EVENTS}}"
    tasks = fix_events.build_tasks(rows, template, mode="per-article")
    assert len(tasks) == 1
    payload = json.loads(tasks[0].prompt)
    assert len(payload) == 2


# ---------------------------------------------------------------------------
# check_event (re-validation)
# ---------------------------------------------------------------------------


def test_check_event_valid_returns_no_errors() -> None:
    event = {
        "event_type": "drought conditions",
        "event_location_text": "the Sahel region",
        "event_location": "Sahel",
        "event_time_text": "2022",
        "event_time": "2022",
        "time_status": "past",
        "affected_entity": "not_stated",
        "affected_group": "millions of people",
        "severity": "high",
        "modality": "asserted",
        "grounding_quote": "A severe drought hit the Sahel region in 2022",
    }
    errors = fix_events.check_event(event, ARTICLE_TEXT, ONTOLOGY, set())
    assert errors == []


def test_check_event_off_ontology_type() -> None:
    event = {
        "event_type": "drought",
        "event_location_text": "not_stated",
        "event_location": "not_stated",
        "event_time_text": "not_stated",
        "event_time": "not_stated",
        "time_status": "past",
        "affected_entity": "not_stated",
        "affected_group": "not_stated",
        "severity": "not_stated",
        "modality": "asserted",
        "grounding_quote": "A severe drought hit the Sahel region in 2022",
    }
    errors = fix_events.check_event(event, ARTICLE_TEXT, ONTOLOGY, set())
    assert any("not in ontology" in e for e in errors)


def test_check_event_grounding_quote_not_substring() -> None:
    event = {
        "event_type": "drought conditions",
        "event_location_text": "not_stated",
        "event_location": "not_stated",
        "event_time_text": "not_stated",
        "event_time": "not_stated",
        "time_status": "past",
        "affected_entity": "not_stated",
        "affected_group": "not_stated",
        "severity": "not_stated",
        "modality": "asserted",
        "grounding_quote": "This quote does not appear in the article at all XYZ.",
    }
    errors = fix_events.check_event(event, ARTICLE_TEXT, ONTOLOGY, set())
    assert any("grounding_quote" in e for e in errors)


# ---------------------------------------------------------------------------
# make_output_record — revalidation wiring
# ---------------------------------------------------------------------------


def test_make_output_record_valid_fixed_event() -> None:
    row = make_row()
    decision = EventDecision(
        decision="fixed",
        fixed_event=FixedEvent(
            event_type="drought conditions",
            event_location_text="the Sahel region",
            event_location="Sahel",
            event_time_text="2022",
            event_time="2022",
            time_status="past",
            affected_entity="not_stated",
            affected_group="millions of people",
            severity="high",
            modality="asserted",
            grounding_quote="A severe drought hit the Sahel region in 2022",
        ),
        reason="Remapped drought -> drought conditions",
    )
    record = fix_events.make_output_record(row, decision, model="gemini-test", ontology=ONTOLOGY)
    assert record["decision"] == "fixed"
    assert record["revalidation"]["valid"] is True
    assert record["revalidation"]["errors"] == []
    assert record["status"] == "ok"


def test_make_output_record_fixed_event_still_fails_revalidation() -> None:
    row = make_row()
    decision = EventDecision(
        decision="fixed",
        fixed_event=FixedEvent(
            event_type="drought conditions",
            event_location_text="not_stated",
            event_location="not_stated",
            event_time_text="not_stated",
            event_time="not_stated",
            time_status="past",
            affected_entity="not_stated",
            affected_group="not_stated",
            severity="not_stated",
            modality="asserted",
            grounding_quote="This is not in the article.",
        ),
        reason="Remapped but bad quote",
    )
    record = fix_events.make_output_record(row, decision, model="gemini-test", ontology=ONTOLOGY)
    assert record["revalidation"]["valid"] is False
    assert len(record["revalidation"]["errors"]) > 0


def test_make_output_record_dropped_event() -> None:
    row = make_row()
    decision = EventDecision(
        decision="dropped",
        fixed_event=None,
        reason="No ontology label fits: this is a geological event.",
    )
    record = fix_events.make_output_record(row, decision, model="gemini-test", ontology=ONTOLOGY)
    assert record["decision"] == "dropped"
    assert record["fixed_event"] is None
    assert record["revalidation"]["valid"] is False
    assert record["reason"] != ""


# ---------------------------------------------------------------------------
# make_error_record
# ---------------------------------------------------------------------------


def test_make_error_record_has_expected_fields() -> None:
    row = make_row()
    record = fix_events.make_error_record(row, "API timeout", model="gemini-test")
    assert record["status"] == "error"
    assert record["error"] == "API timeout"
    assert record["row"] == 0


# ---------------------------------------------------------------------------
# count-guard in process_task (per-article mode)
# ---------------------------------------------------------------------------

import asyncio


def _make_mock_client(parsed_payload: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = json.dumps(parsed_payload)
    mock_response.metadata = {}

    async def _generate(*args, **kwargs):
        yield mock_response

    mock_client = MagicMock()
    mock_client.generate = _generate
    return mock_client


def test_process_task_per_article_count_guard() -> None:
    rows = [make_row(row=0), make_row(row=1)]
    # Model returns only 1 decision for 2 events
    payload = {
        "decisions": [
            {"decision": "fixed", "fixed_event": None, "reason": "only one returned"}
        ]
    }
    task = FixTask(rows=rows, prompt="test prompt")
    mock_client = _make_mock_client(payload)

    records = asyncio.run(
        fix_events.process_task(mock_client, task, model="gemini-test", ontology=ONTOLOGY, mode="per-article")
    )
    assert len(records) == 2
    assert records[0]["status"] == "ok"
    # second row gets error because decision is missing
    assert records[1]["status"] == "error"
    assert "missing" in records[1]["error"].lower()


def test_process_task_per_event() -> None:
    row = make_row(row=0)
    payload = {
        "decision": "dropped",
        "fixed_event": None,
        "reason": "earthquake has no ontology home",
    }
    task = FixTask(rows=[row], prompt="test prompt")
    mock_client = _make_mock_client(payload)

    records = asyncio.run(
        fix_events.process_task(mock_client, task, model="gemini-test", ontology=ONTOLOGY, mode="per-event")
    )
    assert len(records) == 1
    assert records[0]["decision"] == "dropped"
    assert records[0]["fixed_event"] is None


# ---------------------------------------------------------------------------
# completed_rows resume logic
# ---------------------------------------------------------------------------


def test_completed_rows_reads_existing_output(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    records = [
        {"row": 0, "status": "ok"},
        {"row": 1, "status": "ok"},
        {"row": 2, "status": "error"},
    ]
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    done = fix_events.completed_rows(out)
    assert done == {0, 1}  # error rows are not counted as done


def test_completed_rows_empty_if_no_file(tmp_path: Path) -> None:
    assert fix_events.completed_rows(tmp_path / "nonexistent.jsonl") == set()
