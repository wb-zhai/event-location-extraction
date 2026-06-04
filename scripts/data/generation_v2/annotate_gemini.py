"""Gemini annotation CLI for generation_v2."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2.gemini_common import (
    GeneratedPayload,
    GeminiTask,
    add_gemini_args,
    load_env_file,
    prompt_settings,
    run_async,
    run_batch_tasks,
    run_interactive_tasks,
)
from scripts.data.generation_v2.io_utils import (
    append_jsonl_row,
    completed_ids,
    completed_task_keys,
    has_completed_task_keys,
    iter_jsonl,
    load_json_tolerant,
    prune_retryable_error_rows,
    resolve_path,
    write_jsonl,
)
from scripts.data.generation_v2.recover_annotations import recover_offsets
from scripts.data.generation_v2.schema import AnnotationRecord, validate_record_against_ontology
from scripts.data.generation_v2.usage import normalize_usage


SYSTEM_PROMPT = """<role>
You are a precise event-location annotation assistant for humanitarian risk articles.
</role>

<instructions>
1. Analyze the annotation guidelines, ontology, title, and article text.
2. Extract ontology event triggers, linked location arguments, and all explicit location mentions.
3. Validate that every span is an exact substring of article_text at the supplied offsets.
4. Return only the final structured JSON object.
</instructions>

<constraints>
- Use article_text as the only source of evidence.
- Use the title only as context for resolving article_text, never as an offset source.
- Do not infer events, locations, labels, or offsets from world knowledge.
- Omit unsupported or ambiguous events instead of guessing.
- Do not include Markdown, comments, explanations, or reasoning in the response.
</constraints>

<output_format>
Return one JSON object matching the provided response schema with:
- events: ontology event mentions with event_type, start, end, text, and arguments.
- locations: all explicit location mentions with start, end, text, and location_type.
- has_target_event: true only when at least one ontology event is present.
- negative_reason: a concise reason when has_target_event is false, otherwise null.
</output_format>"""


def load_guidelines() -> str:
    path = Path(__file__).with_name("annotation_guidelines.md")
    return path.read_text(encoding="utf-8")


def format_ontology(ontology: dict[str, Any]) -> str:
    events = ontology.get("events") or {}
    roles = ontology.get("argument_roles") or {}
    location_types = ontology.get("location_types") or {}
    event_roles = ontology.get("event_argument_roles") or {}
    return json.dumps(
        {
            "events": events,
            "argument_roles": roles,
            "location_types": location_types,
            "event_argument_roles": event_roles,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_prompt(record: dict[str, Any], ontology: dict[str, Any]) -> str:
    title = str(record.get("title") or "")
    text = str(record.get("text") or "")
    return f"""<context>
<annotation_guidelines>
{load_guidelines()}
</annotation_guidelines>

<ontology>
{format_ontology(ontology)}
</ontology>

<title context_only="true">
{title}
</title>

<article_text offset_source="true">
{text}
</article_text>
</context>

<task>
Extract event triggers, linked location arguments, and all location mentions.
Use only event types, argument roles, and location types from the ontology.
Offsets must be zero-based character offsets into article_text, with end exclusive.
If no ontology event is present, set has_target_event=false and still return all explicit locations.
</task>

<final_instruction>
Think step by step internally, then validate the JSON against the schema and the span rules before responding.
</final_instruction>"""


def _project(start: int, end: int, record: dict[str, Any]) -> tuple[int, int]:
    window = record.get("window") if isinstance(record.get("window"), dict) else None
    if not window:
        return start, end
    offset = int(window.get("start", 0))
    return offset + start, offset + end


def _in_core(start: int, end: int, record: dict[str, Any]) -> bool:
    window = record.get("window") if isinstance(record.get("window"), dict) else None
    if not window:
        return True
    global_start, global_end = _project(start, end, record)
    return int(window.get("core_start", 0)) <= global_start < global_end <= int(window.get("core_end", 0))


def clean_payload(
    *,
    payload: dict[str, Any],
    source_record: dict[str, Any],
    ontology: dict[str, Any],
    model: str,
    api_mode: str,
    settings: dict[str, Any],
    usage: dict[str, int],
    batch_job_name: str | None = None,
    raw_response_path: str | None = None,
    thought_summaries: list[str] | None = None,
) -> dict[str, Any]:
    local_text = str(source_record.get("text") or "")
    article_text = str(source_record.get("article_text") or local_text)
    events: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    event_types = set((ontology.get("events") or {}).keys())
    location_types = set((ontology.get("location_types") or {}).keys())
    role_by_event = ontology.get("event_argument_roles") or {}

    for event in payload.get("events") or []:
        event_type = str(event.get("event_type") or "")
        if event_type not in event_types:
            continue
        recovered = recover_offsets(
            str(event.get("text") or event.get("trigger_text") or ""),
            local_text,
            int(event.get("start", event.get("start_char", -1))),
            int(event.get("end", event.get("end_char", -1))),
        )
        if recovered is None or not _in_core(recovered[0], recovered[1], source_record):
            continue
        start, end = _project(recovered[0], recovered[1], source_record)
        arguments: list[dict[str, Any]] = []
        allowed_roles = set(role_by_event.get(event_type, []))
        for argument in event.get("arguments") or []:
            role = str(argument.get("role") or "")
            location_type = str(argument.get("location_type") or "other")
            if role not in allowed_roles or location_type not in location_types:
                continue
            arg_recovered = recover_offsets(
                str(argument.get("text") or ""),
                local_text,
                int(argument.get("start", argument.get("start_char", -1))),
                int(argument.get("end", argument.get("end_char", -1))),
            )
            if arg_recovered is None:
                continue
            arg_start, arg_end = _project(arg_recovered[0], arg_recovered[1], source_record)
            arguments.append(
                {
                    "role": role,
                    "start": arg_start,
                    "end": arg_end,
                    "text": article_text[arg_start:arg_end],
                    "location_type": location_type,
                }
            )
        events.append(
            {
                "event_type": event_type,
                "start": start,
                "end": end,
                "text": article_text[start:end],
                "arguments": arguments,
            }
        )

    seen_locations: set[tuple[int, int, str]] = set()
    for location in payload.get("locations") or []:
        location_type = str(location.get("location_type") or "other")
        if location_type not in location_types:
            continue
        recovered = recover_offsets(
            str(location.get("text") or ""),
            local_text,
            int(location.get("start", location.get("start_char", -1))),
            int(location.get("end", location.get("end_char", -1))),
        )
        if recovered is None:
            continue
        start, end = _project(recovered[0], recovered[1], source_record)
        key = (start, end, location_type)
        if key in seen_locations:
            continue
        seen_locations.add(key)
        locations.append(
            {
                "start": start,
                "end": end,
                "text": article_text[start:end],
                "location_type": location_type,
            }
        )

    metadata = {
        "annotation_model": model,
        "pipeline_version": "generation_v2",
        "source_bucket": source_record.get("source_bucket"),
        "api_mode": api_mode,
        "run_id": source_record.get("run_id"),
        "task_key": source_record.get("id"),
        "window_indices": [
            int(source_record["window"]["window_index"])
        ]
        if isinstance(source_record.get("window"), dict)
        else [],
        "prompt_settings": {"extractor": settings},
        "usage": {"extractor": normalize_usage(usage)},
        "batch_job_name": batch_job_name,
        "raw_response_path": raw_response_path,
    }
    if thought_summaries:
        metadata["thought_summaries"] = {"extractor": thought_summaries}
    record = AnnotationRecord(
        id=str(source_record.get("article_id") or source_record.get("id")),
        text=article_text,
        events=events,
        locations=locations,
        negatives={
            "has_target_event": bool(events),
            "negative_reason": None if events else payload.get("negative_reason"),
        },
        metadata=metadata,
    )
    validate_record_against_ontology(record, ontology)
    row = record.model_dump()
    row["source"] = {
        "title": source_record.get("title", ""),
        "source_url": source_record.get("source_url") or "",
        "publish_date": source_record.get("publish_date") or "",
    }
    return row


def build_tasks(records: list[dict[str, Any]], ontology: dict[str, Any], run_id: str) -> list[GeminiTask]:
    tasks: list[GeminiTask] = []
    for index, record in enumerate(records):
        row = {**record, "run_id": run_id}
        tasks.append(
            GeminiTask(
                key=str(row.get("id") or index),
                record=row,
                prompt=build_prompt(row, ontology),
            )
        )
    return tasks


def output_row_for_result(
    *,
    task: GeminiTask,
    result: dict[str, Any],
    ontology: dict[str, Any],
    model: str,
    api_mode: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if result.get("error"):
        return {
            "id": str(task.record.get("article_id") or task.record.get("id")),
            "status": "error",
            "error": result["error"],
            "text": task.record.get("article_text") or task.record.get("text"),
            "metadata": {
                "annotation_model": model,
                "api_mode": api_mode,
                "run_id": task.record.get("run_id"),
                "task_key": task.key,
                "usage": {"extractor": normalize_usage(result.get("usage"))},
                **(
                    {"thought_summaries": {"extractor": result["thought_summaries"]}}
                    if result.get("thought_summaries")
                    else {}
                ),
            },
        }
    try:
        return clean_payload(
            payload=result.get("payload") or {},
            source_record=task.record,
            ontology=ontology,
            model=model,
            api_mode=api_mode,
            settings=settings,
            usage=normalize_usage(result.get("usage")),
            batch_job_name=result.get("batch_job_name"),
            raw_response_path=result.get("raw_response_path"),
            thought_summaries=result.get("thought_summaries"),
        )
    except Exception as exc:
        return {
            "id": str(task.record.get("article_id") or task.record.get("id")),
            "status": "error",
            "error": f"failed to validate extraction response: {exc}",
            "text": task.record.get("article_text") or task.record.get("text"),
            "metadata": {
                "annotation_model": model,
                "api_mode": api_mode,
                "run_id": task.record.get("run_id"),
                "task_key": task.key,
                "usage": {"extractor": normalize_usage(result.get("usage"))},
                "batch_job_name": result.get("batch_job_name"),
                "raw_response_path": result.get("raw_response_path"),
                **(
                    {"thought_summaries": {"extractor": result["thought_summaries"]}}
                    if result.get("thought_summaries")
                    else {}
                ),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate generation_v2 windows with Gemini.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("ontologies/zhai/ontology.json"))
    add_gemini_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading environment from {args.env_file}")
    load_env_file(resolve_path(args.env_file))
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    print(f"Loading input windows from {input_path}")
    print(f"Loading ontology from {args.ontology}")
    ontology = load_json_tolerant(resolve_path(args.ontology))
    print(f"Checking existing output at {output_path} for resumable tasks")
    if args.overwrite:
        completed = set()
        source_records = list(iter_jsonl(input_path))
        write_jsonl(output_path, [], overwrite=True)
    elif has_completed_task_keys(output_path, retry_failed=args.retry_failed):
        completed = completed_task_keys(output_path, retry_failed=args.retry_failed)
        source_records = [record for record in iter_jsonl(input_path) if str(record.get("id")) not in completed]
        pruned = prune_retryable_error_rows(output_path, retry_failed=args.retry_failed)
        if pruned:
            print(f"Pruned {pruned} failed output rows before retrying")
    else:
        completed = completed_ids(output_path, retry_failed=args.retry_failed)
        source_records = [
            record
            for record in iter_jsonl(input_path)
            if str(record.get("article_id") or record.get("id")) not in completed
        ]
        pruned = prune_retryable_error_rows(output_path, retry_failed=args.retry_failed)
        if pruned:
            print(f"Pruned {pruned} failed output rows before retrying")
    run_id = uuid.uuid4().hex[:12]
    print(f"Preparing {len(source_records)} extraction tasks; skipping {len(completed)} completed tasks")
    tasks = build_tasks(source_records, ontology, run_id)
    settings = prompt_settings(args, component="extractor")
    run_dir = resolve_path(args.run_dir) / run_id
    task_by_key = {task.key: task for task in tasks}
    saved_count = 0

    def persist_result(result: dict[str, Any], *, api_mode: str) -> None:
        nonlocal saved_count
        task = task_by_key.get(result["key"])
        if task is None:
            return
        row = output_row_for_result(
            task=task,
            result=result,
            ontology=ontology,
            model=args.model,
            api_mode=api_mode,
            settings=settings,
        )
        append_jsonl_row(output_path, row)
        saved_count += 1

    if args.batch_api:
        print(
            "Submitting extraction tasks with Gemini Batch API "
            f"using batch_size={args.batch_size}, workers={args.workers}; run artifacts -> {run_dir}"
        )
        results = run_async(
            run_batch_tasks(
                tasks=tasks,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                response_schema=GeneratedPayload,
                run_dir=run_dir,
                batch_size=args.batch_size,
                batch_display_name=args.batch_display_name,
                poll_interval_seconds=args.batch_poll_interval_seconds,
                workers=args.workers,
                desc="Extraction",
                on_result=lambda result: persist_result(result, api_mode="batch"),
            )
        )
        api_mode = "batch"
    else:
        print(f"Running interactive Gemini extraction with workers={args.workers}")
        results = run_async(
            run_interactive_tasks(
                tasks=tasks,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                response_schema=GeneratedPayload,
                workers=args.workers,
                desc="Extracting",
                on_result=lambda result: persist_result(result, api_mode="interactive"),
            )
        )
        api_mode = "interactive"

    if saved_count != len(results):
        raise RuntimeError(f"saved {saved_count} rows but received {len(results)} {api_mode} results")
    print(f"Saved {saved_count} annotation rows to {output_path}")
    print("Annotation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
