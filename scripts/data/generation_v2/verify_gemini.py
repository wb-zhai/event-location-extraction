"""Gemini verifier CLI for generation_v2 annotations."""

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

from scripts.data.generation_v2.annotate_gemini import SYSTEM_PROMPT, clean_payload, format_ontology
from scripts.data.generation_v2.gemini_common import (
    GeminiTask,
    VerifierDecision,
    add_gemini_args,
    load_env_file,
    prompt_settings,
    run_async,
    run_batch_tasks,
    run_interactive_tasks,
)
from scripts.data.generation_v2.io_utils import (
    append_jsonl_row,
    completed_task_keys,
    iter_jsonl,
    load_json_tolerant,
    prune_retryable_error_rows,
    resolve_path,
    write_jsonl,
)
from scripts.data.generation_v2.usage import add_usage, normalize_usage


def candidate_intervals(record: dict[str, Any]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for event in record.get("events") or []:
        intervals.append(
            (
                int(event.get("start", event.get("start_char", -1))),
                int(event.get("end", event.get("end_char", -1))),
            )
        )
        for argument in event.get("arguments") or []:
            intervals.append(
                (
                    int(argument.get("start", argument.get("start_char", -1))),
                    int(argument.get("end", argument.get("end_char", -1))),
                )
            )
    for location in record.get("locations") or []:
        intervals.append(
            (
                int(location.get("start", location.get("start_char", -1))),
                int(location.get("end", location.get("end_char", -1))),
            )
        )
    text_length = len(str(record.get("text") or ""))
    return [
        (start, end)
        for start, end in intervals
        if 0 <= start < end <= text_length
    ]


def verification_excerpts(
    record: dict[str, Any],
    *,
    context_chars: int = 800,
    fallback_chars: int = 2000,
) -> list[dict[str, Any]]:
    text = str(record.get("text") or "")
    intervals = candidate_intervals(record)
    if not intervals:
        return [{"start_char": 0, "end_char": min(len(text), fallback_chars), "text": text[:fallback_chars]}]

    expanded = [
        (max(0, start - context_chars), min(len(text), end + context_chars))
        for start, end in intervals
    ]
    expanded.sort()
    merged: list[tuple[int, int]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))

    return [
        {"start_char": start, "end_char": end, "text": text[start:end]}
        for start, end in merged
    ]


def build_prompt(record: dict[str, Any], ontology: dict[str, Any]) -> str:
    candidate = {
        "events": record.get("events") or [],
        "locations": record.get("locations") or [],
        "negatives": record.get("negatives") or {},
    }
    excerpts = verification_excerpts(record)
    excerpt_blocks = "\n\n".join(
        f'<article_excerpt start_char="{excerpt["start_char"]}" end_char="{excerpt["end_char"]}">\n'
        f'{excerpt["text"]}\n'
        "</article_excerpt>"
        for excerpt in excerpts
    )
    return f"""<ontology>
{format_ontology(ontology)}
</ontology>

<article_excerpts offset_source="true">
{excerpt_blocks}
</article_excerpts>

<candidate_annotation>
{json.dumps(candidate, ensure_ascii=False, indent=2)}
</candidate_annotation>

Verify the candidate annotation. Return accept if all spans and labels are
supported. Return reject if the annotation should be dropped. Return fix with
corrected events/locations only when the correction is explicit in the excerpts.
All candidate offsets are zero-based article-level offsets. Any fix you return
must also use article-level offsets, not excerpt-local offsets."""


def build_tasks(records: list[dict[str, Any]], ontology: dict[str, Any]) -> list[GeminiTask]:
    return [
        GeminiTask(
            key=str((record.get("metadata") or {}).get("task_key") or record.get("id") or index),
            record=record,
            prompt=build_prompt(record, ontology),
        )
        for index, record in enumerate(records)
    ]


def verified_record(
    *,
    source_record: dict[str, Any],
    result: dict[str, Any],
    ontology: dict[str, Any],
    args: argparse.Namespace,
    settings: dict[str, Any],
    api_mode: str,
    run_id: str,
) -> dict[str, Any]:
    payload = result.get("payload") or {}
    decision = str(payload.get("decision") or "reject").lower()
    metadata = dict(source_record.get("metadata") or {})
    previous_usage = metadata.get("usage") or {}
    verifier_usage = normalize_usage(result.get("usage"))
    metadata.update(
        {
            "verifier_model": args.verifier_model,
            "api_mode": api_mode,
            "run_id": metadata.get("run_id") or run_id,
            "prompt_settings": {
                **(metadata.get("prompt_settings") or {}),
                "verifier": settings,
            },
            "usage": {
                **previous_usage,
                "verifier": verifier_usage,
                "total": add_usage(previous_usage.get("extractor"), verifier_usage),
            },
            "task_key": metadata.get("task_key") or source_record.get("id"),
            "verifier_decision": decision,
            "batch_job_name": result.get("batch_job_name") or metadata.get("batch_job_name"),
            "raw_response_path": result.get("raw_response_path") or metadata.get("raw_response_path"),
        }
    )
    if decision == "accept":
        return {**source_record, "metadata": metadata}
    if decision == "fix":
        fixed_source = {
            **source_record,
            "article_text": source_record.get("text"),
            "text": source_record.get("text"),
            "run_id": metadata["run_id"],
        }
        fixed = clean_payload(
            payload=payload,
            source_record=fixed_source,
            ontology=ontology,
            model=str(source_record.get("metadata", {}).get("annotation_model") or args.model),
            api_mode=api_mode,
            settings=(metadata.get("prompt_settings") or {}).get("extractor") or {},
            usage=previous_usage.get("extractor") or {},
            batch_job_name=metadata.get("batch_job_name"),
            raw_response_path=metadata.get("raw_response_path"),
        )
        fixed["metadata"] = metadata
        return fixed
    return {
        **source_record,
        "events": [],
        "negatives": {"has_target_event": False, "negative_reason": payload.get("reason") or "verifier_rejected"},
        "metadata": metadata,
    }


def output_row_for_result(
    *,
    task: GeminiTask,
    result: dict[str, Any],
    ontology: dict[str, Any],
    args: argparse.Namespace,
    settings: dict[str, Any],
    api_mode: str,
    run_id: str,
) -> dict[str, Any]:
    def error_row(error: str) -> dict[str, Any]:
        metadata = dict(task.record.get("metadata") or {})
        previous_usage = metadata.get("usage") or {}
        verifier_usage = normalize_usage(result.get("usage"))
        metadata.update(
            {
                "verifier_model": args.verifier_model,
                "api_mode": api_mode,
                "run_id": metadata.get("run_id") or run_id,
                "prompt_settings": {
                    **(metadata.get("prompt_settings") or {}),
                    "verifier": settings,
                },
                "usage": {
                    **previous_usage,
                    "verifier": verifier_usage,
                    "total": add_usage(previous_usage.get("extractor"), verifier_usage),
                },
                "task_key": metadata.get("task_key") or task.key,
                "verifier_decision": "error",
                "batch_job_name": result.get("batch_job_name") or metadata.get("batch_job_name"),
                "raw_response_path": result.get("raw_response_path") or metadata.get("raw_response_path"),
            }
        )
        return {**task.record, "status": "error", "error": error, "metadata": metadata}

    if result.get("error"):
        return error_row(str(result["error"]))
    try:
        return verified_record(
            source_record=task.record,
            result=result,
            ontology=ontology,
            args=args,
            settings=settings,
            api_mode=api_mode,
            run_id=run_id,
        )
    except Exception as exc:
        return error_row(f"failed to apply verifier response: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generation_v2 annotations with Gemini.")
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
    print(f"Loading raw annotations from {input_path}")
    print(f"Loading ontology from {args.ontology}")
    ontology = load_json_tolerant(resolve_path(args.ontology))
    print(f"Checking existing output at {output_path} for resumable tasks")
    if args.overwrite:
        completed = set()
        write_jsonl(output_path, [], overwrite=True)
    else:
        completed = completed_task_keys(output_path, retry_failed=args.retry_failed)
        pruned = prune_retryable_error_rows(output_path, retry_failed=args.retry_failed)
        if pruned:
            print(f"Pruned {pruned} failed output rows before retrying")
    source_records = [
        record
        for record in iter_jsonl(input_path)
        if str((record.get("metadata") or {}).get("task_key") or record.get("id")) not in completed
    ]
    source_error_tasks = build_tasks(
        [record for record in source_records if record.get("status") == "error"],
        ontology,
    )
    verifiable_records = [record for record in source_records if record.get("status") != "error"]
    print(
        f"Preparing {len(verifiable_records)} verification tasks; "
        f"propagating {len(source_error_tasks)} source errors; "
        f"skipping {len(completed)} completed tasks"
    )
    tasks = build_tasks(verifiable_records, ontology)
    settings = prompt_settings(args, component="verifier")
    run_id = uuid.uuid4().hex[:12]
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
            args=args,
            settings=settings,
            api_mode=api_mode,
            run_id=run_id,
        )
        append_jsonl_row(output_path, row)
        saved_count += 1

    for task in source_error_tasks:
        row = output_row_for_result(
            task=task,
            result={
                "key": task.key,
                "error": f"source annotation error: {task.record.get('error', 'unknown error')}",
                "usage": normalize_usage({}),
            },
            ontology=ontology,
            args=args,
            settings=settings,
            api_mode="skipped",
            run_id=run_id,
        )
        append_jsonl_row(output_path, row)
        saved_count += 1

    if args.batch_api:
        print(
            "Submitting verification tasks with Gemini Batch API "
            f"using batch_size={args.batch_size}, workers={args.workers}; run artifacts -> {run_dir}"
        )
        results = run_async(
            run_batch_tasks(
                tasks=tasks,
                model=args.verifier_model,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                response_schema=VerifierDecision,
                run_dir=run_dir,
                batch_size=args.batch_size,
                batch_display_name=args.batch_display_name,
                poll_interval_seconds=args.batch_poll_interval_seconds,
                workers=args.workers,
                desc="Verification",
                on_result=lambda result: persist_result(result, api_mode="batch"),
            )
        )
        api_mode = "batch"
    else:
        print(f"Running interactive Gemini verification with workers={args.workers}")
        results = run_async(
            run_interactive_tasks(
                tasks=tasks,
                model=args.verifier_model,
                system_prompt=SYSTEM_PROMPT,
                settings=settings,
                response_schema=VerifierDecision,
                workers=args.workers,
                desc="Verifying",
                on_result=lambda result: persist_result(result, api_mode="interactive"),
            )
        )
        api_mode = "interactive"

    expected_count = len(results) + len(source_error_tasks)
    if saved_count != expected_count:
        raise RuntimeError(f"saved {saved_count} rows but expected {expected_count} {api_mode} results")
    print(f"Saved {saved_count} verified annotation rows to {output_path}")
    print("Verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
