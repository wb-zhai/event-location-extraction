import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from google.genai import _transformers as genai_transformers
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import GeminiLLMClient

LOGGER = logging.getLogger("fix_events")

DEFAULT_MODEL = "gemini-3.1-pro-preview"

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "fixer" / "system_prompt.txt"
USER_PROMPT_PATH = Path(__file__).parent / "prompts" / "fixer" / "user_prompt.txt"
ONTOLOGY_PATH = REPO_ROOT / "ontologies" / "zhai" / "risk.label.description.training.json"


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class FixedEvent(BaseModel):
    event_type: str
    event_location_text: str
    event_location: str
    event_time_text: str
    event_time: str
    time_status: Literal["past", "ongoing", "forecast", "not_stated"]
    affected_entity: str
    affected_group: str
    severity: Literal["low", "medium", "high", "extreme", "not_stated"]
    modality: Literal["asserted", "reported", "projected", "hedged"]
    grounding_quote: str


class EventDecision(BaseModel):
    decision: Literal["fixed", "dropped"]
    fixed_event: FixedEvent | None = None
    reason: str


class ArticleFixResult(BaseModel):
    decisions: list[EventDecision]


# ---------------------------------------------------------------------------
# Reused helpers (mirrors generate.py style)
# ---------------------------------------------------------------------------


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def load_ontology_labels() -> set[str]:
    data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    return set(data["events"].keys())


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def empty_stats() -> dict[str, int]:
    return {
        "total": 0,
        "fixed": 0,
        "fixed_revalidated_ok": 0,
        "fixed_revalidated_fail": 0,
        "dropped": 0,
        "error": 0,
    }


def tally(stats: dict[str, int], record: dict[str, Any]) -> None:
    stats["total"] += 1
    if record.get("status") == "error":
        stats["error"] += 1
        return
    decision = record.get("decision")
    if decision == "fixed":
        stats["fixed"] += 1
        if record.get("revalidation", {}).get("valid"):
            stats["fixed_revalidated_ok"] += 1
        else:
            stats["fixed_revalidated_fail"] += 1
    elif decision == "dropped":
        stats["dropped"] += 1


def print_summary(stats: dict[str, int], skipped: int, output: Path) -> None:
    t = stats["total"]
    print("\n=== Fix summary ===")
    print(f"Skipped (already done) : {skipped}")
    print(f"Processed this run     : {t}")
    if t == 0:
        return
    print(f"  Fixed                : {stats['fixed']}  ({100 * stats['fixed'] / t:.1f}%)")
    print(f"    revalidation ok    : {stats['fixed_revalidated_ok']}")
    print(f"    revalidation fail  : {stats['fixed_revalidated_fail']}")
    print(f"  Dropped              : {stats['dropped']}  ({100 * stats['dropped'] / t:.1f}%)")
    print(f"  API errors           : {stats['error']}")
    print(f"\nOutput: {output}")


def completed_rows(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    for result in iter_jsonl(path):
        row = result.get("row")
        if row is not None and result.get("status") != "error":
            seen.add(int(row))
    return seen


def render_user_prompt(template: str, publish_date: str, source_text: str, events: list[dict[str, Any]]) -> str:
    article = source_text
    return (
        template
        .replace("{{PUBLISH_DATE}}", publish_date or "not_stated")
        .replace("{{ARTICLE_TEXT}}", article)
        .replace("{{INVALID_EVENTS}}", json.dumps(events, ensure_ascii=False, indent=2))
    )


def check_event(
    event: dict[str, Any],
    source_text: str,
    ontology: set[str],
    seen_keys: set[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    event_type = event.get("event_type", "")
    if event_type not in ontology:
        errors.append(f"event_type {event_type!r} not in ontology")

    gq = event.get("grounding_quote", "")
    if not gq:
        errors.append("grounding_quote is empty")
    elif gq not in source_text:
        errors.append("grounding_quote not found in source.text")

    loc_text = event.get("event_location_text", "")
    if loc_text and loc_text != "not_stated" and loc_text not in source_text:
        errors.append("event_location_text not found in source.text")

    time_text = event.get("event_time_text", "")
    if time_text and time_text != "not_stated" and time_text not in source_text:
        errors.append("event_time_text not found in source.text")

    valid_time_status = {"past", "ongoing", "forecast", "not_stated"}
    valid_severity = {"low", "medium", "high", "extreme", "not_stated"}
    valid_modality = {"asserted", "reported", "projected", "hedged"}
    if event.get("time_status") not in valid_time_status:
        errors.append(f"invalid time_status: {event.get('time_status')!r}")
    if event.get("severity") not in valid_severity:
        errors.append(f"invalid severity: {event.get('severity')!r}")
    if event.get("modality") not in valid_modality:
        errors.append(f"invalid modality: {event.get('modality')!r}")

    key = (event_type, gq)
    if key in seen_keys:
        errors.append("duplicate event (same event_type + grounding_quote)")
    else:
        seen_keys.add(key)

    return errors


# ---------------------------------------------------------------------------
# Task dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EventRow:
    row: int
    doc_id: str
    source_text: str
    publish_date: str
    event: dict[str, Any]
    errors: list[str]


@dataclass
class FixTask:
    rows: list[EventRow]
    prompt: str


# ---------------------------------------------------------------------------
# Building tasks
# ---------------------------------------------------------------------------


def build_tasks(rows: list[EventRow], template: str, *, mode: str) -> list[FixTask]:
    if mode == "per-event":
        return [
            FixTask(
                rows=[row],
                prompt=render_user_prompt(
                    template,
                    row.publish_date,
                    row.source_text,
                    [{"event": row.event, "errors": row.errors}],
                ),
            )
            for row in rows
        ]
    # per-article: group by doc_id (preserve order)
    groups: dict[str, list[EventRow]] = {}
    for row in rows:
        groups.setdefault(row.doc_id, []).append(row)
    tasks = []
    for doc_rows in groups.values():
        events_payload = [{"event": r.event, "errors": r.errors} for r in doc_rows]
        tasks.append(
            FixTask(
                rows=doc_rows,
                prompt=render_user_prompt(
                    template,
                    doc_rows[0].publish_date,
                    doc_rows[0].source_text,
                    events_payload,
                ),
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Result record construction
# ---------------------------------------------------------------------------


def make_output_record(
    row: EventRow,
    decision: EventDecision,
    *,
    model: str,
    ontology: set[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    revalidation: dict[str, Any] = {"valid": False, "errors": []}
    if decision.decision == "fixed" and decision.fixed_event is not None:
        errs = check_event(
            decision.fixed_event.model_dump(),
            row.source_text,
            ontology,
            set(),
        )
        revalidation = {"valid": len(errs) == 0, "errors": errs}

    return {
        "row": row.row,
        "doc_id": row.doc_id,
        "publish_date": row.publish_date,
        "original_event": row.event,
        "errors": row.errors,
        "decision": decision.decision,
        "fixed_event": decision.fixed_event.model_dump() if decision.fixed_event else None,
        "revalidation": revalidation,
        "reason": decision.reason,
        "llm": {"model": model, "metadata": metadata or {}},
        "status": "ok",
    }


def make_error_record(row: EventRow, error: str, *, model: str) -> dict[str, Any]:
    return {
        "row": row.row,
        "doc_id": row.doc_id,
        "publish_date": row.publish_date,
        "original_event": row.event,
        "errors": row.errors,
        "status": "error",
        "error": error,
        "llm": {"model": model},
    }


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


async def call_llm(
    client: GeminiLLMClient,
    prompt: str,
    response_format: type,
    *,
    include_thoughts: bool = False,
) -> tuple[Any, dict[str, Any]]:
    async for response in client.generate(
        prompt,
        response_format=response_format,
        add_cot_field=False,
        include_thoughts=include_thoughts,
    ):
        parsed = (
            response.parsed.model_dump()
            if response.parsed
            else json.loads(response.text)
        )
        return parsed, response.metadata
    raise RuntimeError("Gemini returned no response.")


async def process_task(
    client: GeminiLLMClient,
    task: FixTask,
    *,
    model: str,
    ontology: set[str],
    mode: str,
) -> list[dict[str, Any]]:
    response_format = EventDecision if mode == "per-event" else ArticleFixResult
    parsed, metadata = await call_llm(client, task.prompt, response_format, include_thoughts=True)

    if mode == "per-event":
        row = task.rows[0]
        try:
            decision = EventDecision.model_validate(parsed)
        except Exception as exc:
            return [make_error_record(row, str(exc), model=model)]
        return [make_output_record(row, decision, model=model, ontology=ontology, metadata=metadata)]

    # per-article
    try:
        result = ArticleFixResult.model_validate(parsed)
    except Exception as exc:
        return [make_error_record(row, str(exc), model=model) for row in task.rows]

    decisions = result.decisions
    if len(decisions) != len(task.rows):
        LOGGER.warning(
            "doc_id=%s: expected %d decisions, got %d",
            task.rows[0].doc_id,
            len(task.rows),
            len(decisions),
        )

    records = []
    for i, row in enumerate(task.rows):
        if i < len(decisions):
            records.append(
                make_output_record(row, decisions[i], model=model, ontology=ontology, metadata=metadata)
            )
        else:
            records.append(make_error_record(row, "Decision missing from model response.", model=model))
    return records


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------


async def run_sync(
    args: argparse.Namespace,
    tasks: list[FixTask],
    ontology: set[str],
    system_prompt: str,
) -> dict[str, int]:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    semaphore = asyncio.Semaphore(args.workers)

    async def run_task(task: FixTask) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                return await process_task(
                    client, task, model=args.model, ontology=ontology, mode=args.mode
                )
            except Exception as exc:
                LOGGER.exception("Failed task rows=%s", [r.row for r in task.rows])
                return [make_error_record(row, str(exc), model=args.model) for row in task.rows]

    stats = empty_stats()
    coroutines = [run_task(task) for task in tasks]
    with args.output.open("a", encoding="utf-8") as handle:
        for future in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="fixing",
        ):
            for record in await future:
                append_jsonl(handle, record)
                tally(stats, record)
    return stats


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


def thinking_budget(reasoning_effort: str | None) -> int:
    if reasoning_effort is None:
        return 0
    try:
        return int(reasoning_effort)
    except ValueError:
        return {"disable": 0, "low": 1024, "medium": 2048, "high": 4096}.get(
            reasoning_effort, 0
        )


def thinking_level(reasoning_effort: str | None) -> str:
    if reasoning_effort in {"minimal", "low", "medium", "high"}:
        return reasoning_effort
    budget = thinking_budget(reasoning_effort)
    if budget <= 1024:
        return "low"
    if budget <= 2048:
        return "medium"
    return "high"


def batch_request_config(args: argparse.Namespace, response_format: type) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": genai_transformers.t_schema(None, response_format).model_dump(
            exclude_none=True
        ),
        "maxOutputTokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if "2.5" in args.model and "gemini" in args.model:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": thinking_budget(args.reasoning_effort),
            "includeThoughts": True,
        }
    elif "3" in args.model and "gemini" in args.model:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level(args.reasoning_effort),
            "includeThoughts": True,
        }
    return {"generationConfig": generation_config}


def batch_request_line(key: str, prompt: str, request_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            **request_config,
        },
    }


def batch_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("text"), str):
        return response["text"]
    fallback_text: str | None = None
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if not isinstance(text, str):
                continue
            if part.get("thought"):
                fallback_text = fallback_text or text
                continue
            return text
    if fallback_text is not None:
        return fallback_text
    raise ValueError("Batch response does not contain generated text.")


def batch_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage_metadata") or response.get("usageMetadata") or {}
    metadata: dict[str, Any] = {}
    if isinstance(usage, dict):
        metadata.update(
            {
                "prompt_tokens": int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0),
                "completion_tokens": int(usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or 0),
                "cached_tokens": int(usage.get("cached_content_token_count") or usage.get("cachedContentTokenCount") or 0),
                "thoughts_token_count": int(usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount") or 0),
            }
        )
    thought_summaries: list[str] = []
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("thought") and isinstance(part.get("text"), str):
                thought_summaries.append(part["text"])
    metadata["thought_summaries"] = thought_summaries
    return metadata


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def poll_batch_job(client: GeminiLLMClient, name: str, interval: int) -> Any:
    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    job = await asyncio.to_thread(client.client.batches.get, name=name)
    while job.state.name not in done:
        LOGGER.info("batch=%s state=%s", name, job.state.name)
        await asyncio.sleep(interval)
        job = await asyncio.to_thread(client.client.batches.get, name=name)
    return job


async def execute_batch_chunk(
    *,
    client: GeminiLLMClient,
    args: argparse.Namespace,
    request_config: dict[str, Any],
    task_chunk: list[FixTask],
    chunk_index: int,
    ontology: set[str],
) -> list[dict[str, Any]]:
    request_path = args.output.with_suffix(f".batch.part-{chunk_index:04d}.requests.jsonl")
    result_path = args.output.with_suffix(f".batch.part-{chunk_index:04d}.results.jsonl")

    task_keys = [str(chunk_index * 100000 + i) for i in range(len(task_chunk))]
    request_path.write_text(
        "".join(
            json.dumps(batch_request_line(key, task.prompt, request_config), ensure_ascii=False) + "\n"
            for key, task in zip(task_keys, task_chunk)
        ),
        encoding="utf-8",
    )

    uploaded = await asyncio.to_thread(
        client.client.files.upload,
        file=str(request_path),
        config=genai_types.UploadFileConfig(
            display_name=request_path.stem,
            mime_type="jsonl",
        ),
    )
    job = await asyncio.to_thread(
        client.client.batches.create,
        model=args.model,
        src=uploaded.name,
        config={"display_name": f"{args.output.stem}-fix-part-{chunk_index:04d}"},
    )
    job = await poll_batch_job(client, job.name, args.batch_poll_interval_seconds)

    def all_error(msg: str) -> list[dict[str, Any]]:
        return [make_error_record(row, msg, model=args.model) for task in task_chunk for row in task.rows]

    if job.state.name != "JOB_STATE_SUCCEEDED":
        return all_error(f"Batch job {job.name} ended with {job.state.name}: {job.error}")
    if not job.dest or not job.dest.file_name:
        return all_error(f"Batch job {job.name} succeeded without a result file.")

    data = await asyncio.to_thread(client.client.files.download, file=job.dest.file_name)
    result_path.write_bytes(data)

    task_by_key = {key: task for key, task in zip(task_keys, task_chunk)}
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for line in iter_jsonl(result_path):
        key = str(line.get("key") or (line.get("metadata") or {}).get("key") or "")
        task = task_by_key.get(key)
        if task is None:
            LOGGER.warning("Skipping unexpected batch result key=%s", key)
            continue
        seen_keys.add(key)
        try:
            response = line.get("response")
            if not isinstance(response, dict):
                raise ValueError(str(line.get("error") or "Missing batch response."))
            parsed = json.loads(batch_response_text(response))
            meta = {**batch_response_metadata(response), "batch_job_name": job.name}
            if args.mode == "per-event":
                decision = EventDecision.model_validate(parsed)
                records.append(
                    make_output_record(task.rows[0], decision, model=args.model, ontology=ontology, metadata=meta)
                )
            else:
                result = ArticleFixResult.model_validate(parsed)
                decisions = result.decisions
                for i, row in enumerate(task.rows):
                    if i < len(decisions):
                        records.append(
                            make_output_record(row, decisions[i], model=args.model, ontology=ontology, metadata=meta)
                        )
                    else:
                        records.append(make_error_record(row, "Decision missing from model response.", model=args.model))
        except Exception as exc:
            records.extend(make_error_record(row, str(exc), model=args.model) for row in task.rows)

    for key, task in task_by_key.items():
        if key not in seen_keys:
            records.extend(
                make_error_record(row, "Batch result missing.", model=args.model) for row in task.rows
            )

    request_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    return records


async def run_batch(
    args: argparse.Namespace,
    tasks: list[FixTask],
    ontology: set[str],
    system_prompt: str,
) -> dict[str, int]:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    response_format = EventDecision if args.mode == "per-event" else ArticleFixResult
    request_config = batch_request_config(args, response_format)
    chunks = list(enumerate(chunked(tasks, args.batch_size), start=1))
    semaphore = asyncio.Semaphore(args.workers)

    async def run_chunk(chunk_index: int, task_chunk: list[FixTask]) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                return await execute_batch_chunk(
                    client=client,
                    args=args,
                    request_config=request_config,
                    task_chunk=task_chunk,
                    chunk_index=chunk_index,
                    ontology=ontology,
                )
            except Exception as exc:
                LOGGER.exception("Failed batch chunk=%s", chunk_index)
                return [
                    make_error_record(row, str(exc), model=args.model)
                    for task in task_chunk
                    for row in task.rows
                ]

    stats = empty_stats()
    coroutines = [run_chunk(ci, tc) for ci, tc in chunks]
    with args.output.open("a", encoding="utf-8") as handle:
        for future in tqdm(asyncio.as_completed(coroutines), total=len(coroutines), desc="batch chunks"):
            for record in await future:
                append_jsonl(handle, record)
                tally(stats, record)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix invalid v3 annotations using Gemini.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["per-article", "per-event"], default="per-article")
    parser.add_argument("--prompt", type=Path, default=USER_PROMPT_PATH)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-api", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-poll-interval-seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    return args


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    load_env_file(args.env_file)

    ontology = load_ontology_labels()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    template = args.prompt.read_text(encoding="utf-8")

    raw_rows = iter_jsonl(args.input)
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]

    already_done = completed_rows(args.output)
    if already_done:
        LOGGER.info("Skipping %d rows already in %s", len(already_done), args.output)

    rows = [
        EventRow(
            row=i,
            doc_id=str(rec.get("doc_id", i)),
            source_text=str(rec.get("source_text", "")),
            publish_date=str(rec.get("publish_date", "")),
            event=rec.get("event", {}),
            errors=rec.get("errors", []),
        )
        for i, rec in enumerate(raw_rows)
        if i not in already_done
    ]

    tasks = build_tasks(rows, template, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_api:
        stats = await run_batch(args, tasks, ontology, system_prompt)
    else:
        stats = await run_sync(args, tasks, ontology, system_prompt)

    print_summary(stats, skipped=len(already_done), output=args.output)


if __name__ == "__main__":
    asyncio.run(main())
