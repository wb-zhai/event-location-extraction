import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
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

LOGGER = logging.getLogger("generation_v3")

DEFAULT_PROMPT = Path(__file__).with_name("annotation_prompt.txt")
DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
USER_PROMPT_PATH = Path(__file__).with_name("user_prompt.txt")

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
USER_PROMPT = USER_PROMPT_PATH.read_text(encoding="utf-8").strip()


class AnnotationEvent(BaseModel):
    event_type: str
    event_location: str
    event_time: str
    time_status: Literal["past", "ongoing", "forecast", "not_stated"]
    affected_entity: str
    affected_group: str
    severity: Literal["low", "medium", "high", "extreme", "not_stated"]
    modality: Literal["asserted", "reported", "projected", "hedged"]
    grounding_quote: str


class Annotation(BaseModel):
    document_relevance: Literal["relevant", "not_relevant"]
    events: list[AnnotationEvent] = Field(default_factory=list)


@dataclass
class BatchTask:
    key: str
    record: dict[str, Any]
    prompt: str


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
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
    return records


def load_input_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Expected JSON object or list at {path}.")
    return iter_jsonl(path)


def record_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id") or index)


def source_title(record: dict[str, Any]) -> str:
    return str(record.get("title") or "")


def source_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or "")


def source_publish_date(record: dict[str, Any]) -> str:
    return str(record.get("publish_date") or "")


def source_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "title": source_title(record),
        "text": source_text(record),
        "source_url": str(record.get("source_url") or ""),
        "publish_date": source_publish_date(record),
    }


def render_prompt(template: str, record: dict[str, Any]) -> str:
    article = (
        f"Title: {source_title(record)}\n"
        f"publish_date: {source_publish_date(record) or 'not_stated'}\n\n"
        f"{source_text(record)}"
    )
    return template.replace("{{ARTICLE_TEXT}}", article)


def output_record(
    record: dict[str, Any],
    annotation: dict[str, Any],
    *,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "status": "ok",
        "source": source_record(record),
        "annotation": annotation,
        "llm": {"model": model, "metadata": metadata or {}},
    }


def error_record(record: dict[str, Any], error: str, *, model: str) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "status": "error",
        "source": source_record(record),
        "error": error,
        "llm": {"model": model},
    }


def completed_output_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for result in iter_jsonl(path):
        if result.get("status") != "ok" or "llm" not in result:
            continue
        result_id = result.get("id")
        if result_id is not None:
            completed.add(str(result_id))
    return completed


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


async def generate_one(
    client: GeminiLLMClient,
    record: dict[str, Any],
    prompt: str,
    *,
    include_thoughts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async for response in client.generate(
        prompt,
        response_format=Annotation,
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


async def run_sync(
    args: argparse.Namespace, records: list[dict[str, Any]], template: str
) -> None:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_output_ids(args.output)
    pending = [
        (index, record)
        for index, record in enumerate(records)
        if record_id(record, index) not in completed
    ]
    if completed:
        LOGGER.info(
            "Skipping %s records already present in %s", len(completed), args.output
        )

    semaphore = asyncio.Semaphore(args.workers)

    async def process_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                annotation, metadata = await generate_one(
                    client,
                    record,
                    render_prompt(template, record),
                    include_thoughts=args.include_thoughts,
                )
                return output_record(
                    record, annotation, model=args.model, metadata=metadata
                )
            except Exception as exc:
                LOGGER.exception("Failed id=%s", record_id(record, index))
                return error_record(record, str(exc), model=args.model)

    coroutines = [process_record(index, record) for index, record in pending]
    with args.output.open("a", encoding="utf-8") as handle:
        for future in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="annotating",
        ):
            append_jsonl(handle, await future)


async def run_interactive(args: argparse.Namespace, template: str) -> None:
    if sys.stdin.isatty():
        print("Paste article text, then press Ctrl-D:", file=sys.stderr)
    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("No text provided on stdin.")

    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    record = {"id": "interactive", "source": {"title": args.title, "text": text}}
    annotation, metadata = await generate_one(
        client,
        record,
        render_prompt(template, record),
        include_thoughts=True,
    )
    traces = {
        "thoughts_token_count": metadata.get("thoughts_token_count", 0),
        "thought_summaries": metadata.get("thought_summaries", []),
    }
    print("THINKING_TRACES")
    print(json.dumps(traces, ensure_ascii=False, indent=2))
    print("\nOUTPUT")
    print(json.dumps(annotation, ensure_ascii=False, indent=2))


def thinking_budget(reasoning_effort: str | None) -> int:
    if reasoning_effort is None:
        return 0
    try:
        return int(reasoning_effort)
    except ValueError:
        return {"disable": 0, "low": 1024, "medium": 2048, "high": 4096}[
            reasoning_effort
        ]


def thinking_level(reasoning_effort: str | None) -> str:
    if reasoning_effort in {"minimal", "low", "medium", "high"}:
        return reasoning_effort
    budget = thinking_budget(reasoning_effort)
    if budget <= 1024:
        return "low"
    if budget <= 2048:
        return "medium"
    return "high"


def batch_request_config(args: argparse.Namespace) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": genai_transformers.t_schema(None, Annotation).model_dump(
            exclude_none=True
        ),
        "maxOutputTokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if "2.5" in args.model and "gemini" in args.model:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": thinking_budget(args.reasoning_effort)
        }
    elif "3" in args.model and "gemini" in args.model:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level(args.reasoning_effort)
        }
    if args.include_thoughts:
        generation_config.setdefault("thinkingConfig", {})["includeThoughts"] = True
    return {"generationConfig": generation_config}


def build_batch_tasks(records: list[dict[str, Any]], template: str) -> list[BatchTask]:
    return [
        BatchTask(
            key=record_id(record, index),
            record=record,
            prompt=render_prompt(template, record),
        )
        for index, record in enumerate(records)
    ]


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def batch_request_line(
    task: BatchTask, request_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "key": task.key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": task.prompt}]}],
            **request_config,
        },
    }


async def poll_batch_job(client: GeminiLLMClient, name: str, interval: int) -> Any:
    done = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    job = await asyncio.to_thread(client.client.batches.get, name=name)
    while job.state.name not in done:
        LOGGER.info("batch=%s state=%s", name, job.state.name)
        await asyncio.sleep(interval)
        job = await asyncio.to_thread(client.client.batches.get, name=name)
    return job


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
                "prompt_tokens": int(
                    usage.get("prompt_token_count")
                    or usage.get("promptTokenCount")
                    or 0
                ),
                "completion_tokens": int(
                    usage.get("candidates_token_count")
                    or usage.get("candidatesTokenCount")
                    or 0
                ),
                "cached_tokens": int(
                    usage.get("cached_content_token_count")
                    or usage.get("cachedContentTokenCount")
                    or 0
                ),
                "thoughts_token_count": int(
                    usage.get("thoughts_token_count")
                    or usage.get("thoughtsTokenCount")
                    or 0
                ),
            }
        )

    summaries: list[str] = []
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if text and part.get("thought"):
                summaries.append(str(text))
    metadata["thought_summaries"] = summaries
    return metadata


def batch_result_key(line: dict[str, Any]) -> str:
    if line.get("key") is not None:
        return str(line["key"])
    metadata = line.get("metadata") or {}
    return str(metadata.get("key") or "")


async def execute_batch_chunk(
    *,
    client: GeminiLLMClient,
    args: argparse.Namespace,
    request_config: dict[str, Any],
    task_chunk: list[BatchTask],
    chunk_index: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    request_path = args.output.with_suffix(
        f".batch.part-{chunk_index:04d}.requests.jsonl"
    )
    result_path = args.output.with_suffix(
        f".batch.part-{chunk_index:04d}.results.jsonl"
    )
    request_path.write_text(
        "".join(
            json.dumps(batch_request_line(task, request_config), ensure_ascii=False)
            + "\n"
            for task in task_chunk
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
        config={"display_name": f"{args.output.stem}-part-{chunk_index:04d}"},
    )
    job = await poll_batch_job(client, job.name, args.batch_poll_interval_seconds)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        error = f"Batch job {job.name} ended with {job.state.name}: {job.error}"
        return [
            error_record(task.record, error, model=args.model) for task in task_chunk
        ], {task.key for task in task_chunk}
    if not job.dest or not job.dest.file_name:
        error = f"Batch job {job.name} succeeded without a result file."
        return [
            error_record(task.record, error, model=args.model) for task in task_chunk
        ], {task.key for task in task_chunk}

    data = await asyncio.to_thread(
        client.client.files.download, file=job.dest.file_name
    )
    result_path.write_bytes(data)

    task_by_key = {task.key: task for task in task_chunk}
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for line in iter_jsonl(result_path):
        key = batch_result_key(line)
        task = task_by_key.get(key)
        if task is None:
            LOGGER.warning("Skipping unexpected batch result key=%s", key)
            continue
        try:
            response = line.get("response")
            if not isinstance(response, dict):
                raise ValueError(str(line.get("error") or "Missing batch response."))
            annotation = json.loads(batch_response_text(response))
            annotation = Annotation.model_validate(annotation).model_dump()
            records.append(
                output_record(
                    task.record,
                    annotation,
                    model=args.model,
                    metadata={
                        **batch_response_metadata(response),
                        "batch_job_name": job.name,
                    },
                )
            )
        except Exception as exc:
            records.append(error_record(task.record, str(exc), model=args.model))
        seen.add(key)

    for task in task_chunk:
        if task.key not in seen:
            records.append(
                error_record(task.record, "Batch result missing.", model=args.model)
            )
    return records, seen


async def run_batch(
    args: argparse.Namespace, records: list[dict[str, Any]], template: str
) -> None:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    request_config = batch_request_config(args)
    completed = completed_output_ids(args.output)
    tasks = [
        task
        for task in build_batch_tasks(records, template)
        if task.key not in completed
    ]
    if completed:
        LOGGER.info(
            "Skipping %s records already present in %s", len(completed), args.output
        )
    chunks = list(enumerate(chunked(tasks, args.batch_size), start=1))
    semaphore = asyncio.Semaphore(args.workers)

    async def run_chunk(
        chunk_index: int, task_chunk: list[BatchTask]
    ) -> tuple[list[dict[str, Any]], set[str]]:
        async with semaphore:
            try:
                return await execute_batch_chunk(
                    client=client,
                    args=args,
                    request_config=request_config,
                    task_chunk=task_chunk,
                    chunk_index=chunk_index,
                )
            except Exception as exc:
                LOGGER.exception("Failed batch chunk=%s", chunk_index)
                return [
                    error_record(task.record, str(exc), model=args.model)
                    for task in task_chunk
                ], {task.key for task in task_chunk}

    coroutines = [
        run_chunk(chunk_index, task_chunk) for chunk_index, task_chunk in chunks
    ]
    with args.output.open("a", encoding="utf-8") as handle:
        for future in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="batch chunks",
        ):
            chunk_records, _seen = await future
            for result in chunk_records:
                append_jsonl(handle, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=USER_PROMPT_PATH)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--include-thoughts", action="store_true")
    parser.add_argument("--batch-api", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-poll-interval-seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    if not args.interactive:
        if args.input is None:
            raise ValueError("Input path is required when not in interactive mode.")

    if args.output is None:
        raise ValueError("Output path is required.")
    if args.input is not None and args.input.resolve() == args.output.resolve():
        raise ValueError("--output must be different from --input.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_poll_interval_seconds < 1:
        raise ValueError("--batch-poll-interval-seconds must be >= 1")
    return args


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    load_env_file(args.env_file)
    template = args.prompt.read_text(encoding="utf-8")
    if args.interactive:
        await run_interactive(args, template)
        return

    records = load_input_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if args.batch_api:
        await run_batch(args, records, template)
    else:
        await run_sync(args, records, template)


if __name__ == "__main__":
    asyncio.run(main())
