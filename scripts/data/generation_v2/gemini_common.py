"""Shared Gemini helpers for generation_v2 annotation scripts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from google.genai import types as genai_types
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from scripts.data.generation_v2.io_utils import dump_json, resolve_path
from scripts.data.generation_v2.schema import EventArgument, EventMention, LocationMention
from scripts.data.generation_v2.usage import normalize_usage
from src.llms.llm_client import GeminiLLMClient


DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_ROUTER_MODEL = "gemini-2.5-flash"


class GeneratedPayload(BaseModel):
    events: list[EventMention] = Field(default_factory=list)
    locations: list[LocationMention] = Field(default_factory=list)
    has_target_event: bool = False
    negative_reason: str | None = None


class VerifierDecision(BaseModel):
    decision: str = Field(description="'accept', 'reject', or 'fix'")
    reason: str = ""
    events: list[EventMention] = Field(default_factory=list)
    locations: list[LocationMention] = Field(default_factory=list)
    has_target_event: bool = False
    negative_reason: str | None = None


@dataclass(frozen=True)
class GeminiTask:
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


def add_gemini_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--verifier-model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--router-reasoning-effort", default="disable")
    parser.add_argument("--verifier-reasoning-effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-api", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--batch-display-name", default=None)
    parser.add_argument("--batch-poll-interval-seconds", type=int, default=30)
    parser.add_argument("--run-dir", type=Path, default=Path("output/generation_v2/runs"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--save-thought-summaries", action="store_true")


def prompt_settings(args: argparse.Namespace, *, component: str) -> dict[str, Any]:
    if component == "verifier":
        return {
            "temperature": args.verifier_temperature,
            "reasoning_effort": args.verifier_reasoning_effort,
            "max_tokens": args.max_tokens,
            "save_thought_summaries": args.save_thought_summaries,
        }
    if component == "router":
        return {
            "temperature": args.router_temperature,
            "reasoning_effort": args.router_reasoning_effort,
            "max_tokens": args.max_tokens,
            "save_thought_summaries": args.save_thought_summaries,
        }
    return {
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "save_thought_summaries": args.save_thought_summaries,
    }


def interactive_override(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": settings["temperature"],
        "max_output_tokens": settings["max_tokens"],
    }


def interactive_response_format(response_schema: type[BaseModel]) -> type[BaseModel]:
    return response_schema


def _thinking_config_for_batch(model: str, reasoning_effort: str | int | None) -> dict[str, Any] | None:
    if reasoning_effort in {None, "", "disable", 0, "0"}:
        return {"thinkingBudget": 0} if "2.5" in model else None
    try:
        budget = int(reasoning_effort)
    except (TypeError, ValueError):
        budget = 1024 if str(reasoning_effort).lower() == "low" else 4096
    if "3" in model and "gemini" in model:
        return {"thinkingLevel": "low" if budget <= 1024 else "high"}
    return {"thinkingBudget": budget}


def batch_generation_config(
    *,
    model: str,
    settings: dict[str, Any],
    response_schema: type[BaseModel],
) -> dict[str, Any]:
    schema = response_schema.model_json_schema()
    config: dict[str, Any] = {
        "temperature": settings["temperature"],
        "maxOutputTokens": settings["max_tokens"],
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }
    thinking = _thinking_config_for_batch(model, settings.get("reasoning_effort"))
    if settings.get("save_thought_summaries"):
        thinking = {**(thinking or {}), "includeThoughts": True}
    if thinking is not None:
        config["thinkingConfig"] = thinking
    return config


def build_batch_request(
    task: GeminiTask,
    *,
    system_prompt: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": task.key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": task.prompt}],
                }
            ],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": generation_config,
        },
    }


def parse_json_response_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _text_from_response_parts(parts: list[dict[str, Any]]) -> str:
    return "".join(str(part.get("text") or "") for part in parts if not part.get("thought"))


def _thought_summaries_from_response_parts(parts: list[dict[str, Any]]) -> list[str]:
    return [str(part.get("text") or "") for part in parts if part.get("thought") and part.get("text")]


def extract_batch_response(
    line: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, dict[str, int], str | None, list[str]]:
    key = str(line.get("key") or line.get("custom_id") or "")
    if line.get("error") or line.get("status"):
        return key, None, normalize_usage({}), json.dumps(line.get("error") or line.get("status")), []

    response = line.get("response") or line
    usage = normalize_usage(response.get("usageMetadata") or response.get("usage_metadata") or {})
    text = ""
    thought_summaries: list[str] = []
    try:
        candidates = response.get("candidates") or []
        parts = candidates[0]["content"]["parts"]
        thought_summaries = _thought_summaries_from_response_parts(parts)
        text = _text_from_response_parts(parts)
        return key, parse_json_response_text(text), usage, None, thought_summaries
    except Exception as exc:
        return key, None, usage, f"failed to parse batch response: {exc}; text={text[:500]}", thought_summaries


async def call_interactive(
    *,
    client: GeminiLLMClient,
    task: GeminiTask,
    system_prompt: str,
    settings: dict[str, Any],
    response_schema: type[BaseModel],
) -> dict[str, Any]:
    attempt_settings = [settings]
    if settings.get("temperature") not in {0, 0.0, "0", "0.0"}:
        attempt_settings.append({**settings, "temperature": 0.0})

    last_parse_error: Exception | None = None
    last_response_text = ""
    for current_settings in attempt_settings:
        responses = []
        async for response in client.generate(
            task.prompt,
            system_prompt=system_prompt,
            override_settings=interactive_override(current_settings),
            response_format=interactive_response_format(response_schema),
            add_cot_field=False,
            reasoning_effort=current_settings.get("reasoning_effort"),
            include_thoughts=bool(current_settings.get("save_thought_summaries")),
        ):
            responses.append(response)
        if not responses:
            raise RuntimeError("Gemini returned no response.")
        response = responses[-1]
        if response.parsed is not None:
            payload = response.parsed.model_dump()
        else:
            try:
                payload = parse_json_response_text(response.text)
            except Exception as exc:
                last_parse_error = exc
                last_response_text = response.text
                continue
        return {
            "key": task.key,
            "payload": payload,
            "usage": normalize_usage(response.metadata),
            "raw_text": response.text,
            "thought_summaries": response.metadata.get("thought_summaries") or [],
        }

    snippet = last_response_text[:500].replace("\n", "\\n")
    raise RuntimeError(
        "failed to parse structured Gemini response "
        f"after {len(attempt_settings)} attempt(s): {last_parse_error}; text={snippet}"
    )


async def run_interactive_tasks(
    *,
    tasks: list[GeminiTask],
    model: str,
    system_prompt: str,
    settings: dict[str, Any],
    response_schema: type[BaseModel],
    workers: int,
    desc: str,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    client = GeminiLLMClient(
        model_name=model,
        system_prompt=None,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
    )
    semaphore = asyncio.Semaphore(max(workers, 1))

    async def guarded(task: GeminiTask) -> dict[str, Any]:
        async with semaphore:
            try:
                result = await call_interactive(
                    client=client,
                    task=task,
                    system_prompt=system_prompt,
                    settings=settings,
                    response_schema=response_schema,
                )
            except Exception as exc:
                result = {"key": task.key, "error": str(exc), "usage": normalize_usage({})}
            if on_result is not None:
                on_result(result)
            return result

    coroutines = [guarded(task) for task in tasks]
    return await tqdm.gather(*coroutines, desc=desc)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


async def run_batch_tasks(
    *,
    tasks: list[GeminiTask],
    model: str,
    system_prompt: str,
    settings: dict[str, Any],
    response_schema: type[BaseModel],
    run_dir: Path,
    batch_size: int,
    batch_display_name: str | None,
    poll_interval_seconds: int,
    workers: int,
    desc: str,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    client = GeminiLLMClient(
        model_name=model,
        system_prompt=None,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        reasoning_effort=settings["reasoning_effort"],
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_config = batch_generation_config(
        model=model,
        settings=settings,
        response_schema=response_schema,
    )
    chunks = chunked(tasks, max(batch_size, 1))
    semaphore = asyncio.Semaphore(max(workers, 1))

    async def run_chunk(chunk_index: int, chunk_tasks: list[GeminiTask]) -> dict[str, Any]:
        async with semaphore:
            request_path = run_dir / f"{desc.lower()}_batch_{chunk_index:04d}.requests.jsonl"
            request_lines = [
                build_batch_request(
                    task,
                    system_prompt=system_prompt,
                    generation_config=generation_config,
                )
                for task in chunk_tasks
            ]
            request_path.write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in request_lines) + "\n",
                encoding="utf-8",
            )
            uploaded_file = await asyncio.to_thread(
                client.client.files.upload,
                file=str(request_path),
                config=genai_types.UploadFileConfig(
                    display_name=request_path.name,
                    mime_type="jsonl",
                ),
            )
            display_name = batch_display_name or f"generation-v2-{desc}-{uuid.uuid4().hex[:8]}"
            batch_job = await asyncio.to_thread(
                client.client.batches.create,
                model=f"models/{model}" if not model.startswith("models/") else model,
                src=uploaded_file.name,
                config={"display_name": display_name},
            )
            dump_json(run_dir / f"{desc.lower()}_batch_{chunk_index:04d}.job.json", {"name": batch_job.name})
            while True:
                batch_job = await asyncio.to_thread(client.client.batches.get, name=batch_job.name)
                state = str(getattr(getattr(batch_job, "metadata", None), "state", ""))
                if state.endswith("SUCCEEDED") or state == "JOB_STATE_SUCCEEDED":
                    break
                if state.endswith("FAILED") or state.endswith("CANCELLED"):
                    return {
                        "chunk_index": chunk_index,
                        "error": f"batch job ended in {state}",
                        "job_name": batch_job.name,
                        "task_keys": [task.key for task in chunk_tasks],
                    }
                await asyncio.sleep(poll_interval_seconds)

            output_file = getattr(batch_job, "dest", None)
            output_file = getattr(output_file, "file_name", None) or getattr(output_file, "fileName", None)
            output_path = run_dir / f"{desc.lower()}_batch_{chunk_index:04d}.responses.jsonl"
            if output_file:
                content = await asyncio.to_thread(client.client.files.download, file=output_file)
                if isinstance(content, bytes):
                    output_path.write_bytes(content)
                else:
                    output_path.write_text(str(content), encoding="utf-8")
            return {
                "chunk_index": chunk_index,
                "job_name": batch_job.name,
                "request_path": str(request_path),
                "output_path": str(output_path),
                "task_keys": [task.key for task in chunk_tasks],
            }

    by_key: dict[str, dict[str, Any]] = {}
    emitted: set[str] = set()

    def record_result(result: dict[str, Any]) -> None:
        by_key[result["key"]] = result
        emitted.add(result["key"])
        if on_result is not None:
            on_result(result)

    coroutines = [run_chunk(index + 1, chunk) for index, chunk in enumerate(chunks)]
    for future in tqdm(asyncio.as_completed(coroutines), total=len(coroutines), desc=f"{desc} batches"):
        result = await future
        if result.get("error"):
            for key in result.get("task_keys", []):
                record_result({"key": key, "error": result["error"], "usage": normalize_usage({})})
            continue
        output_path = Path(result["output_path"])
        if not output_path.exists():
            for key in result.get("task_keys", []):
                record_result({"key": key, "error": "missing batch output", "usage": normalize_usage({})})
            continue
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key, payload, usage, error, thought_summaries = extract_batch_response(json.loads(line))
            record_result(
                {
                    "key": key,
                    "payload": payload,
                    "usage": usage,
                    "error": error,
                    "thought_summaries": thought_summaries,
                    "batch_job_name": result.get("job_name"),
                    "raw_response_path": str(output_path),
                }
            )

    for task in tasks:
        if task.key not in emitted:
            record_result({"key": task.key, "error": "missing batch response", "usage": normalize_usage({})})

    return [by_key[task.key] for task in tasks]


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)
