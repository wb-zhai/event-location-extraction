import argparse
import asyncio
import collections
import copy
import json
import logging
import os
import random
import re
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

LOGGER = logging.getLogger("generation")

DEFAULT_PROMPT = Path(__file__).with_name("annotation_prompt.txt")
DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "teacher" / "system_prompt.txt"
USER_PROMPT_PATH = Path(__file__).parent / "prompts" / "teacher" / "user_prompt.txt"
SYSTEM_PROMPT_PATH_FR = Path(__file__).parent / "prompts" / "teacher" / "system_prompt.fr.txt"
USER_PROMPT_PATH_FR = Path(__file__).parent / "prompts" / "teacher" / "user_prompt.fr.txt"
ONTOLOGY_PATH = REPO_ROOT / "ontologies" / "zhai" / "bona.v4.json"


class AnnotationEvent(BaseModel):
    event_type: str
    grounding_quote: str
    event_location_text: str
    event_location: str
    event_time_text: str
    event_time: str
    time_status: Literal["past", "ongoing", "forecast", "not_stated"]
    severity: Literal["low", "medium", "high", "extreme", "not_stated"]


class Annotation(BaseModel):
    events: list[AnnotationEvent] = Field(default_factory=list)


# Pre-built schema dict (same as batch_request_config). Used in both sync and batch
# paths to guarantee identical wire format. Replicating the batch path's explicit
# t_schema(...).model_dump(exclude_none=True) call avoids the SDK's internal
# Schema-object serialization, which produces a different (and rejected) wire format.
_ANNOTATION_SCHEMA_DICT = genai_transformers.t_schema(None, Annotation).model_dump(
    exclude_none=True
)


@dataclass
class BatchTask:
    key: str
    record: dict[str, Any]
    prompt: str
    system_prompt: str
    response_schema: dict[str, Any]
    candidate_labels: list[str] | None = None
    error: str | None = None


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


def _source_dict(record: dict[str, Any]) -> dict[str, Any]:
    # Records may carry article fields nested under "source" (e.g. relevance-filter
    # pipeline output) or flat at the top level (e.g. raw scrape dumps).
    source = record.get("source")
    return source if isinstance(source, dict) else record


def source_title(record: dict[str, Any]) -> str:
    return str(_source_dict(record).get("title") or "")


def source_text(record: dict[str, Any]) -> str:
    return str(_source_dict(record).get("text") or "")


def source_publish_date(record: dict[str, Any]) -> str:
    source = _source_dict(record)
    return str(source.get("publish_date") or source.get("published_at") or "")


def source_url(record: dict[str, Any]) -> str:
    return str(_source_dict(record).get("source_url") or "")


def normalized_source(record: dict[str, Any]) -> dict[str, str]:
    return {
        "title": source_title(record),
        "text": source_text(record),
        "source_url": source_url(record),
        "publish_date": source_publish_date(record),
    }


def render_prompt(template: str, record: dict[str, Any]) -> str:
    article = f"Title: {source_title(record)}\n" "\n" f"{source_text(record)}"
    return template.replace(
        "{{PUBLISH_DATE}}", source_publish_date(record) or "not_stated"
    ).replace("{{ARTICLE_TEXT}}", article)


def candidate_labels(
    record: dict[str, Any], top_k: int | None = None
) -> list[str] | None:
    if "candidates" not in record:
        return None

    candidates = record["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list of strings")

    if top_k is not None:
        candidates = candidates[:top_k]

    labels: list[str] = []
    seen: set[str] = set()
    for index, label in enumerate(candidates):
        if not isinstance(label, str):
            raise ValueError(f"candidates[{index}] must be a string")
        label = label.strip()
        if not label:
            raise ValueError(f"candidates[{index}] must be a non-empty string")
        if label not in seen:
            labels.append(label)
            seen.add(label)

    if not labels:
        raise ValueError("candidates must contain at least one label")
    return labels


def load_ontology_definitions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["events"])


def annotation_schema(labels: list[str] | None = None) -> dict[str, Any]:
    if labels is None:
        return _ANNOTATION_SCHEMA_DICT

    schema = copy.deepcopy(_ANNOTATION_SCHEMA_DICT)
    event_type_schema = schema["properties"]["events"]["items"]["properties"][
        "event_type"
    ]
    event_type_schema["enum"] = labels
    return schema


def strip_output_schema_block(system_prompt: str) -> str:
    return re.sub(
        r"\n?<output_schema>.*?</output_schema>\n?",
        "\n",
        system_prompt,
        flags=re.DOTALL,
    ).strip()


def strip_event_type_rules_block(system_prompt: str) -> str:
    return re.sub(
        r"\n?<event_type_rules>.*?</event_type_rules>\n?",
        "\n",
        system_prompt,
        flags=re.DOTALL,
    ).strip()


def replace_allowed_event_types(
    system_prompt: str,
    labels: list[str],
    definitions: dict[str, str] | None = None,
) -> str:
    lines = (
        labels
        if not definitions
        else [
            f"{label}: {definitions[label]}" if label in definitions else label
            for label in labels
        ]
    )
    replacement = (
        "<allowed_event_types>\n" + "\n".join(lines) + "\n</allowed_event_types>"
    )
    updated, count = re.subn(
        r"<allowed_event_types>.*?</allowed_event_types>",
        replacement,
        system_prompt,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("system prompt must contain one <allowed_event_types> block")
    return updated


def render_system_prompt(
    system_prompt: str,
    record: dict[str, Any],
    top_k: int | None = None,
    definitions: dict[str, str] | None = None,
) -> tuple[str, list[str] | None]:
    labels = candidate_labels(record, top_k=top_k)
    rendered = strip_output_schema_block(system_prompt)
    if labels is not None:
        rendered = replace_allowed_event_types(rendered, labels, definitions)
    elif definitions is not None:
        rendered = replace_allowed_event_types(
            rendered, list(definitions.keys()), definitions
        )
    if definitions is not None:
        rendered = strip_event_type_rules_block(rendered)
    return rendered, labels


def validate_candidate_event_types(
    annotation: dict[str, Any], labels: list[str] | None
) -> None:
    if labels is None:
        return
    allowed = set(labels)
    for index, event in enumerate(annotation.get("events") or []):
        event_type = event.get("event_type")
        if event_type not in allowed:
            raise ValueError(
                f"events[{index}].event_type {event_type!r} is not in candidates"
            )


def apply_grounding_verification(
    annotation: dict[str, Any], article_text: str
) -> None:
    _VERBATIM_FIELDS = ("grounding_quote", "event_location_text", "event_time_text")
    article_text_lower = article_text.lower()
    verified: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    for event in annotation.get("events") or []:
        failed_field = None
        for field in _VERBATIM_FIELDS:
            value = event.get(field)
            if value and value != "not_stated" and value.lower() not in article_text_lower:
                failed_field = field
                break
        if failed_field is None:
            verified.append(event)
        else:
            unverified.append({**event, "_unverified_field": failed_field})
    annotation["events"] = verified
    if unverified:
        annotation["unverified_events"] = unverified


def _with_normalized_source(record: dict[str, Any]) -> dict[str, Any]:
    # Preserve the entire original row (all input fields, including any richer
    # nested "source" dict) and layer the title/text/source_url/publish_date
    # keys downstream steps (validate.py) expect on top of it.
    existing_source = record.get("source")
    merged_source = {
        **(existing_source if isinstance(existing_source, dict) else {}),
        **normalized_source(record),
    }
    return {**record, "source": merged_source}


def output_record(
    record: dict[str, Any],
    annotation: dict[str, Any],
    *,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # "generation_llm" (not "llm") avoids clobbering an upstream pipeline's own
    # "llm" field (e.g. relevance-filter metadata already present on the row).
    return {
        **_with_normalized_source(record),
        "status": "ok",
        "annotation": annotation,
        "generation_llm": {"model": model, "metadata": metadata or {}},
    }


def error_record(record: dict[str, Any], error: str, *, model: str) -> dict[str, Any]:
    return {
        **_with_normalized_source(record),
        "status": "error",
        "error": error,
        "generation_llm": {"model": model},
    }


def completed_output_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for result in iter_jsonl(path):
        if result.get("status") != "ok" or "generation_llm" not in result:
            continue
        result_id = result.get("id")
        if result_id is not None:
            completed.add(str(result_id))
    return completed


def _stratified_sample(
    records: list[dict[str, Any]], limit: int, *, key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "")].append(record)

    total = len(records)
    group_keys = list(groups.keys())
    floats = [limit * len(groups[k]) / total for k in group_keys]
    floors = [int(f) for f in floats]
    remainders = sorted(
        range(len(group_keys)), key=lambda i: floats[i] - floors[i], reverse=True
    )
    for i in remainders[: limit - sum(floors)]:
        floors[i] += 1

    sampled: list[dict[str, Any]] = []
    sampled_ids: set[int] = set()
    for i, k in enumerate(group_keys):
        chosen = random.sample(groups[k], min(floors[i], len(groups[k])))
        sampled.extend(chosen)
        sampled_ids.update(id(r) for r in chosen)

    if len(sampled) < limit:
        pool = [r for r in records if id(r) not in sampled_ids]
        sampled.extend(random.sample(pool, min(limit - len(sampled), len(pool))))

    return sampled


def filter_relevant_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered = []
    n_skipped = 0
    for record in records:
        relevance = record.get("relevance")
        if isinstance(relevance, dict) and relevance.get("is_relevant") is False:
            n_skipped += 1
            continue
        filtered.append(record)
    if n_skipped:
        LOGGER.info("Skipping %s records with relevance.is_relevant=False", n_skipped)
    return filtered


def sample_records(
    records: list[dict[str, Any]],
    limit: int | None,
    *,
    random_sample: bool,
    stratified: str | None,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    if stratified == "url":
        return _stratified_sample(records, limit, key="source_url")
    if random_sample:
        return random.sample(records, limit)
    return records[:limit]


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


async def generate_one(
    client: GeminiLLMClient,
    record: dict[str, Any],
    prompt: str,
    *,
    system_prompt: str,
    response_schema: dict[str, Any],
    candidate_labels: list[str] | None = None,
    include_thoughts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Pass the pre-built schema dict via override_settings so both sync and batch
    # paths send the same wire format (matching batch_request_config).
    async for response in client.generate(
        prompt,
        system_prompt=system_prompt,
        override_settings={
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        },
        add_cot_field=False,
        include_thoughts=include_thoughts,
        reasoning_effort=client.reasoning_effort,
    ):
        annotation = Annotation.model_validate(json.loads(response.text)).model_dump()
        validate_candidate_event_types(annotation, candidate_labels)
        apply_grounding_verification(
            annotation,
            f"Title: {source_title(record)}\n\n{source_text(record)}",
        )
        return annotation, response.metadata
    raise RuntimeError("Gemini returned no response.")


def _print_verification_stats(
    n_records: int,
    total_events: int,
    total_unverified: int,
    field_counts: collections.Counter,
) -> None:
    total = total_events + total_unverified
    if not n_records or not total:
        return
    pct = 100 * total_unverified / total
    avg = total_unverified / n_records
    field_summary = ", ".join(
        f"{field}: {count}" for field, count in sorted(field_counts.items())
    )
    print(
        f"\nGrounding verification: {total_unverified}/{total} events unverified"
        f" ({pct:.1f}%), avg {avg:.2f} per record"
        + (f"\n  by field — {field_summary}" if field_summary else "")
    )


async def run_sync(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    template: str,
    system_prompt: str,
) -> None:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=system_prompt,
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
                record_system_prompt, labels = render_system_prompt(
                    system_prompt,
                    record,
                    top_k=args.top_k_candidates,
                    definitions=args.definitions,
                )
                annotation, metadata = await generate_one(
                    client,
                    record,
                    render_prompt(template, record),
                    system_prompt=record_system_prompt,
                    response_schema=annotation_schema(labels),
                    candidate_labels=labels,
                    include_thoughts=should_include_thoughts(args),
                )
                return output_record(
                    record, annotation, model=args.model, metadata=metadata
                )
            except Exception as exc:
                LOGGER.exception("Failed id=%s", record_id(record, index))
                return error_record(record, str(exc), model=args.model)

    coroutines = [process_record(index, record) for index, record in pending]
    n_ok = total_events = total_unverified = 0
    field_counts: collections.Counter = collections.Counter()
    with args.output.open("a", encoding="utf-8") as handle:
        for future in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="annotating",
        ):
            result = await future
            append_jsonl(handle, result)
            if result.get("status") == "ok":
                n_ok += 1
                ann = result.get("annotation") or {}
                total_events += len(ann.get("events") or [])
                for ev in ann.get("unverified_events") or []:
                    total_unverified += 1
                    field_counts[ev.get("_unverified_field") or "unknown"] += 1
    _print_verification_stats(n_ok, total_events, total_unverified, field_counts)


async def run_interactive(
    args: argparse.Namespace, template: str, system_prompt: str
) -> None:
    if sys.stdin.isatty():
        print("Paste article text, then press Ctrl-D:", file=sys.stderr)
    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("No text provided on stdin.")

    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    record = {"id": "interactive", "source": {"title": args.title, "text": text}}
    record_system_prompt, labels = render_system_prompt(
        system_prompt, record, definitions=args.definitions
    )
    annotation, metadata = await generate_one(
        client,
        record,
        render_prompt(template, record),
        system_prompt=record_system_prompt,
        response_schema=annotation_schema(labels),
        candidate_labels=labels,
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


def reasoning_enabled(reasoning_effort: str | int | None) -> bool:
    return reasoning_effort not in {None, "", "disable", 0, "0"}


def should_include_thoughts(args: argparse.Namespace) -> bool:
    return bool(args.include_thoughts) or reasoning_enabled(args.reasoning_effort)


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
    if should_include_thoughts(args):
        generation_config.setdefault("thinkingConfig", {})["includeThoughts"] = True
    return {"generationConfig": generation_config}


def build_batch_tasks(
    records: list[dict[str, Any]],
    template: str,
    system_prompt: str,
    top_k_candidates: int | None = None,
    definitions: dict[str, str] | None = None,
) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    for index, record in enumerate(records):
        key = record_id(record, index)
        prompt = render_prompt(template, record)
        try:
            record_system_prompt, labels = render_system_prompt(
                system_prompt,
                record,
                top_k=top_k_candidates,
                definitions=definitions,
            )
            response_schema = annotation_schema(labels)
            error = None
        except Exception as exc:
            record_system_prompt = ""
            labels = None
            response_schema = _ANNOTATION_SCHEMA_DICT
            error = str(exc)
        tasks.append(
            BatchTask(
                key=key,
                record=record,
                prompt=prompt,
                system_prompt=record_system_prompt,
                response_schema=response_schema,
                candidate_labels=labels,
                error=error,
            )
        )
    return tasks


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def batch_request_line(
    task: BatchTask, request_config: dict[str, Any]
) -> dict[str, Any]:
    generation_config = {
        **request_config["generationConfig"],
        "responseSchema": task.response_schema,
    }
    return {
        "key": task.key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": task.prompt}]}],
            "systemInstruction": {"parts": [{"text": task.system_prompt}]},
            "generationConfig": generation_config,
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
            validate_candidate_event_types(annotation, task.candidate_labels)
            apply_grounding_verification(
                annotation,
                f"Title: {source_title(task.record)}\n\n{source_text(task.record)}",
            )
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
    request_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    return records, seen


async def run_batch(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    template: str,
    system_prompt: str,
) -> None:
    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    request_config = batch_request_config(args)
    completed = completed_output_ids(args.output)
    tasks = [
        task
        for task in build_batch_tasks(
            records,
            template,
            system_prompt,
            args.top_k_candidates,
            args.definitions,
        )
        if task.key not in completed
    ]
    invalid_tasks = [task for task in tasks if task.error is not None]
    tasks = [task for task in tasks if task.error is None]
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
    n_ok = total_events = total_unverified = 0
    field_counts: collections.Counter = collections.Counter()
    with args.output.open("a", encoding="utf-8") as handle:
        for task in invalid_tasks:
            append_jsonl(
                handle, error_record(task.record, task.error or "", model=args.model)
            )
        for future in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="batch chunks",
        ):
            chunk_records, _seen = await future
            for result in chunk_records:
                append_jsonl(handle, result)
                if result.get("status") == "ok":
                    n_ok += 1
                    ann = result.get("annotation") or {}
                    total_events += len(ann.get("events") or [])
                    for ev in ann.get("unverified_events") or []:
                        total_unverified += 1
                        field_counts[ev.get("_unverified_field") or "unknown"] += 1
    _print_verification_stats(n_ok, total_events, total_unverified, field_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument(
        "--french",
        action="store_true",
        help="Use the French translation of the teacher system/user prompts "
        "(event categories and output format stay in English).",
    )
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random", action="store_true", dest="random_sample")
    parser.add_argument("--stratified", default=None, choices=["url"])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--include-thoughts", action="store_true")
    parser.add_argument("--batch-api", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--batch-poll-interval-seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-k-candidates", type=int, default=None)
    parser.add_argument(
        "--skip-not-relevant",
        action="store_true",
        help="Skip records whose relevance.is_relevant is False (if present).",
    )
    parser.add_argument(
        "--use-definitions",
        action="store_true",
        help="Append the ontology's event_type definitions inline in <allowed_event_types>.",
    )
    parser.add_argument("--ontology-path", type=Path, default=ONTOLOGY_PATH)

    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = USER_PROMPT_PATH_FR if args.french else USER_PROMPT_PATH
    args.definitions = (
        load_ontology_definitions(args.ontology_path) if args.use_definitions else None
    )
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
    system_prompt_path = SYSTEM_PROMPT_PATH_FR if args.french else SYSTEM_PROMPT_PATH
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    print("System prompt:")
    print(system_prompt)
    template = args.prompt.read_text(encoding="utf-8")
    if args.interactive:
        await run_interactive(args, template, system_prompt)
        return

    records = load_input_records(args.input)
    if args.skip_not_relevant:
        records = filter_relevant_records(records)
    records = sample_records(
        records,
        args.limit,
        random_sample=args.random_sample,
        stratified=args.stratified,
    )
    if args.batch_api:
        await run_batch(args, records, template, system_prompt)
    else:
        await run_sync(args, records, template, system_prompt)


if __name__ == "__main__":
    asyncio.run(main())
