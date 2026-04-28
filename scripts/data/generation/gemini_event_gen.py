import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.llms.llm_client import GeminiLLMClient

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SYSTEM_PROMPT = """Role: You are a precise information extraction annotator for food-security news.

Goal:
- Extract only explicit evidence spans from the article text.
- Focus on food insecurity events or risk factors using the provided ontology labels.

Critical constraints:
- Use only text that appears verbatim in the article body.
- Do not use title text for offsets.
- Do not infer, paraphrase, or combine non-contiguous text.
- If evidence is absent, return zero spans.

Span granularity (very important):
- Prefer the narrowest meaningful event/risk phrase, not full clauses.
- Keep only the core trigger phrase and essential modifiers.
- Remove location/time/context words unless required to preserve meaning.

Examples of narrow spans:
- "the current desert locust outbreak in the Horn of Africa" -> "desert locust outbreak"
- "recent drought crisis in South Africa's Cape Town region" -> "drought crisis"

Quality checks before finalizing each span:
1) Is the label in the ontology?
2) Is the span verbatim and contiguous in article text?
3) Is this the narrowest phrase that still preserves event/risk meaning?
4) Do start_char/end_char exactly match that span in article text?

Output discipline:
- Return only spans that satisfy all checks.
- Keep rationale short and factual."""

DEFAULT_USER_PROMPT = """Task: Extract food-insecurity and risk-factor evidence spans from the article.

Ontology labels (allowed labels only):
{ontology}

Extraction rules:
- Select only explicit evidence from the article text.
- span_text must be an exact contiguous substring from article text.
- Each span gets exactly one ontology label.
- Use the narrowest valid span. Avoid long clauses when a short core phrase is sufficient.
- Exclude surrounding context (location, date, attribution, background) unless essential to preserve event meaning.
- Do not output duplicates.
- If no valid evidence exists, return an empty spans list.

Narrow-span examples:
- "the current desert locust outbreak in the Horn of Africa" -> "desert locust outbreak"
- "recent drought crisis in South Africa's Cape Town region" -> "drought crisis"

Offset rules:
- start_char and end_char must index article text only (not title).
- end_char is exclusive.
- Offsets must match span_text exactly.

Title (context only, do not offset against this field):
{title}

Article text (offset source):
{text}
"""

LOGGER = logging.getLogger("gemini_event_gen")


class ExtractedSpan(BaseModel):
    span_text: str = Field(..., description="Verbatim span copied from article text.")
    label: str = Field(..., description="One label from the ontology.")
    start_char: int = Field(..., description="Start character offset in article text.")
    end_char: int = Field(
        ..., description="Exclusive end character offset in article text."
    )
    rationale: str = Field(
        default="", description="Brief reason for assigning the label."
    )


class SampleResult(BaseModel):
    spans: list[dict[str, Any]]
    metadata: dict[str, Any]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_json_tolerant(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some ontology files in this repo are JSON-like and contain trailing commas.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)


def normalize_ontology(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict) and isinstance(raw.get("events"), dict):
        raw = raw["events"]
    if not isinstance(raw, dict):
        raise ValueError(
            "Ontology must be a JSON object or contain an 'events' object."
        )

    ontology = {str(label): str(description) for label, description in raw.items()}
    if not ontology:
        raise ValueError("Ontology is empty.")
    return ontology


def format_ontology(ontology: dict[str, str]) -> str:
    return "\n".join(
        f"- {label}: {description}" for label, description in sorted(ontology.items())
    )


def load_prompt(path: Path | None, default: str) -> str:
    if path is None:
        return default
    return path.read_text(encoding="utf-8")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def normalize_input_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = dict(record)
    normalized["id"] = str(
        record.get("id") or record.get("uri") or record.get("doc_id") or index
    )
    normalized["title"] = str(record.get("title") or "")
    normalized["text"] = str(record.get("text") or record.get("body") or "")
    normalized["source_url"] = (
        record.get("source_url") or record.get("url") or record.get("source_uri")
    )
    normalized["publish_date"] = record.get("publish_date") or record.get(
        "published_at"
    )
    return normalized


def records_from_json_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("records", "samples", "articles", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"Input JSON must contain an object or list of objects: {path}")

    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Input record at {path}[{index}] is not a JSON object.")
        normalized_records.append(normalize_input_record(record, index))
    return normalized_records


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            normalize_input_record(record, index)
            for index, record in enumerate(iter_jsonl(path))
        ]
    return records_from_json_payload(load_json_tolerant(path), path)


def completed_ids(path: Path, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()

    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = record.get("id")
            if record_id is None:
                continue
            if retry_failed and record.get("status") == "error":
                continue
            done.add(str(record_id))
    return done


def render_user_prompt(template: str, ontology_text: str, title: str, text: str) -> str:
    return template.format(ontology=ontology_text, title=title, text=text)


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return json.loads(str(response))


def find_offsets(
    span_text: str, text: str, start_char: int, end_char: int
) -> tuple[int, int]:
    if (
        0 <= start_char < end_char <= len(text)
        and text[start_char:end_char] == span_text
    ):
        return start_char, end_char

    found = text.find(span_text)
    if found >= 0:
        return found, found + len(span_text)

    return -1, -1


def clamp_offsets(start_char: int, end_char: int, text_length: int) -> tuple[int, int]:
    start_char = min(max(start_char, 0), text_length)
    end_char = min(max(end_char, start_char), text_length)
    return start_char, end_char


def clean_spans(
    parsed: dict[str, Any], text: str, labels: set[str], strict_offsets: bool
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for span in parsed.get("spans", []) or []:
        if hasattr(span, "model_dump"):
            span = span.model_dump()
        if not isinstance(span, dict):
            continue

        span_text = str(span.get("span_text", "")).strip()
        label = str(span.get("label", "")).strip()
        if not span_text or label not in labels:
            continue

        try:
            start_char = int(span.get("start_char", -1))
            end_char = int(span.get("end_char", -1))
        except (TypeError, ValueError):
            start_char, end_char = -1, -1

        raw_start_char, raw_end_char = start_char, end_char
        start_char, end_char = find_offsets(span_text, text, start_char, end_char)
        if strict_offsets and start_char < 0:
            continue
        if start_char < 0:
            start_char, end_char = clamp_offsets(
                raw_start_char, raw_end_char, len(text)
            )

        key = (start_char, end_char, label)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "span_text": span_text,
                "label": label,
                "start_char": start_char,
                "end_char": end_char,
                "rationale": str(span.get("rationale", "")).strip(),
            }
        )
    return cleaned


def make_source(record: dict[str, Any], title: str, text: str) -> dict[str, Any]:
    return {
        "title": title,
        "text": text,
        "source_url": record.get("source_url"),
        "publish_date": record.get("publish_date"),
    }


def merge_self_consistency_spans(
    sample_spans: list[list[dict[str, Any]]], text: str
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], int], int]:
    successful_samples = len(sample_spans)
    threshold = (successful_samples // 2) + 1
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)

    for spans in sample_spans:
        seen_in_sample: set[tuple[int, int, str]] = set()
        for span in spans:
            key = (
                int(span["start_char"]),
                int(span["end_char"]),
                str(span["label"]),
            )
            if key in seen_in_sample:
                continue
            seen_in_sample.add(key)
            grouped[key].append(span)

    support_by_key = {key: len(spans) for key, spans in grouped.items()}
    merged: list[dict[str, Any]] = []
    for key in sorted(grouped):
        support = support_by_key[key]
        if support < threshold:
            continue

        start_char, end_char, label = key
        rationales = [
            str(span.get("rationale", "")).strip()
            for span in grouped[key]
            if str(span.get("rationale", "")).strip()
        ]
        rationale = ""
        if rationales:
            counts = Counter(rationales)
            rationale = max(
                range(len(rationales)),
                key=lambda index: (counts[rationales[index]], -index),
            )
            rationale = rationales[rationale]

        merged.append(
            {
                "span_text": text[start_char:end_char],
                "label": label,
                "start_char": start_char,
                "end_char": end_char,
                "rationale": rationale,
                "support": support,
            }
        )

    return merged, support_by_key, threshold


async def generate_sample(
    client: GeminiLLMClient,
    prompt: str,
    text: str,
    labels: set[str],
    system_prompt: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    strict_offsets: bool,
    record_id: str,
    override_settings: dict[str, Any] | None = None,
) -> SampleResult:
    response_format = {"spans": list[ExtractedSpan]}

    for attempt in range(max_retries + 1):
        try:
            response = None
            async for candidate in client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                override_settings=override_settings,
                response_format=response_format,
                add_cot_field=False,
            ):
                response = candidate
                break

            if response is None:
                raise RuntimeError("Gemini returned no response.")

            parsed = (
                response_to_dict(response.parsed)
                if response.parsed
                else json.loads(response.text)
            )
            spans = clean_spans(parsed, text, labels, strict_offsets)
            return SampleResult(spans=spans, metadata=response.metadata)
        except Exception as exc:
            if attempt >= max_retries:
                raise

            delay = min(max_backoff, initial_backoff * (2**attempt))
            delay = delay * (0.5 + random.random())
            LOGGER.warning(
                "Retrying id=%s after error on attempt %s/%s: %s",
                record_id,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Gemini returned no response.")


def combine_metadata(sample_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    numeric_keys = {
        key
        for metadata in sample_metadata
        for key, value in metadata.items()
        if isinstance(value, int | float)
    }
    for key in sorted(numeric_keys):
        combined[key] = sum(
            metadata.get(key, 0)
            for metadata in sample_metadata
            if isinstance(metadata.get(key, 0), int | float)
        )
    return combined


async def generate_one(
    client: GeminiLLMClient,
    record: dict[str, Any],
    ontology_text: str,
    labels: set[str],
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    strict_offsets: bool,
    self_consistency: bool = False,
    self_consistency_samples: int = 5,
    self_consistency_temperature: float = 0.7,
    self_consistency_min_successful_samples: int = 3,
) -> dict[str, Any]:
    record_id = str(record.get("id"))
    title = str(record.get("title") or "")
    text = str(record.get("text") or "")
    source = make_source(record, title, text)
    prompt = render_user_prompt(user_prompt_template, ontology_text, title, text)

    if not self_consistency:
        try:
            sample = await generate_sample(
                client=client,
                prompt=prompt,
                text=text,
                labels=labels,
                system_prompt=system_prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                strict_offsets=strict_offsets,
                record_id=record_id,
            )
            return {
                "id": record_id,
                "status": "ok",
                "source": source,
                "spans": sample.spans,
                "llm": {
                    "model": client.model_name,
                    "metadata": sample.metadata,
                },
            }
        except Exception as exc:
            return {
                "id": record_id,
                "status": "error",
                "error": str(exc),
                "source": source,
            }

    sample_spans: list[list[dict[str, Any]]] = []
    sample_metadata: list[dict[str, Any]] = []
    sample_errors: list[dict[str, Any]] = []
    override_settings = {"temperature": self_consistency_temperature}

    for sample_index in range(self_consistency_samples):
        try:
            sample = await generate_sample(
                client=client,
                prompt=prompt,
                text=text,
                labels=labels,
                system_prompt=system_prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                strict_offsets=strict_offsets,
                record_id=record_id,
                override_settings=override_settings,
            )
            sample_spans.append(sample.spans)
            sample_metadata.append(sample.metadata)
        except Exception as exc:
            sample_errors.append({"sample": sample_index, "error": str(exc)})

    samples_succeeded = len(sample_spans)
    if samples_succeeded < self_consistency_min_successful_samples:
        return {
            "id": record_id,
            "status": "error",
            "error": (
                "Self-consistency failed: "
                f"{samples_succeeded}/{self_consistency_samples} samples succeeded; "
                f"minimum required is {self_consistency_min_successful_samples}."
            ),
            "source": source,
            "llm": {
                "model": client.model_name,
                "metadata": combine_metadata(sample_metadata),
                "self_consistency": {
                    "enabled": True,
                    "samples_requested": self_consistency_samples,
                    "samples_succeeded": samples_succeeded,
                    "threshold": None,
                    "temperature": self_consistency_temperature,
                    "sample_errors": sample_errors,
                    "span_support": [],
                },
            },
        }

    spans, support_by_key, threshold = merge_self_consistency_spans(sample_spans, text)
    span_support = [
        {
            "start_char": start_char,
            "end_char": end_char,
            "label": label,
            "support": support,
        }
        for (start_char, end_char, label), support in sorted(support_by_key.items())
        if support >= threshold
    ]
    return {
        "id": record_id,
        "status": "ok",
        "source": source,
        "spans": spans,
        "llm": {
            "model": client.model_name,
            "metadata": combine_metadata(sample_metadata),
            "self_consistency": {
                "enabled": True,
                "samples_requested": self_consistency_samples,
                "samples_succeeded": samples_succeeded,
                "threshold": threshold,
                "temperature": self_consistency_temperature,
                "sample_errors": sample_errors,
                "span_support": span_support,
            },
        },
    }


async def writer(
    output_path: Path, queue: asyncio.Queue[dict[str, Any] | None]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            finally:
                queue.task_done()


async def worker(
    worker_id: int,
    input_queue: asyncio.Queue[dict[str, Any] | None],
    output_queue: asyncio.Queue[dict[str, Any] | None],
    args: argparse.Namespace,
    system_prompt: str,
    ontology_text: str,
    labels: set[str],
    user_prompt_template: str,
) -> None:
    client: GeminiLLMClient | None = None

    while True:
        record = await input_queue.get()
        try:
            if record is None:
                return
            if client is None:
                try:
                    client = GeminiLLMClient(
                        model_name=args.model,
                        system_prompt=None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                        verbose=args.verbose,
                    )
                except Exception as exc:
                    await output_queue.put(
                        {
                            "id": str(record.get("id")),
                            "status": "error",
                            "error": f"Could not initialize Gemini client: {exc}",
                            "source": {
                                "title": record.get("title"),
                                "source_url": record.get("source_url"),
                                "publish_date": record.get("publish_date"),
                            },
                        }
                    )
                    continue
            result = await generate_one(
                client=client,
                record=record,
                ontology_text=ontology_text,
                labels=labels,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                max_retries=args.max_retries,
                initial_backoff=args.initial_backoff,
                max_backoff=args.max_backoff,
                strict_offsets=args.strict_offsets,
                self_consistency=args.self_consistency,
                self_consistency_samples=args.self_consistency_samples,
                self_consistency_temperature=args.self_consistency_temperature,
                self_consistency_min_successful_samples=(
                    args.self_consistency_min_successful_samples
                ),
            )
            await output_queue.put(result)
            LOGGER.info(
                "worker=%s id=%s status=%s", worker_id, result["id"], result["status"]
            )
        finally:
            input_queue.task_done()


async def run(args: argparse.Namespace) -> None:
    if args.env_file:
        load_env_file(args.env_file)

    LOGGER.info("Using Gemini model: %s", args.model)
    LOGGER.info("Project ID: %s", os.environ.get("PROJECT_ID", "not set"))

    ontology = normalize_ontology(load_json_tolerant(args.ontology))
    ontology_text = format_ontology(ontology)
    labels = set(ontology)
    system_prompt = load_prompt(args.system_prompt_file, DEFAULT_SYSTEM_PROMPT)
    user_prompt_template = load_prompt(args.user_prompt_file, DEFAULT_USER_PROMPT)

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        done = set()
    else:
        done = completed_ids(args.output, retry_failed=args.retry_failed)
    pending = [record for record in records if str(record.get("id")) not in done]
    LOGGER.info(
        "loaded=%s completed=%s pending=%s", len(records), len(done), len(pending)
    )
    if not pending:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch(exist_ok=True)
        return

    input_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    writer_task = asyncio.create_task(writer(args.output, output_queue))
    workers = [
        asyncio.create_task(
            worker(
                worker_id=i,
                input_queue=input_queue,
                output_queue=output_queue,
                args=args,
                system_prompt=system_prompt,
                ontology_text=ontology_text,
                labels=labels,
                user_prompt_template=user_prompt_template,
            )
        )
        for i in range(args.workers)
    ]

    for record in pending:
        if not str(record.get("text") or "").strip():
            await output_queue.put(
                {
                    "id": str(record.get("id")),
                    "status": "error",
                    "error": "Input record is missing required article text/body.",
                }
            )
            continue
        await input_queue.put(record)

    for _ in workers:
        await input_queue.put(None)

    await input_queue.join()
    await asyncio.gather(*workers)
    await output_queue.put(None)
    await output_queue.join()
    await writer_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract food-security event/risk-factor spans from news JSON/JSONL "
            "with Gemini."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL or JSON with records containing title plus text or body.",
    )
    parser.add_argument(
        "output", type=Path, help="Output JSONL. Existing IDs are skipped."
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("ontologies/risk-factors/risk.cluster.description.json"),
        help="Ontology JSON file containing label descriptions.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel Gemini workers."
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", default="disable")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=60.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input records.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Do not treat previous status=error records as completed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file and process selected input records from scratch.",
    )
    parser.add_argument(
        "--strict-offsets",
        action="store_true",
        help="Drop spans whose text cannot be exactly located in the article text.",
    )
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Run multiple Gemini samples per record and keep majority-voted spans.",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=5,
        help="Number of Gemini samples per record when --self-consistency is enabled.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used only for self-consistency calls.",
    )
    parser.add_argument(
        "--self-consistency-min-successful-samples",
        type=int,
        default=3,
        help="Minimum successful samples required for a self-consistency result.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional file overriding the default system prompt.",
    )
    parser.add_argument(
        "--user-prompt-file",
        type=Path,
        default=None,
        help="Optional .format template overriding the default user prompt. Must include {ontology}, {title}, {text}.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional env file to load before constructing the Gemini client.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log prompts through the LLM client."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.self_consistency_samples < 1:
        raise ValueError("--self-consistency-samples must be >= 1")
    if args.self_consistency_min_successful_samples < 1:
        raise ValueError("--self-consistency-min-successful-samples must be >= 1")
    if args.self_consistency_min_successful_samples > args.self_consistency_samples:
        raise ValueError(
            "--self-consistency-min-successful-samples cannot exceed "
            "--self-consistency-samples"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    started = time.monotonic()
    asyncio.run(run(args))
    LOGGER.info("finished in %.1fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
