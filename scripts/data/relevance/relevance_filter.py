import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from google.genai import types as genai_types
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import GeminiLLMClient

DEFAULT_RELEVANCE_SYSTEM_PROMPT = """<role>
You are a high-recall relevance gate for food-security risk-event extraction.
</role>

<goal>
Decide whether the article is likely to contain explicit food-insecurity events or risk-factor evidence worth sending to the full extraction pipeline.
</goal>

<event_categories>
The pipeline extracts events in these categories:
- agricultural production issues
- conflicts and violence
- economic issues
- environmental issues
- food crisis
- forced displacement
- humanitarian aid
- land-related issues
- pests and diseases
- political instability
- weather shocks
</event_categories>

<policy>
- Favor recall over precision.
- If the article is borderline, ambiguous, or only partially visible in the preview, mark it relevant.
- Mark it irrelevant only when the title and preview strongly indicate the article is outside all of the event categories above.
- Use only the provided title and article preview.
</policy>
"""

DEFAULT_RELEVANCE_USER_PROMPT = """<context>
<title>
{title}
</title>

<article_preview>
{text}
</article_preview>
</context>

<task>
Return whether this article should proceed to the full food-security risk/event extraction pipeline.
</task>

<decision_rule>
- is_relevant=true if the article likely contains evidence of any of the following event categories: agricultural production issues, conflicts and violence, economic issues, environmental issues, food crisis, forced displacement, humanitarian aid, land-related issues, pests and diseases, political instability, or weather shocks.
- If uncertain, return is_relevant=true.
- is_relevant=false only when the article is clearly unrelated to all of the above categories.
</decision_rule>
"""


food_insecurity_regex = re.compile(
    r"\b(?:"
    r"food insecurity|acute food insecurity|food security crisis|food crisis|"
    r"hunger crisis|acute hunger|hunger|famine|malnutrition|undernourishment|"
    r"food scarcity|food shortage(?:s)?|lack of food|"
    r"food access|food availability|food affordability|"
    r"food aid|food assistance|emergency food aid|humanitarian food assistance|"
    r"rising food prices|high food prices|food price inflation|"
    r"cereal prices|wheat prices|maize prices|rice prices|"
    r"fertilizer shortage|fertilizer prices|"
    r"crop failure|harvest failure|"
    r"drought|flood(?:ing)?|climate shock(?:s)?|"
    r"conflict|displacement"
    r")\b",
    re.IGNORECASE,
)

exhaustive_food_insecurity_regex = re.compile(
    r"\b(?:"
    r"food insecurity|acute food insecurity|chronic food insecurity|severe food insecurity|"
    r"food security|food security crisis|food crisis|nutrition crisis|"
    r"hunger crisis|acute hunger|chronic hunger|hunger|famine|near famine|"
    r"malnutrition|acute malnutrition|severe acute malnutrition|child malnutrition|"
    r"undernourishment|undernutrition|stunting|wasting|"
    r"food scarcity|food shortage(?:s)?|grain shortage(?:s)?|lack of food|"
    r"food access|food availability|food affordability|food consumption|"
    r"food aid|food assistance|emergency food aid|humanitarian food assistance|"
    r"cash assistance|nutrition assistance|school feeding|"
    r"rising food prices|high food prices|food price inflation|food inflation|"
    r"cereal prices|wheat prices|maize prices|rice prices|bread prices|"
    r"fertilizer shortage|fertilizer prices|input costs|"
    r"crop failure|harvest failure|poor harvest|failed harvest|yield loss(?:es)?|"
    r"livestock deaths|pasture shortage|water shortage(?:s)?|"
    r"drought|dry spell(?:s)?|heatwave(?:s)?|flood(?:ing)?|flash flood(?:s)?|"
    r"storm(?:s)?|cyclone(?:s)?|hurricane(?:s)?|typhoon(?:s)?|landslide(?:s)?|"
    r"climate shock(?:s)?|weather shock(?:s)?|el nino|la nina|"
    r"conflict|armed conflict|violence|insecurity|displacement|forced displacement|"
    r"refugee(?:s)?|internally displaced|idp(?:s)?|"
    r"locust(?:s)?|desert locust(?:s)?|fall armyworm|crop pest(?:s)?|livestock disease(?:s)?|"
    r"cholera outbreak(?:s)?|market disruption(?:s)?|supply chain disruption(?:s)?"
    r")\b",
    re.IGNORECASE,
)


class RelevanceDecision(BaseModel):
    reason: str = Field(default="")
    confidence: float = Field(
        default=0.0, description="Confidence from 0.0 to 1.0 in the relevance decision."
    )
    is_relevant: bool = Field(
        ..., description="True when the article should proceed to full extraction."
    )


def clean_relevance_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if not isinstance(parsed, dict):
        parsed = {}

    is_relevant = bool(parsed.get("is_relevant", True))
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    reason = str(parsed.get("reason", "")).strip()
    return {
        "is_relevant": is_relevant,
        "confidence": confidence,
        "reason": reason,
    }


def should_filter_by_relevance(
    decision: dict[str, Any], confidence_threshold: float
) -> bool:
    return (
        not bool(decision.get("is_relevant", True))
        and float(decision.get("confidence", 0.0) or 0.0) >= confidence_threshold
    )


async def classify_article_relevance(
    client: GeminiLLMClient,
    title: str,
    text: str,
    record_id: str,
    max_chars: int,
    confidence_threshold: float,
    system_prompt: str = DEFAULT_RELEVANCE_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_RELEVANCE_USER_PROMPT,
    verbose: bool = False,
) -> dict[str, Any]:
    from scripts.data.generation.gemini_event_gen import (
        log_llm_call,
        response_to_dict,
        truncate_text,
    )

    preview_text = truncate_text(text, max_chars)
    prompt = user_prompt_template.format(title=title, text=preview_text)
    log_llm_call(
        enabled=verbose,
        record_id=record_id,
        step="relevance_filter",
        call_type="relevance",
        system_prompt=system_prompt,
        prompt=prompt,
    )
    response = None
    async for candidate in client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        override_settings={"temperature": 0.0},
        response_format={
            "reason": str,
            "is_relevant": bool,
            "confidence": float,
        },
        add_cot_field=False,
        reasoning_effort="minimal" if "3" in client.model_name else None,
    ):
        response = candidate
        break
    if response is None:
        raise RuntimeError("Gemini returned no relevance response.")

    raw_answer = response_to_dict(response.parsed) if response.parsed else response.text
    log_llm_call(
        enabled=verbose,
        record_id=record_id,
        step="relevance_filter",
        call_type="relevance",
        system_prompt=system_prompt,
        prompt=prompt,
        answer=raw_answer,
    )
    parsed = raw_answer if isinstance(raw_answer, dict) else json.loads(raw_answer)
    decision = clean_relevance_decision(parsed)
    return {
        "decision": "relevant" if decision["is_relevant"] else "irrelevant",
        "is_relevant": decision["is_relevant"],
        "confidence": decision["confidence"],
        "reason": decision["reason"],
        "filtered": should_filter_by_relevance(decision, confidence_threshold),
        "threshold": confidence_threshold,
        "model": client.model_name,
        "max_chars": max_chars,
        "text_chars_used": len(preview_text),
        "metadata": response.metadata,
    }


async def process_record(client, record, args):
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    record_id = str(record.get("id", record.get("url", "")))

    try:
        relevance_info = {}
        if args.use_regex:
            full_text = title + " " + text
            regex_match = bool(food_insecurity_regex.search(full_text))

            if not regex_match:
                relevance_info = {
                    "decision": "irrelevant",
                    "is_relevant": False,
                    "confidence": 1.0,
                    "reason": "Regex filter mismatch",
                    "filtered": True,
                    "threshold": 1.0,
                    "model": "regex",
                }
            else:
                relevance_info = {
                    "decision": "relevant",
                    "is_relevant": True,
                    "confidence": 1.0,
                    "reason": "Regex match",
                    "filtered": False,
                    "threshold": 1.0,
                    "model": "regex",
                }
        elif args.use_keywords:
            text_lower = (title + " " + text).lower()
            keyword_match = any(kw.lower() in text_lower for kw in args.keywords)
            if not keyword_match:
                relevance_info = {
                    "decision": "irrelevant",
                    "is_relevant": False,
                    "confidence": 1.0,
                    "reason": "Keyword filter mismatch",
                    "filtered": True,
                    "threshold": 1.0,
                    "model": "keyword",
                }
            else:
                relevance_info = {
                    "decision": "relevant",
                    "is_relevant": True,
                    "confidence": 1.0,
                    "reason": "Keyword match",
                    "filtered": False,
                    "threshold": 1.0,
                    "model": "keyword",
                }

        if args.use_llm and (not relevance_info.get("filtered", False)):
            relevance_info = await classify_article_relevance(
                client=client,
                title=title,
                text=text,
                record_id=record_id,
                max_chars=args.max_chars,
                confidence_threshold=args.confidence_threshold,
                verbose=args.verbose,
            )

        record["relevance"] = relevance_info
        if "metadata" in relevance_info and relevance_info.get("model") not in (None, "regex", "keyword"):
            record["llm"] = {"model": relevance_info["model"], "metadata": relevance_info["metadata"]}
    except Exception as e:
        record["relevance"] = {"error": str(e), "filtered": False}
        print(f"Error processing record {record_id}: {e}")

    return record


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("url") or "")


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

_RELEVANCE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reason": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "is_relevant": {"type": "BOOLEAN"},
    },
    "required": ["reason", "confidence", "is_relevant"],
}

@dataclass
class _BatchTask:
    key: str
    record: dict[str, Any]
    prompt: str


def _build_batch_request(task: "_BatchTask", system_prompt: str) -> dict[str, Any]:
    return {
        "key": task.key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": task.prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 256,
                "temperature": 0.0,
                "responseSchema": _RELEVANCE_RESPONSE_SCHEMA,
            },
        },
    }


async def _poll_batch_job(client: GeminiLLMClient, name: str, interval: int) -> Any:
    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    job = await asyncio.to_thread(client.client.batches.get, name=name)
    while job.state.name not in done:
        print(f"  batch={name} state={job.state.name}")
        await asyncio.sleep(interval)
        job = await asyncio.to_thread(client.client.batches.get, name=name)
    return job


def _batch_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("text"), str):
        return response["text"]
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and not part.get("thought"):
                return text
    raise ValueError("Batch response contains no generated text.")


def _batch_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage_metadata") or response.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0),
        "completion_tokens": int(usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or 0),
        "cached_tokens": int(usage.get("cached_content_token_count") or usage.get("cachedContentTokenCount") or 0),
        "thoughts_token_count": int(usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount") or 0),
    }


def _batch_result_key(line: dict[str, Any]) -> str:
    if line.get("key") is not None:
        return str(line["key"])
    return str((line.get("metadata") or {}).get("key") or "")


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _execute_batch_chunk(
    client: GeminiLLMClient,
    tasks: list["_BatchTask"],
    system_prompt: str,
    model: str,
    output_path: Path,
    chunk_index: int,
    poll_interval: int,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    request_path = output_path.with_suffix(f".batch.part-{chunk_index:04d}.requests.jsonl")
    result_path = output_path.with_suffix(f".batch.part-{chunk_index:04d}.results.jsonl")

    request_path.write_text(
        "".join(
            json.dumps(_build_batch_request(t, system_prompt), ensure_ascii=False) + "\n"
            for t in tasks
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
        model=model,
        src=uploaded.name,
        config={"display_name": f"{output_path.stem}-part-{chunk_index:04d}"},
    )
    print(f"Chunk {chunk_index}: batch job {job.name} submitted ({len(tasks)} records).")
    job = await _poll_batch_job(client, job.name, poll_interval)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Chunk {chunk_index}: batch job {job.name} ended with {job.state.name}.")
        return [_error_batch_record(t.record, f"Batch job ended with {job.state.name}", model) for t in tasks]

    if not job.dest or not job.dest.file_name:
        return [_error_batch_record(t.record, "Batch job succeeded without a result file.", model) for t in tasks]

    data = await asyncio.to_thread(client.client.files.download, file=job.dest.file_name)
    result_path.write_bytes(data)

    task_by_key = {t.key: t for t in tasks}
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    with result_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            key = _batch_result_key(line)
            task = task_by_key.get(key)
            if task is None:
                continue
            try:
                response = line.get("response")
                if not isinstance(response, dict):
                    raise ValueError(str(line.get("error") or "Missing batch response."))
                parsed = json.loads(_batch_response_text(response))
                decision = clean_relevance_decision(parsed)
                metadata = _batch_response_metadata(response)
                relevance_info = {
                    "decision": "relevant" if decision["is_relevant"] else "irrelevant",
                    "is_relevant": decision["is_relevant"],
                    "confidence": decision["confidence"],
                    "reason": decision["reason"],
                    "filtered": should_filter_by_relevance(decision, confidence_threshold),
                    "threshold": confidence_threshold,
                    "model": model,
                    "metadata": metadata,
                }
                rec = dict(task.record)
                rec["relevance"] = relevance_info
                rec["llm"] = {"model": model, "metadata": metadata}
            except Exception as exc:
                rec = dict(task.record)
                rec["relevance"] = {"error": str(exc), "filtered": False}
            results.append(rec)
            seen.add(key)

    for task in tasks:
        if task.key not in seen:
            results.append(_error_batch_record(task.record, "Batch result missing.", model))

    request_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    return results


def _error_batch_record(record: dict[str, Any], error: str, model: str) -> dict[str, Any]:
    rec = dict(record)
    rec["relevance"] = {"error": error, "filtered": False, "model": model}
    return rec


async def run_batch_relevance(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    done_keys: set[str],
) -> list[dict[str, Any]]:
    from scripts.data.generation_v3.costs import aggregate, report

    client = GeminiLLMClient(model_name=args.model, system_prompt=None)
    system_prompt = args.system_prompt if hasattr(args, "system_prompt") and args.system_prompt else DEFAULT_RELEVANCE_SYSTEM_PROMPT
    user_prompt_template = DEFAULT_RELEVANCE_USER_PROMPT

    tasks = [
        _BatchTask(
            key=_record_key(r) or str(i),
            record=r,
            prompt=user_prompt_template.format(
                title=str(r.get("title") or (r.get("source") or {}).get("title", "")),
                text=(str(r.get("text") or (r.get("source") or {}).get("text", "")))[:args.max_chars],
            ),
        )
        for i, r in enumerate(records)
    ]

    chunks = list(enumerate(_chunked(tasks, args.batch_size), start=1))
    all_results: list[dict[str, Any]] = []

    for chunk_index, chunk in chunks:
        print(f"Processing chunk {chunk_index}/{len(chunks)} ({len(chunk)} records)...")
        chunk_results = await _execute_batch_chunk(
            client=client,
            tasks=chunk,
            system_prompt=system_prompt,
            model=args.model,
            output_path=output_path,
            chunk_index=chunk_index,
            poll_interval=args.batch_poll_interval_seconds,
            confidence_threshold=args.confidence_threshold,
        )
        all_results.extend(chunk_results)

        write_mode = "a" if (args.resume and done_keys) or chunk_index > 1 else "w"
        with output_path.open(write_mode, encoding="utf-8") as fh:
            for r in chunk_results:
                if args.filter_only and r.get("relevance", {}).get("filtered", False):
                    continue
                fh.write(json.dumps(r) + "\n")

    totals = aggregate(all_results)
    report(output_path.name, totals, batch=True)
    return all_results


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

async def process_file(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output file {output_path} already exists. Use --resume to continue "
            "or --overwrite to replace it."
        )

    # Load already-done keys when resuming
    done_keys: set[str] = set()
    if args.resume and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_keys.add(_record_key(json.loads(line)))
        print(f"Resuming: {len(done_keys)} records already done, skipping.")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if _record_key(rec) not in done_keys:
                    records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    if args.batch_api:
        results = await run_batch_relevance(args, records, output_path, done_keys)
    else:
        client = None
        if args.use_llm:
            client = GeminiLLMClient(model_name=args.model, system_prompt=None)

        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_process(record):
            async with sem:
                return await process_record(client, record, args)

        # Optional progress bar if tqdm is available
        try:
            from tqdm.asyncio import tqdm

            tasks = [bounded_process(r) for r in records]
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering"):
                results.append(await f)
        except ImportError:
            results = await asyncio.gather(*(bounded_process(r) for r in records))

        # Append when resuming, overwrite otherwise
        write_mode = "a" if args.resume and done_keys else "w"
        with open(output_path, write_mode, encoding="utf-8") as f:
            for r in results:
                if args.filter_only and r.get("relevance", {}).get("filtered", False):
                    continue
                f.write(json.dumps(r) + "\n")

        if args.use_llm:
            from scripts.data.generation_v3.costs import aggregate, report
            totals = aggregate(results)
            report(output_path.name, totals, batch=False)

    # print some summary stats
    total = len(results)
    filtered = sum(1 for r in results if r.get("relevance", {}).get("filtered", False))
    print(f"Total records: {total}")
    if total:
        print(f"Filtered out: {filtered} ({filtered/total:.2%})")


DEFAULT_KEYWORDS = [
    "food insecurity",
    "famine",
    "starvation",
    "malnutrition",
    "undernourished",
    "hunger",
    "drought",
    "crop failure",
    "food shortage",
    "starve",
    "food price",
    "food crisis",
    "acute food",
    "food assistance",
    "food aid",
    "food rationing",
    "locust",
    "flood",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run relevance gate on articles standalone."
    )
    parser.add_argument("--input", required=True, type=str, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL file")
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash", help="Model name"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters to use for relevance",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold to filter",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent requests"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Do not write filtered records to output",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="List of keywords to filter by (case-insensitive). Only used when --use-keywords is provided.",
    )
    parser.add_argument(
        "--use-keywords",
        action="store_true",
        help="Filter articles by keyword match (see --keywords) before any other processing. If not provided, keyword filtering is skipped entirely.",
    )
    parser.add_argument(
        "--use-regex",
        action="store_true",
        help="Use the precompiled food_insecurity_regex instead of the keyword list. This is faster.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for relevance classification.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in the output file and append new results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file. Required if the output file "
        "already exists and --resume is not set.",
    )
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Use Gemini Batch API instead of streaming (~50%% off, but async/slower).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records per batch job chunk (default: 1000).",
    )
    parser.add_argument(
        "--batch-poll-interval-seconds",
        type=int,
        default=30,
        help="Seconds between batch job status polls (default: 30).",
    )
    args = parser.parse_args()

    # Import log_llm_call if needed globally, but it's handled inside classify_article_relevance
    asyncio.run(process_file(args))


if __name__ == "__main__":
    main()
