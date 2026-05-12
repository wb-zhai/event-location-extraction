import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.llms.llm_client import GeminiLLMClient

DEFAULT_RELEVANCE_SYSTEM_PROMPT = """<role>
You are a high-recall relevance gate for food-security risk-event extraction.
</role>

<goal>
Decide whether the article is likely to contain explicit food-insecurity events or risk-factor evidence worth sending to the full extraction pipeline.
</goal>

<policy>
- Favor recall over precision.
- If the article is borderline, ambiguous, or only partially visible in the preview, mark it relevant.
- Mark it irrelevant only when the title and preview strongly indicate the article is outside food insecurity and risk-factor event detection.
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
- is_relevant=true if the article likely contains explicit food-insecurity events, shocks, hazards, conflict, displacement, market disruption, pests/disease, climate/weather shocks, or other risk-factor evidence.
- If uncertain, return is_relevant=true.
- is_relevant=false only when the article is clearly unrelated.
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


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ..., description="True when the article should proceed to full extraction."
    )
    confidence: float = Field(
        default=0.0, description="Confidence from 0.0 to 1.0 in the relevance decision."
    )
    reason: str = Field(default="")


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
        override_settings={"temperature": 0.0, "max_output_tokens": 256},
        response_format={
            "is_relevant": bool,
            "confidence": float,
            "reason": str,
        },
        add_cot_field=False,
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
        elif args.keywords:
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
    except Exception as e:
        record["relevance"] = {"error": str(e), "filtered": False}
        print(f"Error processing record {record_id}: {e}")

    return record


async def process_file(args):
    client = None
    if args.use_llm:
        client = GeminiLLMClient(model_name=args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

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

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if args.filter_only and r.get("relevance", {}).get("filtered", False):
                continue
            f.write(json.dumps(r) + "\n")

    # print some summary stats
    total = len(results)
    filtered = sum(1 for r in results if r.get("relevance", {}).get("filtered", False))
    print(f"Total records: {total}")
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
        default=3000,
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
        help="List of keywords to filter by (case-insensitive). If provided, only articles containing at least one keyword will be kept. Ignored if --use-regex is provided.",
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
    args = parser.parse_args()

    # Import log_llm_call if needed globally, but it's handled inside classify_article_relevance
    asyncio.run(process_file(args))


if __name__ == "__main__":
    main()
