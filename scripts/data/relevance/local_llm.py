"""Run the relevance gate against a local OpenAI-compatible LLM server.

Point this at a local llama.cpp `llama-server` (or any other OpenAI-compatible
endpoint, e.g. vLLM) to test small local models — e.g. LiquidAI/LFM2.5-350M — as a
stand-in for the Gemini relevance gate in `relevance_filter.py`.

Small models struggle with the full Gemini prompt (long policy text, free-form
`reason`/`confidence` fields they can't ground). This script uses a deliberately
minimal prompt and a boolean-only `{is_relevant}` schema instead — self-contained
here rather than sharing `relevance_filter.py`'s schema, since that one is tuned
for a much larger reasoning model. Output still has the same `relevance` shape
(`decision`, `is_relevant`, `filtered`, `model`, ...), so it's diffable against a
Gemini-labeled file with `agreement.py`.

    # 1. Serve the model (see scripts/data/relevance/README.md for OS-specific commands):
    #    llama-server -hf LiquidAI/LFM2.5-350M-GGUF --host 127.0.0.1 --port 8080

    python scripts/data/relevance/local.py \\
        --input dataset/zhai/v3/articles.jsonl \\
        --output /tmp/articles.local.jsonl \\
        --model LFM2.5-350M --base-url http://127.0.0.1:8080/v1 --limit 50

    python scripts/data/relevance/agreement.py \\
        --a dataset/zhai/v3/articles.gemini.jsonl --b /tmp/articles.local.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import OpenAILLMClient
from scripts.data.relevance.relevance_filter import _record_key, food_insecurity_regex

SIMPLE_SYSTEM_PROMPT = """You classify news articles for a food-security monitoring system.

Relevant topics: famine, hunger, malnutrition, food shortages, food prices, drought, flood, \
crop failure, crop pests or disease, war or armed conflict, refugees or displacement, \
humanitarian aid, political instability causing food problems.

Not relevant: sports, entertainment, celebrity news, general business or tech, general politics, \
dictionary or word definitions, or anything that only mentions a relevant topic in passing."""

SIMPLE_USER_PROMPT_TEMPLATE = """Title: {title}
Article: {text}

Is the MAIN subject of this article one of the relevant topics listed above?"""

SIMPLE_RESPONSE_FORMAT = {"is_relevant": bool}


async def classify_relevance_simple(
    client: OpenAILLMClient,
    title: str,
    text: str,
    max_chars: int,
    override_settings: dict[str, Any],
) -> dict[str, Any]:
    preview_text = text[:max_chars]
    prompt = SIMPLE_USER_PROMPT_TEMPLATE.format(title=title, text=preview_text)

    response = None
    async for candidate in client.generate(
        prompt=prompt,
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        override_settings=override_settings,
        response_format=SIMPLE_RESPONSE_FORMAT,
        add_cot_field=False,
    ):
        response = candidate
        break
    if response is None:
        raise RuntimeError("Local model returned no relevance response.")

    is_relevant = bool(getattr(response.parsed, "is_relevant", True))
    return {
        "decision": "relevant" if is_relevant else "irrelevant",
        "is_relevant": is_relevant,
        "filtered": not is_relevant,
        "model": client.model_name,
        "max_chars": max_chars,
        "text_chars_used": len(preview_text),
        "metadata": response.metadata,
    }


async def process_record(client: OpenAILLMClient, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    record_id = str(record.get("id", record.get("url", "")))

    try:
        if args.use_regex and not food_insecurity_regex.search(title + " " + text):
            relevance_info = {
                "decision": "irrelevant",
                "is_relevant": False,
                "filtered": True,
                "model": "regex",
            }
        else:
            relevance_info = await classify_relevance_simple(
                client=client, title=title, text=text, max_chars=args.max_chars,
                override_settings=args.override_settings,
            )
        record["relevance"] = relevance_info
    except Exception as e:
        record["relevance"] = {"error": str(e), "filtered": False}
        print(f"Error processing record {record_id}: {e}")

    return record


async def process_file(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output file {output_path} already exists. Use --resume to continue "
            "or --overwrite to replace it."
        )

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

    client = OpenAILLMClient(
        model_name=args.model, base_url=args.base_url, api_key=args.api_key, system_prompt=None
    )
    sem = asyncio.Semaphore(args.concurrency)

    write_mode = "a" if args.resume and done_keys else "w"
    out_f = open(output_path, write_mode, encoding="utf-8")
    write_lock = asyncio.Lock()
    counts = {"total": 0, "filtered": 0}

    async def bounded_process(record: dict[str, Any]) -> None:
        async with sem:
            result = await process_record(client, record, args)
        is_filtered = bool(result.get("relevance", {}).get("filtered", False))
        async with write_lock:
            if not (args.filter_only and is_filtered):
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            counts["total"] += 1
            counts["filtered"] += int(is_filtered)

    tasks = [bounded_process(r) for r in records]
    try:
        from tqdm.asyncio import tqdm

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering"):
            await coro
    except ImportError:
        for coro in asyncio.as_completed(tasks):
            await coro
    finally:
        out_f.close()

    total = counts["total"]
    filtered = counts["filtered"]
    print(f"Total records: {total}")
    if total:
        print(f"Filtered out: {filtered} ({filtered/total:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the relevance gate against a local OpenAI-compatible LLM server."
    )
    parser.add_argument("--input", required=True, type=str, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="LFM2.5-350M", help="Model name as reported by the server")
    parser.add_argument(
        "--base-url", type=str, default="http://127.0.0.1:8080/v1", help="OpenAI-compatible server base URL"
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key, if the server checks one")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max characters to use for relevance")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--filter-only", action="store_true", help="Do not write filtered records to output")
    parser.add_argument(
        "--use-regex",
        action="store_true",
        help="Pre-filter with the food_insecurity_regex before calling the local model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process.")
    parser.add_argument(
        "--resume", action="store_true", help="Skip records already present in the output file and append new results."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file. Required if the output file "
        "already exists and --resume is not set.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (llama.cpp/vLLM extra)")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling (llama.cpp/vLLM extra)")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (llama.cpp/vLLM extra)")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty (standard OpenAI sampling param)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle Qwen3 thinking mode via chat_template_kwargs (--no-enable-thinking to disable)",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate for the decision")
    args = parser.parse_args()

    override_settings: dict[str, Any] = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    for flag, key in (
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("repetition_penalty", "repeat_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("seed", "seed"),
        ("enable_thinking", "enable_thinking"),
    ):
        value = getattr(args, flag)
        if value is not None:
            override_settings[key] = value
    args.override_settings = override_settings

    asyncio.run(process_file(args))


if __name__ == "__main__":
    main()
