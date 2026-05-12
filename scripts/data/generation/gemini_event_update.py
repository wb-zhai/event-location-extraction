import argparse
import asyncio
import json
import logging
import os
import random
import time
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import GeminiLLMClient
from scripts.data.generation import gemini_event_gen

DEFAULT_MODEL = "gemini-2.5-flash"
LOGGER = logging.getLogger("gemini_event_update")

DEFAULT_SYSTEM_PROMPT = """<role>
You are a highly analytical data extractor and data updater for food-security news.
Your task is to take existing annotations and an updated ontology, and refine the annotations accordingly.
</role>

<goal>
Output updated annotations grounded entirely in the article text and the provided ontology. 
Focus primarily on fixing the event labels and argument labels (e.g., changing location types like country, city, etc.) to match the new ontology.
Avoid aggressively extracting new events unless they are blatantly missing and highly relevant.
</goal>

<grounding>
- Use only the user-provided article text as evidence.
- Existing annotations are provided as a starting point. Review each, adjust its event label strictly to the new ontology, adjust its argument roles and labels (such as location types) to match the new ontology, and adjust its offset/trigger to be optimal. Drop it if it is no longer relevant under the new ontology.
- Focus on correcting existing annotations rather than finding new ones.
- The offsets in your output must precisely match exactly the substrings from the text.
</grounding>

<quality_checks>
1. Event and argument labels must be in the new ontology.
2. Copied text must be verbatim and contiguous.
3. Keep narrow core trigger phrases.
4. Extracted events and arguments must have correct offsets.
</quality_checks>

<output_constraints>
Return the complete updated list of annotations.
</output_constraints>"""

DEFAULT_USER_PROMPT = """<context>
<title usage="context_only_do_not_offset">
{title}
</title>

<article_text>
{text}
</article_text>

<new_ontology allowed_labels_only="true">
{ontology}
</new_ontology>

<current_annotations>
{current_annotations}
</current_annotations>
</context>

<task>
Update, correct, or add to the current annotations based on the provided new_ontology and article text.
</task>
"""


class UpdatedSpan(BaseModel):
    span_text: str = Field(..., description="Verbatim span from article.")
    label: str = Field(..., description="Label from new ontology.")
    start_char: int = Field(..., description="Start character offset.")
    end_char: int = Field(..., description="Exclusive end character offset.")


class UpdatedArgument(BaseModel):
    role: str = Field(..., description="Argument role.")
    text: str = Field(..., description="Verbatim argument span.")
    start_char: int = Field(..., description="Start character offset.")
    end_char: int = Field(..., description="Exclusive end character offset.")
    location_type: str | None = Field(default=None)


class UpdatedEvent(BaseModel):
    event_type: str = Field(..., description="Event label from new ontology.")
    trigger_text: str = Field(..., description="Verbatim trigger.")
    start_char: int = Field(..., description="Start offset.")
    end_char: int = Field(..., description="End offset.")
    arguments: list[UpdatedArgument] = Field(default_factory=list)


async def generate_update(
    client: GeminiLLMClient,
    record: dict[str, Any],
    ontology_text: str,
    labels: set[str],
    system_prompt: str,
    user_prompt_template: str,
    output_mode: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    verbose: bool,
    strict_offsets: bool,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
) -> dict[str, Any]:
    text = str(record.get("text", ""))
    title = str(record.get("title", ""))
    record_id = str(record.get("id"))

    current_annotations_str = ""
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        current_annotations_str = json.dumps(
            record.get("spans", []), ensure_ascii=False, indent=2
        )
    else:
        current_annotations_str = json.dumps(
            record.get("events", []), ensure_ascii=False, indent=2
        )

    prompt = user_prompt_template.format(
        title=title,
        text=text,
        ontology=ontology_text,
        current_annotations=current_annotations_str,
    )

    response_format = (
        {"spans": list[UpdatedSpan]}
        if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS
        else {"events": list[UpdatedEvent]}
    )

    response = None
    for attempt in range(max_retries + 1):
        try:
            gemini_event_gen.log_llm_call(
                enabled=verbose,
                record_id=record_id,
                step="update",
                call_type=f"update attempt={attempt + 1}",
                system_prompt=system_prompt,
                prompt=prompt,
            )
            async for candidate in client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                add_cot_field=False,
            ):
                response = candidate
                break

            if not response:
                raise RuntimeError("No response from LLM")
            break
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = min(max_backoff, initial_backoff * (2**attempt)) * (
                0.5 + random.random()
            )
            LOGGER.warning(
                "Retrying id=%s after error on attempt %s/%s: %s",
                record_id,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            await asyncio.sleep(delay)

    raw_answer = (
        gemini_event_gen.response_to_dict(response.parsed)
        if response.parsed
        else json.loads(response.text)
    )
    gemini_event_gen.log_llm_call(
        enabled=verbose,
        record_id=record_id,
        step="update",
        call_type="update",
        system_prompt=system_prompt,
        prompt=prompt,
        answer=raw_answer,
    )

    updated_record = dict(record)
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        updated_record["spans"] = gemini_event_gen.clean_spans(
            raw_answer, text, labels, strict_offsets
        )
    else:
        updated_record["events"] = gemini_event_gen.clean_events_with_args(
            raw_answer,
            text,
            labels,
            strict_offsets,
            argument_roles=argument_roles or set(),
            event_argument_roles=event_argument_roles or {},
            location_types=location_types,
        )

    return updated_record


async def process_batch(
    client: GeminiLLMClient,
    batch: list[dict[str, Any]],
    output_path: Path,
    ontology_text: str,
    labels: set[str],
    system_prompt: str,
    user_prompt_template: str,
    output_mode: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    verbose: bool,
    strict_offsets: bool,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
) -> None:
    tasks = [
        generate_update(
            client=client,
            record=record,
            ontology_text=ontology_text,
            labels=labels,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_mode=output_mode,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            verbose=verbose,
            strict_offsets=strict_offsets,
            argument_roles=argument_roles,
            event_argument_roles=event_argument_roles,
            location_types=location_types,
        )
        for record in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for idx, result in enumerate(results):
            record = batch[idx]
            if isinstance(result, Exception):
                LOGGER.error("Failed to process id=%s: %s", record.get("id"), result)
                # Save the original or a failed state
                record["status"] = "error"
                record["error"] = str(result)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                result["status"] = "ok"
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace) -> None:
    gemini_event_gen.load_env_file(REPO_ROOT / ".env")

    client = GeminiLLMClient(
        model_name=args.model,
        system_prompt=None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    raw_ontology = gemini_event_gen.load_json_tolerant(args.ontology)
    ontology = gemini_event_gen.normalize_ontology(raw_ontology)
    ontology_text = gemini_event_gen.format_ontology(ontology)
    labels = set(ontology.keys())

    argument_roles, event_argument_roles, location_types = None, None, None
    if args.output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS:
        raw_roles = gemini_event_gen.normalize_argument_roles(raw_ontology)
        argument_roles = set(raw_roles.keys())
        raw_event_roles = gemini_event_gen.normalize_event_argument_roles(
            raw_ontology, labels, argument_roles
        )
        event_argument_roles = raw_event_roles
        location_types = set(
            gemini_event_gen.normalize_location_types(raw_ontology).keys()
        )

    records = gemini_event_gen.load_records(args.input)
    done_ids = gemini_event_gen.completed_ids(args.output, False)
    pending = [r for r in records if str(r.get("id")) not in done_ids]

    if args.limit:
        pending = pending[: args.limit]

    LOGGER.info(
        "Processing %s records (skipped %s done)",
        len(pending),
        len(records) - len(pending),
    )

    system_prompt = gemini_event_gen.load_prompt(
        args.system_prompt, DEFAULT_SYSTEM_PROMPT
    )
    user_prompt = gemini_event_gen.load_prompt(args.user_prompt, DEFAULT_USER_PROMPT)

    with tqdm(total=len(pending), desc="Updating annotations") as pbar:
        for i in range(0, len(pending), args.workers):
            batch = pending[i : i + args.workers]
            await process_batch(
                client=client,
                batch=batch,
                output_path=args.output,
                ontology_text=ontology_text,
                labels=labels,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt,
                output_mode=args.output_mode,
                max_retries=args.max_retries,
                initial_backoff=args.initial_backoff,
                max_backoff=args.max_backoff,
                verbose=args.verbose,
                strict_offsets=not args.fast_offsets,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                location_types=location_types,
            )
            pbar.update(len(batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update existing annotations with a new prompt/ontology."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL/JSON with records and existing spans/events.",
    )
    parser.add_argument("output", type=Path, help="Output JSONL.")
    parser.add_argument(
        "--ontology", type=Path, required=True, help="New ontology JSON file."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--output-mode",
        choices=gemini_event_gen.OUTPUT_MODES,
        default=gemini_event_gen.OUTPUT_MODE_SPANS,
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--user-prompt", type=Path, default=None)
    parser.add_argument("--fast-offsets", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.monotonic()
    asyncio.run(run(args))
    LOGGER.info("finished in %.1fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
