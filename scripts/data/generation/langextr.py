from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import langextract as lx

from langextract import schema


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.data.generation.gemini_event_gen import (  # noqa: E402
    DEFAULT_MODEL,
    iter_jsonl,
    load_env_file,
    load_json_tolerant,
    load_records,
    make_source,
    normalize_argument_roles,
    normalize_event_argument_roles,
    normalize_ontology,
)
from scripts.data.generation.validation import find_offsets  # noqa: E402

DEFAULT_ONTOLOGY = Path("ontologies/risk-factors/risk.cluster.description.json")
DEFAULT_EXAMPLES = Path("dataset/manual/manual_fixes.jsonl")


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_input_records(path: Path | None, text: str | None) -> list[dict[str, Any]]:
    if text is not None:
        return [
            {
                "id": "text_0",
                "title": "",
                "text": text,
                "source_url": None,
                "publish_date": None,
            }
        ]

    if path is None:
        raise ValueError("Provide either --text or --input.")

    path = resolve_path(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return load_records(path)

    raw_text = path.read_text(encoding="utf-8")
    return [
        {
            "id": path.stem or "text_0",
            "title": "",
            "text": raw_text,
            "source_url": str(path),
            "publish_date": None,
        }
    ]


def load_ontology(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    raw = load_json_tolerant(resolve_path(path))
    events = normalize_ontology(raw)
    argument_roles = normalize_argument_roles(raw)
    event_argument_roles = normalize_event_argument_roles(
        raw, set(events), set(argument_roles)
    )
    return events, argument_roles, event_argument_roles


def format_allowed_schema(
    events: dict[str, str],
    argument_roles: dict[str, str],
    event_argument_roles: dict[str, list[str]],
) -> str:
    lines = []
    for event, description in sorted(events.items()):
        roles = event_argument_roles.get(event, [])
        role_text = ", ".join(
            (f"{role} ({argument_roles[role]})" if argument_roles.get(role) else role)
            for role in roles
        )
        lines.append(f"- {event}: {description} | roles: {role_text or 'none'}")
    return "\n".join(lines)


def gemini_response_schema(
    events: dict[str, str],
    event_argument_roles: dict[str, list[str]],
) -> dict[str, Any]:
    extraction_properties: dict[str, Any] = {}
    for event in sorted(events):
        extraction_properties[event] = {"type": "string"}
        roles = event_argument_roles.get(event, [])
        role_properties = {
            role: {"type": "string"}
            for role in sorted(roles)
        }
        if not role_properties:
            role_properties["_unused"] = {"type": "string"}
        extraction_properties[f"{event}_attributes"] = {
            "type": "object",
            "properties": role_properties,
            "nullable": True,
        }

    return {
        "type": "object",
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": extraction_properties,
                },
            },
        },
        "required": ["extractions"],
    }


def prompt_description(
    events: dict[str, str],
    argument_roles: dict[str, str],
    event_argument_roles: dict[str, list[str]],
) -> str:
    return f"""Extract explicit food-insecurity and risk-factor events from the document.

Rules:
- Use only labels and roles from the schema.
- extraction_text must be the narrowest contiguous trigger phrase copied verbatim from the document.
- Argument values must be exact contiguous document spans linked to the trigger.
- Do not infer, paraphrase, combine non-contiguous text, or output duplicates.
- Keep locations, participants, dates, attribution, sources, and targets out of the trigger unless essential.
- If no valid event is explicit, return zero extractions.

Schema:
{format_allowed_schema(events, argument_roles, event_argument_roles)}
"""


def record_text(record: dict[str, Any]) -> str:
    source = record.get("source")
    if isinstance(source, dict):
        return str(source.get("text") or record.get("text") or "")
    return str(record.get("text") or record.get("body") or "")


def record_title(record: dict[str, Any]) -> str:
    source = record.get("source")
    if isinstance(source, dict):
        return str(source.get("title") or record.get("title") or "")
    return str(record.get("title") or "")


def record_source(record: dict[str, Any], title: str, text: str) -> dict[str, Any]:
    source = make_source(record, title, text)
    raw_source = record.get("source")
    if not isinstance(raw_source, dict):
        return source
    source["source_url"] = raw_source.get("source_url") or source.get("source_url")
    source["publish_date"] = raw_source.get("publish_date") or source.get(
        "publish_date"
    )
    return source


def example_attributes(event: dict[str, Any]) -> dict[str, str | list[str]]:
    attributes: dict[str, str | list[str]] = {}
    by_role: dict[str, list[str]] = {}
    for argument in event.get("arguments", []) or []:
        if not isinstance(argument, dict):
            continue
        role = str(argument.get("role") or "").strip()
        text = str(argument.get("text") or "").strip()
        if role and text:
            by_role.setdefault(role, []).append(text)

    for role, texts in by_role.items():
        attributes[role] = texts[0] if len(texts) == 1 else texts

    return attributes


def langextract_example_json(example: Any) -> dict[str, Any]:
    return {
        "text": str(getattr(example, "text", "")),
        "extractions": [
            {
                "extraction_class": str(
                    getattr(extraction, "extraction_class", "")
                ),
                "extraction_text": str(getattr(extraction, "extraction_text", "")),
                "attributes": getattr(extraction, "attributes", None) or {},
            }
            for extraction in getattr(example, "extractions", []) or []
        ],
    }


def load_examples(path: Path) -> list[Any]:
    examples: list[Any] = []
    for record in iter_jsonl(resolve_path(path)):
        text = record_text(record)
        if not text:
            continue

        extractions = []
        for event in record.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or "").strip()
            trigger_text = str(event.get("trigger_text") or "").strip()
            if not event_type or not trigger_text or trigger_text not in text:
                continue
            extractions.append(
                lx.data.Extraction(
                    extraction_class=event_type,
                    extraction_text=trigger_text,
                    attributes=example_attributes(event),
                )
            )

        if extractions:
            example = lx.data.ExampleData(text=text, extractions=extractions)
            if not examples:
                print("Example format:")
                print(
                    json.dumps(
                        langextract_example_json(example),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file=sys.stderr,
                )
            examples.append(example)

    if not examples:
        raise ValueError(f"No usable LangExtract examples found in {path}.")

    # Define custom schema
    # custom_schema = schema.StructuredSchema.from_examples(examples)
    # # Print the custom schema for debugging
    # print("Inferred custom schema from examples:")
    # print(custom_schema)

    return examples


def sample_examples(
    examples: list[Any], sample_size: int | None, seed: int | None
) -> list[Any]:
    if sample_size is None:
        return examples
    if sample_size >= len(examples):
        return examples
    return random.Random(seed).sample(examples, sample_size)


def attribute_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def extraction_offsets(extraction: Any) -> tuple[int, int] | None:
    interval = getattr(extraction, "char_interval", None)
    if interval is None:
        return None
    start = getattr(interval, "start_pos", None)
    end = getattr(interval, "end_pos", None)
    if start is None or end is None:
        return None
    return int(start), int(end)


def convert_extractions(
    extractions: list[Any],
    text: str,
    labels: set[str],
    argument_roles: set[str],
    event_argument_roles: dict[str, list[str]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    for extraction in extractions:
        event_type = str(getattr(extraction, "extraction_class", "")).strip()
        trigger_text = str(getattr(extraction, "extraction_text", "")).strip()
        offsets = extraction_offsets(extraction)
        if event_type not in labels or not trigger_text or offsets is None:
            continue

        start_char, end_char = offsets
        if not (0 <= start_char < end_char <= len(text)):
            continue
        if text[start_char:end_char] != trigger_text:
            repaired = find_offsets(trigger_text, text, start_char, end_char)
            if repaired[0] < 0:
                continue
            start_char, end_char = repaired

        key = (start_char, end_char, event_type)
        if key in seen:
            continue
        seen.add(key)

        attributes = getattr(extraction, "attributes", None) or {}
        if not isinstance(attributes, dict):
            attributes = {}

        allowed_roles = set(event_argument_roles.get(event_type, argument_roles))
        arguments: list[dict[str, Any]] = []
        seen_arguments: set[tuple[int, int, str]] = set()
        for role in sorted(allowed_roles):
            for argument_text in attribute_values(attributes.get(role)):
                arg_start, arg_end = find_offsets(argument_text, text, -1, -1)
                if arg_start < 0:
                    continue
                argument_key = (arg_start, arg_end, role)
                if argument_key in seen_arguments:
                    continue
                seen_arguments.add(argument_key)
                arguments.append(
                    {
                        "role": role,
                        "text": text[arg_start:arg_end],
                        "start_char": arg_start,
                        "end_char": arg_end,
                    }
                )

        event = {
            "event_type": event_type,
            "trigger_text": text[start_char:end_char],
            "start_char": start_char,
            "end_char": end_char,
            "arguments": arguments,
        }
        events.append(event)

    return events


def annotate_record(
    record: dict[str, Any],
    prompt: str,
    examples: list[Any],
    model: str,
    response_schema: dict[str, Any],
    labels: set[str],
    argument_roles: set[str],
    event_argument_roles: dict[str, list[str]],
    max_workers: int,
    max_char_buffer: int,
    extraction_passes: int,
    context_window_chars: int | None = None,
) -> tuple[dict[str, Any], Any | None]:
    record_id = str(record.get("id") or "record")
    title = record_title(record)
    text = record_text(record)
    source = record_source(record, title, text)

    try:
        config = lx.factory.ModelConfig(
            model_id=model,
            provider_kwargs={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            config=config,
            use_schema_constraints=False,
            fence_output=False,
            max_workers=max_workers,
            max_char_buffer=max_char_buffer,
            extraction_passes=extraction_passes,
            # prompt_validation_level=lx.prompt_validation.PromptValidationLevel.OFF,
            show_progress=True,
            context_window_chars=context_window_chars,
        )
        events = convert_extractions(
            list(result.extractions or []),
            text,
            labels,
            argument_roles,
            event_argument_roles,
        )
        return (
            {
                "id": record_id,
                "status": "ok",
                "source": source,
                "events": events,
                "llm": {"model": model, "output_mode": "events-with-args"},
            },
            result,
        )
    except Exception as exc:
        return (
            {
                "id": record_id,
                "status": "error",
                "error": str(exc),
                "source": source,
            },
            None,
        )


def write_visualization(lx: Any, documents: list[Any], html_path: Path) -> None:
    if not documents:
        return
    html_path = resolve_path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = html_path.with_suffix(".langextract.jsonl")
    lx.io.save_annotated_documents(
        documents,
        output_name=jsonl_path.name,
        output_dir=str(jsonl_path.parent),
    )
    html_content = lx.visualize(str(jsonl_path))
    if hasattr(html_content, "data"):
        html_content = html_content.data
    html_path.write_text(str(html_content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate food-security risk events with Google LangExtract."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw text to annotate as one document.")
    input_group.add_argument(
        "--input",
        type=Path,
        help="Input .json/.jsonl file, or a raw text file.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--example-sample-size",
        type=int,
        default=None,
        help="Randomly sample N usable examples. Defaults to all examples.",
    )
    parser.add_argument(
        "--example-seed",
        type=int,
        default=None,
        help="Optional seed for --example-sample-size.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-char-buffer", type=int, default=1000)
    parser.add_argument("--extraction-passes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.example_sample_size is not None and args.example_sample_size < 1:
        raise ValueError("--example-sample-size must be >= 1")

    load_env_file(resolve_path(args.env_file))

    records = load_input_records(args.input, args.text)
    if args.limit is not None:
        records = records[: args.limit]

    events, argument_roles, event_argument_roles = load_ontology(args.ontology)
    response_schema = gemini_response_schema(events, event_argument_roles)
    examples = sample_examples(
        load_examples(args.examples),
        args.example_sample_size,
        args.example_seed,
    )
    prompt = prompt_description(events, argument_roles, event_argument_roles)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    annotated_documents: list[Any] = []
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            result, annotated_document = annotate_record(
                record=record,
                prompt=prompt,
                examples=examples,
                model=args.model,
                response_schema=response_schema,
                labels=set(events),
                argument_roles=set(argument_roles),
                event_argument_roles=event_argument_roles,
                max_workers=args.max_workers,
                max_char_buffer=args.max_char_buffer,
                extraction_passes=args.extraction_passes,
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if annotated_document is not None:
                annotated_documents.append(annotated_document)

    if args.html is not None:
        write_visualization(lx, annotated_documents, args.html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
