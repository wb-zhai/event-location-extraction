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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
- Treat locations, participants, sources, targets, dates, and attribution as context, not event trigger text.
- If an event is expressed by a non-contiguous noun plus predicate, select the contiguous trigger predicate rather than the whole clause.

Examples of narrow spans:
- "the current desert locust outbreak in the Horn of Africa" -> "desert locust outbreak"
- "recent drought crisis in South Africa's Cape Town region" -> "drought crisis"
- "missiles from Gaza again raining down on Israel" -> "raining down"

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
- Treat locations, participants, sources, targets, dates, and attribution as context, not event trigger text.
- If an event is expressed by a non-contiguous noun plus predicate, select the contiguous trigger predicate rather than the whole clause.
- Do not output duplicates.
- If no valid evidence exists, return an empty spans list.

Narrow-span examples:
- "the current desert locust outbreak in the Horn of Africa" -> "desert locust outbreak"
- "recent drought crisis in South Africa's Cape Town region" -> "drought crisis"
- "missiles from Gaza again raining down on Israel" -> "raining down"

Offset rules:
- start_char and end_char must index article text only (not title).
- end_char is exclusive.
- Offsets must match span_text exactly.

Title (context only, do not offset against this field):
{title}

Article text (offset source):
{text}
"""

DEFAULT_EVENTS_WITH_ARGS_USER_PROMPT = """Task: Extract food-insecurity and risk-factor events from the article, including linked argument spans.

Ontology labels (allowed event labels only):
{ontology}

Extraction rules:
- Select only explicit evidence from the article text.
- trigger_text and argument text must be exact contiguous substrings from article text.
- Each event gets exactly one ontology event_type.
- Event trigger = the minimal contiguous word or phrase that evokes the event/risk.
- Arguments = linked entities, locations, times, participants, affected populations, sources, or targets.
- Do not include locations or participants inside trigger_text unless they are part of the trigger expression itself.
- Do not infer arguments that are not explicit in the article text.
- Do not output duplicates.
- If no valid evidence exists, return an empty events list.

Argument roles (allowed roles only):
{argument_roles}

Allowed roles by event label:
{event_argument_roles}

Examples:
- "the current desert locust outbreak in the Horn of Africa" -> trigger_text "desert locust outbreak", argument "Horn of Africa" with role "location"
- "recent drought crisis in South Africa's Cape Town region" -> trigger_text "drought crisis", argument "South Africa's Cape Town region" with role "location"
- "missiles from Gaza again raining down on Israel" -> trigger_text "raining down", argument "Gaza" with role "source_location", argument "Israel" with role "target_location"

Offset rules:
- start_char and end_char must index article text only (not title).
- end_char is exclusive.
- Offsets must match the copied text exactly.

Title (context only, do not offset against this field):
{title}

Article text (offset source):
{text}
"""

LOGGER = logging.getLogger("gemini_event_gen")
OUTPUT_MODE_SPANS = "spans"
OUTPUT_MODE_EVENTS_WITH_ARGS = "events-with-args"
OUTPUT_MODES = (OUTPUT_MODE_SPANS, OUTPUT_MODE_EVENTS_WITH_ARGS)


@dataclass(frozen=True)
class ArticleWindow:
    window_index: int
    window_count: int
    start_char: int
    end_char: int
    text: str
    overlap_prev: bool
    overlap_next: bool


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


class ExtractedArgument(BaseModel):
    role: str = Field(..., description="One role from the allowed argument roles.")
    text: str = Field(
        ..., description="Verbatim argument span copied from article text."
    )
    start_char: int = Field(..., description="Start character offset in article text.")
    end_char: int = Field(
        ..., description="Exclusive end character offset in article text."
    )


class ExtractedEvent(BaseModel):
    event_type: str = Field(..., description="One event label from the ontology.")
    trigger_text: str = Field(
        ..., description="Verbatim event trigger span copied from article text."
    )
    start_char: int = Field(..., description="Start character offset in article text.")
    end_char: int = Field(
        ..., description="Exclusive end character offset in article text."
    )
    arguments: list[ExtractedArgument] = Field(default_factory=list)
    rationale: str = Field(
        default="", description="Brief reason for assigning the event label."
    )


class SampleResult(BaseModel):
    spans: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
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


def normalize_argument_roles(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Ontology must be a JSON object to define argument_roles.")

    raw_roles = raw.get("argument_roles")
    if raw_roles is None:
        raise ValueError("Ontology must define 'argument_roles'.")
    if isinstance(raw_roles, dict):
        roles = {str(role): str(description) for role, description in raw_roles.items()}
    elif isinstance(raw_roles, list):
        roles = {str(role): "" for role in raw_roles}
    else:
        raise ValueError("'argument_roles' must be a JSON object or list.")

    if not roles:
        raise ValueError("'argument_roles' must not be empty.")
    return roles


def normalize_event_argument_roles(
    raw: Any, event_labels: set[str], argument_roles: set[str]
) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        raise ValueError(
            "Ontology must be a JSON object to define event_argument_roles."
        )
    if raw.get("event_argument_roles") is None:
        raise ValueError("Ontology must define 'event_argument_roles'.")

    raw_mapping = raw["event_argument_roles"]
    if not isinstance(raw_mapping, dict):
        raise ValueError("'event_argument_roles' must be a JSON object.")

    mapping: dict[str, list[str]] = {}
    for event, roles in raw_mapping.items():
        event = str(event)
        if event not in event_labels:
            continue
        if not isinstance(roles, list) or not all(
            isinstance(role, str) for role in roles
        ):
            raise ValueError(
                f"'event_argument_roles.{event}' must be a list of role strings."
            )
        unknown_roles = [role for role in roles if role not in argument_roles]
        if unknown_roles:
            raise ValueError(
                f"'event_argument_roles.{event}' contains unknown roles: {unknown_roles}"
            )
        mapping[event] = list(dict.fromkeys(roles))

    for event in event_labels:
        if event not in mapping:
            raise ValueError(f"'event_argument_roles' is missing event label: {event}")
    return mapping


def format_ontology(ontology: dict[str, str]) -> str:
    return "\n".join(
        f"- {label}: {description}" for label, description in sorted(ontology.items())
    )


def format_argument_roles(argument_roles: dict[str, str]) -> str:
    return "\n".join(
        f"- {role}: {description}" if description else f"- {role}"
        for role, description in sorted(argument_roles.items())
    )


def format_event_argument_roles(event_argument_roles: dict[str, list[str]]) -> str:
    return "\n".join(
        f"- {event}: {', '.join(roles) if roles else 'none'}"
        for event, roles in sorted(event_argument_roles.items())
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
        raise ValueError(
            f"Input JSON must contain an object or list of objects: {path}"
        )

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


def render_user_prompt(
    template: str,
    ontology_text: str,
    title: str,
    text: str,
    argument_roles_text: str = "",
    event_argument_roles_text: str = "",
) -> str:
    return template.format(
        ontology=ontology_text,
        argument_roles=argument_roles_text,
        event_argument_roles=event_argument_roles_text,
        title=title,
        text=text,
    )


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


def trim_span_to_non_whitespace(
    text: str, start_char: int, end_char: int
) -> tuple[int, int] | None:
    while start_char < end_char and text[start_char].isspace():
        start_char += 1
    while end_char > start_char and text[end_char - 1].isspace():
        end_char -= 1
    if start_char >= end_char:
        return None
    return start_char, end_char


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start_char = 0
    for match in re.finditer(r"\n\s*\n", text):
        trimmed = trim_span_to_non_whitespace(text, start_char, match.start())
        if trimmed is not None:
            spans.append(trimmed)
        start_char = match.end()

    trimmed = trim_span_to_non_whitespace(text, start_char, len(text))
    if trimmed is not None:
        spans.append(trimmed)
    return spans


def build_article_windows(text: str) -> list[ArticleWindow]:
    if not text:
        return []

    ranges = paragraph_spans(text)
    if not ranges:
        return []

    window_count = len(ranges)
    windows: list[ArticleWindow] = []
    for index, (start_char, end_char) in enumerate(ranges):
        windows.append(
            ArticleWindow(
                window_index=index,
                window_count=window_count,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
                overlap_prev=False,
                overlap_next=False,
            )
        )
    return windows


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


def clean_events_with_args(
    parsed: dict[str, Any],
    text: str,
    labels: set[str],
    strict_offsets: bool,
    argument_roles: set[str],
    event_argument_roles: dict[str, list[str]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    event_index_by_key: dict[tuple[int, int, str], int] = {}
    seen_arguments_by_event: dict[tuple[int, int, str], set[tuple[int, int, str]]] = {}

    for event in parsed.get("events", []) or []:
        if hasattr(event, "model_dump"):
            event = event.model_dump()
        if not isinstance(event, dict):
            continue

        trigger_text = str(event.get("trigger_text", "")).strip()
        event_type = str(event.get("event_type", "")).strip()
        if not trigger_text or event_type not in labels:
            continue

        try:
            start_char = int(event.get("start_char", -1))
            end_char = int(event.get("end_char", -1))
        except (TypeError, ValueError):
            start_char, end_char = -1, -1

        raw_start_char, raw_end_char = start_char, end_char
        start_char, end_char = find_offsets(trigger_text, text, start_char, end_char)
        if strict_offsets and start_char < 0:
            continue
        if start_char < 0:
            start_char, end_char = clamp_offsets(
                raw_start_char, raw_end_char, len(text)
            )

        event_key = (start_char, end_char, event_type)
        event_index = event_index_by_key.get(event_key)
        if event_index is None:
            event_index = len(cleaned)
            event_index_by_key[event_key] = event_index
            seen_arguments_by_event[event_key] = set()
            cleaned.append(
                {
                    "event_type": event_type,
                    "trigger_text": trigger_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "arguments": [],
                    "rationale": str(event.get("rationale", "")).strip(),
                }
            )

        arguments = event.get("arguments", []) or []
        if not isinstance(arguments, list):
            continue

        for argument in arguments:
            if hasattr(argument, "model_dump"):
                argument = argument.model_dump()
            if not isinstance(argument, dict):
                continue

            role = str(argument.get("role", "")).strip()
            argument_text = str(argument.get("text", "")).strip()
            allowed_roles = set(event_argument_roles.get(event_type, argument_roles))
            if role not in allowed_roles or not argument_text:
                continue

            try:
                arg_start_char = int(argument.get("start_char", -1))
                arg_end_char = int(argument.get("end_char", -1))
            except (TypeError, ValueError):
                arg_start_char, arg_end_char = -1, -1

            raw_arg_start_char, raw_arg_end_char = arg_start_char, arg_end_char
            arg_start_char, arg_end_char = find_offsets(
                argument_text, text, arg_start_char, arg_end_char
            )
            if strict_offsets and arg_start_char < 0:
                continue
            if arg_start_char < 0:
                arg_start_char, arg_end_char = clamp_offsets(
                    raw_arg_start_char, raw_arg_end_char, len(text)
                )

            argument_key = (arg_start_char, arg_end_char, role)
            if argument_key in seen_arguments_by_event[event_key]:
                continue
            seen_arguments_by_event[event_key].add(argument_key)
            cleaned[event_index]["arguments"].append(
                {
                    "role": role,
                    "text": argument_text,
                    "start_char": arg_start_char,
                    "end_char": arg_end_char,
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


def project_local_offsets_to_article(
    span_text: str,
    local_start_char: int,
    local_end_char: int,
    window: ArticleWindow,
    article_text: str,
) -> tuple[int, int] | None:
    if (
        0 <= local_start_char < local_end_char <= len(window.text)
        and window.text[local_start_char:local_end_char] == span_text
    ):
        global_start = window.start_char + local_start_char
        global_end = window.start_char + local_end_char
        if article_text[global_start:global_end] == span_text:
            return global_start, global_end

    local_start_char, local_end_char = find_offsets(
        span_text, window.text, local_start_char, local_end_char
    )
    if local_start_char < 0:
        return None

    global_start = window.start_char + local_start_char
    global_end = window.start_char + local_end_char
    if article_text[global_start:global_end] != span_text:
        return None
    return global_start, global_end


def project_window_spans(
    spans: list[dict[str, Any]], window: ArticleWindow, article_text: str
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for span in spans:
        offsets = project_local_offsets_to_article(
            str(span.get("span_text", "")),
            int(span.get("start_char", -1)),
            int(span.get("end_char", -1)),
            window,
            article_text,
        )
        if offsets is None:
            continue
        start_char, end_char = offsets
        projected.append(
            {
                **span,
                "start_char": start_char,
                "end_char": end_char,
                "window_indices": [window.window_index],
            }
        )
    return projected


def project_window_events(
    events: list[dict[str, Any]], window: ArticleWindow, article_text: str
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for event in events:
        event_offsets = project_local_offsets_to_article(
            str(event.get("trigger_text", "")),
            int(event.get("start_char", -1)),
            int(event.get("end_char", -1)),
            window,
            article_text,
        )
        if event_offsets is None:
            continue

        arguments: list[dict[str, Any]] = []
        for argument in event.get("arguments", []) or []:
            argument_offsets = project_local_offsets_to_article(
                str(argument.get("text", "")),
                int(argument.get("start_char", -1)),
                int(argument.get("end_char", -1)),
                window,
                article_text,
            )
            if argument_offsets is None:
                continue
            arg_start_char, arg_end_char = argument_offsets
            arguments.append(
                {
                    **argument,
                    "start_char": arg_start_char,
                    "end_char": arg_end_char,
                    "window_indices": [window.window_index],
                }
            )

        start_char, end_char = event_offsets
        projected.append(
            {
                **event,
                "start_char": start_char,
                "end_char": end_char,
                "arguments": arguments,
                "window_indices": [window.window_index],
            }
        )
    return projected


def merge_window_spans(
    spans: list[dict[str, Any]], article_text: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    for span in spans:
        key = (
            int(span["start_char"]),
            int(span["end_char"]),
            str(span["label"]),
        )
        if key not in grouped:
            grouped[key] = {
                **span,
                "span_text": article_text[key[0] : key[1]],
                "window_indices": [],
            }
        grouped[key]["window_indices"].extend(span.get("window_indices", []))

    merged: list[dict[str, Any]] = []
    for key in sorted(grouped):
        span = grouped[key]
        span["window_indices"] = sorted(set(span["window_indices"]))
        merged.append(span)
    return merged


def merge_window_events(
    events: list[dict[str, Any]], article_text: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
    seen_arguments: dict[
        tuple[int, int, str], set[tuple[int, int, str]]
    ] = defaultdict(set)

    for event in events:
        event_key = (
            int(event["start_char"]),
            int(event["end_char"]),
            str(event["event_type"]),
        )
        if event_key not in grouped:
            grouped[event_key] = {
                **event,
                "trigger_text": article_text[event_key[0] : event_key[1]],
                "arguments": [],
                "window_indices": [],
            }
        grouped[event_key]["window_indices"].extend(event.get("window_indices", []))

        for argument in event.get("arguments", []) or []:
            argument_key = (
                int(argument["start_char"]),
                int(argument["end_char"]),
                str(argument["role"]),
            )
            if argument_key in seen_arguments[event_key]:
                continue
            seen_arguments[event_key].add(argument_key)
            grouped[event_key]["arguments"].append(
                {
                    **argument,
                    "text": article_text[argument_key[0] : argument_key[1]],
                    "window_indices": sorted(
                        set(argument.get("window_indices", []))
                    ),
                }
            )

    merged: list[dict[str, Any]] = []
    for key in sorted(grouped):
        event = grouped[key]
        event["window_indices"] = sorted(set(event["window_indices"]))
        event["arguments"].sort(
            key=lambda argument: (
                int(argument["start_char"]),
                int(argument["end_char"]),
                str(argument["role"]),
            )
        )
        merged.append(event)
    return merged


def format_window_metadata(windows: list[ArticleWindow]) -> list[dict[str, Any]]:
    return [
        {
            "window_index": window.window_index,
            "window_count": window.window_count,
            "start_char": window.start_char,
            "end_char": window.end_char,
            "overlap_prev": window.overlap_prev,
            "overlap_next": window.overlap_next,
        }
        for window in windows
    ]


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


def most_common_first_seen(values: list[str]) -> str:
    if not values:
        return ""

    counts = Counter(values)
    value_index = max(
        range(len(values)),
        key=lambda index: (counts[values[index]], -index),
    )
    return values[value_index]


def merge_self_consistency_events(
    sample_events: list[list[dict[str, Any]]], text: str
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int, str], int],
    dict[tuple[int, int, str], dict[tuple[int, int, str], int]],
    int,
]:
    successful_samples = len(sample_events)
    threshold = (successful_samples // 2) + 1
    grouped_events: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    grouped_arguments: dict[
        tuple[int, int, str], dict[tuple[int, int, str], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))

    for events in sample_events:
        seen_events_in_sample: set[tuple[int, int, str]] = set()
        seen_arguments_in_sample: dict[
            tuple[int, int, str], set[tuple[int, int, str]]
        ] = defaultdict(set)

        for event in events:
            event_key = (
                int(event["start_char"]),
                int(event["end_char"]),
                str(event["event_type"]),
            )
            if event_key not in seen_events_in_sample:
                seen_events_in_sample.add(event_key)
                grouped_events[event_key].append(event)

            for argument in event.get("arguments", []) or []:
                argument_key = (
                    int(argument["start_char"]),
                    int(argument["end_char"]),
                    str(argument["role"]),
                )
                if argument_key in seen_arguments_in_sample[event_key]:
                    continue
                seen_arguments_in_sample[event_key].add(argument_key)
                grouped_arguments[event_key][argument_key].append(argument)

    event_support_by_key = {
        event_key: len(events) for event_key, events in grouped_events.items()
    }
    argument_support_by_event_key = {
        event_key: {
            argument_key: len(arguments)
            for argument_key, arguments in arguments_by_key.items()
        }
        for event_key, arguments_by_key in grouped_arguments.items()
    }

    merged: list[dict[str, Any]] = []
    for event_key in sorted(grouped_events):
        event_support = event_support_by_key[event_key]
        if event_support < threshold:
            continue

        start_char, end_char, event_type = event_key
        rationales = [
            str(event.get("rationale", "")).strip()
            for event in grouped_events[event_key]
            if str(event.get("rationale", "")).strip()
        ]
        argument_threshold = (event_support // 2) + 1
        arguments: list[dict[str, Any]] = []
        for argument_key in sorted(grouped_arguments[event_key]):
            argument_support = len(grouped_arguments[event_key][argument_key])
            if argument_support < argument_threshold:
                continue

            arg_start_char, arg_end_char, role = argument_key
            arguments.append(
                {
                    "role": role,
                    "text": text[arg_start_char:arg_end_char],
                    "start_char": arg_start_char,
                    "end_char": arg_end_char,
                    "support": argument_support,
                }
            )

        merged.append(
            {
                "event_type": event_type,
                "trigger_text": text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
                "arguments": arguments,
                "rationale": most_common_first_seen(rationales),
                "support": event_support,
            }
        )

    return merged, event_support_by_key, argument_support_by_event_key, threshold


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
    output_mode: str = OUTPUT_MODE_SPANS,
    argument_roles: set[str] | None = None,
    event_argument_roles: dict[str, list[str]] | None = None,
    override_settings: dict[str, Any] | None = None,
) -> SampleResult:
    if output_mode == OUTPUT_MODE_SPANS:
        response_format = {"spans": list[ExtractedSpan]}
    elif output_mode == OUTPUT_MODE_EVENTS_WITH_ARGS:
        response_format = {"events": list[ExtractedEvent]}
    else:
        raise ValueError(f"Unsupported output mode: {output_mode}")

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
            if output_mode == OUTPUT_MODE_SPANS:
                spans = clean_spans(parsed, text, labels, strict_offsets)
                return SampleResult(spans=spans, metadata=response.metadata)

            if argument_roles is None or event_argument_roles is None:
                raise ValueError(
                    "Argument roles are required with output_mode=events-with-args."
                )

            events = clean_events_with_args(
                parsed,
                text,
                labels,
                strict_offsets,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
            )
            return SampleResult(events=events, metadata=response.metadata)
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


async def generate_windowed_one(
    client: GeminiLLMClient,
    record_id: str,
    title: str,
    text: str,
    ontology_text: str,
    labels: set[str],
    system_prompt: str,
    user_prompt_template: str,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    strict_offsets: bool,
    output_mode: str,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    argument_roles_text: str,
    event_argument_roles_text: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    windows = build_article_windows(text)
    if not windows:
        windows = [
            ArticleWindow(
                window_index=0,
                window_count=1,
                start_char=0,
                end_char=len(text),
                text=text,
                overlap_prev=False,
                overlap_next=False,
            )
        ]

    sample_metadata: list[dict[str, Any]] = []
    projected_spans: list[dict[str, Any]] = []
    projected_events: list[dict[str, Any]] = []
    window_errors: list[dict[str, Any]] = []

    for window in windows:
        prompt = render_user_prompt(
            user_prompt_template,
            ontology_text,
            title,
            window.text,
            argument_roles_text=argument_roles_text,
            event_argument_roles_text=event_argument_roles_text,
        )
        try:
            sample = await generate_sample(
                client=client,
                prompt=prompt,
                text=window.text,
                labels=labels,
                system_prompt=system_prompt,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                strict_offsets=strict_offsets,
                record_id=f"{record_id}__w{window.window_index}",
                output_mode=output_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
            )
            sample_metadata.append(sample.metadata)
            if output_mode == OUTPUT_MODE_SPANS:
                projected_spans.extend(
                    project_window_spans(sample.spans, window, text)
                )
            else:
                projected_events.extend(
                    project_window_events(sample.events, window, text)
                )
        except Exception as exc:
            window_errors.append(
                {"window_index": window.window_index, "error": str(exc)}
            )

    if output_mode == OUTPUT_MODE_SPANS:
        payload = merge_window_spans(projected_spans, text)
    else:
        payload = merge_window_events(projected_events, text)

    metadata = combine_metadata(sample_metadata)
    metadata["long_document"] = {
        "enabled": True,
        "threshold_triggered": True,
        "window_boundary": "paragraph",
        "window_count": len(windows),
        "windows": format_window_metadata(windows),
        "window_errors": window_errors,
    }
    return payload, metadata


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
    output_mode: str = OUTPUT_MODE_SPANS,
    argument_roles: set[str] | None = None,
    event_argument_roles: dict[str, list[str]] | None = None,
    argument_roles_text: str = "",
    event_argument_roles_text: str = "",
    long_document_mode: bool = False,
    long_document_threshold_chars: int = 12000,
) -> dict[str, Any]:
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output mode: {output_mode}")

    record_id = str(record.get("id"))
    title = str(record.get("title") or "")
    text = str(record.get("text") or "")
    source = make_source(record, title, text)

    if long_document_mode and len(text) >= long_document_threshold_chars:
        if self_consistency:
            return {
                "id": record_id,
                "status": "error",
                "error": (
                    "Long-document windowing does not support "
                    "--self-consistency yet."
                ),
                "source": source,
            }
        try:
            payload, metadata = await generate_windowed_one(
                client=client,
                record_id=record_id,
                title=title,
                text=text,
                ontology_text=ontology_text,
                labels=labels,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                strict_offsets=strict_offsets,
                output_mode=output_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                argument_roles_text=argument_roles_text,
                event_argument_roles_text=event_argument_roles_text,
            )
            payload_key = "spans" if output_mode == OUTPUT_MODE_SPANS else "events"
            return {
                "id": record_id,
                "status": "ok",
                "source": source,
                payload_key: payload,
                "llm": {
                    "model": client.model_name,
                    "metadata": metadata,
                    "output_mode": output_mode,
                },
            }
        except Exception as exc:
            return {
                "id": record_id,
                "status": "error",
                "error": str(exc),
                "source": source,
            }

    prompt = render_user_prompt(
        user_prompt_template,
        ontology_text,
        title,
        text,
        argument_roles_text=argument_roles_text,
        event_argument_roles_text=event_argument_roles_text,
    )

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
                output_mode=output_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
            )
            payload_key = "spans" if output_mode == OUTPUT_MODE_SPANS else "events"
            payload = (
                sample.spans if output_mode == OUTPUT_MODE_SPANS else sample.events
            )
            return {
                "id": record_id,
                "status": "ok",
                "source": source,
                payload_key: payload,
                "llm": {
                    "model": client.model_name,
                    "metadata": sample.metadata,
                    "output_mode": output_mode,
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
    sample_events: list[list[dict[str, Any]]] = []
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
                output_mode=output_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                override_settings=override_settings,
            )
            if output_mode == OUTPUT_MODE_SPANS:
                sample_spans.append(sample.spans)
            else:
                sample_events.append(sample.events)
            sample_metadata.append(sample.metadata)
        except Exception as exc:
            sample_errors.append({"sample": sample_index, "error": str(exc)})

    samples_succeeded = len(
        sample_spans if output_mode == OUTPUT_MODE_SPANS else sample_events
    )
    if samples_succeeded < self_consistency_min_successful_samples:
        support_key = (
            "span_support"
            if output_mode == OUTPUT_MODE_SPANS
            else "event_support"
        )
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
                "output_mode": output_mode,
                "self_consistency": {
                    "enabled": True,
                    "samples_requested": self_consistency_samples,
                    "samples_succeeded": samples_succeeded,
                    "threshold": None,
                    "temperature": self_consistency_temperature,
                    "sample_errors": sample_errors,
                    support_key: [],
                    **(
                        {"argument_support": []}
                        if output_mode == OUTPUT_MODE_EVENTS_WITH_ARGS
                        else {}
                    ),
                },
            },
        }

    if output_mode == OUTPUT_MODE_EVENTS_WITH_ARGS:
        events, event_support_by_key, argument_support_by_event_key, threshold = (
            merge_self_consistency_events(sample_events, text)
        )
        event_support = [
            {
                "start_char": start_char,
                "end_char": end_char,
                "event_type": event_type,
                "support": support,
            }
            for (start_char, end_char, event_type), support in sorted(
                event_support_by_key.items()
            )
            if support >= threshold
        ]
        argument_support = []
        for event_key, arguments_by_key in sorted(
            argument_support_by_event_key.items()
        ):
            event_support_count = event_support_by_key.get(event_key, 0)
            if event_support_count < threshold:
                continue
            argument_threshold = (event_support_count // 2) + 1
            event_start_char, event_end_char, event_type = event_key
            for (start_char, end_char, role), support in sorted(
                arguments_by_key.items()
            ):
                if support < argument_threshold:
                    continue
                argument_support.append(
                    {
                        "event_start_char": event_start_char,
                        "event_end_char": event_end_char,
                        "event_type": event_type,
                        "start_char": start_char,
                        "end_char": end_char,
                        "role": role,
                        "support": support,
                        "threshold": argument_threshold,
                    }
                )
        return {
            "id": record_id,
            "status": "ok",
            "source": source,
            "events": events,
            "llm": {
                "model": client.model_name,
                "metadata": combine_metadata(sample_metadata),
                "output_mode": output_mode,
                "self_consistency": {
                    "enabled": True,
                    "samples_requested": self_consistency_samples,
                    "samples_succeeded": samples_succeeded,
                    "threshold": threshold,
                    "temperature": self_consistency_temperature,
                    "sample_errors": sample_errors,
                    "event_support": event_support,
                    "argument_support": argument_support,
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
            "output_mode": output_mode,
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
    argument_roles: set[str],
    event_argument_roles: dict[str, list[str]],
    argument_roles_text: str,
    event_argument_roles_text: str,
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
                output_mode=args.output_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                argument_roles_text=argument_roles_text,
                event_argument_roles_text=event_argument_roles_text,
                long_document_mode=args.long_document_mode,
                long_document_threshold_chars=args.long_document_threshold_chars,
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

    raw_ontology = load_json_tolerant(args.ontology)
    ontology = normalize_ontology(raw_ontology)
    ontology_text = format_ontology(ontology)
    labels = set(ontology)
    argument_role_descriptions = normalize_argument_roles(raw_ontology)
    argument_roles = set(argument_role_descriptions)
    event_argument_roles = normalize_event_argument_roles(
        raw_ontology, labels, argument_roles
    )
    argument_roles_text = format_argument_roles(argument_role_descriptions)
    event_argument_roles_text = format_event_argument_roles(event_argument_roles)
    system_prompt = load_prompt(args.system_prompt_file, DEFAULT_SYSTEM_PROMPT)
    default_user_prompt = (
        DEFAULT_EVENTS_WITH_ARGS_USER_PROMPT
        if args.output_mode == OUTPUT_MODE_EVENTS_WITH_ARGS
        else DEFAULT_USER_PROMPT
    )
    user_prompt_template = load_prompt(args.user_prompt_file, default_user_prompt)

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
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                argument_roles_text=argument_roles_text,
                event_argument_roles_text=event_argument_roles_text,
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
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default=OUTPUT_MODE_SPANS,
        help=(
            "Output schema to request from Gemini. 'spans' preserves the existing "
            "flat span output; 'events-with-args' emits event triggers with linked "
            "arguments."
        ),
    )
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
        "--long-document-mode",
        action="store_true",
        help="Use paragraph windows for articles at or above the long-document threshold.",
    )
    parser.add_argument(
        "--long-document-threshold-chars",
        type=int,
        default=1000,
        help="Minimum article character length that triggers windowed extraction.",
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
    if args.long_document_threshold_chars < 1:
        raise ValueError("--long-document-threshold-chars must be >= 1")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    started = time.monotonic()
    asyncio.run(run(args))
    LOGGER.info("finished in %.1fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
