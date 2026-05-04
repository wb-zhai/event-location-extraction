"""Deterministic validation helpers for generated IE annotations."""

from __future__ import annotations

from typing import Any


def find_offsets(
    span_text: str, text: str, start_char: int, end_char: int
) -> tuple[int, int]:
    """Return exact offsets, repairing by substring search when possible."""
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
    """Validate flat span outputs against labels and article text."""
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
                # "rationale": str(span.get("rationale", "")).strip(),
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
    """Validate event trigger and linked argument outputs."""
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
                    # "rationale": str(event.get("rationale", "")).strip(),
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
