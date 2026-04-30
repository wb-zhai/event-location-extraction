"""Merge repeated teacher samples into one deterministic annotation set."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def most_common_first_seen(values: list[str]) -> str:
    """Return the most common value, breaking ties by first occurrence."""
    if not values:
        return ""

    counts = Counter(values)
    value_index = max(
        range(len(values)),
        key=lambda index: (counts[values[index]], -index),
    )
    return values[value_index]


def merge_self_consistency_spans(
    sample_spans: list[list[dict[str, Any]]], text: str
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], int], int]:
    """Majority-vote flat span samples by exact offset and label."""
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
        merged.append(
            {
                "span_text": text[start_char:end_char],
                "label": label,
                "start_char": start_char,
                "end_char": end_char,
                "rationale": most_common_first_seen(rationales),
                "support": support,
            }
        )

    return merged, support_by_key, threshold


def merge_self_consistency_events(
    sample_events: list[list[dict[str, Any]]], text: str
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int, str], int],
    dict[tuple[int, int, str], dict[tuple[int, int, str], int]],
    int,
]:
    """Majority-vote event samples, then vote arguments within kept events."""
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


def apply_verifier_decisions(
    events: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filter accepted events using verifier decisions keyed by event offset/type."""
    accepted: set[tuple[int, int, str]] = set()
    for decision in decisions:
        if str(decision.get("decision")) != "accept":
            continue
        accepted.add(
            (
                int(decision.get("start_char", -1)),
                int(decision.get("end_char", -1)),
                str(decision.get("event_type", "")),
            )
        )
    if not accepted:
        return []
    return [
        event
        for event in events
        if (
            int(event.get("start_char", -1)),
            int(event.get("end_char", -1)),
            str(event.get("event_type", "")),
        )
        in accepted
    ]
