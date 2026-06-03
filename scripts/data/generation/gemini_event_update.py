import argparse
import asyncio
import json
import logging
import random
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import GeminiLLMClient
from scripts.data.generation import adjudication, gemini_event_gen

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_EXAMPLES = gemini_event_gen.DEFAULT_EXAMPLES
LOGGER = logging.getLogger("gemini_event_update")
TARGET_MODE_ALL = "all"
TARGET_MODE_EVENTS = "events"
TARGET_MODE_ARGUMENTS = "arguments"
TARGET_MODES = (TARGET_MODE_ALL, TARGET_MODE_EVENTS, TARGET_MODE_ARGUMENTS)
EVENT_REVIEW_GRANULARITY_RECORD = "record"
EVENT_REVIEW_GRANULARITY_SINGLE = "single"
EVENT_REVIEW_GRANULARITIES = (
    EVENT_REVIEW_GRANULARITY_RECORD,
    EVENT_REVIEW_GRANULARITY_SINGLE,
)
GENERIC_TRIGGER_FILTER_WINDOW_CHARS = 160
GENERIC_TRIGGER_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "war": {
        "military conflict": (
            "army",
            "troops",
            "offensive",
            "battle",
            "fighting",
            "combat",
            "hostilities",
            "military",
        ),
        "israel gaza conflict": ("gaza", "israel", "hamas", "palestinian"),
    },
    "conflict": {
        "military conflict": (
            "fighting",
            "troops",
            "army",
            "armed",
            "battle",
            "offensive",
            "clashes",
            "shelling",
        ),
        "conflict risk": (
            "risk",
            "could",
            "may",
            "threat",
            "warning",
            "fear",
            "possible",
            "escalat",
        ),
        "geopolitical risk": ("tension", "sanctions", "border", "regional"),
    },
    "crisis": {
        "existing humanitarian crisis": (
            "humanitarian",
            "aid",
            "emergency",
            "hunger",
            "famine",
            "displaced",
            "shelter",
        ),
        "political instability": (
            "government",
            "protests",
            "election",
            "unrest",
            "political",
            "coup",
        ),
        "geopolitical risk": ("sanctions", "regional", "border", "tension"),
        "refugee crisis": ("refugee", "camp", "asylum", "displaced", "fled"),
    },
    "violence": {
        "violence": (
            "killed",
            "wounded",
            "attacked",
            "beatings",
            "shooting",
            "abuse",
            "assault",
        ),
        "military conflict": ("army", "troops", "clashes", "combat", "shelling"),
    },
    "attack": {
        "threat of attack": (
            "threat",
            "risk",
            "warning",
            "possible",
            "could",
            "may",
            "fear",
            "imminent",
        ),
        "violence": (
            "killed",
            "wounded",
            "struck",
            "hit",
            "bomb",
            "launched",
            "shell",
            "dead",
        ),
        "military conflict": ("army", "troops", "military", "combat", "shelling"),
    },
    "attacks": {
        "threat of attack": (
            "threat",
            "risk",
            "warning",
            "possible",
            "could",
            "may",
            "fear",
            "imminent",
        ),
        "violence": (
            "killed",
            "wounded",
            "struck",
            "hit",
            "bomb",
            "launched",
            "shell",
            "dead",
        ),
        "military conflict": ("army", "troops", "military", "combat", "shelling"),
    },
    "attacked": {
        "threat of attack": (
            "threat",
            "risk",
            "warning",
            "possible",
            "could",
            "may",
            "fear",
            "imminent",
        ),
        "violence": (
            "killed",
            "wounded",
            "struck",
            "hit",
            "bomb",
            "launched",
            "shell",
            "dead",
        ),
        "military conflict": ("army", "troops", "military", "combat", "shelling"),
    },
    "refugees": {
        "refugee crisis": (
            "crisis",
            "camp",
            "asylum",
            "displaced",
            "fled",
            "crossing",
            "humanitarian",
            "thousands",
        ),
        "fled home population": ("fled", "escaped", "left", "crossed", "evacuated"),
        "population affected by displacement": (
            "displaced",
            "uprooted",
            "relocated",
            "shelter",
        ),
    },
    "climate change": {
        "temperature increase climate change": (
            "warming",
            "temperature",
            "heat",
            "emissions",
            "global warming",
        ),
        "climate event": ("rainfall", "storm", "flood", "drought", "cyclone"),
    },
}
SUMMARY_GENERIC_TRIGGERS = set(GENERIC_TRIGGER_RULES)

DEFAULT_SYSTEM_PROMPT = """<role>
You are an expert annotation reviewer specializing in repairing food insecurity event annotations from news articles.
</role>

<objective>
Your goal is to repair existing event annotations for maximum precision and consistency. Review each candidate event against the article text and the fixed ontology, then keep, relabel, narrow, repair, or drop it.
</objective>

<primary_directives>
1. Strict Contextual Grounding: Keep or modify an annotation ONLY if it is explicitly supported by the provided `<article_text>`. Never infer facts or use background knowledge.
2. Repair, Do Not Re-Extract: Treat `<current_annotations>` as the candidate set. Do not broadly add new events. Only repair, relabel, narrow, or drop the provided candidates.
3. Exact Verbatim Extraction: Every kept trigger and argument span MUST be an exact, contiguous substring from `<article_text>`.
4. Offset Precision: Use zero-based character offsets (`start_char`, `end_char`) that exactly match the extracted substring in `<article_text>`. Do not use the title for offsets.
5. Narrow Granularity: Keep the narrowest meaningful trigger phrase. Exclude surrounding location, time, attribution, and background context unless essential to event meaning.
6. Ontology Adherence: Use ONLY the event labels, argument roles, and location types provided in the ontology context.
7. Precision Over Recall: If an annotation is ambiguous, weakly grounded, too generic, duplicated, or unsupported, drop it.
</primary_directives>

<update_process>
For each existing annotation in `<current_annotations>`:
1. Verify that the event mention is explicit, current, and article-grounded. Drop historical, hypothetical, rhetorical, or background-only mentions.
2. If valid, assign the single best ontology label for that mention.
3. Refine the trigger span to the minimal distinct phrase and correct offsets.
4. Review arguments conservatively. Keep only explicit, directly linked location arguments and assign one of `country`, `state`, `county`, `city`, or `other`.
5. Keep one event per event mention. Do not keep repeated generic mentions across the article unless they are clearly distinct event instances.
</update_process>

<consistency_rules>
- Generic triggers such as `war`, `conflict`, `crisis`, `violence`, `attack`, `attacks`, `attacked`, `refugees`, and `climate change` must only be kept when the local sentence clearly supports one ontology label over plausible alternatives.
- Do not label `attack`, `attacks`, or `attacked` as `threat of attack` when the text describes a completed attack.
- Do not label `conflict` as `conflict risk` when the text describes active fighting or ongoing hostilities.
- Do not label `refugees` alone as `refugee crisis` unless the text describes acute refugee conditions, large-scale displacement, or a humanitarian crisis.
- Do not keep background mentions such as `war crimes`, `post-conflict`, historical references, or abstract discussion unless they are direct evidence of the annotated event mention.
</consistency_rules>

<quality_assurance>
- A valid updated extraction must have an exact text match and a valid ontology label.
- Spans must consist of complete words.
- Event-argument links must be explicitly supported by the local text.
- Resolve overlapping or nested triggers for the same mention to one best span.
- Prefer dropping an uncertain annotation over keeping a noisy one.
</quality_assurance>"""

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

<argument_roles>
{argument_roles}
</argument_roles>

<event_argument_roles>
{event_argument_roles}
</event_argument_roles>

<location_types>
{location_types}
</location_types>

<current_annotations>
{current_annotations}
</current_annotations>
</context>

<task>
Review the provided `<current_annotations>` against the article and return a repaired, precision-first annotation set. Do not broadly re-extract the article.
</task>

<extraction_rules>
- Triggers: Keep only explicit food-insecurity and risk-factor triggers. The `trigger_text` must be the minimal distinct phrase for that event mention.
- Candidate policy: Use `<current_annotations>` as the candidate list. Repair or drop candidates; do not broadly add new events.
- Arguments: Keep only explicitly stated location arguments that are directly linked to the event mention. If multiple explicit candidates exist for the same role, prefer the closest one to the trigger.
- Locations: Arguments with location roles MUST have a `location_type` assigned from exactly one of: `country`, `state`, `county`, `city`, or `other`. Use `other` for regions, camps, facilities, border areas, or ambiguous place levels. Non-location arguments must not have a location type.
- Generic-trigger caution: Drop ambiguous generic triggers unless the local sentence clearly disambiguates the label.
- Constraints:
  - `start_char` and `end_char` must correspond exactly to the extracted text within the `<article_text>`.
  - `end_char` is exclusive.
  - Do not use the title for any extraction.
  - No inferred entities; everything must be explicitly in the text.
  - Output must not contain duplicate event-argument structures.
  - Prefer dropping weak or background-only candidates over preserving them.
</extraction_rules>

Take into account any reference examples provided before the current article. Use them only to learn annotation style and strictness, never as evidence for this article.
"""

TARGET_MODE_SYSTEM_INSTRUCTIONS = {
    TARGET_MODE_ALL: """<target_mode>
Refine the provided annotations fully. You may correct event labels, trigger spans, argument roles, location types, and offsets, and you may drop unsupported annotations.
</target_mode>""",
    TARGET_MODE_EVENTS: """<target_mode>
Focus on event review only. Correct event validity, event labels, trigger spans, and trigger offsets. Arguments are fixed metadata from the current annotations and must not be added, removed, or revised in this pass.
</target_mode>""",
    TARGET_MODE_ARGUMENTS: """<target_mode>
Focus on argument review only. Treat each provided event trigger span and event label as fixed context. Only repair, add, or remove arguments, their roles, location types, and offsets. Do not alter event trigger text, event offsets, or event label.
</target_mode>""",
}

TARGET_MODE_USER_INSTRUCTIONS = {
    TARGET_MODE_ALL: """<target_mode_task>
Fully refine the current annotations against the ontology and article text.
</target_mode_task>""",
    TARGET_MODE_EVENTS: """<target_mode_task>
Update events only. Review event validity, event labels, trigger text, and trigger offsets. Keep arguments unchanged from the current annotations for any event that remains after review.
</target_mode_task>""",
    TARGET_MODE_ARGUMENTS: """<target_mode_task>
Update arguments only. Preserve the provided event identity order whenever possible. Keep each event trigger text, event offsets, and event label fixed; only repair, add, or remove arguments and their labels/location types/offsets.
</target_mode_task>""",
}


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


class BatchUpdateSpansPayload(BaseModel):
    spans: list[UpdatedSpan] = Field(default_factory=list)


class BatchUpdateEventsPayload(BaseModel):
    events: list[UpdatedEvent] = Field(default_factory=list)


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.output_mode == gemini_event_gen.OUTPUT_MODE_SPANS
        and args.target_mode == TARGET_MODE_ARGUMENTS
    ):
        raise ValueError(
            "--target-mode arguments requires --output-mode events-with-args"
        )
    if (
        args.event_review_granularity == EVENT_REVIEW_GRANULARITY_SINGLE
        and args.target_mode != TARGET_MODE_EVENTS
    ):
        raise ValueError(
            "--event-review-granularity single requires --target-mode events"
        )


def _append_instruction_block(base_prompt: str, instruction_block: str) -> str:
    if not instruction_block.strip():
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{instruction_block}\n"


def _event_key(event: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(event.get("start_char", -1)),
        int(event.get("end_char", -1)),
        str(event.get("event_type", "")).strip(),
    )


def _copy_arguments(arguments: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not isinstance(arguments, list):
        return copied
    for argument in arguments:
        if isinstance(argument, dict):
            copied.append(dict(argument))
    return copied


def _record_with_selected_events(
    record: dict[str, Any], selected_events: list[dict[str, Any]]
) -> dict[str, Any]:
    subset_record = dict(record)
    subset_record["events"] = selected_events
    return subset_record


def _record_for_event_index(
    record: dict[str, Any], event_index: int
) -> dict[str, Any] | None:
    events = record.get("events", [])
    if (
        not isinstance(events, list)
        or event_index < 0
        or event_index >= len(events)
        or not isinstance(events[event_index], dict)
    ):
        return None
    return _record_with_selected_events(record, [events[event_index]])


def _sentence_window(text: str, start_char: int, end_char: int) -> str:
    if not text:
        return ""
    left = max(0, start_char - GENERIC_TRIGGER_FILTER_WINDOW_CHARS)
    right = min(len(text), end_char + GENERIC_TRIGGER_FILTER_WINDOW_CHARS)
    while left > 0 and text[left - 1] not in ".!?\n":
        left -= 1
    while right < len(text) and text[right] not in ".!?\n":
        right += 1
    return text[left:right].lower()


def _generic_trigger_allowed(event: dict[str, Any], text: str) -> bool:
    trigger = str(event.get("trigger_text", "")).strip().lower()
    event_type = str(event.get("event_type", "")).strip()
    if trigger not in GENERIC_TRIGGER_RULES:
        return True
    evidence_terms = GENERIC_TRIGGER_RULES[trigger].get(event_type)
    if not evidence_terms:
        return False
    start_char = int(event.get("start_char", -1))
    end_char = int(event.get("end_char", -1))
    if start_char < 0 or end_char <= start_char:
        return False
    sentence = _sentence_window(text, start_char, end_char)
    return any(term in sentence for term in evidence_terms)


def _dedupe_and_rank_arguments(
    arguments: list[dict[str, Any]],
    *,
    trigger_start: int,
    trigger_end: int,
) -> list[dict[str, Any]]:
    unique_arguments: dict[tuple[int, int, str], dict[str, Any]] = {}
    for argument in arguments:
        key = (
            int(argument.get("start_char", -1)),
            int(argument.get("end_char", -1)),
            str(argument.get("role", "")).strip(),
        )
        if key not in unique_arguments:
            unique_arguments[key] = argument

    ranked_by_role: dict[str, list[dict[str, Any]]] = {}
    for argument in unique_arguments.values():
        role = str(argument.get("role", "")).strip()
        ranked_by_role.setdefault(role, []).append(argument)

    selected: list[dict[str, Any]] = []
    for role, role_arguments in ranked_by_role.items():
        role_arguments.sort(
            key=lambda argument: (
                min(
                    abs(int(argument.get("start_char", -1)) - trigger_end),
                    abs(trigger_start - int(argument.get("end_char", -1))),
                ),
                int(argument.get("start_char", -1)),
                int(argument.get("end_char", -1)),
            )
        )
        if role in gemini_event_gen.LOCATION_ARGUMENT_ROLES:
            selected.append(role_arguments[0])
        else:
            selected.extend(role_arguments)

    selected.sort(
        key=lambda argument: (
            int(argument.get("start_char", -1)),
            int(argument.get("end_char", -1)),
            str(argument.get("role", "")).strip(),
        )
    )
    return selected


def _post_filter_events(
    events: list[dict[str, Any]],
    *,
    text: str,
    strict_offsets: bool,
    filter_generic_triggers: bool = True,
    collapse_same_mention: bool = True,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen_events: set[tuple[int, int, str]] = set()
    seen_mentions: set[tuple[int, int]] = set()

    for event in sorted(
        events,
        key=lambda item: (
            int(item.get("start_char", -1)),
            int(item.get("end_char", -1)),
            str(item.get("event_type", "")).strip(),
        ),
    ):
        start_char = int(event.get("start_char", -1))
        end_char = int(event.get("end_char", -1))
        event_type = str(event.get("event_type", "")).strip()
        trigger_text = str(event.get("trigger_text", "")).strip()
        event_key = (start_char, end_char, event_type)
        mention_key = (start_char, end_char)

        if event_key in seen_events:
            continue
        if strict_offsets and (
            start_char < 0
            or end_char <= start_char
            or end_char > len(text)
            or text[start_char:end_char] != trigger_text
        ):
            continue
        if collapse_same_mention and mention_key in seen_mentions:
            continue
        if filter_generic_triggers and not _generic_trigger_allowed(event, text):
            continue

        filtered_event = dict(event)
        filtered_event["arguments"] = _dedupe_and_rank_arguments(
            _copy_arguments(event.get("arguments", [])),
            trigger_start=start_char,
            trigger_end=end_char,
        )
        seen_events.add(event_key)
        seen_mentions.add(mention_key)
        filtered.append(filtered_event)

    return filtered


def _merge_metadata(sample_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    if not sample_metadata:
        return combined
    numeric_keys = {
        key
        for metadata in sample_metadata
        for key, value in metadata.items()
        if isinstance(value, int | float)
    }
    for key in sorted(numeric_keys):
        combined[key] = sum(metadata.get(key, 0) for metadata in sample_metadata)
    return combined


USAGE_METADATA_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "thoughts_token_count",
)


def _usage_metadata_only(metadata: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: int(metadata.get(key, 0))
        for key in USAGE_METADATA_KEYS
        if isinstance(metadata.get(key), int | float)
    }


def _attach_cost_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    update_usage: dict[str, int] = {}
    update_block = metadata.get("update")
    if isinstance(update_block, dict):
        if isinstance(update_block.get("per_event"), list):
            per_event_usage = [
                _usage_metadata_only(item.get("metadata"))
                for item in update_block["per_event"]
                if isinstance(item, dict)
            ]
            update_usage = _merge_metadata(per_event_usage)
        else:
            update_usage = _usage_metadata_only(update_block)
    elif isinstance(metadata.get("self_consistency"), dict):
        update_usage = _usage_metadata_only(metadata["self_consistency"].get("metadata"))

    self_consistency_usage = {}
    if isinstance(metadata.get("self_consistency"), dict):
        self_consistency_usage = _usage_metadata_only(
            metadata["self_consistency"].get("metadata")
        )
        if self_consistency_usage:
            update_usage = self_consistency_usage

    verifier_usage = {}
    if isinstance(metadata.get("verifier"), dict):
        verifier_usage = _usage_metadata_only(metadata["verifier"].get("metadata"))

    total_usage = _merge_metadata(
        [usage for usage in (update_usage, verifier_usage) if usage]
    )
    metadata["cost"] = {
        "update": update_usage,
        **({"verifier": verifier_usage} if verifier_usage else {}),
        "total": total_usage,
    }
    return metadata


def _event_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    arguments = tuple(
        sorted(
            (
                str(argument.get("role", "")).strip(),
                int(argument.get("start_char", -1)),
                int(argument.get("end_char", -1)),
                str(argument.get("text", "")).strip(),
                str(argument.get("location_type", "")).strip(),
            )
            for argument in event.get("arguments", []) or []
            if isinstance(argument, dict)
        )
    )
    return (
        str(event.get("event_type", "")).strip(),
        int(event.get("start_char", -1)),
        int(event.get("end_char", -1)),
        str(event.get("trigger_text", "")).strip(),
        arguments,
    )


def _trigger_length(event: dict[str, Any]) -> int:
    return len(str(event.get("trigger_text", "")).strip())


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = max(
        0,
        min(len(sorted_values) - 1, round((len(sorted_values) - 1) * percentile)),
    )
    return sorted_values[index]


def _event_summary_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    trigger_lengths: list[int] = []
    generic_trigger_events = 0
    argument_total = 0
    empty_argument_events = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        label_counts[str(event.get("event_type", "")).strip()] += 1
        trigger = str(event.get("trigger_text", "")).strip().lower()
        if trigger in SUMMARY_GENERIC_TRIGGERS:
            generic_trigger_events += 1
        trigger_length = _trigger_length(event)
        if trigger_length > 0:
            trigger_lengths.append(trigger_length)
        arguments = [
            argument
            for argument in event.get("arguments", []) or []
            if isinstance(argument, dict)
        ]
        argument_total += len(arguments)
        if not arguments:
            empty_argument_events += 1

    trigger_lengths.sort()
    avg_trigger_length = (
        round(sum(trigger_lengths) / len(trigger_lengths), 2)
        if trigger_lengths
        else 0.0
    )
    return {
        "events": len(events),
        "arguments": argument_total,
        "empty_argument_events": empty_argument_events,
        "generic_trigger_events": generic_trigger_events,
        "trigger_length": {
            "avg": avg_trigger_length,
            "median": _percentile(trigger_lengths, 0.5),
            "p90": _percentile(trigger_lengths, 0.9),
            "max": max(trigger_lengths) if trigger_lengths else 0,
        },
        "label_counts": dict(label_counts),
    }


def _summarize_record_update(
    original_record: dict[str, Any],
    updated_record: dict[str, Any] | None,
    *,
    output_mode: str,
    error: str | None = None,
) -> dict[str, Any]:
    original_events = (
        [
            event
            for event in original_record.get("events", []) or []
            if isinstance(event, dict)
        ]
        if output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        else []
    )
    updated_events = (
        [
            event
            for event in (updated_record or {}).get("events", []) or []
            if isinstance(event, dict)
        ]
        if output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        else []
    )

    original_signatures = {_event_signature(event) for event in original_events}
    updated_signatures = {_event_signature(event) for event in updated_events}
    original_payload = _event_summary_payload(original_events)
    updated_payload = _event_summary_payload(updated_events)

    verifier_rejections = 0
    if updated_record:
        verifier = (updated_record.get("metadata") or {}).get("verifier")
        decisions = verifier.get("decisions", []) if isinstance(verifier, dict) else []
        verifier_rejections = sum(
            1
            for decision in decisions
            if str(decision.get("decision", "")).strip() != "accept"
        )

    return {
        "record_id": str(original_record.get("id")),
        "status": "error" if error else "ok",
        "error": error,
        "changed": original_signatures != updated_signatures,
        "events_removed": max(0, len(original_signatures - updated_signatures)),
        "events_added": max(0, len(updated_signatures - original_signatures)),
        "event_count_delta": updated_payload["events"] - original_payload["events"],
        "argument_count_delta": updated_payload["arguments"]
        - original_payload["arguments"],
        "empty_argument_event_delta": updated_payload["empty_argument_events"]
        - original_payload["empty_argument_events"],
        "generic_trigger_event_delta": updated_payload["generic_trigger_events"]
        - original_payload["generic_trigger_events"],
        "verifier_rejections": verifier_rejections,
        "before": original_payload,
        "after": updated_payload,
    }


def _merge_label_counts(destination: Counter[str], source: dict[str, Any]) -> None:
    for label, count in source.items():
        destination[str(label)] += int(count)


def _build_run_summary(
    record_summaries: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ok_records = [summary for summary in record_summaries if summary["status"] == "ok"]
    error_records = [
        {
            "record_id": summary["record_id"],
            "error": summary["error"],
        }
        for summary in record_summaries
        if summary["status"] == "error"
    ]

    before_labels: Counter[str] = Counter()
    after_labels: Counter[str] = Counter()
    before_events = before_arguments = before_empty = before_generic = 0
    after_events = after_arguments = after_empty = after_generic = 0
    verifier_rejections = 0

    for summary in ok_records:
        before = summary["before"]
        after = summary["after"]
        before_events += int(before["events"])
        before_arguments += int(before["arguments"])
        before_empty += int(before["empty_argument_events"])
        before_generic += int(before["generic_trigger_events"])
        after_events += int(after["events"])
        after_arguments += int(after["arguments"])
        after_empty += int(after["empty_argument_events"])
        after_generic += int(after["generic_trigger_events"])
        verifier_rejections += int(summary["verifier_rejections"])
        _merge_label_counts(before_labels, before["label_counts"])
        _merge_label_counts(after_labels, after["label_counts"])

    changed_records = sum(1 for summary in ok_records if summary["changed"])
    event_reduction_records = sum(
        1 for summary in ok_records if summary["event_count_delta"] < 0
    )
    argument_reduction_records = sum(
        1 for summary in ok_records if summary["argument_count_delta"] < 0
    )

    def top_label_deltas(limit: int = 15) -> list[dict[str, Any]]:
        labels = sorted(set(before_labels) | set(after_labels))
        deltas = []
        for label in labels:
            before_count = before_labels[label]
            after_count = after_labels[label]
            delta = after_count - before_count
            if delta == 0:
                continue
            deltas.append(
                {
                    "label": label,
                    "before": before_count,
                    "after": after_count,
                    "delta": delta,
                }
            )
        deltas.sort(key=lambda item: (abs(int(item["delta"])), item["label"]), reverse=True)
        return deltas[:limit]

    def top_changed_records(limit: int = 20) -> list[dict[str, Any]]:
        changed = [summary for summary in ok_records if summary["changed"]]
        changed.sort(
            key=lambda summary: (
                abs(int(summary["event_count_delta"])),
                abs(int(summary["argument_count_delta"])),
                abs(int(summary["generic_trigger_event_delta"])),
                str(summary["record_id"]),
            ),
            reverse=True,
        )
        return [
            {
                "record_id": summary["record_id"],
                "event_count_delta": summary["event_count_delta"],
                "argument_count_delta": summary["argument_count_delta"],
                "empty_argument_event_delta": summary["empty_argument_event_delta"],
                "generic_trigger_event_delta": summary["generic_trigger_event_delta"],
                "events_removed": summary["events_removed"],
                "events_added": summary["events_added"],
                "verifier_rejections": summary["verifier_rejections"],
            }
            for summary in changed[:limit]
        ]

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "processed_records": len(record_summaries),
        "ok_records": len(ok_records),
        "error_records": len(error_records),
        "changed_records": changed_records,
        "changed_record_ratio": ratio(changed_records, len(ok_records)),
        "event_reduction_records": event_reduction_records,
        "argument_reduction_records": argument_reduction_records,
        "verifier_rejections": verifier_rejections,
        "before": {
            "events": before_events,
            "arguments": before_arguments,
            "empty_argument_events": before_empty,
            "empty_argument_event_ratio": ratio(before_empty, before_events),
            "generic_trigger_events": before_generic,
            "generic_trigger_ratio": ratio(before_generic, before_events),
            "top_labels": before_labels.most_common(15),
        },
        "after": {
            "events": after_events,
            "arguments": after_arguments,
            "empty_argument_events": after_empty,
            "empty_argument_event_ratio": ratio(after_empty, after_events),
            "generic_trigger_events": after_generic,
            "generic_trigger_ratio": ratio(after_generic, after_events),
            "top_labels": after_labels.most_common(15),
        },
        "deltas": {
            "events": after_events - before_events,
            "arguments": after_arguments - before_arguments,
            "empty_argument_events": after_empty - before_empty,
            "generic_trigger_events": after_generic - before_generic,
            "top_label_deltas": top_label_deltas(),
        },
        "top_changed_records": top_changed_records(),
        "errors": error_records[:20],
        "settings": {
            "output_mode": args.output_mode,
            "target_mode": args.target_mode,
            "event_review_granularity": args.event_review_granularity,
            "example_sample_size": args.example_sample_size,
            "example_retrieval": args.example_retrieval,
            "enable_verifier": args.enable_verifier,
            "self_consistency": args.self_consistency,
            "self_consistency_samples": args.self_consistency_samples,
            "self_consistency_temperature": args.self_consistency_temperature,
        },
    }


def _summary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.summary.json")


def _select_current_annotations(
    record: dict[str, Any],
    output_mode: str,
    target_mode: str,
) -> list[dict[str, Any]]:
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        return list(record.get("spans", []))

    events = record.get("events", [])
    if target_mode != TARGET_MODE_EVENTS:
        return list(events)

    selected_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        selected_events.append(
            {
                "event_type": event.get("event_type"),
                "trigger_text": event.get("trigger_text"),
                "start_char": event.get("start_char"),
                "end_char": event.get("end_char"),
            }
        )
    return selected_events


def _merge_event_updates(
    original_events: list[dict[str, Any]],
    cleaned_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    original_arguments = {
        _event_key(event): _copy_arguments(event.get("arguments", []))
        for event in original_events
        if isinstance(event, dict)
    }
    unmatched_originals: list[dict[str, Any]] = [
        event for event in original_events if isinstance(event, dict)
    ]
    merged_events: list[dict[str, Any]] = []
    for event in cleaned_events:
        event_key = _event_key(event)
        merged_event = dict(event)
        matched_arguments = original_arguments.get(event_key)
        if matched_arguments is None:
            best_index = -1
            best_score: tuple[int, int, int, int] | None = None
            event_start = int(event.get("start_char", -1))
            event_end = int(event.get("end_char", -1))
            event_trigger = str(event.get("trigger_text", "")).strip().lower()

            for index, original_event in enumerate(unmatched_originals):
                original_start = int(original_event.get("start_char", -1))
                original_end = int(original_event.get("end_char", -1))
                original_trigger = (
                    str(original_event.get("trigger_text", "")).strip().lower()
                )
                overlap_penalty = 0
                if not (
                    event_start < original_end and original_start < event_end
                ):
                    overlap_penalty = 1
                trigger_penalty = 0 if event_trigger == original_trigger else 1
                score = (
                    overlap_penalty,
                    abs(event_start - original_start),
                    abs(event_end - original_end),
                    trigger_penalty,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = index

            if best_index >= 0:
                matched_arguments = _copy_arguments(
                    unmatched_originals.pop(best_index).get("arguments", [])
                )
            else:
                matched_arguments = []
        else:
            for index, original_event in enumerate(unmatched_originals):
                if _event_key(original_event) == event_key:
                    unmatched_originals.pop(index)
                    break

        merged_event["arguments"] = matched_arguments
        merged_events.append(merged_event)
    return merged_events


def _clean_argument_updates_for_original_events(
    raw_answer: dict[str, Any],
    original_events: list[dict[str, Any]],
    text: str,
    labels: set[str],
    strict_offsets: bool,
    argument_roles: set[str],
    event_argument_roles: dict[str, list[str]],
    location_types: set[str] | None,
) -> list[dict[str, Any]]:
    raw_events = raw_answer.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    merged_events: list[dict[str, Any]] = []
    for index, original_event in enumerate(original_events):
        if not isinstance(original_event, dict):
            continue

        candidate_arguments: list[dict[str, Any]] = []
        has_candidate_event = index < len(raw_events) and isinstance(raw_events[index], dict)
        if has_candidate_event:
            raw_arguments = raw_events[index].get("arguments", [])
            if isinstance(raw_arguments, list):
                candidate_arguments = raw_arguments

        original_arguments = _copy_arguments(original_event.get("arguments", []))
        cleaned_events = gemini_event_gen.clean_events_with_args(
            {
                "events": [
                    {
                        "event_type": original_event.get("event_type"),
                        "trigger_text": original_event.get("trigger_text"),
                        "start_char": original_event.get("start_char"),
                        "end_char": original_event.get("end_char"),
                        "arguments": candidate_arguments,
                    }
                ]
            },
            text,
            labels,
            strict_offsets,
            argument_roles=argument_roles,
            event_argument_roles=event_argument_roles,
            location_types=location_types,
        )

        merged_event = dict(original_event)
        if not has_candidate_event:
            merged_event["arguments"] = original_arguments
        elif cleaned_events:
            merged_event["arguments"] = cleaned_events[0].get("arguments", [])
        else:
            merged_event["arguments"] = []
        merged_events.append(merged_event)

    return merged_events


def _clean_updated_payload(
    raw_answer: dict[str, Any],
    record: dict[str, Any],
    text: str,
    labels: set[str],
    output_mode: str,
    target_mode: str,
    strict_offsets: bool,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
) -> list[dict[str, Any]]:
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        return gemini_event_gen.clean_spans(raw_answer, text, labels, strict_offsets)

    cleaned_events = gemini_event_gen.clean_events_with_args(
        raw_answer,
        text,
        labels,
        strict_offsets,
        argument_roles=argument_roles or set(),
        event_argument_roles=event_argument_roles or {},
        location_types=location_types,
    )
    original_events = [
        event for event in record.get("events", []) if isinstance(event, dict)
    ]
    if target_mode == TARGET_MODE_ALL:
        return _post_filter_events(
            cleaned_events, text=text, strict_offsets=strict_offsets
        )
    if target_mode == TARGET_MODE_EVENTS:
        return _post_filter_events(
            _merge_event_updates(original_events, cleaned_events),
            text=text,
            strict_offsets=strict_offsets,
        )
    repaired_events = _clean_argument_updates_for_original_events(
        raw_answer,
        original_events,
        text,
        labels,
        strict_offsets,
        argument_roles=argument_roles or set(),
        event_argument_roles=event_argument_roles or {},
        location_types=location_types,
    )
    return _post_filter_events(
        repaired_events,
        text=text,
        strict_offsets=strict_offsets,
        filter_generic_triggers=False,
        collapse_same_mention=False,
    )


def _render_update_prompt(
    *,
    title: str,
    text: str,
    ontology_text: str,
    current_annotations_str: str,
    argument_roles_text: str,
    event_argument_roles_text: str,
    location_types_text: str,
    user_prompt_template: str,
    target_mode: str,
    examples_text: str,
) -> str:
    format_kwargs = {
        "title": title,
        "text": text,
        "ontology": ontology_text,
        "current_annotations": current_annotations_str,
        "argument_roles": argument_roles_text,
        "event_argument_roles": event_argument_roles_text,
        "location_types": location_types_text,
    }
    prompt = user_prompt_template.format(**format_kwargs)
    prompt = _append_instruction_block(prompt, TARGET_MODE_USER_INSTRUCTIONS[target_mode])
    if examples_text:
        return f"{examples_text}\n\n{prompt}"
    return prompt


def _build_update_prompt(
    *,
    record: dict[str, Any],
    output_mode: str,
    target_mode: str,
    ontology_text: str,
    user_prompt_template: str,
    examples: list[dict[str, Any]] | None,
    example_sample_size: int | None,
    example_retrieval: str,
    example_retriever: Any | None,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
) -> str:
    text = str(record.get("text", ""))
    title = str(record.get("title", ""))
    current_annotations = _select_current_annotations(record, output_mode, target_mode)
    current_annotations_str = json.dumps(
        current_annotations, ensure_ascii=False, indent=2
    )

    argument_roles_text = ""
    if argument_roles:
        argument_roles_text = gemini_event_gen.format_argument_roles(
            {r: "" for r in argument_roles}
        )

    event_argument_roles_text = ""
    if event_argument_roles:
        event_argument_roles_text = gemini_event_gen.format_event_argument_roles(
            event_argument_roles
        )

    location_types_text = ""
    if location_types:
        location_types_text = gemini_event_gen.format_location_types(
            {t: "" for t in location_types}
        )

    example_records = gemini_event_gen.retrieve_examples_for_query(
        text,
        title,
        examples or [],
        output_mode=output_mode,
        sample_size=example_sample_size,
        retrieval_mode=example_retrieval,
        example_retriever=example_retriever,
    )
    return _render_update_prompt(
        title=title,
        text=text,
        ontology_text=ontology_text,
        current_annotations_str=current_annotations_str,
        argument_roles_text=argument_roles_text,
        event_argument_roles_text=event_argument_roles_text,
        location_types_text=location_types_text,
        user_prompt_template=user_prompt_template,
        target_mode=target_mode,
        examples_text=gemini_event_gen.format_examples(example_records, output_mode),
    )


def _response_format_for_output_mode(output_mode: str) -> dict[str, Any]:
    return (
        {"spans": list[UpdatedSpan]}
        if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS
        else {"events": list[UpdatedEvent]}
    )


def _batch_response_schema(output_mode: str) -> type[BaseModel]:
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        return BatchUpdateSpansPayload
    return BatchUpdateEventsPayload


def _build_update_batch_tasks(
    records: list[dict[str, Any]],
    *,
    output_mode: str,
    target_mode: str,
    ontology_text: str,
    user_prompt_template: str,
    examples: list[dict[str, Any]] | None,
    example_sample_size: int | None,
    example_retrieval: str,
    example_retriever: Any | None,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
    self_consistency: bool,
    self_consistency_samples: int,
    event_review_granularity: str,
) -> list[gemini_event_gen.ExtractionBatchTask]:
    tasks: list[gemini_event_gen.ExtractionBatchTask] = []
    for record in records:
        record_id = str(record.get("id"))
        text = str(record.get("text", ""))
        task_records: list[tuple[int | None, dict[str, Any]]] = []
        if (
            event_review_granularity == EVENT_REVIEW_GRANULARITY_SINGLE
            and target_mode == TARGET_MODE_EVENTS
            and output_mode != gemini_event_gen.OUTPUT_MODE_SPANS
        ):
            for event_index, event in enumerate(record.get("events", []) or []):
                if not isinstance(event, dict):
                    continue
                subset_record = _record_with_selected_events(record, [event])
                task_records.append((event_index, subset_record))
        else:
            task_records.append((None, record))

        sample_count = self_consistency_samples if self_consistency else 1
        for event_index, task_record in task_records:
            prompt = _build_update_prompt(
                record=task_record,
                output_mode=output_mode,
                target_mode=target_mode,
                ontology_text=ontology_text,
                user_prompt_template=user_prompt_template,
                examples=examples,
                example_sample_size=example_sample_size,
                example_retrieval=example_retrieval,
                example_retriever=example_retriever,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                location_types=location_types,
            )
            for sample_index in range(sample_count):
                batch_key = gemini_event_gen.task_batch_key(
                    record_id,
                    window_index=event_index,
                    sample_index=sample_index if self_consistency else None,
                )
                tasks.append(
                    gemini_event_gen.ExtractionBatchTask(
                        batch_key=batch_key,
                        record_id=record_id,
                        prompt=prompt,
                        text=text,
                        sample_index=sample_index if self_consistency else None,
                        event_index=event_index,
                    )
                )
    return tasks


def _parse_batch_update_payload(
    *,
    line: dict[str, Any],
    record: dict[str, Any],
    event_index: int | None,
    output_mode: str,
    target_mode: str,
    labels: set[str],
    strict_offsets: bool,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = line.get("response")
    if not isinstance(response, dict):
        error = line.get("error") or line.get("status") or "Missing batch response."
        raise ValueError(str(error))
    raw_answer = json.loads(gemini_event_gen.batch_response_text(response))
    metadata = gemini_event_gen.batch_usage_metadata(response)
    payload_record = (
        _record_for_event_index(record, event_index)
        if event_index is not None
        else record
    )
    if payload_record is None:
        raise ValueError(f"Unknown event index: {event_index}")
    payload = _clean_updated_payload(
        raw_answer,
        payload_record,
        str(payload_record.get("text", "")),
        labels,
        output_mode,
        target_mode,
        strict_offsets,
        argument_roles,
        event_argument_roles,
        location_types,
    )
    return payload, metadata


def _aggregate_batch_update_record(
    *,
    record: dict[str, Any],
    tasks: list[gemini_event_gen.ExtractionBatchTask],
    task_results: dict[str, dict[str, Any]],
    output_mode: str,
    target_mode: str,
    strict_offsets: bool,
    self_consistency: bool,
    self_consistency_samples: int,
    self_consistency_min_successful_samples: int,
    event_review_granularity: str,
) -> dict[str, Any]:
    text = str(record.get("text", ""))
    record_id = str(record.get("id"))
    updated_record = dict(record)
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
        successful_payloads: list[list[dict[str, Any]]] = []
        metadata_list: list[dict[str, Any]] = []
        for task in tasks:
            task_result = task_results.get(task.batch_key)
            if not task_result or task_result.get("status") != "ok":
                continue
            successful_payloads.append(task_result["payload"])
            metadata_list.append(task_result.get("metadata", {}))
        if not successful_payloads:
            error = "Batch result did not include a response for this record."
            return {**updated_record, "status": "error", "error": error}
        updated_record["spans"] = successful_payloads[0]
        updated_record["metadata"] = _attach_cost_metadata(
            {"update": _merge_metadata(metadata_list)}
        )
        return updated_record

    if (
        event_review_granularity == EVENT_REVIEW_GRANULARITY_SINGLE
        and target_mode == TARGET_MODE_EVENTS
    ):
        event_tasks: dict[int, list[gemini_event_gen.ExtractionBatchTask]] = {}
        for task in tasks:
            if task.event_index is None:
                continue
            event_tasks.setdefault(task.event_index, []).append(task)

        merged_events: list[dict[str, Any]] = []
        per_event_metadata: list[dict[str, Any]] = []
        for event_index, task_group in sorted(event_tasks.items()):
            successful_payloads: list[list[dict[str, Any]]] = []
            event_metadata: list[dict[str, Any]] = []
            event_errors: list[str] = []
            for task in task_group:
                task_result = task_results.get(task.batch_key)
                if not task_result or task_result.get("status") != "ok":
                    event_errors.append(
                        str(
                            (task_result or {}).get("error")
                            or "Batch result did not include a response for this sample."
                        )
                    )
                    continue
                successful_payloads.append(task_result["payload"])
                event_metadata.append(task_result.get("metadata", {}))
            if not successful_payloads:
                continue
            if self_consistency and self_consistency_samples > 1:
                if len(successful_payloads) < self_consistency_min_successful_samples:
                    continue
                event_merged, _, _, _ = gemini_event_gen.merge_self_consistency_events(
                    successful_payloads, text
                )
                merged_events.extend(event_merged)
            else:
                merged_events.extend(successful_payloads[0])
            per_event_metadata.append(
                {
                    "event_index": event_index,
                    "samples_succeeded": len(successful_payloads),
                    "sample_errors": event_errors,
                    "metadata": _merge_metadata(event_metadata),
                }
            )

        updated_record["events"] = _post_filter_events(
            merged_events,
            text=text,
            strict_offsets=strict_offsets,
        )
        updated_record["metadata"] = _attach_cost_metadata(
            {
            "update": {
                "granularity": EVENT_REVIEW_GRANULARITY_SINGLE,
                "events_reviewed": len(per_event_metadata),
                "per_event": per_event_metadata,
            }
            }
        )
        return updated_record

    successful_events: list[list[dict[str, Any]]] = []
    metadata_list: list[dict[str, Any]] = []
    sample_errors: list[str] = []
    for task in tasks:
        task_result = task_results.get(task.batch_key)
        if not task_result or task_result.get("status") != "ok":
            sample_errors.append(
                str(
                    (task_result or {}).get("error")
                    or "Batch result did not include a response for this sample."
                )
            )
            continue
        successful_events.append(task_result["payload"])
        metadata_list.append(task_result.get("metadata", {}))

    if not successful_events:
        error = sample_errors[0] if sample_errors else "No successful batch samples."
        return {**updated_record, "status": "error", "error": error}

    if (
        self_consistency
        and target_mode != TARGET_MODE_ARGUMENTS
        and self_consistency_samples > 1
    ):
        if len(successful_events) < self_consistency_min_successful_samples:
            return {
                **updated_record,
                "status": "error",
                "error": (
                    "Self-consistency failed: "
                    f"{len(successful_events)}/{self_consistency_samples} successful "
                    f"samples; minimum required is {self_consistency_min_successful_samples}."
                ),
            }
        merged_events, event_support_by_key, argument_support_by_event_key, threshold = (
            gemini_event_gen.merge_self_consistency_events(successful_events, text)
        )
        merged_events = _post_filter_events(
            merged_events,
            text=text,
            strict_offsets=strict_offsets,
        )
        updated_record["events"] = merged_events
        updated_record["metadata"] = _attach_cost_metadata(
            {
            "self_consistency": {
                "enabled": True,
                "samples_requested": self_consistency_samples,
                "successful_samples": len(successful_events),
                "minimum_successful_samples": self_consistency_min_successful_samples,
                "threshold": threshold,
                "sample_errors": sample_errors,
                "event_support": [
                    {
                        "start_char": start_char,
                        "end_char": end_char,
                        "event_type": event_type,
                        "support": support,
                    }
                    for (start_char, end_char, event_type), support in sorted(
                        event_support_by_key.items()
                    )
                ],
                "argument_support": [
                    {
                        "event_start_char": event_start,
                        "event_end_char": event_end,
                        "event_type": event_type,
                        "role": role,
                        "start_char": arg_start,
                        "end_char": arg_end,
                        "support": support,
                    }
                    for (event_start, event_end, event_type), support_by_argument in sorted(
                        argument_support_by_event_key.items()
                    )
                    for (arg_start, arg_end, role), support in sorted(
                        support_by_argument.items()
                    )
                ],
                "metadata": _merge_metadata(metadata_list),
            }
            }
        )
    else:
        updated_record["events"] = successful_events[0]
        updated_record["metadata"] = _attach_cost_metadata(
            {"update": _merge_metadata(metadata_list)}
        )

    return updated_record


async def _call_update_once(
    *,
    client: GeminiLLMClient,
    record_id: str,
    system_prompt: str,
    prompt: str,
    response_format: dict[str, Any],
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    verbose: bool,
    call_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = None
    for attempt in range(max_retries + 1):
        try:
            gemini_event_gen.log_llm_call(
                enabled=verbose,
                record_id=record_id,
                step="update",
                call_type=f"{call_type} attempt={attempt + 1}",
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
        call_type=call_type,
        system_prompt=system_prompt,
        prompt=prompt,
        answer=raw_answer,
    )
    return raw_answer, response.metadata or {}


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
    target_mode: str,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
    examples: list[dict[str, Any]] | None,
    example_sample_size: int | None,
    example_retrieval: str,
    example_retriever: Any | None,
    enable_verifier: bool,
    self_consistency: bool,
    self_consistency_samples: int,
    self_consistency_temperature: float,
    self_consistency_min_successful_samples: int,
    event_review_granularity: str,
) -> dict[str, Any]:
    text = str(record.get("text", ""))
    title = str(record.get("title", ""))
    record_id = str(record.get("id"))
    prompt = _build_update_prompt(
        record=record,
        output_mode=output_mode,
        target_mode=target_mode,
        ontology_text=ontology_text,
        user_prompt_template=user_prompt_template,
        examples=examples,
        example_sample_size=example_sample_size,
        example_retrieval=example_retrieval,
        example_retriever=example_retriever,
        argument_roles=argument_roles,
        event_argument_roles=event_argument_roles,
        location_types=location_types,
    )
    system_prompt = _append_instruction_block(
        system_prompt,
        TARGET_MODE_SYSTEM_INSTRUCTIONS[target_mode],
    )
    response_format = _response_format_for_output_mode(output_mode)
    verifier_enabled = (
        enable_verifier
        and output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        and target_mode != TARGET_MODE_ARGUMENTS
    )
    self_consistency_enabled = (
        self_consistency
        and output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
        and target_mode != TARGET_MODE_ARGUMENTS
        and self_consistency_samples > 1
    )

    if (
        event_review_granularity == EVENT_REVIEW_GRANULARITY_SINGLE
        and target_mode == TARGET_MODE_EVENTS
        and output_mode != gemini_event_gen.OUTPUT_MODE_SPANS
    ):
        merged_events: list[dict[str, Any]] = []
        per_event_metadata: list[dict[str, Any]] = []
        for event_index, event in enumerate(record.get("events", []) or []):
            if not isinstance(event, dict):
                continue
            event_record = _record_with_selected_events(record, [event])
            event_prompt = _build_update_prompt(
                record=event_record,
                output_mode=output_mode,
                target_mode=target_mode,
                ontology_text=ontology_text,
                user_prompt_template=user_prompt_template,
                examples=examples,
                example_sample_size=example_sample_size,
                example_retrieval=example_retrieval,
                example_retriever=example_retriever,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                location_types=location_types,
            )
            if not self_consistency_enabled:
                raw_answer, metadata = await _call_update_once(
                    client=client,
                    record_id=f"{record_id}#{event_index}",
                    system_prompt=system_prompt,
                    prompt=event_prompt,
                    response_format=response_format,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    verbose=verbose,
                    call_type="update",
                )
                merged_events.extend(
                    _clean_updated_payload(
                        raw_answer,
                        event_record,
                        text,
                        labels,
                        output_mode,
                        target_mode,
                        strict_offsets,
                        argument_roles,
                        event_argument_roles,
                        location_types,
                    )
                )
                per_event_metadata.append(
                    {
                        "event_index": event_index,
                        "samples_succeeded": 1,
                        "metadata": metadata,
                    }
                )
                continue

            sample_events: list[list[dict[str, Any]]] = []
            sample_metadata: list[dict[str, Any]] = []
            for sample_index in range(self_consistency_samples):
                override_client = (
                    client
                    if sample_index == 0
                    and client.temperature == self_consistency_temperature
                    else GeminiLLMClient(
                        model_name=client.model_name,
                        system_prompt=None,
                        temperature=self_consistency_temperature,
                        max_tokens=client.max_tokens,
                    )
                )
                raw_answer, metadata = await _call_update_once(
                    client=override_client,
                    record_id=f"{record_id}#{event_index}",
                    system_prompt=system_prompt,
                    prompt=event_prompt,
                    response_format=response_format,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    max_backoff=max_backoff,
                    verbose=verbose,
                    call_type=(
                        "self_consistency_update "
                        f"sample={sample_index + 1}/{self_consistency_samples}"
                    ),
                )
                sample_events.append(
                    _clean_updated_payload(
                        raw_answer,
                        event_record,
                        text,
                        labels,
                        output_mode,
                        target_mode,
                        strict_offsets,
                        argument_roles,
                        event_argument_roles,
                        location_types,
                    )
                )
                sample_metadata.append(metadata)

            if len(sample_events) < self_consistency_min_successful_samples:
                continue
            event_merged, _, _, _ = gemini_event_gen.merge_self_consistency_events(
                sample_events, text
            )
            merged_events.extend(event_merged)
            per_event_metadata.append(
                {
                    "event_index": event_index,
                    "samples_succeeded": len(sample_events),
                    "metadata": _merge_metadata(sample_metadata),
                }
            )

        updated_record = dict(record)
        updated_record["events"] = _post_filter_events(
            merged_events,
            text=text,
            strict_offsets=strict_offsets,
        )
        metadata: dict[str, Any] = {
            "update": {
                "granularity": EVENT_REVIEW_GRANULARITY_SINGLE,
                "events_reviewed": len(per_event_metadata),
                "per_event": per_event_metadata,
            }
        }
        if verifier_enabled:
            decisions, verifier_metadata = await gemini_event_gen.verify_events(
                client=client,
                ontology_text=ontology_text,
                title=title,
                text=text,
                events=updated_record["events"],
                system_prompt=system_prompt,
                verifier_prompt_template=gemini_event_gen.DEFAULT_VERIFIER_PROMPT,
                verbose=verbose,
                record_id=record_id,
                step="update_verifier",
            )
            updated_record["events"] = _post_filter_events(
                adjudication.apply_verifier_decisions(
                    updated_record["events"], decisions
                ),
                text=text,
                strict_offsets=strict_offsets,
            )
            metadata["verifier"] = {
                "enabled": True,
                "decisions": decisions,
                "metadata": verifier_metadata,
            }
        updated_record["metadata"] = _attach_cost_metadata(metadata)
        return updated_record

    updated_record = dict(record)
    if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS or not self_consistency_enabled:
        raw_answer, metadata = await _call_update_once(
            client=client,
            record_id=record_id,
            system_prompt=system_prompt,
            prompt=prompt,
            response_format=response_format,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            verbose=verbose,
            call_type="update",
        )
        if output_mode == gemini_event_gen.OUTPUT_MODE_SPANS:
            updated_record["spans"] = _clean_updated_payload(
                raw_answer,
                record,
                text,
                labels,
                output_mode,
                target_mode,
                strict_offsets,
                argument_roles,
                event_argument_roles,
                location_types,
            )
        else:
            updated_record["events"] = _clean_updated_payload(
                raw_answer,
                record,
                text,
                labels,
                output_mode,
                target_mode,
                strict_offsets,
                argument_roles,
                event_argument_roles,
                location_types,
            )
        if verifier_enabled:
            decisions, verifier_metadata = await gemini_event_gen.verify_events(
                client=client,
                ontology_text=ontology_text,
                title=title,
                text=text,
                events=updated_record["events"],
                system_prompt=system_prompt,
                verifier_prompt_template=gemini_event_gen.DEFAULT_VERIFIER_PROMPT,
                verbose=verbose,
                record_id=record_id,
                step="update_verifier",
            )
            updated_record["events"] = _post_filter_events(
                adjudication.apply_verifier_decisions(
                    updated_record["events"], decisions
                ),
                text=text,
                strict_offsets=strict_offsets,
            )
            updated_record["metadata"] = _attach_cost_metadata(
                {
                "update": metadata,
                "verifier": {
                    "enabled": True,
                    "decisions": decisions,
                    "metadata": verifier_metadata,
                },
                }
            )
        elif metadata:
            updated_record["metadata"] = _attach_cost_metadata({"update": metadata})
        return updated_record

    sample_events: list[list[dict[str, Any]]] = []
    sample_metadata: list[dict[str, Any]] = []
    for sample_index in range(self_consistency_samples):
        override_client = (
            client
            if sample_index == 0 and client.temperature == self_consistency_temperature
            else GeminiLLMClient(
                model_name=client.model_name,
                system_prompt=None,
                temperature=self_consistency_temperature,
                max_tokens=client.max_tokens,
            )
        )
        raw_answer, metadata = await _call_update_once(
            client=override_client,
            record_id=record_id,
            system_prompt=system_prompt,
            prompt=prompt,
            response_format=response_format,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            verbose=verbose,
            call_type=f"self_consistency_update sample={sample_index + 1}/{self_consistency_samples}",
        )
        cleaned_events = _clean_updated_payload(
            raw_answer,
            record,
            text,
            labels,
            output_mode,
            target_mode,
            strict_offsets,
            argument_roles,
            event_argument_roles,
            location_types,
        )
        sample_events.append(cleaned_events)
        sample_metadata.append(metadata)

    if len(sample_events) < self_consistency_min_successful_samples:
        raise RuntimeError(
            "Self-consistency failed: "
            f"{len(sample_events)}/{self_consistency_samples} successful samples; "
            f"minimum required is {self_consistency_min_successful_samples}."
        )

    merged_events, event_support_by_key, argument_support_by_event_key, threshold = (
        gemini_event_gen.merge_self_consistency_events(sample_events, text)
    )
    merged_events = _post_filter_events(
        merged_events,
        text=text,
        strict_offsets=strict_offsets,
    )
    metadata = {
        "self_consistency": {
            "enabled": True,
            "samples_requested": self_consistency_samples,
            "temperature": self_consistency_temperature,
            "minimum_successful_samples": self_consistency_min_successful_samples,
            "successful_samples": len(sample_events),
            "threshold": threshold,
            "event_support": [
                {
                    "start_char": start_char,
                    "end_char": end_char,
                    "event_type": event_type,
                    "support": support,
                }
                for (start_char, end_char, event_type), support in sorted(
                    event_support_by_key.items()
                )
            ],
            "argument_support": [
                {
                    "event_start_char": event_start,
                    "event_end_char": event_end,
                    "event_type": event_type,
                    "role": role,
                    "start_char": arg_start,
                    "end_char": arg_end,
                    "support": support,
                }
                for (event_start, event_end, event_type), support_by_argument in sorted(
                    argument_support_by_event_key.items()
                )
                for (arg_start, arg_end, role), support in sorted(
                    support_by_argument.items()
                )
            ],
            "metadata": _merge_metadata(sample_metadata),
        }
    }
    if verifier_enabled:
        decisions, verifier_metadata = await gemini_event_gen.verify_events(
            client=client,
            ontology_text=ontology_text,
            title=title,
            text=text,
            events=merged_events,
            system_prompt=system_prompt,
            verifier_prompt_template=gemini_event_gen.DEFAULT_VERIFIER_PROMPT,
            verbose=verbose,
            record_id=record_id,
            step="self_consistency_update_verifier",
        )
        merged_events = _post_filter_events(
            adjudication.apply_verifier_decisions(merged_events, decisions),
            text=text,
            strict_offsets=strict_offsets,
        )
        metadata["verifier"] = {
            "enabled": True,
            "decisions": decisions,
            "metadata": verifier_metadata,
        }
    updated_record["events"] = merged_events
    updated_record["metadata"] = _attach_cost_metadata(metadata)

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
    target_mode: str,
    argument_roles: set[str] | None,
    event_argument_roles: dict[str, list[str]] | None,
    location_types: set[str] | None,
    examples: list[dict[str, Any]] | None,
    example_sample_size: int | None,
    example_retrieval: str,
    example_retriever: Any | None,
    enable_verifier: bool,
    self_consistency: bool,
    self_consistency_samples: int,
    self_consistency_temperature: float,
    self_consistency_min_successful_samples: int,
    event_review_granularity: str,
) -> list[dict[str, Any]]:
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
            target_mode=target_mode,
            argument_roles=argument_roles,
            event_argument_roles=event_argument_roles,
            location_types=location_types,
            examples=examples,
            example_sample_size=example_sample_size,
            example_retrieval=example_retrieval,
            example_retriever=example_retriever,
            enable_verifier=enable_verifier,
            self_consistency=self_consistency,
            self_consistency_samples=self_consistency_samples,
            self_consistency_temperature=self_consistency_temperature,
            self_consistency_min_successful_samples=self_consistency_min_successful_samples,
            event_review_granularity=event_review_granularity,
        )
        for record in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    batch_summaries: list[dict[str, Any]] = []

    with output_path.open("a", encoding="utf-8") as handle:
        for idx, result in enumerate(results):
            record = batch[idx]
            if isinstance(result, Exception):
                LOGGER.error("Failed to process id=%s: %s", record.get("id"), result)
                # Save the original or a failed state
                record["status"] = "error"
                record["error"] = str(result)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                batch_summaries.append(
                    _summarize_record_update(
                        record,
                        None,
                        output_mode=output_mode,
                        error=str(result),
                    )
                )
            else:
                result["status"] = "ok"
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                batch_summaries.append(
                    _summarize_record_update(
                        record,
                        result,
                        output_mode=output_mode,
                    )
                )

    return batch_summaries


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
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
    example_count = gemini_event_gen.effective_example_count(
        args.example_sample_size, args.example_top_k
    )
    raw_examples = (
        gemini_event_gen.load_examples(args.examples)
        if example_count is not None
        else []
    )
    examples = (
        gemini_event_gen.compact_example_records(raw_examples, args.output_mode)
        if raw_examples
        else []
    )
    example_retriever = (
        gemini_event_gen.build_example_retriever(examples, args.output_mode)
        if args.example_retrieval == gemini_event_gen.EXAMPLE_RETRIEVAL_BM25 and examples
        else None
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

    if args.batch_api:
        record_summaries: list[dict[str, Any]] = []
        batch_system_prompt = _append_instruction_block(
            system_prompt,
            TARGET_MODE_SYSTEM_INSTRUCTIONS[args.target_mode],
        )
        extraction_request_config = gemini_event_gen.batch_request_config(
            args,
            batch_system_prompt,
            _batch_response_schema(args.output_mode),
            temperature=(
                args.self_consistency_temperature
                if (
                    args.self_consistency
                    and args.output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
                    and args.target_mode != TARGET_MODE_ARGUMENTS
                )
                else args.temperature
            ),
        )
        extraction_tasks = _build_update_batch_tasks(
            pending,
            output_mode=args.output_mode,
            target_mode=args.target_mode,
            ontology_text=ontology_text,
            user_prompt_template=user_prompt,
            examples=examples,
            example_sample_size=example_count,
            example_retrieval=args.example_retrieval,
            example_retriever=example_retriever,
            argument_roles=argument_roles,
            event_argument_roles=event_argument_roles,
            location_types=location_types,
            self_consistency=(
                args.self_consistency
                and args.output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
                and args.target_mode != TARGET_MODE_ARGUMENTS
            ),
            self_consistency_samples=args.self_consistency_samples,
            event_review_granularity=args.event_review_granularity,
        )
        extraction_task_by_key = {task.batch_key: task for task in extraction_tasks}
        extraction_tasks_by_record: dict[str, list[gemini_event_gen.ExtractionBatchTask]] = {}
        for task in extraction_tasks:
            extraction_tasks_by_record.setdefault(task.record_id, []).append(task)
        extraction_chunks = gemini_event_gen.build_batch_chunks(
            tasks=extraction_tasks,
            stage="update",
            batch_size=args.batch_size,
            request_config=extraction_request_config,
        )
        extraction_task_results: dict[str, dict[str, Any]] = {}
        extraction_chunk_results = await gemini_event_gen.execute_batch_stage(
            client=client,
            chunks=extraction_chunks,
            output_path=args.output,
            model_name=args.model,
            batch_display_name=args.batch_display_name,
            poll_interval_seconds=args.batch_poll_interval_seconds,
            workers=args.workers,
        ) if extraction_chunks else []
        for chunk_result in extraction_chunk_results:
            chunk_keys = set(chunk_result["task_keys"])
            if chunk_result["status"] != "ok":
                for batch_key in chunk_keys:
                    extraction_task_results[batch_key] = {
                        "status": "error",
                        "error": str(chunk_result.get("error")),
                    }
                continue
            seen: set[str] = set()
            for line in gemini_event_gen.iter_jsonl(Path(chunk_result["result_path"])):
                batch_key = gemini_event_gen.batch_result_key(line)
                if batch_key not in chunk_keys:
                    LOGGER.warning(
                        "Skipping unexpected batch result stage=update key=%s",
                        batch_key,
                    )
                    continue
                task = extraction_task_by_key[batch_key]
                record = next(
                    item for item in pending if str(item.get("id")) == task.record_id
                )
                try:
                    payload, metadata = _parse_batch_update_payload(
                        line=line,
                        record=record,
                        event_index=task.event_index,
                        output_mode=args.output_mode,
                        target_mode=args.target_mode,
                        labels=labels,
                        strict_offsets=not args.fast_offsets,
                        argument_roles=argument_roles,
                        event_argument_roles=event_argument_roles,
                        location_types=location_types,
                    )
                    extraction_task_results[batch_key] = {
                        "status": "ok",
                        "payload": payload,
                        "metadata": metadata,
                    }
                except Exception as exc:
                    extraction_task_results[batch_key] = {
                        "status": "error",
                        "error": f"Failed to parse batch response: {exc}",
                    }
                seen.add(batch_key)
            for batch_key in chunk_keys - seen:
                extraction_task_results[batch_key] = {
                    "status": "error",
                    "error": "Batch result did not include a response for this task.",
                }

        aggregated_results: list[dict[str, Any]] = []
        final_results: dict[str, dict[str, Any]] = {}
        for record in pending:
            record_id = str(record.get("id"))
            result = _aggregate_batch_update_record(
                record=record,
                tasks=extraction_tasks_by_record.get(record_id, []),
                task_results=extraction_task_results,
                output_mode=args.output_mode,
                target_mode=args.target_mode,
                strict_offsets=not args.fast_offsets,
                self_consistency=args.self_consistency,
                self_consistency_samples=args.self_consistency_samples,
                self_consistency_min_successful_samples=args.self_consistency_min_successful_samples,
                event_review_granularity=args.event_review_granularity,
            )
            aggregated_results.append(result)
            final_results[record_id] = result

        verifier_enabled = (
            args.enable_verifier
            and args.output_mode == gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS
            and args.target_mode != TARGET_MODE_ARGUMENTS
        )
        verifier_chunk_results: list[dict[str, Any]] = []
        if verifier_enabled:
            verifier_request_config = gemini_event_gen.batch_request_config(
                args,
                batch_system_prompt,
                gemini_event_gen.BatchVerifierPayload,
            )
            verifier_tasks = gemini_event_gen.build_verifier_tasks(
                aggregated_results,
                args=args,
                ontology_text=ontology_text,
                verifier_prompt_template=gemini_event_gen.DEFAULT_VERIFIER_PROMPT,
            )
            verifier_task_by_key = {task.batch_key: task for task in verifier_tasks}
            verifier_chunks = gemini_event_gen.build_batch_chunks(
                tasks=verifier_tasks,
                stage="verifier",
                batch_size=args.batch_size,
                request_config=verifier_request_config,
            )
            verifier_task_results: dict[str, dict[str, Any]] = {}
            if verifier_chunks:
                verifier_chunk_results = await gemini_event_gen.execute_batch_stage(
                    client=client,
                    chunks=verifier_chunks,
                    output_path=args.output,
                    model_name=args.model,
                    batch_display_name=args.batch_display_name,
                    poll_interval_seconds=args.batch_poll_interval_seconds,
                    workers=args.workers,
                )
                for chunk_result in verifier_chunk_results:
                    chunk_keys = set(chunk_result["task_keys"])
                    if chunk_result["status"] != "ok":
                        for batch_key in chunk_keys:
                            verifier_task_results[batch_key] = {
                                "status": "error",
                                "error": str(chunk_result.get("error")),
                            }
                        continue
                    seen: set[str] = set()
                    for line in gemini_event_gen.iter_jsonl(Path(chunk_result["result_path"])):
                        batch_key = gemini_event_gen.batch_result_key(line)
                        if batch_key not in chunk_keys:
                            LOGGER.warning(
                                "Skipping unexpected batch result stage=verifier key=%s",
                                batch_key,
                            )
                            continue
                        task = verifier_task_by_key[batch_key]
                        try:
                            decisions, metadata = gemini_event_gen.parse_batch_verifier_payload(
                                line, task.events
                            )
                            verifier_task_results[batch_key] = {
                                "status": "ok",
                                "decisions": decisions,
                                "metadata": metadata,
                            }
                        except Exception as exc:
                            verifier_task_results[batch_key] = {
                                "status": "error",
                                "error": f"Failed to parse verifier batch response: {exc}",
                            }
                        seen.add(batch_key)
                    for batch_key in chunk_keys - seen:
                        verifier_task_results[batch_key] = {
                            "status": "error",
                            "error": "Batch result did not include a verifier response.",
                        }

            for result in aggregated_results:
                if result.get("status") == "error":
                    continue
                result_id = str(result.get("id"))
                events = result.get("events") or []
                metadata = result.setdefault("metadata", {})
                if not events:
                    metadata["verifier"] = {
                        "enabled": True,
                        "decisions": [],
                        "metadata": {},
                    }
                    final_results[result_id] = result
                    continue
                verifier_result = verifier_task_results.get(result_id)
                if verifier_result is None or verifier_result.get("status") != "ok":
                    final_results[result_id] = {
                        **result,
                        "status": "error",
                        "error": (
                            "Batch result did not include a verifier response."
                            if verifier_result is None
                            else str(verifier_result.get("error"))
                        ),
                    }
                    continue
                decisions = verifier_result["decisions"]
                result["events"] = _post_filter_events(
                    adjudication.apply_verifier_decisions(events, decisions),
                    text=str(result.get("text", "")),
                    strict_offsets=not args.fast_offsets,
                )
                metadata["verifier"] = {
                    "enabled": True,
                    "decisions": decisions,
                    "metadata": verifier_result["metadata"],
                }
                result["metadata"] = _attach_cost_metadata(metadata)
                final_results[result_id] = result

        extraction_chunk_metadata = [
            {
                key: chunk_result.get(key)
                for key in (
                    "stage",
                    "chunk_index",
                    "status",
                    "request_path",
                    "result_path",
                    "uploaded_file",
                    "batch_job_name",
                    "batch_output_file",
                    "attempts",
                )
                if key in chunk_result
            }
            | {"task_count": len(chunk_result.get("task_keys", []))}
            for chunk_result in extraction_chunk_results
        ]
        verifier_chunk_metadata = [
            {
                key: chunk_result.get(key)
                for key in (
                    "stage",
                    "chunk_index",
                    "status",
                    "request_path",
                    "result_path",
                    "uploaded_file",
                    "batch_job_name",
                    "batch_output_file",
                    "attempts",
                )
                if key in chunk_result
            }
            | {"task_count": len(chunk_result.get("task_keys", []))}
            for chunk_result in verifier_chunk_results
        ]
        pipeline_metadata = {
            "batch_api": True,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "update_chunks": extraction_chunk_metadata,
            **(
                {"verifier_chunks": verifier_chunk_metadata}
                if verifier_chunk_metadata
                else {}
            ),
        }
        with args.output.open("a", encoding="utf-8") as handle:
            for record in pending:
                result = final_results[str(record.get("id"))]
                if result.get("status") == "error":
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    record_summaries.append(
                        _summarize_record_update(
                            record,
                            None,
                            output_mode=args.output_mode,
                            error=str(result.get("error")),
                        )
                    )
                    continue
                result["status"] = "ok"
                result.setdefault("metadata", {})
                result["metadata"]["pipeline"] = dict(pipeline_metadata)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                record_summaries.append(
                    _summarize_record_update(
                        record,
                        result,
                        output_mode=args.output_mode,
                    )
                )

        summary = _build_run_summary(
            record_summaries,
            args=args,
            input_path=args.input,
            output_path=args.output,
        )
        summary_path = _summary_output_path(args.output)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "update summary records=%s changed=%s before_events=%s after_events=%s "
            "before_empty_arg_ratio=%.4f after_empty_arg_ratio=%.4f "
            "before_generic_ratio=%.4f after_generic_ratio=%.4f summary=%s",
            summary["processed_records"],
            summary["changed_records"],
            summary["before"]["events"],
            summary["after"]["events"],
            summary["before"]["empty_argument_event_ratio"],
            summary["after"]["empty_argument_event_ratio"],
            summary["before"]["generic_trigger_ratio"],
            summary["after"]["generic_trigger_ratio"],
            summary_path,
        )
        return
    record_summaries: list[dict[str, Any]] = []

    with tqdm(total=len(pending), desc="Updating annotations") as pbar:
        for i in range(0, len(pending), args.workers):
            batch = pending[i : i + args.workers]
            batch_summaries = await process_batch(
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
                target_mode=args.target_mode,
                argument_roles=argument_roles,
                event_argument_roles=event_argument_roles,
                location_types=location_types,
                examples=examples,
                example_sample_size=example_count,
                example_retrieval=args.example_retrieval,
                example_retriever=example_retriever,
                enable_verifier=args.enable_verifier,
                self_consistency=args.self_consistency,
                self_consistency_samples=args.self_consistency_samples,
                self_consistency_temperature=args.self_consistency_temperature,
                self_consistency_min_successful_samples=args.self_consistency_min_successful_samples,
                event_review_granularity=args.event_review_granularity,
            )
            record_summaries.extend(batch_summaries)
            pbar.update(len(batch))

    summary = _build_run_summary(
        record_summaries,
        args=args,
        input_path=args.input,
        output_path=args.output,
    )
    summary_path = _summary_output_path(args.output)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "update summary records=%s changed=%s before_events=%s after_events=%s "
        "before_empty_arg_ratio=%.4f after_empty_arg_ratio=%.4f "
        "before_generic_ratio=%.4f after_generic_ratio=%.4f summary=%s",
        summary["processed_records"],
        summary["changed_records"],
        summary["before"]["events"],
        summary["after"]["events"],
        summary["before"]["empty_argument_event_ratio"],
        summary["after"]["empty_argument_event_ratio"],
        summary["before"]["generic_trigger_ratio"],
        summary["after"]["generic_trigger_ratio"],
        summary_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--batch-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit update and optional verifier requests through Gemini Batch API. "
            "Stages are chunked by --batch-size and submitted in parallel up to "
            "--workers."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Maximum number of requests per Gemini batch JSONL file.",
    )
    parser.add_argument(
        "--batch-poll-interval-seconds",
        type=int,
        default=30,
        help="Polling interval for --batch-api jobs.",
    )
    parser.add_argument(
        "--batch-display-name",
        default=None,
        help="Optional display name for the Gemini batch job.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--temperature", type=float, default=0.0)
    # parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="disable")
    parser.add_argument(
        "--output-mode",
        choices=gemini_event_gen.OUTPUT_MODES,
        default=gemini_event_gen.OUTPUT_MODE_EVENTS_WITH_ARGS,
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default=TARGET_MODE_ALL,
        help="Restrict updates to all annotations, only events, or only arguments.",
    )
    parser.add_argument(
        "--event-review-granularity",
        choices=EVENT_REVIEW_GRANULARITIES,
        default=EVENT_REVIEW_GRANULARITY_RECORD,
        help="When --target-mode events is used, review all candidate events together or one event per request.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--user-prompt", type=Path, default=None)
    parser.add_argument(
        "--examples",
        type=Path,
        default=DEFAULT_EXAMPLES,
        help="Manual examples JSONL used when --example-sample-size is set.",
    )
    parser.add_argument(
        "--example-sample-size",
        type=int,
        default=3,
        help="Number of compact example windows to retrieve per update call.",
    )
    parser.add_argument(
        "--example-retrieval",
        choices=[
            gemini_event_gen.EXAMPLE_RETRIEVAL_RANDOM,
            gemini_event_gen.EXAMPLE_RETRIEVAL_BM25,
        ],
        default=gemini_event_gen.EXAMPLE_RETRIEVAL_BM25,
        help="How to choose few-shot examples when examples are enabled.",
    )
    parser.add_argument(
        "--example-top-k",
        type=int,
        default=None,
        help="Optional explicit top-k for example retrieval; defaults to --example-sample-size.",
    )
    parser.add_argument(
        "--enable-verifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a verifier pass over updated events.",
    )
    parser.add_argument(
        "--self-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run multiple repair samples and majority-merge them.",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=3,
        help="Number of repair samples when --self-consistency is enabled.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used for self-consistency repair calls.",
    )
    parser.add_argument(
        "--self-consistency-min-successful-samples",
        type=int,
        default=2,
        help="Minimum successful samples required for a self-consistency result.",
    )
    parser.add_argument("--fast-offsets", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.example_sample_size is not None and args.example_sample_size < 1:
        parser.error("--example-sample-size must be >= 1")
    if args.example_top_k is not None and args.example_top_k < 1:
        parser.error("--example-top-k must be >= 1")
    if args.batch_poll_interval_seconds < 1:
        parser.error("--batch-poll-interval-seconds must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.self_consistency:
        if args.self_consistency_samples < 1:
            parser.error("--self-consistency-samples must be >= 1")
        if args.self_consistency_min_successful_samples < 1:
            parser.error("--self-consistency-min-successful-samples must be >= 1")
        if (
            args.self_consistency_min_successful_samples
            > args.self_consistency_samples
        ):
            parser.error(
                "--self-consistency-min-successful-samples cannot exceed "
                "--self-consistency-samples"
            )
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


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
