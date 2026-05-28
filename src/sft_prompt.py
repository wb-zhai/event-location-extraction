from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT_WITH_ARGS = (
    "You extract risk-factor events and their location arguments from text.\n"
)
SYSTEM_PROMPT_EVENTS_ONLY = "You extract risk-factor events from text.\n"

SCHEMA_WITH_ARGS_OFFSETS = (
    '{"events":[{"event_type":"...", "trigger":{"start":0, "end":1, "text":"..."}, '
    '"arguments":[{"role":"...", "span":{"start":0, "end":1, "text":"..."}, "location_type":"..."}]}]}'
)
SCHEMA_EVENTS_ONLY_OFFSETS = (
    '{"events":[{"event_type":"...", "trigger":{"start":0, "end":1, "text":"..."}}]}'
)
SCHEMA_WITH_ARGS_CONTEXT = (
    '{"events":[{"event_type":"...", "trigger":{"text":"...", "left_context":"...", "right_context":"..."}, '
    '"arguments":[{"role":"...", "span":{"text":"...", "left_context":"...", "right_context":"..."}, "location_type":"..."}]}]}'
)
SCHEMA_EVENTS_ONLY_CONTEXT = (
    '{"events":[{"event_type":"...", "trigger":{"text":"...", "left_context":"...", "right_context":"..."}}]}'
)

USER_TEMPLATE_EVENTS_ONLY = (
    "Extract all risk-factor events that clearly match the provided event labels.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Return valid JSON only. Use the exact output schema described above."
)

USER_TEMPLATE = (
    "Extract all risk-factor events that clearly match the provided event labels and attach only their location arguments.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Select argument roles from the following set: {argument_roles}\n"
    "Select location types from the following set: {location_types}\n"
    "Return valid JSON only. Use the exact output schema described above."
)


def _build_system_prompt(*, events_only: bool, omit_offsets: bool) -> str:
    task_line = SYSTEM_PROMPT_EVENTS_ONLY if events_only else SYSTEM_PROMPT_WITH_ARGS
    if events_only and omit_offsets:
        schema = SCHEMA_EVENTS_ONLY_CONTEXT
        rules = [
            "1) trigger.text must be an exact substring from the document.",
            "2) Put trigger text under trigger.text and include left_context and right_context.",
            "3) left_context and right_context must be exact verbatim text immediately before and after that trigger mention in the document.",
            "4) Target about 4 words for left_context and about 4 words for right_context; use fewer only when the mention is near a boundary or fewer words are needed.",
            "5) Use the shortest context that uniquely identifies that occurrence. Do not include the trigger text itself inside left_context or right_context.",
            "6) Either context field may be empty when the mention touches a document boundary.",
            "7) Use only the provided event labels.",
            "8) Extract only events that clearly match the provided event labels.",
            "9) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.",
            '10) If no valid event matches the provided labels, return {"events":[]}.',
            "11) Do not paraphrase. Do not add unsupported events.",
        ]
    elif events_only:
        schema = SCHEMA_EVENTS_ONLY_OFFSETS
        rules = [
            "1) trigger.text must be an exact substring from the document.",
            "2) start/end must be character offsets in the provided document.",
            "3) Put trigger offsets under trigger.",
            "4) When start/end are present, do not output left_context or right_context.",
            "5) Use only the provided event labels.",
            "6) Extract only events that clearly match the provided event labels.",
            "7) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.",
            '8) If no valid event matches the provided labels, return {"events":[]}.',
            "9) Do not paraphrase. Do not add unsupported events.",
        ]
    elif omit_offsets:
        schema = SCHEMA_WITH_ARGS_CONTEXT
        rules = [
            "1) trigger.text and argument span.text must be exact substrings from the document.",
            "2) Put trigger text under trigger.text and argument text under span.text.",
            "3) For every trigger and argument span, left_context and right_context must be exact verbatim text immediately before and after that mention in the document.",
            "4) Target about 4 words for each left_context and about 4 words for each right_context; use fewer only when the mention is near a boundary or fewer words are needed.",
            "5) Use the shortest context that uniquely identifies that occurrence. Do not include the trigger text or span text itself inside the context fields.",
            "6) Either context field may be empty when the mention touches a document boundary.",
            "7) Use only the provided event labels, argument roles, and location types.",
            "8) Extract only events that clearly match the provided event labels.",
            "9) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.",
            "10) Argument spans must be the shortest exact location phrase.",
            '11) If an event has no valid location arguments, return "arguments":[] for that event.',
            '12) If no valid event matches the provided labels, return {"events":[]}.',
            "13) Do not paraphrase. Do not add unsupported events or arguments.",
            "14) Never place event labels in the argument role field.",
        ]
    else:
        schema = SCHEMA_WITH_ARGS_OFFSETS
        rules = [
            "1) trigger.text and argument span.text must be exact substrings from the document.",
            "2) start/end must be character offsets in the provided document.",
            "3) Put trigger offsets under trigger and argument offsets under span.",
            "4) When start/end are present, do not output left_context or right_context.",
            "5) Use only the provided event labels, argument roles, and location types.",
            "6) Extract only events that clearly match the provided event labels.",
            "7) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.",
            "8) Argument spans must be the shortest exact location phrase.",
            '9) If an event has no valid location arguments, return "arguments":[] for that event.',
            '10) If no valid event matches the provided labels, return {"events":[]}.',
            "11) Do not paraphrase. Do not add unsupported events or arguments.",
            "12) Never place event labels in the argument role field.",
        ]

    return task_line + "Return JSON only with this shape:\n" + schema + "\nRules:\n" + "\n".join(rules)


def build_user_prompt(
    document: str,
    event_labels: str | list[str] | dict[str, str],
    argument_roles: str | list[str] | dict[str, str],
    location_types: str | list[str] | dict[str, str],
    *,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> str:
    def _render_candidates(value: str | list[str] | dict[str, str]) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    if events_only:
        return USER_TEMPLATE_EVENTS_ONLY.format(
            document=document,
            event_labels=_render_candidates(event_labels),
        )
    return USER_TEMPLATE.format(
        document=document,
        event_labels=_render_candidates(event_labels),
        argument_roles=_render_candidates(argument_roles),
        location_types=_render_candidates(location_types),
    )

def build_messages(
    document: str,
    event_labels: str | list[str] | dict[str, str],
    argument_roles: str | list[str] | dict[str, str],
    location_types: str | list[str] | dict[str, str],
    *,
    answer_obj: dict[str, Any] | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> list[dict[str, str]]:
    system_prompt = _build_system_prompt(
        events_only=events_only,
        omit_offsets=omit_offsets,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(
                document,
                event_labels,
                argument_roles,
                location_types,
                events_only=events_only,
            ),
        },
    ]
    if answer_obj is not None:
        messages.append(
            {"role": "assistant", "content": json.dumps(answer_obj, ensure_ascii=False)}
        )
    return messages


def _chat_template_kwargs(tokenizer) -> dict[str, Any]:
    # name_or_path = getattr(tokenizer, "name_or_path", "")
    # print(f"Tokenizer name_or_path: {name_or_path}")
    # if isinstance(name_or_path, str) and (
    #     "qwen3" in name_or_path.lower() or "qwen3.5" in name_or_path.lower()
    # ):
    #     # Qwen3 hybrid-thinking models think by default unless explicitly disabled.
    # return {}
    return {"enable_thinking": False}


def render_chat(
    tokenizer,
    document: str,
    event_labels: str | list[str] | dict[str, str],
    argument_roles: str | list[str] | dict[str, str],
    location_types: str | list[str] | dict[str, str],
    *,
    answer_obj: dict[str, Any] | None = None,
    add_generation_prompt: bool,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        build_messages(
            document,
            event_labels,
            argument_roles,
            location_types,
            answer_obj=answer_obj,
            events_only=events_only,
            omit_offsets=omit_offsets,
        ),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **_chat_template_kwargs(tokenizer),
    )
