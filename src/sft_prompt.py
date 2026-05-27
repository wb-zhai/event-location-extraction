from __future__ import annotations

import json
from typing import Any

import re

SYSTEM_PROMPT = (
    "You extract risk-factor events and their location arguments from text.\n"
    "Return JSON only with this shape:\n"
    '{"events":[{"event_type":"...", "trigger":{"start":0, "end":1, "text":"..."}, "arguments":[{"role":"...", "span":{"start":0, "end":1, "text":"..."}, "location_type":"..."}]}]}\n'
    "Rules:\n"
    "1) trigger.text and argument span.text must be exact substrings from the document.\n"
    "2) start/end must be character offsets in the provided document.\n"
    "3) Put trigger offsets under trigger and argument offsets under span.\n"
    "4) Use only the provided event labels, argument roles, and location types.\n"
    "5) Extract only events that clearly match the provided event labels.\n"
    "6) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.\n"
    "7) Argument spans must be the shortest exact location phrase.\n"
    "8) If no valid event matches the ontology, return {\"events\":[]}.\n"
    "9) Do not paraphrase. Do not add unsupported events or arguments.\n"
    "10) Never place event labels in the argument role field."
)

SYSTEM_PROMPT_EVENTS_ONLY = (
    "You extract risk-factor events from text.\n"
    "Return JSON only with this shape:\n"
    '{"events":[{"event_type":"...", "trigger":{"start":0, "end":1, "text":"..."}}]}\n'
    "Rules:\n"
    "1) trigger.text must be an exact substring from the document.\n"
    "2) start/end must be character offsets in the provided document.\n"
    "3) Put trigger offsets under trigger.\n"
    "4) Use only the provided event labels.\n"
    "5) Extract only events that clearly match the provided event labels.\n"
    "6) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.\n"
    "7) If no valid event matches the ontology, return {\"events\":[]}.\n"
    "8) Do not paraphrase. Do not add unsupported events."
)

USER_TEMPLATE_EVENTS_ONLY = (
    "Extract all ontology-matching risk-factor events.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Return valid JSON only. Use short trigger spans."
)


USER_TEMPLATE = (
    "Extract all ontology-matching risk-factor events and their location arguments.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Select argument roles from the following set: {argument_roles}\n"
    "Select location types from the following set: {location_types}\n"
    "Return valid JSON only. Use short trigger spans and short location spans."
)


def build_user_prompt(
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    *,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> str:
    if events_only:
        return USER_TEMPLATE_EVENTS_ONLY.format(
            document=document,
            event_labels=json.dumps(event_labels, ensure_ascii=False),
        )
    return USER_TEMPLATE.format(
        document=document,
        event_labels=json.dumps(event_labels, ensure_ascii=False),
        argument_roles=json.dumps(argument_roles, ensure_ascii=False),
        location_types=json.dumps(location_types, ensure_ascii=False),
    )

def build_messages(
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    *,
    answer_obj: dict[str, Any] | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> list[dict[str, str]]:
    system_prompt = SYSTEM_PROMPT_EVENTS_ONLY if events_only else SYSTEM_PROMPT
    if omit_offsets:
        system_prompt = system_prompt.replace('"start":0, "end":1, ', '')
        system_prompt = re.sub(r'\n\d+\) start/end must be character offsets in the provided document.', '', system_prompt)
        system_prompt = re.sub(r'\n\d+\) Put trigger offsets under trigger( and argument offsets under span)?\.', '', system_prompt)
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
    name_or_path = getattr(tokenizer, "name_or_path", "")
    if isinstance(name_or_path, str) and (
        "qwen3" in name_or_path.lower() or "qwen3.5" in name_or_path.lower()
    ):
        # Qwen3 hybrid-thinking models think by default unless explicitly disabled.
        return {"enable_thinking": False}
    return {}


def render_chat(
    tokenizer,
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
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
