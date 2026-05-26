from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You extract risk-factor events and their location arguments from text.\n"
    "Return JSON only with this shape:\n"
    '{"events":[{"event_type":"...", "start":0, "end":1, "text":"...", "arguments":[{"role":"...", "start":0, "end":1, "text":"...", "location_type":"..."}]}]}\n'
    "Rules:\n"
    "1) text must be exact substring from document.\n"
    "2) start/end must be character offsets in the provided document.\n"
    "3) Argument text must be exact substring from document.\n"
    "4) Use only the provided event labels, argument roles, and location types.\n"
    "5) Extract only events that clearly match the provided event labels.\n"
    "6) Event spans must be the shortest exact trigger phrase, not a full clause or sentence.\n"
    "7) Argument spans must be the shortest exact location phrase.\n"
    "8) If no valid event matches the ontology, return {\"events\":[]}.\n"
    "9) Do not paraphrase. Do not add unsupported events or arguments.\n"
    "10) Never place event labels in the argument role field."
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
) -> str:
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
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                document,
                event_labels,
                argument_roles,
                location_types,
            ),
        },
    ]
    if answer_obj is not None:
        messages.append(
            {"role": "assistant", "content": json.dumps(answer_obj, ensure_ascii=False)}
        )
    return messages


def render_chat(
    tokenizer,
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    *,
    answer_obj: dict[str, Any] | None = None,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        build_messages(
            document,
            event_labels,
            argument_roles,
            location_types,
            answer_obj=answer_obj,
        ),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
