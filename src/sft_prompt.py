from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You extract events and location arguments from text.\n"
    "Return JSON only with this shape:\n"
    '{"events":[{"event_type":"...", "start":0, "end":1, "text":"...", "arguments":[{"role":"...", "start":0, "end":1, "text":"...", "location_type":"..."}]}]}\n'
    "Rules:\n"
    "1) text must be exact substring from document.\n"
    "2) start/end must be character offsets in the provided document.\n"
    "3) Argument text must be exact substring from document.\n"
    "4) Use only the provided event labels, argument roles, and location types.\n"
    "5) Do not paraphrase. Do not add unsupported events or arguments."
)

USER_TEMPLATE = (
    "Extract all events and their location arguments.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Select argument roles from the following set: {argument_roles}\n"
    "Select location types from the following set: {location_types}\n"
    "Return valid JSON only."
)


def build_user_prompt(
    document: str,
    event_labels: dict[str, str],
    argument_roles: dict[str, str],
    location_types: dict[str, str],
) -> str:
    return USER_TEMPLATE.format(
        document=document,
        event_labels=json.dumps(event_labels, ensure_ascii=False),
        argument_roles=json.dumps(argument_roles, ensure_ascii=False),
        location_types=json.dumps(location_types, ensure_ascii=False),
    )


def build_messages(
    document: str,
    event_labels: dict[str, str],
    argument_roles: dict[str, str],
    location_types: dict[str, str],
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
    event_labels: dict[str, str],
    argument_roles: dict[str, str],
    location_types: dict[str, str],
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
