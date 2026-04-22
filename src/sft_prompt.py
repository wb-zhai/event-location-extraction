from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You extract events from text.\n"
    "Return JSON only with this shape:\n"
    '{"events":[{"event_type":"...", "start":0, "end":1, "text":"..."}]}\n'
    "Rules:\n"
    "1) text must be exact substring from document.\n"
    "2) start/end must be character offsets in the provided document.\n"
    "3) Do not paraphrase. Do not add unsupported events."
)

USER_TEMPLATE = (
    "Extract all events.\n\n"
    "Document:\n{document}\n\n"
    "Select event labels from the following set: {event_labels}\n"
    "Return valid JSON only."
)


def build_user_prompt(document: str, event_labels: list[str]) -> str:
    return USER_TEMPLATE.format(
        document=document,
        event_labels=json.dumps(event_labels, ensure_ascii=False),
    )


def build_messages(
    document: str,
    event_labels: list[str],
    *,
    answer_obj: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(document, event_labels)},
    ]
    if answer_obj is not None:
        messages.append(
            {"role": "assistant", "content": json.dumps(answer_obj, ensure_ascii=False)}
        )
    return messages


def render_chat(
    tokenizer,
    document: str,
    event_labels: list[str],
    *,
    answer_obj: dict[str, Any] | None = None,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        build_messages(document, event_labels, answer_obj=answer_obj),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
