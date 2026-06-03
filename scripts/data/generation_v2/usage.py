"""Token usage normalization and aggregation."""

from __future__ import annotations

from typing import Any


USAGE_KEYS = ("input_tokens", "output_tokens", "cached_input_tokens", "thoughts_tokens")


def as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(metadata: dict[str, Any] | None) -> dict[str, int]:
    metadata = metadata or {}
    usage_metadata = metadata.get("usageMetadata") or metadata.get("usage_metadata")
    if isinstance(usage_metadata, dict):
        metadata = {**metadata, **usage_metadata}

    return {
        "input_tokens": as_int(
            metadata.get("input_tokens")
            or metadata.get("prompt_tokens")
            or metadata.get("promptTokenCount")
            or metadata.get("prompt_token_count")
        ),
        "output_tokens": as_int(
            metadata.get("output_tokens")
            or metadata.get("completion_tokens")
            or metadata.get("candidatesTokenCount")
            or metadata.get("candidates_token_count")
        ),
        "cached_input_tokens": as_int(
            metadata.get("cached_input_tokens")
            or metadata.get("cached_tokens")
            or metadata.get("cachedContentTokenCount")
            or metadata.get("cached_content_token_count")
        ),
        "thoughts_tokens": as_int(
            metadata.get("thoughts_tokens")
            or metadata.get("thoughts_token_count")
            or metadata.get("thoughtsTokenCount")
        ),
    }


def add_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    left = normalize_usage(left)
    right = normalize_usage(right)
    return {key: left[key] + right[key] for key in USAGE_KEYS}


def aggregate_component_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_component: dict[str, dict[str, int]] = {}
    total = {key: 0 for key in USAGE_KEYS}
    missing_usage = 0

    for record in records:
        metadata = record.get("metadata") or {}
        usage = metadata.get("usage") or {}
        if not usage:
            missing_usage += 1
        for component, component_usage in usage.items():
            normalized = normalize_usage(component_usage)
            bucket = by_component.setdefault(component, {key: 0 for key in USAGE_KEYS})
            for key in USAGE_KEYS:
                bucket[key] += normalized[key]
                total[key] += normalized[key]

    return {
        "total": total,
        "by_component": by_component,
        "records_missing_usage": missing_usage,
    }
