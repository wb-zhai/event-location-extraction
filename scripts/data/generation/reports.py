"""Small dataset reports for generated IE annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count labels, roles, statuses, and token metadata in JSONL records."""
    status_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    relevance_decision_counts: Counter[str] = Counter()
    synthetic_count = 0
    relevance_filtered_records = 0
    extraction_prompt_tokens = 0
    extraction_completion_tokens = 0
    relevance_prompt_tokens = 0
    relevance_completion_tokens = 0

    for record in records:
        status_counts[str(record.get("status", "missing"))] += 1
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        if source.get("synthetic"):
            synthetic_count += 1

        for event in record.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            event_counts[str(event.get("event_type", ""))] += 1
            for argument in event.get("arguments", []) or []:
                if isinstance(argument, dict):
                    role_counts[str(argument.get("role", ""))] += 1

        llm = record.get("llm") or {}
        metadata = llm.get("metadata") or {}
        extraction_prompt_tokens += int(metadata.get("prompt_tokens", 0) or 0)
        extraction_completion_tokens += int(metadata.get("completion_tokens", 0) or 0)

        relevance = llm.get("relevance") or {}
        decision = str(relevance.get("decision", "")).strip()
        if decision:
            relevance_decision_counts[decision] += 1
        if bool(relevance.get("filtered")):
            relevance_filtered_records += 1
        relevance_metadata = relevance.get("metadata") or {}
        relevance_prompt_tokens += int(relevance_metadata.get("prompt_tokens", 0) or 0)
        relevance_completion_tokens += int(
            relevance_metadata.get("completion_tokens", 0) or 0
        )

    return {
        "records": len(records),
        "synthetic_records": synthetic_count,
        "relevance_filtered_records": relevance_filtered_records,
        "relevance_decision_counts": dict(sorted(relevance_decision_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "token_usage": {
            "prompt_tokens": extraction_prompt_tokens + relevance_prompt_tokens,
            "completion_tokens": (
                extraction_completion_tokens + relevance_completion_tokens
            ),
            "extraction": {
                "prompt_tokens": extraction_prompt_tokens,
                "completion_tokens": extraction_completion_tokens,
            },
            "relevance": {
                "prompt_tokens": relevance_prompt_tokens,
                "completion_tokens": relevance_completion_tokens,
            },
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_report(output_path: Path, report_path: Path) -> dict[str, Any]:
    """Write a JSON report next to a generated JSONL file."""
    records = read_jsonl(output_path)
    report = summarize_records(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def synthetic_gap_requests(
    report: dict[str, Any],
    ontology: dict[str, str],
    minimum_per_event_type: int = 10,
) -> list[dict[str, Any]]:
    """Describe event types that need synthetic examples; generation stays separate."""
    event_counts = report.get("event_counts") or {}
    requests: list[dict[str, Any]] = []
    for event_type, description in sorted(ontology.items()):
        count = int(event_counts.get(event_type, 0) or 0)
        if count >= minimum_per_event_type:
            continue
        requests.append(
            {
                "event_type": event_type,
                "description": description,
                "current_count": count,
                "target_count": minimum_per_event_type,
                "needed": minimum_per_event_type - count,
            }
        )
    return requests
