"""Minimal config support for the event generation CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineMode:
    output_mode: str
    self_consistency: bool
    self_consistency_samples: int
    self_consistency_temperature: float
    strict_offsets: bool
    long_document_mode: bool
    enable_verifier: bool
    enable_synthetic_gaps: bool


MODE_DEFAULTS = {
    "quality_first": PipelineMode(
        output_mode="events-with-args",
        self_consistency=True,
        self_consistency_samples=5,
        self_consistency_temperature=0.7,
        strict_offsets=True,
        long_document_mode=True,
        enable_verifier=True,
        enable_synthetic_gaps=True,
    ),
    "balanced": PipelineMode(
        output_mode="events-with-args",
        self_consistency=True,
        self_consistency_samples=3,
        self_consistency_temperature=0.5,
        strict_offsets=True,
        long_document_mode=True,
        enable_verifier=False,
        enable_synthetic_gaps=False,
    ),
    "low_cost": PipelineMode(
        output_mode="events-with-args",
        self_consistency=False,
        self_consistency_samples=0,
        self_consistency_temperature=0.0,
        strict_offsets=True,
        long_document_mode=True,
        enable_verifier=False,
        enable_synthetic_gaps=False,
    ),
}


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a tiny JSON/YAML-like config without adding a dependency."""
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(text)

    config: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        config[key.strip()] = _parse_scalar(value.strip())
    return config


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")
