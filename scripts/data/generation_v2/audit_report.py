"""Generate compact run reports for generation_v2 outputs."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v2.io_utils import dump_json, iter_jsonl, resolve_path
from scripts.data.generation_v2.usage import aggregate_component_usage
from scripts.data.generation.gemini_cost_estimator import (
    PRICE_SOURCE_CHECKED_AT,
    PRICE_SOURCE_URL,
    estimate_usage_cost,
    money,
    normalize_model_name,
)


def cost_usage(component_usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(component_usage.get("input_tokens") or 0)
    cached_tokens = min(int(component_usage.get("cached_input_tokens") or 0), input_tokens)
    return {
        "prompt_tokens": input_tokens,
        "non_cached_prompt_tokens": max(input_tokens - cached_tokens, 0),
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": int(component_usage.get("output_tokens") or 0),
        "thoughts_token_count": int(component_usage.get("thoughts_tokens") or 0),
    }


def component_model(metadata: dict[str, Any], component: str) -> str:
    if component == "verifier":
        return str(metadata.get("verifier_model") or metadata.get("annotation_model") or "")
    return str(metadata.get("annotation_model") or "")


def estimate_costs(records: list[dict[str, Any]], *, pricing_mode: str) -> dict[str, Any]:
    total_cost = Decimal("0")
    components: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []

    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        for component, component_usage in usage.items():
            if component == "total" or not isinstance(component_usage, dict):
                continue
            raw_model = component_model(metadata, component)
            model = normalize_model_name(raw_model)
            if not model:
                skipped.append(
                    {
                        "id": record.get("id"),
                        "component": component,
                        "reason": "missing_model",
                    }
                )
                continue
            try:
                estimate = estimate_usage_cost(cost_usage(component_usage), model, pricing_mode)
            except KeyError:
                skipped.append(
                    {
                        "id": record.get("id"),
                        "component": component,
                        "model": raw_model,
                        "normalized_model": model,
                        "reason": "unknown_model",
                    }
                )
                continue

            usage_for_cost = cost_usage(component_usage)
            cost = Decimal(estimate["cost_breakdown_usd"]["total"])
            total_cost += cost
            component_bucket = components.setdefault(
                component,
                {
                    "records": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_cost_usd": Decimal("0"),
                    "pricing_details": estimate["pricing_tier"],
                },
            )
            component_bucket["records"] += 1
            component_bucket["input_tokens"] += usage_for_cost["prompt_tokens"]
            component_bucket["cached_input_tokens"] += usage_for_cost["cached_prompt_tokens"]
            component_bucket["output_tokens"] += usage_for_cost["completion_tokens"]
            component_bucket["thoughts_tokens"] += usage_for_cost["thoughts_token_count"]
            component_bucket["total_cost_usd"] += cost

            model_bucket = models.setdefault(
                model,
                {
                    "records": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_cost_usd": Decimal("0"),
                    "components": {},
                },
            )
            model_bucket["records"] += 1
            model_bucket["input_tokens"] += usage_for_cost["prompt_tokens"]
            model_bucket["cached_input_tokens"] += usage_for_cost["cached_prompt_tokens"]
            model_bucket["output_tokens"] += usage_for_cost["completion_tokens"]
            model_bucket["thoughts_tokens"] += usage_for_cost["thoughts_token_count"]
            model_bucket["total_cost_usd"] += cost
            model_component = model_bucket["components"].setdefault(
                component,
                {
                    "records": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_cost_usd": Decimal("0"),
                    "pricing_details": estimate["pricing_tier"],
                },
            )
            model_component["records"] += 1
            model_component["input_tokens"] += usage_for_cost["prompt_tokens"]
            model_component["cached_input_tokens"] += usage_for_cost["cached_prompt_tokens"]
            model_component["output_tokens"] += usage_for_cost["completion_tokens"]
            model_component["thoughts_tokens"] += usage_for_cost["thoughts_token_count"]
            model_component["total_cost_usd"] += cost

    def serialize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (str(money(value)) if key == "total_cost_usd" else value)
            for key, value in bucket.items()
        }

    serialized_models: dict[str, Any] = {}
    for model, bucket in models.items():
        serialized_models[model] = serialize_bucket(
            {
                **bucket,
                "components": {
                    component: serialize_bucket(component_bucket)
                    for component, component_bucket in bucket["components"].items()
                },
            }
        )

    return {
        "pricing_mode": pricing_mode,
        "price_source_url": PRICE_SOURCE_URL,
        "price_source_checked_at": PRICE_SOURCE_CHECKED_AT,
        "total_cost_usd": str(money(total_cost)),
        "average_cost_per_record_usd": str(
            money(total_cost / Decimal(len(records))) if records else money(Decimal("0"))
        ),
        "components": {
            component: serialize_bucket(bucket)
            for component, bucket in components.items()
        },
        "models": serialized_models,
        "skipped": skipped,
        "notes": [
            "Estimates are token-based and assume text pricing for the selected Gemini API mode.",
            "Thought tokens are reported but not added separately unless included in output token usage.",
            "The aggregate metadata.usage.total bucket is ignored to avoid double-counting component usage.",
        ],
    }


def build_report(records: list[dict[str, Any]], *, pricing_mode: str = "standard") -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    api_modes: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for record in tqdm(records, desc="Reporting"):
        status_counts[str(record.get("status", "ok"))] += 1
        metadata = record.get("metadata") or {}
        if metadata.get("api_mode"):
            api_modes[str(metadata["api_mode"])] += 1
        if metadata.get("annotation_model"):
            models[str(metadata["annotation_model"])] += 1
        for event in record.get("events") or []:
            event_counts[str(event.get("event_type"))] += 1
            for argument in event.get("arguments") or []:
                role_counts[str(argument.get("role"))] += 1
        for location in record.get("locations") or []:
            location_counts[str(location.get("location_type"))] += 1

    return {
        "records": len(records),
        "status_counts": dict(status_counts),
        "event_counts": dict(event_counts),
        "location_type_counts": dict(location_counts),
        "argument_role_counts": dict(role_counts),
        "api_modes": dict(api_modes),
        "models": dict(models),
        "token_usage": aggregate_component_usage(records),
        "cost_estimate": estimate_costs(records, pricing_mode=pricing_mode),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generation_v2 audit report.")
    parser.add_argument("input", type=Path)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument(
        "--pricing-mode",
        choices=("standard", "batch", "flex", "priority"),
        default="standard",
        help="Gemini API pricing mode to apply. Default: standard.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(iter_jsonl(resolve_path(args.input)))
    report_dir = resolve_path(args.report_dir)
    dump_json(report_dir / "summary.json", build_report(records, pricing_mode=args.pricing_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
