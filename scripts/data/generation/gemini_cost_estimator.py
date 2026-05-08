"""Estimate per-record Gemini API costs from generated JSONL metadata.

This utility reads records produced by ``gemini_event_gen.py`` and computes a
cost estimate for each article using:

- ``record["llm"]["model"]``
- ``record["llm"]["metadata"]["prompt_tokens"]``
- ``record["llm"]["metadata"]["completion_tokens"]``
- optional ``record["llm"]["relevance"]`` metadata and model
- optional cached-input and verifier metadata

Pricing defaults are based on the Gemini Developer API pricing page:
https://ai.google.dev/gemini-api/docs/pricing
Checked on 2026-05-05.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

PRICE_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICE_SOURCE_CHECKED_AT = "2026-05-05"

PRICING_MODES = ("standard", "batch", "flex", "priority")
ONE_MILLION = Decimal("1000000")
USD_PLACES = Decimal("0.00000001")


@dataclass(frozen=True)
class PriceTier:
    max_prompt_tokens: int | None
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None


def money(value: Decimal) -> Decimal:
    return value.quantize(USD_PLACES, rounding=ROUND_HALF_UP)


def d(value: str) -> Decimal:
    return Decimal(value)


MODEL_PRICING: dict[str, dict[str, list[PriceTier]]] = {
    "gemini-3.1-pro-preview": {
        "standard": [
            PriceTier(200_000, d("2.00"), d("12.00"), d("0.20")),
            PriceTier(None, d("4.00"), d("18.00"), d("0.40")),
        ],
        "batch": [
            PriceTier(200_000, d("1.00"), d("6.00"), d("0.20")),
            PriceTier(None, d("2.00"), d("9.00"), d("0.40")),
        ],
        "flex": [
            PriceTier(200_000, d("1.00"), d("6.00"), d("0.20")),
            PriceTier(None, d("2.00"), d("9.00"), d("0.40")),
        ],
        "priority": [
            PriceTier(200_000, d("3.60"), d("21.60"), d("0.36")),
            PriceTier(None, d("7.20"), d("32.40"), d("0.72")),
        ],
    },
    "gemini-3.1-flash-lite-preview": {
        "standard": [PriceTier(None, d("0.25"), d("1.50"), d("0.025"))],
        "batch": [PriceTier(None, d("0.125"), d("0.75"), d("0.0125"))],
        "flex": [PriceTier(None, d("0.125"), d("0.75"), d("0.0125"))],
        "priority": [PriceTier(None, d("0.45"), d("2.70"), d("0.045"))],
    },
    "gemini-3-flash-preview": {
        "standard": [PriceTier(None, d("0.50"), d("3.00"), d("0.05"))],
        "batch": [PriceTier(None, d("0.25"), d("1.50"), d("0.05"))],
        "flex": [PriceTier(None, d("0.25"), d("1.50"), d("0.05"))],
        "priority": [PriceTier(None, d("0.90"), d("5.40"), d("0.09"))],
    },
    "gemini-2.5-flash": {
        "standard": [PriceTier(None, d("0.30"), d("2.50"), d("0.03"))],
        "batch": [PriceTier(None, d("0.15"), d("1.25"), d("0.03"))],
        "flex": [PriceTier(None, d("0.15"), d("1.25"), d("0.03"))],
        "priority": [PriceTier(None, d("0.54"), d("4.50"), d("0.054"))],
    },
    "gemini-2.5-flash-lite": {
        "standard": [PriceTier(None, d("0.10"), d("0.40"), d("0.01"))],
        "batch": [PriceTier(None, d("0.05"), d("0.20"), d("0.01"))],
        "flex": [PriceTier(None, d("0.05"), d("0.20"), d("0.01"))],
        "priority": [PriceTier(None, d("0.18"), d("0.72"), d("0.018"))],
    },
    "gemini-2.5-pro": {
        "standard": [
            PriceTier(200_000, d("1.25"), d("10.00"), d("0.125")),
            PriceTier(None, d("2.50"), d("15.00"), d("0.25")),
        ],
        "batch": [
            PriceTier(200_000, d("0.625"), d("5.00"), d("0.125")),
            PriceTier(None, d("1.25"), d("7.50"), d("0.25")),
        ],
        "flex": [
            PriceTier(200_000, d("0.625"), d("5.00"), d("0.125")),
            PriceTier(None, d("1.25"), d("7.50"), d("0.25")),
        ],
        "priority": [
            PriceTier(200_000, d("2.25"), d("18.00"), d("0.225")),
            PriceTier(None, d("4.50"), d("27.00"), d("0.45")),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Gemini API costs for each record in a generated JSONL."
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "output_jsonl",
        type=Path,
        nargs="?",
        help="Optional output JSONL. Defaults to INPUT with .costs.jsonl suffix.",
    )
    parser.add_argument(
        "--pricing-mode",
        choices=PRICING_MODES,
        default="standard",
        help="Gemini API pricing mode to apply. Default: standard.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional model override for pricing all records. "
            "Example: gemini-2.5-flash or gemini-3.1-pro-preview."
        ),
    )
    parser.add_argument(
        "--exclude-verifier",
        action="store_true",
        help="Do not include verifier token usage in the total estimate.",
    )
    parser.add_argument(
        "--unknown-model",
        choices=("error", "skip"),
        default="error",
        help="How to handle records whose llm.model is not in the built-in pricing table.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write a JSON summary report.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    if not model_name:
        return model_name
    known_exact = set(MODEL_PRICING)
    if model_name in known_exact:
        return model_name
    for known in known_exact:
        if model_name.startswith(f"{known}-"):
            return known
    return model_name


def pick_price_tier(
    model_name: str, pricing_mode: str, prompt_tokens: int
) -> PriceTier:
    model_pricing = MODEL_PRICING.get(model_name)
    if model_pricing is None:
        raise KeyError(model_name)
    for tier in model_pricing[pricing_mode]:
        if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens:
            return tier
    raise RuntimeError(
        f"No pricing tier matched model={model_name} mode={pricing_mode}"
    )


def as_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def extract_usage(metadata: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = as_int(metadata.get("prompt_tokens"))
    completion_tokens = as_int(metadata.get("completion_tokens"))
    cached_tokens = min(as_int(metadata.get("cached_tokens")), prompt_tokens)
    non_cached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)
    thoughts_token_count = as_int(metadata.get("thoughts_token_count"))
    return {
        "prompt_tokens": prompt_tokens,
        "non_cached_prompt_tokens": non_cached_prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "thoughts_token_count": thoughts_token_count,
    }


def empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "non_cached_prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_token_count": 0,
    }


def estimate_usage_cost(
    usage: dict[str, int],
    model_name: str,
    pricing_mode: str,
) -> dict[str, Any]:
    tier = pick_price_tier(model_name, pricing_mode, usage["prompt_tokens"])
    input_cost = (
        Decimal(usage["non_cached_prompt_tokens"])
        * tier.input_per_million
        / ONE_MILLION
    )
    cached_input_cost = Decimal("0")
    if usage["cached_prompt_tokens"] and tier.cached_input_per_million is not None:
        cached_input_cost = (
            Decimal(usage["cached_prompt_tokens"])
            * tier.cached_input_per_million
            / ONE_MILLION
        )
    output_cost = (
        Decimal(usage["completion_tokens"]) * tier.output_per_million / ONE_MILLION
    )
    total_cost = input_cost + cached_input_cost + output_cost
    return {
        "pricing_tier": {
            "max_prompt_tokens": tier.max_prompt_tokens,
            "input_per_million_usd": str(tier.input_per_million),
            "output_per_million_usd": str(tier.output_per_million),
            "cached_input_per_million_usd": (
                str(tier.cached_input_per_million)
                if tier.cached_input_per_million is not None
                else None
            ),
        },
        "cost_breakdown_usd": {
            "input": str(money(input_cost)),
            "cached_input": str(money(cached_input_cost)),
            "output": str(money(output_cost)),
            "total": str(money(total_cost)),
        },
    }


def estimate_record_cost(
    record: dict[str, Any],
    pricing_mode: str = "standard",
    include_verifier: bool = True,
    model_override: str | None = None,
) -> dict[str, Any]:
    llm = record.get("llm") if isinstance(record.get("llm"), dict) else {}
    metadata = llm.get("metadata") if isinstance(llm.get("metadata"), dict) else {}
    raw_model_name = str(llm.get("model") or "").strip()
    effective_raw_model_name = str(model_override or raw_model_name).strip()
    model_name = normalize_model_name(effective_raw_model_name)
    if not model_name:
        raise KeyError("missing_model")

    extraction_usage = extract_usage(metadata)
    extraction_cost = estimate_usage_cost(extraction_usage, model_name, pricing_mode)

    relevance = llm.get("relevance") if isinstance(llm.get("relevance"), dict) else {}
    has_relevance = bool(relevance)
    relevance_usage = None
    relevance_cost = None
    raw_relevance_model_name = str(relevance.get("model") or "").strip()
    effective_relevance_raw_model_name = None
    relevance_model_name = None
    if has_relevance:
        relevance_metadata = (
            relevance.get("metadata")
            if isinstance(relevance.get("metadata"), dict)
            else {}
        )
        effective_relevance_raw_model_name = str(
            model_override or raw_relevance_model_name or raw_model_name
        ).strip()
        relevance_model_name = normalize_model_name(effective_relevance_raw_model_name)
        if not relevance_model_name:
            raise KeyError("missing_relevance_model")
        relevance_usage = extract_usage(relevance_metadata)
        relevance_cost = estimate_usage_cost(
            relevance_usage, relevance_model_name, pricing_mode
        )

    verifier_usage = empty_usage()
    verifier_cost = None
    if include_verifier:
        verifier = (
            metadata.get("verifier")
            if isinstance(metadata.get("verifier"), dict)
            else {}
        )
        verifier_metadata = (
            verifier.get("metadata")
            if isinstance(verifier.get("metadata"), dict)
            else {}
        )
        verifier_usage = extract_usage(verifier_metadata)
        verifier_cost = estimate_usage_cost(verifier_usage, model_name, pricing_mode)

    total_input_tokens = (
        extraction_usage["prompt_tokens"]
        + as_int((relevance_usage or {}).get("prompt_tokens"))
        + verifier_usage["prompt_tokens"]
    )
    total_output_tokens = (
        extraction_usage["completion_tokens"]
        + as_int((relevance_usage or {}).get("completion_tokens"))
        + verifier_usage["completion_tokens"]
    )
    total_cached_tokens = (
        extraction_usage["cached_prompt_tokens"]
        + as_int((relevance_usage or {}).get("cached_prompt_tokens"))
        + verifier_usage["cached_prompt_tokens"]
    )
    total_cost = Decimal(extraction_cost["cost_breakdown_usd"]["total"])
    if relevance_cost is not None:
        total_cost += Decimal(relevance_cost["cost_breakdown_usd"]["total"])
    if verifier_cost is not None:
        total_cost += Decimal(verifier_cost["cost_breakdown_usd"]["total"])

    return {
        "model": effective_raw_model_name,
        "record_model": raw_model_name,
        "model_override": model_override,
        "normalized_model": model_name,
        "relevance_model": effective_relevance_raw_model_name,
        "record_relevance_model": raw_relevance_model_name,
        "normalized_relevance_model": relevance_model_name,
        "pricing_mode": pricing_mode,
        "price_source_url": PRICE_SOURCE_URL,
        "price_source_checked_at": PRICE_SOURCE_CHECKED_AT,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_tokens,
            "output_tokens": total_output_tokens,
            "extraction": extraction_usage,
            "relevance": relevance_usage,
            "verifier": verifier_usage if include_verifier else None,
        },
        "cost_usd": {
            "extraction": extraction_cost["cost_breakdown_usd"],
            "relevance": (
                relevance_cost["cost_breakdown_usd"] if relevance_cost else None
            ),
            "verifier": verifier_cost["cost_breakdown_usd"] if verifier_cost else None,
            "total": str(money(total_cost)),
        },
        "pricing_details": {
            "extraction": extraction_cost["pricing_tier"],
            "relevance": relevance_cost["pricing_tier"] if relevance_cost else None,
            "verifier": verifier_cost["pricing_tier"] if verifier_cost else None,
        },
        "notes": [
            "Estimates are token-based and assume text pricing for the selected Gemini API mode.",
            "For models with prompt-length tiers, pricing is chosen from aggregated prompt tokens in each extraction, relevance, or verifier call summary.",
        ],
    }


def summarize_estimates(
    records: list[dict[str, Any]],
    include_verifier: bool,
) -> dict[str, Any]:
    total_cost = Decimal("0")
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    estimated_records = 0
    skipped_records = 0
    per_model: dict[str, dict[str, Any]] = {}

    def add_component(
        component_name: str,
        usage: dict[str, Any] | None,
        cost: dict[str, Any] | None,
        normalized_model: str,
    ) -> None:
        if not usage or not cost:
            return
        model_bucket = per_model.setdefault(
            normalized_model,
            {
                "records": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": Decimal("0"),
                "components": {},
            },
        )
        component_bucket = model_bucket["components"].setdefault(
            component_name,
            {
                "records": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_cost_usd": Decimal("0"),
            },
        )
        input_tokens = as_int(usage.get("prompt_tokens"))
        cached_input_tokens = as_int(usage.get("cached_prompt_tokens"))
        output_tokens = as_int(usage.get("completion_tokens"))
        total_component_cost = Decimal(str(cost.get("total") or "0"))

        model_bucket["input_tokens"] += input_tokens
        model_bucket["cached_input_tokens"] += cached_input_tokens
        model_bucket["output_tokens"] += output_tokens
        model_bucket["total_cost_usd"] += total_component_cost

        component_bucket["records"] += 1
        component_bucket["input_tokens"] += input_tokens
        component_bucket["cached_input_tokens"] += cached_input_tokens
        component_bucket["output_tokens"] += output_tokens
        component_bucket["total_cost_usd"] += total_component_cost

    for record in records:
        estimate = record.get("cost_estimate")
        if not isinstance(estimate, dict):
            skipped_records += 1
            continue
        estimated_records += 1
        total_cost += Decimal(str((estimate.get("cost_usd") or {}).get("total") or "0"))
        token_usage = estimate.get("token_usage") or {}
        total_input_tokens += as_int(token_usage.get("input_tokens"))
        total_cached_tokens += as_int(token_usage.get("cached_input_tokens"))
        total_output_tokens += as_int(token_usage.get("output_tokens"))
        cost_usage = estimate.get("cost_usd") or {}
        touched_models: set[str] = set()
        primary_model = str(
            estimate.get("normalized_model") or estimate.get("model") or "unknown"
        )
        relevance_model = str(
            estimate.get("normalized_relevance_model")
            or estimate.get("relevance_model")
            or primary_model
        )
        add_component(
            "extraction",
            token_usage.get("extraction"),
            cost_usage.get("extraction"),
            primary_model,
        )
        touched_models.add(primary_model)
        add_component(
            "relevance",
            token_usage.get("relevance"),
            cost_usage.get("relevance"),
            relevance_model,
        )
        if token_usage.get("relevance") and cost_usage.get("relevance"):
            touched_models.add(relevance_model)
        if include_verifier:
            add_component(
                "verifier",
                token_usage.get("verifier"),
                cost_usage.get("verifier"),
                primary_model,
            )
            if token_usage.get("verifier") and cost_usage.get("verifier"):
                touched_models.add(primary_model)
        for model_name in touched_models:
            per_model.setdefault(
                model_name,
                {
                    "records": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": Decimal("0"),
                    "components": {},
                },
            )["records"] += 1

    average_cost_per_article = Decimal("0")
    if estimated_records:
        average_cost_per_article = total_cost / Decimal(estimated_records)

    summary = {
        "price_source_url": PRICE_SOURCE_URL,
        "price_source_checked_at": PRICE_SOURCE_CHECKED_AT,
        "records": len(records),
        "estimated_records": estimated_records,
        "skipped_records": skipped_records,
        "include_verifier": include_verifier,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "cached_input_tokens": total_cached_tokens,
            "output_tokens": total_output_tokens,
        },
        "total_cost_usd": str(money(total_cost)),
        "average_cost_per_article_usd": str(money(average_cost_per_article)),
        "per_model": {},
    }
    for model_name, bucket in sorted(per_model.items()):
        summary["per_model"][model_name] = {
            "records": bucket["records"],
            "input_tokens": bucket["input_tokens"],
            "cached_input_tokens": bucket["cached_input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "total_cost_usd": str(money(bucket["total_cost_usd"])),
            "components": {
                component_name: {
                    "records": component_bucket["records"],
                    "input_tokens": component_bucket["input_tokens"],
                    "cached_input_tokens": component_bucket["cached_input_tokens"],
                    "output_tokens": component_bucket["output_tokens"],
                    "total_cost_usd": str(money(component_bucket["total_cost_usd"])),
                }
                for component_name, component_bucket in sorted(
                    bucket["components"].items()
                )
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    input_path = args.input_jsonl
    output_path = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
    records = read_jsonl(input_path)

    enriched_records: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        try:
            enriched["cost_estimate"] = estimate_record_cost(
                enriched,
                pricing_mode=args.pricing_mode,
                include_verifier=not args.exclude_verifier,
                model_override=args.model,
            )
        except KeyError as exc:
            if args.unknown_model == "error":
                model_name = str(
                    ((record.get("llm") or {}).get("model") or "")
                ).strip() or str(exc)
                raise ValueError(
                    f"Unknown or missing Gemini pricing entry for model '{model_name}'. "
                    "Update MODEL_PRICING or run with --unknown-model skip."
                ) from exc
        enriched_records.append(enriched)

    if output_path:
        write_jsonl(output_path, enriched_records)
    summary = summarize_estimates(
        enriched_records,
        include_verifier=not args.exclude_verifier,
    )
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if output_path:
        print(
            json.dumps({"output_jsonl": str(output_path), "summary": summary}, indent=2)
        )
    else:
        print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
