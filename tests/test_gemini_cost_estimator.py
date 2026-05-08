from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = (
    REPO_ROOT / "scripts" / "data" / "generation" / "gemini_cost_estimator.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_cost_estimator", MODULE_PATH)
gemini_cost_estimator = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = gemini_cost_estimator
SPEC.loader.exec_module(gemini_cost_estimator)


def test_estimate_record_cost_includes_verifier_and_cached_tokens() -> None:
    record = {
        "id": "article-1",
        "llm": {
            "model": "gemini-2.5-flash",
            "metadata": {
                "prompt_tokens": 1_000,
                "completion_tokens": 200,
                "cached_tokens": 100,
                "verifier": {
                    "metadata": {
                        "prompt_tokens": 250,
                        "completion_tokens": 50,
                    }
                },
            },
            "relevance": {
                "model": "gemini-2.5-flash-lite",
                "metadata": {
                    "prompt_tokens": 300,
                    "completion_tokens": 20,
                },
            },
        },
    }

    estimate = gemini_cost_estimator.estimate_record_cost(record)

    assert estimate["normalized_model"] == "gemini-2.5-flash"
    assert estimate["normalized_relevance_model"] == "gemini-2.5-flash-lite"
    assert estimate["token_usage"]["input_tokens"] == 1_550
    assert estimate["token_usage"]["cached_input_tokens"] == 100
    assert estimate["token_usage"]["output_tokens"] == 270
    assert estimate["cost_usd"]["extraction"]["input"] == "0.00027000"
    assert estimate["cost_usd"]["extraction"]["cached_input"] == "0.00000300"
    assert estimate["cost_usd"]["extraction"]["output"] == "0.00050000"
    assert estimate["cost_usd"]["relevance"]["input"] == "0.00003000"
    assert estimate["cost_usd"]["relevance"]["output"] == "0.00000800"
    assert estimate["cost_usd"]["verifier"]["total"] == "0.00020000"
    assert estimate["cost_usd"]["total"] == "0.00101100"


def test_estimate_record_cost_without_relevance_leaves_relevance_empty() -> None:
    record = {
        "id": "article-no-relevance",
        "llm": {
            "model": "gemini-2.5-flash",
            "metadata": {
                "prompt_tokens": 1_000,
                "completion_tokens": 200,
            },
        },
    }

    estimate = gemini_cost_estimator.estimate_record_cost(record)

    assert estimate["token_usage"]["relevance"] is None
    assert estimate["cost_usd"]["relevance"] is None
    assert estimate["pricing_details"]["relevance"] is None


def test_normalize_model_name_maps_versioned_variants() -> None:
    assert (
        gemini_cost_estimator.normalize_model_name("gemini-2.5-flash-preview-06-17")
        == "gemini-2.5-flash"
    )
    assert (
        gemini_cost_estimator.normalize_model_name(
            "gemini-3.1-pro-preview-customtools"
        )
        == "gemini-3.1-pro-preview"
    )
    assert (
        gemini_cost_estimator.normalize_model_name("gemini-3.1-flash-lite-preview-09-2025")
        == "gemini-3.1-flash-lite-preview"
    )


def test_estimate_record_cost_uses_gemini_3_long_prompt_tier() -> None:
    record = {
        "id": "article-2",
        "llm": {
            "model": "gemini-3.1-pro-preview-customtools",
            "metadata": {
                "prompt_tokens": 250_000,
                "completion_tokens": 1_000,
                "cached_tokens": 50_000,
            },
        },
    }

    estimate = gemini_cost_estimator.estimate_record_cost(record, pricing_mode="standard")

    assert estimate["normalized_model"] == "gemini-3.1-pro-preview"
    assert estimate["pricing_details"]["extraction"]["max_prompt_tokens"] is None
    assert estimate["pricing_details"]["extraction"]["input_per_million_usd"] == "4.00"
    assert estimate["pricing_details"]["extraction"]["output_per_million_usd"] == "18.00"
    assert estimate["pricing_details"]["extraction"]["cached_input_per_million_usd"] == "0.40"
    assert estimate["cost_usd"]["extraction"]["input"] == "0.80000000"
    assert estimate["cost_usd"]["extraction"]["cached_input"] == "0.02000000"
    assert estimate["cost_usd"]["extraction"]["output"] == "0.01800000"
    assert estimate["cost_usd"]["total"] == "0.83800000"


def test_estimate_record_cost_supports_model_override() -> None:
    record = {
        "id": "article-3",
        "llm": {
            "model": "gemini-2.5-flash",
            "metadata": {
                "prompt_tokens": 1_000,
                "completion_tokens": 200,
            },
        },
    }

    estimate = gemini_cost_estimator.estimate_record_cost(
        record,
        model_override="gemini-3-flash-preview",
    )

    assert estimate["record_model"] == "gemini-2.5-flash"
    assert estimate["model_override"] == "gemini-3-flash-preview"
    assert estimate["model"] == "gemini-3-flash-preview"
    assert estimate["normalized_model"] == "gemini-3-flash-preview"
    assert estimate["cost_usd"]["extraction"]["input"] == "0.00050000"
    assert estimate["cost_usd"]["extraction"]["output"] == "0.00060000"
    assert estimate["cost_usd"]["total"] == "0.00110000"


def test_summarize_estimates_includes_relevance_model_bucket() -> None:
    record = {
        "cost_estimate": {
            "normalized_model": "gemini-2.5-flash",
            "normalized_relevance_model": "gemini-2.5-flash-lite",
            "token_usage": {
                "input_tokens": 180,
                "cached_input_tokens": 10,
                "output_tokens": 35,
                "extraction": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 10,
                    "completion_tokens": 20,
                },
                "relevance": {
                    "prompt_tokens": 50,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 5,
                },
                "verifier": {
                    "prompt_tokens": 30,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 10,
                },
            },
            "cost_usd": {
                "extraction": {"total": "0.00100000"},
                "relevance": {"total": "0.00010000"},
                "verifier": {"total": "0.00020000"},
                "total": "0.00130000",
            },
        }
    }

    summary = gemini_cost_estimator.summarize_estimates([record], include_verifier=True)

    assert summary["token_usage"] == {
        "input_tokens": 180,
        "cached_input_tokens": 10,
        "output_tokens": 35,
    }
    assert summary["per_model"]["gemini-2.5-flash"]["components"] == {
        "extraction": {
            "records": 1,
            "input_tokens": 100,
            "cached_input_tokens": 10,
            "output_tokens": 20,
            "total_cost_usd": "0.00100000",
        },
        "verifier": {
            "records": 1,
            "input_tokens": 30,
            "cached_input_tokens": 0,
            "output_tokens": 10,
            "total_cost_usd": "0.00020000",
        },
    }
    assert summary["per_model"]["gemini-2.5-flash-lite"]["components"] == {
        "relevance": {
            "records": 1,
            "input_tokens": 50,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "total_cost_usd": "0.00010000",
        }
    }
