from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "scripts" / "data" / "generation" / "reports.py"
SPEC = importlib.util.spec_from_file_location("reports", MODULE_PATH)
reports = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = reports
SPEC.loader.exec_module(reports)


def test_summarize_records_counts_relevance_and_nested_usage() -> None:
    summary = reports.summarize_records(
        [
            {
                "status": "ok",
                "events": [],
                "llm": {
                    "metadata": {"prompt_tokens": 10, "completion_tokens": 2},
                    "relevance": {
                        "decision": "relevant",
                        "filtered": False,
                        "metadata": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                },
            },
            {
                "status": "ok",
                "events": [],
                "llm": {
                    "metadata": {},
                    "relevance": {
                        "decision": "irrelevant",
                        "filtered": True,
                        "metadata": {"prompt_tokens": 4, "completion_tokens": 1},
                    },
                },
            },
        ]
    )

    assert summary["relevance_filtered_records"] == 1
    assert summary["relevance_decision_counts"] == {
        "irrelevant": 1,
        "relevant": 1,
    }
    assert summary["token_usage"] == {
        "prompt_tokens": 17,
        "completion_tokens": 4,
        "extraction": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
        },
        "relevance": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
        },
    }
