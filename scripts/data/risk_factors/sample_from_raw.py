from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT / "dataset" / "zhai" / "raw" / "sample_50000_with_tags_stratified_raw.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "dataset" / "zhai" / "raw" / "sample_50000_with_tags_stratified_raw.balanced.jsonl"
)
BALANCE_MODULE_PATH = REPO_ROOT / "src" / "data" / "balance_event_dataset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a self-balanced raw JSONL subset by event_type count. "
            "This balances event instances while keeping whole documents."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the input raw JSONL file. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the balanced subset. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--target-strategy",
        choices=["min", "median"],
        default="median",
        help=(
            "How to choose the per-label event target when --target-count is omitted. "
            "'min' enforces the smallest available class size; 'median' is usually a better compromise."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Explicit per-label event target. Overrides --target-strategy.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on the number of selected documents.",
    )
    parser.add_argument(
        "--max-overshoot-per-label",
        type=int,
        default=0,
        help=(
            "Allow up to this many extra event instances above target for any label. "
            "Default 0 enforces hard caps."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for tie-breaking.",
    )
    return parser.parse_args()


def load_balance_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "balance_event_dataset", BALANCE_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {BALANCE_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.target_count is not None and args.target_count <= 0:
        raise ValueError("--target-count must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("--max-documents must be positive")
    if args.max_overshoot_per_label < 0:
        raise ValueError("--max-overshoot-per-label must be non-negative")

    balance = load_balance_module()
    pool_records = balance.read_jsonl(args.input_path)
    pool_counts = balance.event_type_counts(pool_records)
    target_count = balance.choose_self_balanced_target_count(
        pool_counts,
        strategy=args.target_strategy,
        explicit_target=args.target_count,
    )

    selected_records, report = balance.select_self_balanced_subset(
        pool_records,
        target_count=target_count,
        seed=args.seed,
        max_documents=args.max_documents,
        max_overshoot_per_label=args.max_overshoot_per_label,
    )

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report_output
        if args.report_output is not None
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )

    balance.write_jsonl(output_path, selected_records)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"input: {format_path(args.input_path)}")
    print(f"output_path: {format_path(output_path)}")
    print(f"report_path: {format_path(report_path)}")
    print(f"target_count: {target_count}")
    print(f"selected_documents: {report['selected_documents']}")
    print(f"selected_min_count: {report['selected_min_count']}")
    print(f"selected_max_count: {report['selected_max_count']}")
    print(f"labels_with_remaining_deficit: {len(report['remaining_deficits'])}")


if __name__ == "__main__":
    main()
