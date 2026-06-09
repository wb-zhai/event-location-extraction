"""Create event-type balanced train/dev splits for Zhai annotation JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[4]
    / "dataset/zhai/v2/annotated/sampled_by_event_label_500_interactive_0.3/recovered.jsonl"
)


@dataclass(frozen=True)
class CandidateRecord:
    index: int
    event_counts: Counter[str]
    has_empty_events: bool


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def record_event_counts(record: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    events = record.get("events", [])
    if not isinstance(events, list):
        return counts

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if isinstance(event_type, str) and event_type.strip():
            counts[event_type.strip()] += 1
    return counts


def has_empty_events(record: dict[str, Any]) -> bool:
    return record.get("events") == []


def empty_event_documents(records: list[dict[str, Any]]) -> int:
    return sum(1 for record in records if has_empty_events(record))


def event_type_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record_event_counts(record))
    return counts


def choose_target_counts(
    total_counts: Counter[str],
    dev_fraction: float,
    *,
    min_dev_per_type: int,
) -> Counter[str]:
    targets: Counter[str] = Counter()
    for event_type, count in total_counts.items():
        target = int(round(count * dev_fraction))
        if min_dev_per_type > 0 and count >= min_dev_per_type:
            target = max(min_dev_per_type, target)
        targets[event_type] = min(count, target)
    return targets


def score_candidate(
    counts: Counter[str],
    has_empty_events_: bool,
    current_counts: Counter[str],
    target_counts: Counter[str],
    current_empty_documents: int,
    target_empty_documents: int,
) -> tuple[float, int, int]:
    benefit = 0.0
    overshoot = 0
    total_events = sum(counts.values())

    for event_type, count in counts.items():
        target = target_counts[event_type]
        before = current_counts[event_type]
        after = before + count
        deficit = max(0, target - before)
        if target > 0:
            benefit += min(count, deficit) / target
        overshoot += max(0, after - target)

    if has_empty_events_:
        before = current_empty_documents
        after = before + 1
        deficit = max(0, target_empty_documents - before)
        if target_empty_documents > 0:
            benefit += min(1, deficit) / target_empty_documents
        overshoot += max(0, after - target_empty_documents)

    return benefit, -overshoot, -total_events


def event_type_balance_objective(counts: Counter[str], target_counts: Counter[str]) -> float:
    objective = 0.0
    for event_type, target in target_counts.items():
        objective += abs(counts[event_type] - target) / max(1, target)
    return objective


def split_objective(
    counts: Counter[str],
    target_counts: Counter[str],
    empty_documents: int,
    target_empty_documents: int,
) -> float:
    objective = event_type_balance_objective(counts, target_counts)
    objective += abs(empty_documents - target_empty_documents) / max(1, target_empty_documents)
    return objective


def swap_counts(
    current_counts: Counter[str],
    remove_counts: Counter[str],
    add_counts_: Counter[str],
) -> Counter[str]:
    swapped = Counter(current_counts)
    swapped.subtract(remove_counts)
    swapped.update(add_counts_)
    return +swapped


def split_records(
    records: list[dict[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
    min_dev_per_type: int,
    max_swap_passes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_event_counts = event_type_counts(records)
    total_empty_documents = empty_event_documents(records)
    target_event_counts = choose_target_counts(
        total_event_counts,
        dev_fraction,
        min_dev_per_type=min_dev_per_type,
    )
    target_empty_documents = int(round(total_empty_documents * dev_fraction))
    target_dev_documents = int(round(len(records) * dev_fraction))
    if records:
        target_dev_documents = max(1, target_dev_documents)
    if len(records) > 1:
        target_dev_documents = min(len(records) - 1, target_dev_documents)
    if target_empty_documents > target_dev_documents:
        target_empty_documents = target_dev_documents

    rng = random.Random(seed)
    candidates: list[CandidateRecord] = []
    for index, record in enumerate(records):
        candidates.append(
            CandidateRecord(
                index=index,
                event_counts=record_event_counts(record),
                has_empty_events=has_empty_events(record),
            )
        )

    rng.shuffle(candidates)

    selected_indices: set[int] = set()
    dev_event_counts: Counter[str] = Counter()
    dev_empty_documents = 0

    while len(selected_indices) < target_dev_documents:
        best_position: int | None = None
        best_score: tuple[float, int, int] | None = None

        for position, candidate in enumerate(candidates):
            if candidate.index in selected_indices:
                continue
            score = score_candidate(
                candidate.event_counts,
                candidate.has_empty_events,
                dev_event_counts,
                target_event_counts,
                dev_empty_documents,
                target_empty_documents,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_position = position

        if best_position is None:
            break

        candidate = candidates[best_position]
        selected_indices.add(candidate.index)
        dev_event_counts.update(candidate.event_counts)
        if candidate.has_empty_events:
            dev_empty_documents += 1

    counts_by_index = {candidate.index: candidate.event_counts for candidate in candidates}
    empty_by_index = {candidate.index: candidate.has_empty_events for candidate in candidates}
    current_objective = split_objective(
        dev_event_counts,
        target_event_counts,
        dev_empty_documents,
        target_empty_documents,
    )
    for _ in range(max_swap_passes):
        improved = False
        selected_order = list(selected_indices)
        unselected_order = [
            candidate.index for candidate in candidates if candidate.index not in selected_indices
        ]
        rng.shuffle(selected_order)
        rng.shuffle(unselected_order)

        best_swap: tuple[int, int] | None = None
        best_counts: Counter[str] | None = None
        best_empty_documents: int | None = None
        best_objective = current_objective

        for selected_index in selected_order:
            selected_counts = counts_by_index[selected_index]
            for unselected_index in unselected_order:
                candidate_counts = swap_counts(
                    dev_event_counts,
                    selected_counts,
                    counts_by_index[unselected_index],
                )
                candidate_empty_documents = (
                    dev_empty_documents
                    - int(empty_by_index[selected_index])
                    + int(empty_by_index[unselected_index])
                )
                candidate_objective = split_objective(
                    candidate_counts,
                    target_event_counts,
                    candidate_empty_documents,
                    target_empty_documents,
                )
                if candidate_objective < best_objective:
                    best_objective = candidate_objective
                    best_swap = (selected_index, unselected_index)
                    best_counts = candidate_counts
                    best_empty_documents = candidate_empty_documents

        if (
            best_swap is not None
            and best_counts is not None
            and best_empty_documents is not None
        ):
            selected_indices.remove(best_swap[0])
            selected_indices.add(best_swap[1])
            dev_event_counts = best_counts
            dev_empty_documents = best_empty_documents
            current_objective = best_objective
            improved = True

        if not improved:
            break

    dev_records = [
        record for index, record in enumerate(records) if index in selected_indices
    ]
    train_records = [
        record for index, record in enumerate(records) if index not in selected_indices
    ]
    train_event_counts = event_type_counts(train_records)
    dev_event_counts = event_type_counts(dev_records)
    train_empty_documents = empty_event_documents(train_records)
    dev_empty_documents = empty_event_documents(dev_records)
    empty_balance_objective = (
        abs(dev_empty_documents - target_empty_documents) / max(1, target_empty_documents)
    )

    report = {
        "input_documents": len(records),
        "train_documents": len(train_records),
        "dev_documents": len(dev_records),
        "requested_dev_fraction": dev_fraction,
        "actual_dev_fraction": len(dev_records) / len(records) if records else 0,
        "seed": seed,
        "min_dev_per_type": min_dev_per_type,
        "max_swap_passes": max_swap_passes,
        "event_balance_objective": event_type_balance_objective(
            dev_event_counts,
            target_event_counts,
        ),
        "combined_balance_objective": split_objective(
            dev_event_counts,
            target_event_counts,
            dev_empty_documents,
            target_empty_documents,
        ),
        "empty_event_balance_objective": empty_balance_objective,
        "total_empty_event_documents": total_empty_documents,
        "target_dev_empty_event_documents": target_empty_documents,
        "actual_dev_empty_event_documents": dev_empty_documents,
        "actual_train_empty_event_documents": train_empty_documents,
        "actual_dev_empty_event_ratio": (
            dev_empty_documents / len(dev_records) if dev_records else 0
        ),
        "actual_train_empty_event_ratio": (
            train_empty_documents / len(train_records) if train_records else 0
        ),
        "remaining_dev_empty_event_deficit": max(
            0,
            target_empty_documents - dev_empty_documents,
        ),
        "dev_empty_event_overshoot": max(0, dev_empty_documents - target_empty_documents),
        "total_event_type_counts": dict(total_event_counts.most_common()),
        "target_dev_event_type_counts": dict(target_event_counts.most_common()),
        "actual_dev_event_type_counts": dict(dev_event_counts.most_common()),
        "actual_train_event_type_counts": dict(train_event_counts.most_common()),
        "remaining_dev_event_deficits": {
            event_type: target - dev_event_counts[event_type]
            for event_type, target in sorted(target_event_counts.items())
            if dev_event_counts[event_type] < target
        },
        "dev_event_overshoots": {
            event_type: dev_event_counts[event_type] - target
            for event_type, target in sorted(target_event_counts.items())
            if dev_event_counts[event_type] > target
        },
    }
    return train_records, dev_records, report


def parse_dev_fraction(value: str) -> float:
    raw_value = value.strip()
    try:
        if raw_value.endswith("%"):
            fraction = float(raw_value[:-1]) / 100
        else:
            number = float(raw_value)
            fraction = number / 100 if number > 1 else number
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dev percentage must be a number") from exc

    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError("dev percentage must be greater than 0 and less than 100")
    return fraction


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split a Zhai recovered/annotated JSONL file into train/dev while balancing "
            "the dev split by events[*].event_type and empty events: [] annotations."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input JSONL path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--dev-percentage",
        "--dev-percent",
        "--dev-fraction",
        dest="dev_fraction",
        type=parse_dev_fraction,
        default=0.2,
        help="Dev split size. Accepts 20, 20%%, or 0.2. Default: 0.2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input file's parent directory.",
    )
    parser.add_argument("--train-name", default="train.jsonl", help="Train output filename.")
    parser.add_argument("--dev-name", default="dev.jsonl", help="Dev output filename.")
    parser.add_argument(
        "--report-name",
        default="train_dev_split_report.json",
        help="JSON report output filename.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for tie-breaking.")
    parser.add_argument(
        "--min-dev-per-type",
        type=int,
        default=1,
        help=(
            "Minimum target dev occurrences for event types with at least this many total "
            "occurrences. Use 0 to allow rare labels to round to zero."
        ),
    )
    parser.add_argument(
        "--max-swap-passes",
        type=int,
        default=10,
        help="Maximum hill-climbing swap passes used to improve event balance. Default: 10.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.min_dev_per_type < 0:
        parser.error("--min-dev-per-type must be non-negative")
    if args.max_swap_passes < 0:
        parser.error("--max-swap-passes must be non-negative")

    records = read_jsonl(args.input)
    train_records, dev_records, report = split_records(
        records,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        min_dev_per_type=args.min_dev_per_type,
        max_swap_passes=args.max_swap_passes,
    )

    output_dir = args.output_dir or args.input.parent
    train_path = output_dir / args.train_name
    dev_path = output_dir / args.dev_name
    report_path = output_dir / args.report_name

    write_jsonl(train_path, train_records)
    write_jsonl(dev_path, dev_records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(train_records)} train documents to {train_path}")
    print(f"Wrote {len(dev_records)} dev documents to {dev_path}")
    print(f"Wrote split report to {report_path}")
    if report["remaining_dev_event_deficits"]:
        print("Some event-type dev targets could not be met exactly; see report.")
    if report["remaining_dev_empty_event_deficit"] or report["dev_empty_event_overshoot"]:
        print("The empty events: [] dev target could not be met exactly; see report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
