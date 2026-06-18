from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateRecord:
    index: int
    record: dict[str, Any]
    counts: Counter[str]
    total_relevant_events: int


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def event_type_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for event in record.get("events", []):
            event_type = event.get("event_type")
            if isinstance(event_type, str) and event_type:
                counts[event_type] += 1
    return counts


def candidate_event_counts(
    record: dict[str, Any],
    allowed_event_types: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in record.get("events", []):
        event_type = event.get("event_type")
        if isinstance(event_type, str) and event_type in allowed_event_types:
            counts[event_type] += 1
    return counts


def choose_target_count(
    reference_counts: Counter[str],
    *,
    strategy: str,
    explicit_target: int | None,
) -> int:
    if explicit_target is not None:
        if explicit_target <= 0:
            raise ValueError("--target-count must be positive")
        return explicit_target
    if not reference_counts:
        raise ValueError("Reference dataset does not contain any event labels")
    values = sorted(reference_counts.values())
    if strategy == "max":
        return values[-1]
    if strategy == "median":
        return values[len(values) // 2]
    raise ValueError(f"Unsupported target strategy: {strategy}")


def choose_self_balanced_target_count(
    pool_counts: Counter[str],
    *,
    strategy: str,
    explicit_target: int | None,
) -> int:
    if explicit_target is not None:
        if explicit_target <= 0:
            raise ValueError("--target-count must be positive")
        return explicit_target
    if not pool_counts:
        raise ValueError("Pool dataset does not contain any event labels")
    values = sorted(pool_counts.values())
    if strategy == "min":
        return values[0]
    if strategy == "median":
        return values[len(values) // 2]
    raise ValueError(f"Unsupported self-balance strategy: {strategy}")


def build_candidates(
    pool_records: list[dict[str, Any]],
    allowed_event_types: set[str],
) -> tuple[list[CandidateRecord], dict[str, list[int]]]:
    candidates: list[CandidateRecord] = []
    postings: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(pool_records):
        counts = candidate_event_counts(record, allowed_event_types)
        if not counts:
            continue
        candidate = CandidateRecord(
            index=index,
            record=record,
            counts=counts,
            total_relevant_events=sum(counts.values()),
        )
        candidate_idx = len(candidates)
        candidates.append(candidate)
        for label in counts:
            postings[label].append(candidate_idx)
    return candidates, postings


def score_candidate(
    candidate: CandidateRecord,
    current_counts: Counter[str],
    target_count: int,
) -> tuple[int, int, int]:
    benefit = 0
    overflow = 0
    for label, count in candidate.counts.items():
        deficit = max(0, target_count - current_counts[label])
        benefit += min(count, deficit)
        overflow += max(0, current_counts[label] + count - target_count)
    return benefit, -overflow, -candidate.total_relevant_events


def score_self_balanced_candidate(
    candidate: CandidateRecord,
    current_counts: Counter[str],
    target_count: int,
    *,
    max_overshoot_per_label: int,
) -> tuple[int, int, int, int] | None:
    benefit = 0
    overflow = 0
    waste = 0
    max_label_overflow = 0

    for label, count in candidate.counts.items():
        new_count = current_counts[label] + count
        label_overflow = max(0, new_count - target_count)
        if label_overflow > max_overshoot_per_label:
            return None
        deficit = max(0, target_count - current_counts[label])
        benefit += min(count, deficit)
        overflow += label_overflow
        max_label_overflow = max(max_label_overflow, label_overflow)
        waste += max(0, count - deficit)

    if benefit <= 0:
        return None
    return benefit, -max_label_overflow, -overflow, -waste


def summarize_self_balance(
    selected_records: list[dict[str, Any]],
    pool_counts: Counter[str],
    target_count: int,
) -> dict[str, Any]:
    selected_counts = event_type_counts(selected_records)
    remaining_deficits = {
        label: max(0, target_count - selected_counts[label])
        for label in pool_counts
        if selected_counts[label] < target_count
    }
    overshoot_by_label = {
        label: max(0, selected_counts[label] - target_count)
        for label in selected_counts
        if selected_counts[label] > target_count
    }
    return {
        "mode": "self_balanced",
        "target_count": target_count,
        "pool_event_type_counts": dict(pool_counts.most_common()),
        "selected_event_type_counts": dict(selected_counts.most_common()),
        "remaining_deficits": remaining_deficits,
        "overshoot_by_label": dict(sorted(overshoot_by_label.items())),
        "selected_min_count": min(selected_counts.values()) if selected_counts else 0,
        "selected_max_count": max(selected_counts.values()) if selected_counts else 0,
        "selected_documents": len(selected_records),
    }


def select_balancing_subset(
    reference_records: list[dict[str, Any]],
    pool_records: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int = 13,
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_counts = event_type_counts(reference_records)
    pool_counts = event_type_counts(pool_records)
    allowed_event_types = set(reference_counts)
    current_counts = Counter(reference_counts)
    candidates, postings = build_candidates(pool_records, allowed_event_types)
    overlap_labels = sorted(allowed_event_types & set(pool_counts))
    pool_overlap_counts = {
        label: pool_counts[label] for label in overlap_labels if pool_counts[label] > 0
    }

    rng = random.Random(seed)
    for label in postings:
        rng.shuffle(postings[label])

    selected_indices: list[int] = []
    selected_mask = [False] * len(candidates)
    exhausted_labels: set[str] = set()

    def remaining_deficits() -> dict[str, int]:
        return {
            label: max(0, target_count - current_counts[label])
            for label in allowed_event_types
            if current_counts[label] < target_count
        }

    while True:
        deficits = remaining_deficits()
        if not deficits:
            break
        if max_documents is not None and len(selected_indices) >= max_documents:
            break

        label = max(
            (
                current_label
                for current_label in deficits
                if current_label not in exhausted_labels
            ),
            key=lambda item: (deficits[item], item),
            default=None,
        )
        if label is None:
            break

        best_candidate_idx: int | None = None
        best_score: tuple[int, int, int] | None = None
        for candidate_idx in postings.get(label, []):
            if selected_mask[candidate_idx]:
                continue
            candidate = candidates[candidate_idx]
            score = score_candidate(candidate, current_counts, target_count)
            if score[0] <= 0:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        if best_candidate_idx is None:
            exhausted_labels.add(label)
            continue

        selected_mask[best_candidate_idx] = True
        selected_indices.append(best_candidate_idx)
        current_counts.update(candidates[best_candidate_idx].counts)
        exhausted_labels.clear()

    selected_records = [candidates[idx].record for idx in selected_indices]
    report = {
        "target_count": target_count,
        "reference_event_type_counts": dict(reference_counts.most_common()),
        "pool_event_type_counts": dict(pool_counts.most_common()),
        "overlap_labels": overlap_labels,
        "pool_overlap_event_type_counts": pool_overlap_counts,
        "selected_event_type_counts": dict(
            event_type_counts(selected_records).most_common()
        ),
        "combined_event_type_counts": dict(current_counts.most_common()),
        "remaining_deficits": remaining_deficits(),
        "selected_documents": len(selected_records),
        "reference_documents": len(reference_records),
        "pool_documents": len(pool_records),
    }
    return selected_records, report


def select_self_balanced_subset(
    pool_records: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int = 13,
    max_documents: int | None = None,
    max_overshoot_per_label: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_counts = event_type_counts(pool_records)
    allowed_event_types = set(pool_counts)
    current_counts: Counter[str] = Counter()
    candidates, postings = build_candidates(pool_records, allowed_event_types)

    rng = random.Random(seed)
    for label in postings:
        rng.shuffle(postings[label])

    selected_indices: list[int] = []
    selected_mask = [False] * len(candidates)
    exhausted_labels: set[str] = set()

    def remaining_deficits() -> dict[str, int]:
        return {
            label: max(0, target_count - current_counts[label])
            for label in allowed_event_types
            if current_counts[label] < target_count
        }

    while True:
        deficits = remaining_deficits()
        if not deficits:
            break
        if max_documents is not None and len(selected_indices) >= max_documents:
            break

        label = max(
            (
                current_label
                for current_label in deficits
                if current_label not in exhausted_labels
            ),
            key=lambda item: (deficits[item], item),
            default=None,
        )
        if label is None:
            break

        best_candidate_idx: int | None = None
        best_score: tuple[int, int, int, int] | None = None
        for candidate_idx in postings.get(label, []):
            if selected_mask[candidate_idx]:
                continue
            candidate = candidates[candidate_idx]
            score = score_self_balanced_candidate(
                candidate,
                current_counts,
                target_count,
                max_overshoot_per_label=max_overshoot_per_label,
            )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_candidate_idx = candidate_idx

        if best_candidate_idx is None:
            exhausted_labels.add(label)
            continue

        selected_mask[best_candidate_idx] = True
        selected_indices.append(best_candidate_idx)
        current_counts.update(candidates[best_candidate_idx].counts)
        exhausted_labels.clear()

    selected_records = [candidates[idx].record for idx in selected_indices]
    report = summarize_self_balance(selected_records, pool_counts, target_count)
    report["pool_documents"] = len(pool_records)
    report["max_overshoot_per_label"] = max_overshoot_per_label
    return selected_records, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a raw JSONL subset that reduces event-type imbalance relative to an "
            "existing parsed training dataset."
        )
    )
    parser.add_argument(
        "--reference",
        required=False,
        help="Existing parsed JSONL dataset. If omitted, self-balance the pool dataset.",
    )
    parser.add_argument(
        "--pool", required=True, help="Raw JSONL pool to subsample from."
    )
    parser.add_argument(
        "--output", required=True, help="Selected subset output JSONL path."
    )
    parser.add_argument(
        "--report-output",
        required=False,
        help="Optional JSON report path. Defaults to <output>.report.json",
    )
    parser.add_argument(
        "--target-strategy",
        choices=["max", "median", "min"],
        default="max",
        help=(
            "How to choose the per-label balancing target. Use max/median with a reference, "
            "or min/median when self-balancing the pool."
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Explicit per-label target count. Overrides --target-strategy.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on the number of raw documents selected.",
    )
    parser.add_argument(
        "--max-overshoot-per-label",
        type=int,
        default=0,
        help=(
            "Self-balance mode only. Allow up to this many extra occurrences above the "
            "target for any single label. Default 0 enforces hard caps."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="Random seed for tie-breaking."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pool_records = read_jsonl(args.pool)
    if args.reference:
        if args.target_strategy == "min":
            raise ValueError(
                "--target-strategy=min is only supported when --reference is omitted"
            )
        reference_records = read_jsonl(args.reference)
        reference_counts = event_type_counts(reference_records)
        target_count = choose_target_count(
            reference_counts,
            strategy=args.target_strategy,
            explicit_target=args.target_count,
        )

        selected_records, report = select_balancing_subset(
            reference_records,
            pool_records,
            target_count=target_count,
            seed=args.seed,
            max_documents=args.max_documents,
        )
        report["mode"] = "reference_balanced"
    else:
        if args.target_strategy == "max" and args.target_count is None:
            raise ValueError(
                "--target-strategy=max requires --reference; use min or median for self-balancing"
            )
        target_count = choose_self_balanced_target_count(
            event_type_counts(pool_records),
            strategy=args.target_strategy,
            explicit_target=args.target_count,
        )
        selected_records, report = select_self_balanced_subset(
            pool_records,
            target_count=target_count,
            seed=args.seed,
            max_documents=args.max_documents,
            max_overshoot_per_label=args.max_overshoot_per_label,
        )

    output_path = Path(args.output)
    report_path = (
        Path(args.report_output)
        if args.report_output
        else output_path.with_suffix(output_path.suffix + ".report.json")
    )

    write_jsonl(output_path, selected_records)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Selected {len(selected_records)} documents")
    if report.get("mode") == "reference_balanced" and not report["overlap_labels"]:
        print("No overlapping event_type labels were found between reference and pool.")
    print(f"Wrote subset to {output_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
