import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the distribution of annotated event trigger text from JSONL records."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a JSONL file with `events[*].trigger_text` annotations.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        help="Optional path to save a horizontal bar chart of trigger text frequency.",
    )
    parser.add_argument(
        "--most-common",
        type=int,
        default=20,
        help="Number of most common trigger texts to print and plot.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def get_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]

    answer = record.get("answer")
    if isinstance(answer, dict):
        nested_events = answer.get("events")
        if isinstance(nested_events, list):
            return [event for event in nested_events if isinstance(event, dict)]

    return []

def count_spans(
    records: list[dict[str, Any]],
) -> tuple[Counter[str], int]:
    span_texts: Counter[str] = Counter()
    event_total = 0

    for record in records:
        for event in get_events(record):
            trigger_text = str(
                event.get("trigger_text") or event.get("text") or ""
            ).strip().lower()
            if trigger_text:
                span_texts[trigger_text] += 1
                event_total += 1

    return (
        span_texts,
        event_total,
    )


def print_distribution(
    title: str,
    counts: Counter[str],
    total: int,
    most_common: int | None = None,
) -> None:
    print(title)
    print(f"total: {total}")
    if total == 0:
        print("  (none)")
        return

    for label, count in counts.most_common(n=most_common):
        pct = (count / total) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")


def plot_distribution(
    title: str,
    counts: Counter[str],
    total: int,
    output_path: Path,
    most_common: int | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or run without --plot-path."
        ) from exc

    if not counts:
        print(f"No data to plot for {title}")
        return

    labels, values = zip(*counts.most_common(n=most_common))
    percentages = [(value / total) * 100 for value in values]

    fig_height = max(4, min(0.5 * len(labels) + 1.5, 14))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(range(len(labels)), values, align="center", color="#2b6cb0")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    max_value = max(values)
    ax.set_xlim(0, max_value * 1.18 if max_value else 1)
    for bar, value, pct in zip(bars, values, percentages):
        ax.text(
            value + max_value * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {title} plot to {output_path}")
    plt.close()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    (
        span_texts,
        event_total,
    ) = count_spans(records)

    articles_no_events = sum(1 for record in records if not get_events(record))
    avg_events = event_total / len(records) if records else 0.0

    print(f"file: {args.input_path}")
    print(f"records: {len(records)}")
    print(f"average events per article: {avg_events:.2f}")
    print(f"articles without events: {articles_no_events}")
    print()
    print_distribution(
        "Trigger text distribution",
        span_texts,
        event_total,
        most_common=args.most_common,
    )
    print()

    if args.plot_path:
        plot_distribution(
            (
                f"Top-{args.most_common} trigger text distribution"
                if args.most_common
                else "Trigger text distribution"
            ),
            span_texts,
            event_total,
            args.plot_path,
            most_common=args.most_common,
        )


if __name__ == "__main__":
    main()
