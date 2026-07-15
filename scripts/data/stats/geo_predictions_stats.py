"""Load and print stats for a `*.geo.jsonl` prediction file.

Each line is an article with `window_predictions`, a list of
{window_start, window_end, window_text, prediction: {events: [...]}}.
Each event has event_type, grounding_quote, event_location, event_time,
time_status, modality.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_events(records: list[dict]) -> Iterator[dict[str, Any]]:
    """Flatten article -> window_predictions -> events into one row per event."""
    for record in records:
        for window in record.get("window_predictions") or []:
            for event in (window.get("prediction") or {}).get("events") or []:
                yield {
                    "id": record.get("id"),
                    "status": record.get("status"),
                    "adm_codes": record.get("adm_codes") or [],
                    "window_start": window.get("window_start"),
                    "window_end": window.get("window_end"),
                    **event,
                }


def print_table(title: str, counts: Counter, total: int, name_width: int = 40, top: int | None = None):
    print()
    print(title)
    print(f"  {'Name':<{name_width}} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*name_width} {'-'*7}  {'-'*6}")
    for name, count in counts.most_common(top):
        pct = count / total * 100 if total else 0
        print(f"  {str(name):<{name_width}} {count:>7,}  {pct:>5.1f}%")


def print_stats(path: Path, top: int) -> None:
    records = load_jsonl(path)
    total = len(records)

    status_counts = Counter(r.get("status", "unknown") for r in records)
    windows = [w for r in records for w in (r.get("window_predictions") or [])]
    events = list(iter_events(records))

    n_windows_with_events = sum(
        1 for w in windows if (w.get("prediction") or {}).get("events")
    )

    event_type_counts = Counter(e.get("event_type", "unknown") for e in events)
    time_status_counts = Counter(e.get("time_status", "unknown") for e in events)
    modality_counts = Counter(e.get("modality", "unknown") for e in events)
    location_counts = Counter(e.get("event_location", "unknown") for e in events)

    print(f"File: {path}")
    print(f"{'='*62}")
    print(f"Articles:                 {total:>8,}")
    print(f"Windows:                  {len(windows):>8,}")
    print(f"Windows with events:      {n_windows_with_events:>8,}"
          + (f"  ({n_windows_with_events/len(windows):.1%})" if windows else ""))
    print(f"Total events:             {len(events):>8,}")

    print_table("Status distribution:", status_counts, total)
    print_table("event_type distribution:", event_type_counts, len(events), top=top)
    print_table("time_status distribution:", time_status_counts, len(events))
    print_table("modality distribution:", modality_counts, len(events))
    print_table("event_location distribution:", location_counts, len(events), top=top)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Print article/window/event distribution stats for a *.geo.jsonl predictions file."
    )
    arg_parser.add_argument("input_path", type=Path)
    arg_parser.add_argument("--top", type=int, default=30, help="Top-N rows to show for high-cardinality tables.")
    args = arg_parser.parse_args()

    print_stats(args.input_path, args.top)
