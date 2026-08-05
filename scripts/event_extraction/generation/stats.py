#!/usr/bin/env python3
"""Print summary stats for relevance+event-annotated JSONL, e.g.:

    dataset/extraction/en_5k.relevance.annotated.v8.jsonl

Each record is expected to look like:

    {
      "relevance": {"is_relevant": bool, "decision": "relevant"/"irrelevant", ...},
      "annotation": {"events": [{"event_type": ..., "event_location": ..., ...}, ...]},
      "status": "ok" / "error",
      ...
    }

Default output:
  - relevant vs. not-relevant article counts (relevance.is_relevant)
  - relevant articles with zero events / not-relevant articles with events
    ("mismatches" between the relevance filter and what the extractor found)
  - event_type distribution
  - top-N event_location distribution (a single event's event_location can be
    multiple ";"-separated places, e.g. "Ethiopia; Indonesia; Haiti" -- each
    place is counted separately)

--complete adds: generation status breakdown, events-per-article histogram,
time_status / severity / event_location_admin_level distributions, and
top-N adm0_code (country) x relevance cross-tab.

Usage:
    python scripts/event_extraction/generation/stats.py \
        --input dataset/extraction/en_5k.relevance.annotated.v8.jsonl

    python scripts/event_extraction/generation/stats.py \
        --input dataset/extraction/en_5k.relevance.annotated.v8.jsonl --complete --top-n 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.event_extraction.generation.io_utils import iter_jsonl, resolve_path

NOT_STATED = "not_stated"


def get_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = rec.get("annotation")
    if not isinstance(annotation, dict):
        return []
    events = annotation.get("events")
    return events if isinstance(events, list) else []


def get_is_relevant(rec: dict[str, Any]) -> bool | None:
    relevance = rec.get("relevance")
    if not isinstance(relevance, dict):
        return None
    is_relevant = relevance.get("is_relevant")
    return is_relevant if isinstance(is_relevant, bool) else None


def split_locations(event: dict[str, Any]) -> list[str]:
    """event_location can hold multiple ";"-separated places for one event."""
    raw = event.get("event_location")
    if not isinstance(raw, str) or not raw.strip():
        return []
    places = [p.strip() for p in raw.split(";")]
    return [p for p in places if p and p.lower() != NOT_STATED]


def print_counter(counter: Counter, total: int, top_n: int | None = None, indent: str = "  ") -> None:
    items = counter.most_common(top_n)
    for key, count in items:
        pct = count / total * 100 if total else 0.0
        print(f"{indent}{key}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats for relevance+event-annotated JSONL data.")
    parser.add_argument("--input", required=True, help="Path to a JSONL file.")
    parser.add_argument("--complete", action="store_true",
        help="Also print status, events-per-article, time_status/severity/admin_level, and country breakdowns.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top locations/countries to show (default 20).")
    args = parser.parse_args()

    path = resolve_path(args.input)

    total = 0
    statuses: Counter = Counter()
    relevant_counts: Counter = Counter()  # "relevant" / "not_relevant" / "unknown"
    event_types: Counter = Counter()
    locations: Counter = Counter()
    events_per_article: Counter = Counter()
    time_status: Counter = Counter()
    severity: Counter = Counter()
    admin_level: Counter = Counter()
    countries: Counter = Counter()
    country_relevant: Counter = Counter()
    relevant_no_events = 0
    not_relevant_with_events = 0

    for rec in iter_jsonl(path):
        total += 1
        statuses[rec.get("status") or "unknown"] += 1

        is_relevant = get_is_relevant(rec)
        bucket = "unknown" if is_relevant is None else ("relevant" if is_relevant else "not_relevant")
        relevant_counts[bucket] += 1

        events = get_events(rec)
        n_events = len(events)
        events_per_article[n_events if n_events < 5 else "5+"] += 1

        if is_relevant is True and n_events == 0:
            relevant_no_events += 1
        elif is_relevant is False and n_events > 0:
            not_relevant_with_events += 1

        country = rec.get("adm0_code") or "unknown"
        countries[country] += 1
        country_relevant[(country, bucket)] += 1

        for event in events:
            event_types[event.get("event_type") or "unknown"] += 1
            time_status[event.get("time_status") or "unknown"] += 1
            severity[event.get("severity") or "unknown"] += 1
            for level in str(event.get("event_location_admin_level") or "").split(";"):
                level = level.strip()
                if level and level.lower() != NOT_STATED:
                    admin_level[level] += 1
            for place in split_locations(event):
                locations[place] += 1

    print(f"Total records: {total} ({path})")
    print()

    print("Relevance (relevance.is_relevant):")
    print_counter(relevant_counts, total)
    print()

    print("Relevance vs. events mismatch:")
    n_relevant = relevant_counts["relevant"]
    n_not_relevant = relevant_counts["not_relevant"]
    pct_a = relevant_no_events / n_relevant * 100 if n_relevant else 0.0
    pct_b = not_relevant_with_events / n_not_relevant * 100 if n_not_relevant else 0.0
    print(f"  relevant with zero events: {relevant_no_events} ({pct_a:.1f}% of relevant)")
    print(f"  not-relevant with events:  {not_relevant_with_events} ({pct_b:.1f}% of not-relevant)")
    print()

    n_events_total = sum(event_types.values())
    print(f"event_type distribution ({n_events_total} events total):")
    print_counter(event_types, n_events_total)
    print()

    n_locations_total = sum(locations.values())
    print(f"Top {args.top_n} event locations ({n_locations_total} location mentions total):")
    print_counter(locations, n_locations_total, args.top_n)

    if not args.complete:
        return
    print()

    print("Generation status:")
    print_counter(statuses, total)
    print()

    print("Events per article:")
    for key in sorted(events_per_article, key=lambda k: (isinstance(k, str), k)):
        count = events_per_article[key]
        pct = count / total * 100 if total else 0.0
        print(f"  {key}: {count} ({pct:.1f}%)")
    print()

    print(f"time_status distribution ({n_events_total} events total):")
    print_counter(time_status, n_events_total)
    print()

    print(f"severity distribution ({n_events_total} events total):")
    print_counter(severity, n_events_total)
    print()

    n_admin_total = sum(admin_level.values())
    print(f"event_location_admin_level distribution ({n_admin_total} location mentions total):")
    print_counter(admin_level, n_admin_total)
    print()

    print(f"Top {args.top_n} countries (adm0_code):")
    for country, country_total in countries.most_common(args.top_n):
        pct = country_total / total * 100 if total else 0.0
        print(f"  {country}: {country_total} ({pct:.1f}%)")
        for bucket in ("relevant", "not_relevant", "unknown"):
            count = country_relevant[(country, bucket)]
            if count == 0:
                continue
            bpct = count / country_total * 100 if country_total else 0.0
            print(f"    {bucket}: {count} ({bpct:.1f}%)")


if __name__ == "__main__":
    main()
