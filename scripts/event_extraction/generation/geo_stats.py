#!/usr/bin/env python3
"""Top-N location frequency table using Photon-resolved locations
(event.geotaxonomy[].resolved_name) instead of the raw event_location text.

Each event can carry multiple geotaxonomy entries (one per place Photon
resolved out of the location text); each resolved place is counted
separately, same convention as stats.py's raw event_location breakdown.

Usage:
    python scripts/event_extraction/generation/geo_stats.py \
        --input dataset/extraction/data/predictions/.../dev_en_5k....geo.jsonl \
        --source predictions --top-n 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.event_extraction.generation.io_utils import iter_jsonl, resolve_path
from scripts.event_extraction.generation.stats import print_counter


def get_events(rec: dict[str, Any], source: str) -> list[dict[str, Any]]:
    if source == "predictions":
        events = rec.get("predictions")
    else:
        annotation = rec.get("annotation")
        events = annotation.get("events") if isinstance(annotation, dict) else None
    return events if isinstance(events, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-N Photon-resolved location distribution.")
    parser.add_argument("--input", required=True, help="Path to a JSONL file with event.geotaxonomy entries.")
    parser.add_argument("--source", choices=["predictions", "annotation"], default="predictions",
        help="Which events to read: model predictions (default) or gold annotation.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top locations to show (default 30).")
    args = parser.parse_args()

    path = resolve_path(args.input)

    total = 0
    n_with_events = 0
    n_events_total = 0
    n_places_total = 0
    resolved_names: Counter = Counter()
    countries: Counter = Counter()
    unresolved_events = 0  # events with a location but no geotaxonomy match

    for rec in iter_jsonl(path):
        total += 1
        events = get_events(rec, args.source)
        if events:
            n_with_events += 1
        for event in events:
            n_events_total += 1
            geotax = event.get("geotaxonomy")
            if not isinstance(geotax, list) or not geotax:
                unresolved_events += 1
                continue
            for place in geotax:
                if not isinstance(place, dict):
                    continue
                name = place.get("resolved_name")
                if not name:
                    continue
                n_places_total += 1
                resolved_names[name] += 1
                country = place.get("country")
                if country:
                    countries[country] += 1

    print(f"Total records: {total} ({path})")
    print(f"Records with {args.source} events: {n_with_events}")
    print(f"Total {args.source} events: {n_events_total}")
    print(f"Events with no geotaxonomy match: {unresolved_events}")
    print()

    print(f"Top {args.top_n} resolved locations (geotaxonomy[].resolved_name, {n_places_total} place mentions total):")
    print_counter(resolved_names, n_places_total, args.top_n)
    print()

    n_countries_total = sum(countries.values())
    print(f"Top {args.top_n} resolved countries (geotaxonomy[].country, {n_countries_total} place mentions total):")
    print_counter(countries, n_countries_total, args.top_n)


if __name__ == "__main__":
    main()
