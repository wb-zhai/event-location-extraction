#!/usr/bin/env python3
"""
Add geotaxonomy labels to event locations in a JSONL predictions file.

Reads annotation.events and predictions event dicts, resolves each
event_location string via the Photon geocoding API, and writes a
geotaxonomy key back to each event.

Photon admin level → our taxonomy:
  type=country  (admin_level 2) → "country"
  type=state    (admin_level 4) → "province"
  type=county   (admin_level 6) → "district"
  type=city/town/village/...    → "city" / "town" / etc.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PHOTON_URL = "https://photon.komoot.io/api/"

PHOTON_TYPE_TO_OURS = {
    "country": "country",
    "state": "province",
    "county": "district",
    "city": "city",
    "town": "city",
    "village": "village",
    "suburb": "suburb",
    "borough": "district",
    "district": "district",
    "municipality": "district",
}


def resolve_location(query: str, session: requests.Session) -> dict | None:
    """Query Photon for a single location string, return a geotaxonomy dict."""
    try:
        resp = session.get(PHOTON_URL, params={"q": query, "limit": 1}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: request failed for '{query}': {e}", file=sys.stderr)
        return None

    data = resp.json()
    features = data.get("features", [])
    if not features:
        return None

    props = features[0]["properties"]
    coords = features[0]["geometry"]["coordinates"]
    photon_type = props.get("type", "")
    our_type = PHOTON_TYPE_TO_OURS.get(photon_type, photon_type)

    result: dict = {
        "query": query,
        "resolved_name": props.get("name", query),
        "type": our_type,
        "photon_type": photon_type,
        "lat": coords[1],
        "lon": coords[0],
    }

    if "country" in props:
        result["country"] = props["country"]
    if "countrycode" in props:
        result["countrycode"] = props["countrycode"]
    if "state" in props:
        result["province"] = props["state"]
    if "county" in props:
        result["district"] = props["county"]
    if "city" in props:
        result["city"] = props["city"]

    return result


def resolve_event_location(
    location_str: str,
    cache: dict,
    session: requests.Session,
    delay: float,
) -> list[dict]:
    """
    Split a potentially semicolon-separated location string and resolve each part.
    Returns a list of geotaxonomy dicts.
    """
    parts = [p.strip() for p in location_str.split(";") if p.strip()]
    results = []

    for part in parts:
        if part.lower() == "not_stated":
            continue
        if part in cache:
            results.append(cache[part])
        else:
            print(f"  Querying: '{part}'", file=sys.stderr)
            geo = resolve_location(part, session)
            cache[part] = geo  # cache even None results
            if geo is not None:
                results.append(geo)
            if delay > 0:
                time.sleep(delay)

    return results


def process_events(events: list[dict], cache: dict, session: requests.Session, delay: float) -> None:
    """Mutate each event dict in-place, adding a geotaxonomy key."""
    for event in events:
        loc = event.get("event_location", "")
        if not loc or loc.lower() == "not_stated":
            continue
        event["geotaxonomy"] = resolve_event_location(loc, cache, session, delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add geotaxonomy to event locations")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "-o", "--output", help="Output JSONL file (default: input with .geo suffix)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds between API requests (default: 0.1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".geo.jsonl")

    cache: dict = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "event-location-extraction/1.0"})
    retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    print(f"Input:  {input_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    with input_path.open() as fin, output_path.open("w") as fout:
        for i, line in enumerate(fin):
            obj = json.loads(line)

            # annotation.events (singular key, nested under "annotation")
            annotation = obj.get("annotation")
            if isinstance(annotation, dict):
                process_events(annotation.get("events", []), cache, session, args.delay)

            # predictions (list of event dicts at top level)
            process_events(obj.get("predictions", []), cache, session, args.delay)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1} lines, cache size: {len(cache)}", file=sys.stderr)

    print(f"Done. Cache size: {len(cache)}", file=sys.stderr)
    print(f"Output written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
