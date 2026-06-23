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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
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

_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "event-location-extraction/1.0"})
        retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _thread_local.session = s
    return _thread_local.session


def resolve_location(query: str) -> dict | None:
    """Query Photon for a single location string, return a geotaxonomy dict."""
    session = _get_session()
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

    if "osm_id" in props:
        result["osm_id"] = props["osm_id"]
    if "osm_type" in props:
        result["osm_type"] = props["osm_type"]

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
    cache_lock: threading.Lock,
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

        with cache_lock:
            if part in cache:
                val = cache[part]
                if val is not None:
                    results.append(val)
                continue

        geo = resolve_location(part)

        with cache_lock:
            cache.setdefault(part, geo)

        if geo is not None:
            results.append(geo)
        if delay > 0:
            time.sleep(delay)

    return results


def process_events(
    events: list[dict], cache: dict, cache_lock: threading.Lock, delay: float
) -> None:
    """Mutate each event dict in-place, adding a geotaxonomy key."""
    for event in events:
        loc = event.get("event_location", "")
        if not loc or loc.lower() == "not_stated":
            continue
        event["geotaxonomy"] = resolve_event_location(loc, cache, cache_lock, delay)


def process_obj(line: str, cache: dict, cache_lock: threading.Lock, delay: float) -> str:
    obj = json.loads(line)

    annotation = obj.get("annotation")
    if isinstance(annotation, dict):
        process_events(annotation.get("events", []), cache, cache_lock, delay)

    process_events(obj.get("predictions", []), cache, cache_lock, delay)

    return json.dumps(obj, ensure_ascii=False)


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
        help="Seconds between API requests per thread (default: 0.1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel geocoding threads (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".geo.jsonl")

    cache: dict = {}
    cache_lock = threading.Lock()

    total = sum(1 for _ in input_path.open())
    print(f"Input:  {input_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Workers: {args.workers}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with input_path.open() as fin, output_path.open("w") as fout:
            results = executor.map(
                lambda line: process_obj(line, cache, cache_lock, args.delay),
                fin,
            )
            for result in tqdm(results, total=total, desc="geocoding", unit="line", file=sys.stderr):
                fout.write(result + "\n")

    print(f"Done. Cache size: {len(cache)}", file=sys.stderr)


if __name__ == "__main__":
    main()
