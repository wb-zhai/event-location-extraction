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

# PHOTON_URL = "https://photon.komoot.io/api/"
PHOTON_URL = "http://localhost:2322/api"

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

# OSM object type is the best proxy for geographic scope available in the API response:
# relations (R) cover large areas, nodes (N) are single points.
_OSM_TYPE_IMPORTANCE = {"R": 1.0, "W": 0.6, "N": 0.3}

# Geographic type scope as a secondary importance signal.
_GEO_SCOPE_IMPORTANCE = {
    "continent": 1.0,
    "country": 0.9,
    "state": 0.7,
    "county": 0.55,
    "borough": 0.55,
    "municipality": 0.55,
    "district": 0.55,
    "city": 0.5,
    "town": 0.4,
    "village": 0.3,
    "suburb": 0.25,
}


def _importance_proxy(props: dict) -> float:
    """Approximate Nominatim importance from OSM type and place type."""
    osm = _OSM_TYPE_IMPORTANCE.get(props.get("osm_type", "N"), 0.3)
    geo = _GEO_SCOPE_IMPORTANCE.get(props.get("type", ""), 0.3)
    return 0.7 * osm + 0.3 * geo


def _reranker_factor(query: str, name: str) -> float:
    """
    Replicates Photon's QueryReranker multiplier.

    Photon applies this factor to the existing OpenSearch score after importance
    weighting, so a high-importance non-matching result can still beat a low-
    importance exact match.  We replicate the same multiplier tiers:
      1.0  exact match
      0.9  name starts with query at a word boundary  ("Africa X" for "Africa")
      0.8  name starts with query (no boundary)
      0.8× word-level char ratio (a word in name equals the full query)
      0.5  no recognisable match (importance alone determines rank)
    """
    q = query.lower().strip()
    n = name.lower().strip()

    if n == q:
        return 1.0

    # Word-boundary prefix: name begins with "query " (space after = boundary)
    if n.startswith(q + " "):
        return 0.9

    # Partial prefix: name begins with query but not at a word boundary
    if n.startswith(q):
        return 0.8

    # Word-level: one or more words in name contain the full query string
    matched_chars = sum(len(w) for w in n.split() if q in w)
    if matched_chars:
        return 0.8 * min(matched_chars / max(len(q), 1), 1.0)

    return 0.5


def _feature_score(feat: dict, query: str) -> float:
    """Composite score replicating Photon's importance × reranker pipeline."""
    props = feat["properties"]
    imp = _importance_proxy(props)
    factor = _reranker_factor(query, props.get("name", ""))
    return factor * imp


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
        resp = session.get(PHOTON_URL, params={"q": query, "limit": 10}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: request failed for '{query}': {e}", file=sys.stderr)
        return None

    data = resp.json()
    features = data.get("features", [])
    if not features:
        return None

    best = max(features, key=lambda f: _feature_score(f, query))
    props = best["properties"]
    coords = best["geometry"]["coordinates"]
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
