#!/usr/bin/env python3
"""
Add geotaxonomy labels to event locations in a JSONL predictions file.

Reads annotation.events and predictions event dicts, resolves each
event_location string via the Photon geocoding API, and writes a
geotaxonomy key back to each event.

Photon admin_level → our taxonomy (in-between levels rounded up to coarser):
  admin_level ≤ 3  → "country"   (2 = country, 3 rounded up to country)
  admin_level 4–5  → "province"  (4 = state, 5 rounded up to province)
  admin_level ≥ 6  → "district"  (6 = county/district)

Non-admin types (city, town, village, …) use PHOTON_TYPE_TO_OURS directly.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geotaxonomy_utils import get_latlon_to_id_from_path

# PHOTON_URL = "https://photon.komoot.io/api/"
PHOTON_URL = "http://localhost:2322/api"

_PHOTON_LAYERS = {"country", "state", "county", "city", "district", "locality", "street", "house", "poi"}

PHOTON_TYPE_TO_OURS = {
    "continent": "country",
    "country": "country",
    "state": "province",
    "county": "district",
    "city": "district",
    "town": "district",
    "village": "district",
    "suburb": "district",
    "borough": "district",
    "district": "district",
    "municipality": "district",
}


def _admin_level_to_ours(level: int) -> str:
    """Map a Photon admin_level integer to our 3-tier taxonomy.

    Levels that fall between our anchor points are rounded up to the coarser
    (higher-hierarchy) tier: 3 → country, 5 → province.
    """
    if level <= 3:
        return "country"
    if level <= 5:
        return "province"
    return "district"


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

    # Partial prefix: name begins with query and next char is a word boundary
    if n.startswith(q) and (len(n) == len(q) or n[len(q)] in (" ", "-", "/")):
        return 0.8

    # Word-level: a word in name exactly equals the full query string.
    # Use equality (not substring) so "Libyan" doesn't match query "Libya".
    matched_chars = sum(len(w) for w in n.split() if w == q)
    if matched_chars:
        return 0.8 * min(matched_chars / max(len(q), 1), 1.0)

    return 0.5


def _feature_score(feat: dict, query: str) -> float:
    """Composite score replicating Photon's importance × reranker pipeline."""
    props = feat["properties"]
    imp = _importance_proxy(props)
    factor = _reranker_factor(query, props.get("name", ""))
    return factor * imp


# Generic administrative-unit words that are often appended/prepended to a place
# name in noisy input ("Gaza City", "City of Gaza") but usually aren't part of
# the name Photon actually indexed ("Gaza"). Stripping them is only attempted as
# a *fallback* — i.e. only once the original query has already failed to find a
# real name match — so genuine multi-word names ("New York City", "Mexico City")
# are left alone, since those score well on the first attempt.
_GENERIC_ADMIN_WORDS = {
    "city", "town", "village", "municipality", "province", "state", "region",
    "district", "county", "governorate", "prefecture", "department", "territory",
    "area", "zone", "capital",
}

# A reranker factor at or below this counts as "no real name match" — the winner
# was picked on importance alone, which is a strong signal the query window
# didn't contain the intended place at all.
_WEAK_MATCH_THRESHOLD = 0.5


def _strip_variants(query: str) -> list[str]:
    """Generate simplified fallback queries by stripping a leading/trailing generic
    admin word (and "<word> of" prefixes like "City of Gaza")."""
    words = query.split()
    variants: list[str] = []
    if len(words) > 1:
        if words[-1].lower().strip(",.") in _GENERIC_ADMIN_WORDS:
            variants.append(" ".join(words[:-1]))
        if words[0].lower().strip(",.") in _GENERIC_ADMIN_WORDS:
            if len(words) > 2 and words[1].lower() == "of":
                variants.append(" ".join(words[2:]))
            else:
                variants.append(" ".join(words[1:]))
    seen: set[str] = set()
    out = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _singularize_words(s: str) -> str:
    """Naive per-word singularization (strip a trailing "s" on words > 3 chars)
    so plural/singular mismatches ("Island" vs "Islands") aren't mistaken for a
    real name mismatch when deciding whether a match is weak."""
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split())


def _is_weak_match(query: str, feat: dict) -> bool:
    """True if `feat` (the top-scored candidate) has no real name match — i.e.
    it was picked on importance alone rather than matching the query text.
    Plural/singular differences ("Island" vs "Islands") are not treated as a
    mismatch."""
    name = feat["properties"].get("name", "")
    if _reranker_factor(query, name) > _WEAK_MATCH_THRESHOLD:
        return False
    return (
        _reranker_factor(_singularize_words(query), _singularize_words(name))
        <= _WEAK_MATCH_THRESHOLD
    )


_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "event-location-extraction/1.0"})
        retry = Retry(
            total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _thread_local.session = s
    return _thread_local.session


def _fetch_features(query: str, layer: str | None) -> list[dict] | None:
    """Query Photon for a location string, returning its raw feature list (None on request failure)."""
    session = _get_session()
    params: list[tuple[str, str]] = [("q", query), ("limit", "10"), ("lang", "en")]
    if layer:
        params.append(("layer", layer))
    try:
        resp = session.get(PHOTON_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: request failed for '{query}': {e}", file=sys.stderr)
        return None
    return resp.json().get("features", [])


def _best_candidate(query: str, layer: str | None) -> dict | None:
    """Fetch candidates for (query, layer) and return the top-scored feature, or None."""
    features = _fetch_features(query, layer)
    if not features:
        return None
    return max(features, key=lambda f: _feature_score(f, query))


def _to_geotaxonomy(query: str, feat: dict) -> dict:
    """Build the geotaxonomy dict for a chosen feature. `query` is always the
    original caller-supplied string, even if `feat` was found via a fallback
    (layer-dropped or word-stripped) retry."""
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]
    photon_type = props.get("type", "")
    admin_level = props.get("admin_level")
    if admin_level is not None:
        our_type = _admin_level_to_ours(int(admin_level))
    else:
        our_type = PHOTON_TYPE_TO_OURS.get(photon_type, "district")

    result: dict = {
        "query": query,
        "resolved_name": props.get("name", query),
        "zhai": our_type,
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


def resolve_location(query: str, layer: str | None = None) -> dict | None:
    """Query Photon for a single location string, return a geotaxonomy dict.

    Photon's own text search — not just our re-ranking — can fail to surface the
    right candidate at all: either the query carries a noisy admin word Photon
    didn't index ("Gaza City" vs. the indexed name "Gaza"), or the caller-supplied
    `layer` doesn't match how Photon actually classifies the entity (e.g. a
    country mislabeled as "city" excludes the real country outright, since layer
    is a hard filter applied before any ranking happens).

    When the first attempt has no real name match, retry in this order and keep
    the first attempt that finds one — always trying a given query text with the
    layer constraint dropped before retrying it with the (already-suspect)
    original layer, since a mismatched `layer` is the more common failure mode
    here and re-applying it can let a same-substring decoy win over the real
    place a layer-free search would have found (e.g. "state of Qatar" + layer
    "city" matching a village literally named "...Qatar..." before the actual
    country is ever considered):
      1. same query, no layer constraint
      2. each generic-admin-word-stripped variant (see _strip_variants), no
         layer constraint
      3. each stripped variant, with the original layer (last resort)
    If nothing improves on the original, its result is kept even if weak, so a
    fallback attempt can only ever replace a bad answer with a better one, never
    turn a bad answer into no answer.
    """
    best = _best_candidate(query, layer)
    if best is not None and not _is_weak_match(query, best):
        return _to_geotaxonomy(query, best)

    attempts: list[tuple[str, str | None]] = []
    if layer is not None:
        attempts.append((query, None))
    variants = _strip_variants(query)
    attempts += [(v, None) for v in variants]
    if layer is not None:
        attempts += [(v, layer) for v in variants]

    for q, lyr in attempts:
        candidate = _best_candidate(q, lyr)
        if candidate is not None and not _is_weak_match(q, candidate):
            return _to_geotaxonomy(query, candidate)

    return _to_geotaxonomy(query, best) if best is not None else None


def resolve_event_location(
    location_str: str,
    cache: dict,
    cache_lock: threading.Lock,
    delay: float,
    admin_level: str | None = None,
) -> list[dict]:
    """Split a potentially semicolon-separated location string and resolve each part."""
    raw_parts = [p.strip() for p in location_str.split(";")]
    raw_levels = [a.strip() for a in (admin_level or "").split(";")]
    # Broadcast a single admin level to all parts; pad shorter lists with "" so no
    # location is silently dropped when counts don't match.
    if len(raw_levels) == 1:
        raw_levels = raw_levels * len(raw_parts)
    elif len(raw_levels) < len(raw_parts):
        raw_levels += [""] * (len(raw_parts) - len(raw_levels))

    results = []

    for part, lvl in zip(raw_parts, raw_levels):
        if not part or part.lower() == "not_stated":
            continue

        layer = lvl if lvl in _PHOTON_LAYERS else None
        cache_key = (part, layer)
        with cache_lock:
            if cache_key in cache:
                val = cache[cache_key]
                if val is not None:
                    results.append(val)
                continue

        geo = resolve_location(part, layer)

        with cache_lock:
            cache.setdefault(cache_key, geo)

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
        admin_level = event.get("event_location_admin_level") or None
        event["geotaxonomy"] = resolve_event_location(loc, cache, cache_lock, delay, admin_level)


def process_obj(
    line: str, cache: dict, cache_lock: threading.Lock, delay: float
) -> dict:
    obj = json.loads(line)

    annotation = obj.get("annotation")
    if isinstance(annotation, dict):
        process_events(annotation.get("events", []), cache, cache_lock, delay)

    process_events(obj.get("predictions", []), cache, cache_lock, delay)

    return obj


def _collect_all_geo_dicts(all_objs: list[dict]) -> list[dict]:
    geo_dicts = []
    for obj in all_objs:
        annotation = obj.get("annotation")
        if isinstance(annotation, dict):
            for event in annotation.get("events", []):
                geo_dicts.extend(event.get("geotaxonomy", []))
        for event in obj.get("predictions", []):
            geo_dicts.extend(event.get("geotaxonomy", []))
    return geo_dicts


_ZHAI_TO_LEVEL = {"country": 0, "province": 1}  # everything else → 2


def _spatial_join_one_level(
    level: int, rows: list[dict], geotaxonomy_dir: str
) -> tuple[int, list[tuple], list[str]]:
    """Run spatial join for one admin level. Top-level so it is picklable for ProcessPoolExecutor."""
    warnings: list[str] = []
    df = pd.DataFrame(rows)
    path = os.path.join(geotaxonomy_dir, f"geotaxonomy_prewb_{level}.geojson")
    try:
        result_df = get_latlon_to_id_from_path(df, path, ["adm_code", "adm_level"], "lat", "lon")
        rows_out = [
            (row.latitude, row.longitude, row.adm_code, row.adm_level)
            for row in result_df.itertuples(index=False)
        ]
    except Exception as e:
        warnings.append(f"  WARNING: adm_code level {level} lookup failed: {e}")
        rows_out = []
    return level, rows_out, warnings


def _build_adm_code_lookup(
    geo_dicts: list[dict],
    geotaxonomy_dir: str,
) -> dict[tuple[float, float], dict[str, object]]:
    # Group unique (lat, lon) pairs by the admin level that matches their zhai type
    level_to_rows: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    seen: set[tuple[float, float]] = set()
    for g in geo_dicts:
        key = (g["lat"], g["lon"])
        if key in seen:
            continue
        seen.add(key)
        level = _ZHAI_TO_LEVEL.get(g.get("zhai", ""), 2)
        level_to_rows[level].append({"lat": g["lat"], "lon": g["lon"]})

    active_levels = [lvl for lvl in [0, 1, 2] if level_to_rows[lvl]]
    lookup: dict[tuple[float, float], dict[str, object]] = {}

    # The three levels are fully independent — run them in parallel processes to
    # overlap GeoJSON loading and the CPU-bound sjoin work.
    n_workers = len(active_levels)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_level = {
            executor.submit(_spatial_join_one_level, lvl, level_to_rows[lvl], geotaxonomy_dir): lvl
            for lvl in active_levels
        }
        for future in tqdm(
            as_completed(future_to_level),
            total=len(active_levels),
            desc="spatial join",
            unit="level",
            file=sys.stderr,
        ):
            _, rows_out, warnings = future.result()
            for w in warnings:
                print(w, file=sys.stderr)
            for lat, lon, adm_code, adm_level in rows_out:
                lookup[(lat, lon)] = {"adm_code": adm_code, "adm_level": adm_level}

    return lookup


def _enrich_geo_dicts_with_adm_codes(
    geo_dicts: list[dict],
    lookup: dict[tuple[float, float], dict[str, object]],
) -> None:
    for geo in geo_dicts:
        entry = lookup.get((geo.get("lat"), geo.get("lon")))
        geo["adm_code"] = entry["adm_code"] if entry is not None else None
        geo["adm_level"] = entry["adm_level"] if entry is not None else None


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
    parser.add_argument(
        "--geotaxonomy-dir",
        default=str(Path(__file__).parent.parent.parent / "dataset" / "geotaxonomy"),
        help="Directory containing geotaxonomy_prewb_0/1/2.geojson files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else input_path.with_suffix(".geo.jsonl")
    )

    cache: dict = {}
    cache_lock = threading.Lock()

    lines = list(input_path.open())
    print(f"Input:  {input_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Workers: {args.workers}", file=sys.stderr)

    # Phase 1 — geocode all lines via Photon
    all_objs: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for obj in tqdm(
            executor.map(
                lambda line: process_obj(line, cache, cache_lock, args.delay), lines
            ),
            total=len(lines),
            desc="geocoding",
            unit="line",
            file=sys.stderr,
        ):
            all_objs.append(obj)

    print(f"Geocoding done. Cache size: {len(cache)}", file=sys.stderr)

    # Phase 2 — batch spatial join to add adm_code_0/1/2
    all_geo_dicts = _collect_all_geo_dicts(all_objs)
    if all_geo_dicts:
        n_unique = len({(g["lat"], g["lon"]) for g in all_geo_dicts})
        print(f"Spatial join: {n_unique} unique coordinates...", file=sys.stderr)
        adm_lookup = _build_adm_code_lookup(all_geo_dicts, args.geotaxonomy_dir)
        _enrich_geo_dicts_with_adm_codes(all_geo_dicts, adm_lookup)

    with output_path.open("w") as fout:
        for obj in all_objs:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
