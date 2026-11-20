#!/usr/bin/env python3
"""
Interactively query Photon and inspect how add_geotaxonomy.py would score/rank
and resolve the results.

Imports the actual scoring and fallback-cascade functions from add_geotaxonomy.py
(importance proxy, reranker factor, weak-match detection, layer/text fallback
tiers) rather than reimplementing them, so what you see here can never drift
from what the real pipeline does. The final answer shown is always produced by
calling add_geotaxonomy.resolve_location() directly; everything printed above it
is just a play-by-play of the same tiers that function walks internally.

Usage:
    # One-off query
    python photon_playground.py "Berlin"
    python photon_playground.py "Berlin" --layer city
    python photon_playground.py "Berlin" --limit 5 --json

    # Interactive REPL (omit the query)
    python photon_playground.py
    > Berlin
    > Berlin --layer city
    > quit
"""

import argparse
import json
import shlex
import sys

from add_geotaxonomy import (
    PHOTON_URL,
    _PHOTON_LAYERS,
    _feature_score,
    _get_session,
    _is_weak_match,
    _strip_variants,
    resolve_location,
)


def query_photon(query: str, layer: str | None, limit: int, url: str) -> dict:
    session = _get_session()
    params: list[tuple[str, str]] = [("q", query), ("limit", str(limit)), ("lang", "en")]
    if layer:
        params.append(("layer", layer))
    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _fetch_and_score(
    query: str, layer: str | None, limit: int, url: str
) -> tuple[dict, list[tuple[float, dict]]] | None:
    """Fetch candidates for `query` and return them sorted by composite score.
    Returns None on request failure."""
    try:
        data = query_photon(query, layer, limit, url)
    except Exception as e:
        print(f"  ERROR: request failed: {e}", file=sys.stderr)
        return None
    features = data.get("features", [])
    scored = sorted(
        ((_feature_score(f, query), f) for f in features), key=lambda x: x[0], reverse=True
    )
    return data, scored


def _print_candidates(query: str, layer: str | None, scored: list[tuple[float, dict]]) -> None:
    header = f'Query: "{query}"' + (f"  layer={layer}" if layer else "  (no layer)")
    print(header)
    print("-" * len(header))
    for rank, (score, feat) in enumerate(scored, start=1):
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        marker = " <-- SELECTED" if rank == 1 else ""
        name = props.get("name", "")
        extra_bits = [
            props.get("state"),
            props.get("country"),
        ]
        extra = ", ".join(b for b in extra_bits if b)
        admin_level = props.get("admin_level")
        admin_str = f" admin_level={admin_level}" if admin_level is not None else ""
        print(
            f"{rank:2d}. score={score:.3f}  {name!r} ({extra})  "
            f"type={props.get('type', '?')}{admin_str}  "
            f"osm={props.get('osm_type', '?')}{props.get('osm_id', '?')}  "
            f"lat/lon=({coords[1]:.4f}, {coords[0]:.4f}){marker}"
        )


def _try_tier(
    label: str, query: str, layer: str | None, limit: int, url: str
) -> bool:
    """Fetch + print one cascade tier. Returns True if it found a real name match."""
    print(f"\n{label}")
    fetched = _fetch_and_score(query, layer, limit, url)
    if fetched is None:
        return False
    _data, scored = fetched
    if not scored:
        print(f'  No results for "{query}"' + (f" (layer={layer})" if layer else ""))
        return False
    _print_candidates(query, layer, scored)
    weak = _is_weak_match(query, scored[0][1])
    if weak:
        print("  (no real name match — picked on importance alone)")
    return not weak


def run_query(
    query: str,
    layer: str | None,
    limit: int,
    url: str,
    show_json: bool,
    fallback: bool = True,
) -> None:
    """Replay add_geotaxonomy.resolve_location's exact cascade order, printing
    each tier it would try, then report the tier's real, authoritative answer."""
    found = _try_tier("Tier 1: original query", query, layer, limit, url)

    if not found and fallback:
        tiers: list[tuple[str, str, str | None]] = []
        if layer is not None:
            tiers.append(("Tier 2: layer dropped", query, None))
        variants = _strip_variants(query)
        tiers += [(f'Tier 3: stripped, no layer ("{v}")', v, None) for v in variants]
        if layer is not None:
            tiers += [
                (f'Tier 4: stripped, original layer ("{v}")', v, layer) for v in variants
            ]

        for label, q, lyr in tiers:
            if _try_tier(label, q, lyr, limit, url):
                print("  --> real name match found, cascade stops here.")
                break
        else:
            if tiers:
                print("\nNo tier found a real name match — falling back to the tier-1 result.")
    elif not found:
        print("\n(fallback disabled with --no-fallback; keeping the tier-1 result as-is)")

    # Whatever happened above, get the authoritative answer straight from the
    # real pipeline function so this can never drift from production behavior.
    if fallback:
        best = resolve_location(query, layer)
    else:
        # --no-fallback: replicate tier-1-only behavior without invoking the cascade.
        fetched = _fetch_and_score(query, layer, limit, url)
        if fetched is None or not fetched[1]:
            best = None
        else:
            from add_geotaxonomy import _to_geotaxonomy

            best = _to_geotaxonomy(query, fetched[1][0][1])

    print("\nWould write to `geotaxonomy`:")
    print(json.dumps(best, indent=2, ensure_ascii=False) if best else "null")

    if show_json:
        fetched = _fetch_and_score(query, layer, limit, url)
        if fetched is not None:
            print("\nRaw Photon response (tier 1):")
            print(json.dumps(fetched[0], indent=2, ensure_ascii=False))
    print()


def parse_line(line: str) -> tuple[str, argparse.Namespace] | None:
    """Parse a REPL line as `<query> [--layer L] [--limit N] [--json] [--no-fallback]`."""
    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("query", nargs="+")
    sub.add_argument("--layer", choices=sorted(_PHOTON_LAYERS), default=None)
    sub.add_argument("--limit", type=int, default=10)
    sub.add_argument("--json", action="store_true")
    sub.add_argument("--no-fallback", action="store_true")
    try:
        tokens = shlex.split(line)
        if not tokens:
            return None
        ns = sub.parse_args(tokens)
    except SystemExit:
        return None
    return " ".join(ns.query), ns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate add_geotaxonomy's Photon query/resolution for one location string."
    )
    parser.add_argument("query", nargs="?", help="Location string to query (omit for REPL mode)")
    parser.add_argument("--layer", choices=sorted(_PHOTON_LAYERS), default=None)
    parser.add_argument("--limit", type=int, default=10, help="Number of candidates to fetch (default: 10)")
    parser.add_argument("--url", default=PHOTON_URL, help=f"Photon API URL (default: {PHOTON_URL})")
    parser.add_argument("--json", action="store_true", help="Also print the raw Photon response")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable the layer-relaxation / generic-word-stripping fallback cascade on weak matches",
    )
    args = parser.parse_args()

    if args.query:
        run_query(args.query, args.layer, args.limit, args.url, args.json, not args.no_fallback)
        return

    print(f"Photon playground — querying {args.url}")
    print(
        'Enter a location string, optionally followed by '
        '"--layer L" / "--limit N" / "--json" / "--no-fallback".'
    )
    print("Type 'quit' or Ctrl-D to exit.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        parsed = parse_line(line)
        if parsed is None:
            continue
        query, ns = parsed
        run_query(query, ns.layer, ns.limit, args.url, ns.json, not ns.no_fallback)


if __name__ == "__main__":
    main()
