import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_geo_path(base_path: Path) -> Path | None:
    geo_dir = base_path.parent.parent / (base_path.parent.name + "_geo")
    geo_file = geo_dir / (base_path.stem + ".geo.jsonl")
    return geo_file if geo_file.exists() else None


def merge_geo(records: list, geo_records: list) -> list:
    geo_by_id = {r["id"]: r for r in geo_records}
    merged = []
    for r in records:
        geo = geo_by_id.get(r["id"])
        if geo:
            merged.append(geo)
        else:
            merged.append(r)
    return merged


def print_bar(label: str, count: int, total: int, width: int = 30):
    pct = count / total if total else 0
    filled = int(pct * width)
    bar = "#" * filled + "." * (width - filled)
    print(f"  [{bar}] {pct:5.1%}  {count:>7,}  {label}")


def has_geotaxonomy(records: list) -> bool:
    return any(
        "geotaxonomy" in e
        for r in records
        for e in r.get("predictions", [])
    )


def load_file_with_geo(base_path: Path) -> list:
    records = load_jsonl(base_path)
    if not has_geotaxonomy(records):
        geo_path = find_geo_path(base_path)
        if geo_path is not None:
            records = merge_geo(records, load_jsonl(geo_path))
    return records


def collect_records(path: Path) -> tuple[list, str]:
    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {path}")
            sys.exit(1)
        records = []
        for f in jsonl_files:
            records.extend(load_file_with_geo(f))
        label = f"Folder: {path}  ({len(jsonl_files)} files)"
    else:
        records = load_file_with_geo(path)
        label = f"File: {path}"
    return records, label


def print_stats(path: str):
    base_path = Path(path)
    records, source_label = collect_records(base_path)

    has_geo = has_geotaxonomy(records)
    total = len(records)

    empty = [r for r in records if not r.get("predictions")]
    n_empty = len(empty)

    all_events = [e for r in records for e in r.get("predictions", [])]
    n_events = len(all_events)

    label_counts = Counter(e.get("event_type", "unknown") for e in all_events)
    location_counts = Counter(
        e.get("event_location", "").strip()
        for e in all_events
        if e.get("event_location", "").strip()
    )

    locs_per_article = [len(r.get("predictions", [])) for r in records if r.get("predictions")]
    avg_locs = sum(locs_per_article) / len(locs_per_article) if locs_per_article else 0.0

    # --- geo stats ---
    all_geo = [
        g
        for e in all_events
        for g in e.get("geotaxonomy", [])
    ]
    resolved_counts = Counter(
        g.get("resolved_name", "").strip()
        for g in all_geo
        if g.get("resolved_name", "").strip()
    )
    geo_type_counts = Counter(g.get("type", "unknown") for g in all_geo)
    n_events_with_geo = sum(1 for e in all_events if e.get("geotaxonomy"))

    # ----------------------------------------------------------------
    print(source_label)
    print(f"{'='*62}")
    print(f"Total articles:          {total:>8,}")
    print(f"Empty predictions:       {n_empty:>8,}  ({n_empty/total:.1%})")
    print(f"Total events:            {n_events:>8,}")
    print(f"Avg locations/article*:  {avg_locs:>8.2f}  (* among non-empty)")
    if has_geo:
        print(f"Events with geo:         {n_events_with_geo:>8,}  ({n_events_with_geo/n_events:.1%} of events)")
        print(f"Total geo resolutions:   {len(all_geo):>8,}")

    print()
    print("Label frequency:")
    print(f"  {'Label':<35} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*35} {'-'*7}  {'-'*6}")
    for label, count in label_counts.most_common():
        pct = count / n_events * 100 if n_events else 0
        print(f"  {label:<35} {count:>7,}  {pct:>5.1f}%")

    print()
    print("Top 10 event_location (raw string):")
    print(f"  {'Location':<40} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*40} {'-'*7}  {'-'*6}")
    total_locs = sum(location_counts.values())
    for loc, count in location_counts.most_common(10):
        pct = count / total_locs * 100 if total_locs else 0
        print(f"  {loc:<40} {count:>7,}  {pct:>5.1f}%")

    if has_geo:
        print()
        print("Top 10 resolved_name (geo):")
        print(f"  {'Resolved name':<40} {'Count':>7}  {'%':>6}")
        print(f"  {'-'*40} {'-'*7}  {'-'*6}")
        total_resolved = sum(resolved_counts.values())
        for name, count in resolved_counts.most_common(10):
            pct = count / total_resolved * 100 if total_resolved else 0
            print(f"  {name:<40} {count:>7,}  {pct:>5.1f}%")

        print()
        print("Geo resolution type breakdown:")
        print(f"  {'Type':<20} {'Count':>7}  {'%':>6}")
        print(f"  {'-'*20} {'-'*7}  {'-'*6}")
        total_geo = sum(geo_type_counts.values())
        for gtype, count in geo_type_counts.most_common():
            pct = count / total_geo * 100 if total_geo else 0
            print(f"  {gtype:<20} {count:>7,}  {pct:>5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_stats_predictions.py <path/to/shard.jsonl|folder/>")
        sys.exit(1)
    print_stats(sys.argv[1])
