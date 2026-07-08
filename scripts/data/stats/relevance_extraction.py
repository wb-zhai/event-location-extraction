"""Print summary stats for a Gemini relevance+extraction JSONL file.

Reports, for each of the "relevant", "partially_relevant", and "not_relevant"
relevance labels, how many rows have at least one extracted event vs. none, plus
the distribution of event_type labels (and their taxonomy clusters) among the
extracted events.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("dataset/db/relevance/matrix_5M.sample_1000.3.1pro.extracted.jsonl")
DEFAULT_TAXONOMY = Path("ontologies/zhai/science.json")
DEFAULT_CLUSTERS = Path("ontologies/zhai/science_clusters.json")

RELEVANCE_LABELS = ("relevant", "partially_relevant", "not_relevant")


def load_label_to_cluster(clusters_path: Path) -> dict[str, str]:
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    return {entry["name"]: entry["cluster"] for entry in clusters}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats for a relevance+extraction JSONL file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the extracted JSONL")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY, help="Path to the event taxonomy JSON")
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS, help="Path to the taxonomy clusters JSON")
    args = parser.parse_args()

    taxonomy_events = set(json.loads(args.taxonomy.read_text(encoding="utf-8"))["events"])
    label_to_cluster = load_label_to_cluster(args.clusters)

    total = 0
    status_counts: Counter[str] = Counter()
    relevance_counts: Counter[str] = Counter()
    with_events: Counter[str] = Counter()
    without_events: Counter[str] = Counter()
    label_dist: dict[str, Counter[str]] = {label: Counter() for label in RELEVANCE_LABELS}
    cluster_dist: dict[str, Counter[str]] = {label: Counter() for label in RELEVANCE_LABELS}
    unknown_labels: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            status_counts[rec.get("status") or "none"] += 1

            annotation = rec.get("annotation") or {}
            relevance_info = rec.get("relevance") or {}
            relevance = relevance_info.get("decision") or "none"
            events = annotation.get("events") or []
            relevance_counts[relevance] += 1

            if relevance in RELEVANCE_LABELS:
                if events:
                    with_events[relevance] += 1
                else:
                    without_events[relevance] += 1

                for event in events:
                    event_type = event.get("event_type") or "none"
                    label_dist[relevance][event_type] += 1
                    if event_type in taxonomy_events:
                        cluster_dist[relevance][label_to_cluster.get(event_type, "none")] += 1
                    else:
                        unknown_labels[event_type] += 1

    print(f"Total records: {total} ({args.input})")
    print()

    print("Status counts:")
    for status, count in status_counts.most_common():
        pct = count / total * 100 if total else 0.0
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    print("document_relevance counts:")
    for label in RELEVANCE_LABELS:
        count = relevance_counts.get(label, 0)
        pct = count / total * 100 if total else 0.0
        print(f"  {label}: {count} ({pct:.1f}%)")
    other = set(relevance_counts) - set(RELEVANCE_LABELS)
    for label in sorted(other):
        print(f"  {label}: {relevance_counts[label]}")
    print()

    for label in RELEVANCE_LABELS:
        rel_total = relevance_counts.get(label, 0)
        n_with = with_events[label]
        n_without = without_events[label]
        print(f"{label} ({rel_total}):")
        print(f"  with extracted events:    {n_with} ({n_with / rel_total * 100 if rel_total else 0.0:.1f}%)")
        print(f"  without extracted events: {n_without} ({n_without / rel_total * 100 if rel_total else 0.0:.1f}%)")
        print()

    for label in RELEVANCE_LABELS:
        dist = label_dist[label]
        event_total = sum(dist.values())
        print(f"Event label distribution for {label} ({event_total} events):")
        for event_type, count in dist.most_common():
            pct = count / event_total * 100 if event_total else 0.0
            print(f"  {event_type}: {count} ({pct:.1f}%)")
        print()

    for label in RELEVANCE_LABELS:
        dist = cluster_dist[label]
        event_total = sum(dist.values())
        print(f"Cluster distribution for {label} ({event_total} events):")
        for cluster, count in dist.most_common():
            pct = count / event_total * 100 if event_total else 0.0
            print(f"  {cluster}: {count} ({pct:.1f}%)")
        print()

    if unknown_labels:
        print("Event labels not found in taxonomy:")
        for event_type, count in unknown_labels.most_common():
            print(f"  {event_type}: {count}")


if __name__ == "__main__":
    main()
