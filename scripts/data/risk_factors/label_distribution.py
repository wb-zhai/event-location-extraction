import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLUSTER_PATH = REPO_ROOT / "ontologies/risk-factors/risks.names.clusters.training.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the distribution of event roles, argument roles, and event clusters from JSONL records."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a JSONL file like dataset/zhai/raw/sample_1000_with_tags.jsonl",
    )
    parser.add_argument(
        "--cluster-path",
        type=Path,
        default=DEFAULT_CLUSTER_PATH,
        help="Path to the ontology JSON mapping event names to clusters.",
    )
    parser.add_argument(
        "--plot-cluster",
        type=Path,
        help="Optional path to save a bar chart of the cluster distribution (e.g., cluster_dist.png).",
    )
    parser.add_argument(
        "--unknown-path",
        type=Path,
        help="Optional path to save the list of unknown event roles.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def load_cluster_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")

    cluster_map: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        cluster = str(item.get("cluster") or "").strip()
        if name and cluster:
            cluster_map[name] = cluster
    return cluster_map


def count_roles(
    records: list[dict[str, Any]], cluster_map: dict[str, str]
) -> tuple[Counter[str], Counter[str], Counter[str], int, int, int]:
    event_roles: Counter[str] = Counter()
    argument_roles: Counter[str] = Counter()
    cluster_roles: Counter[str] = Counter()
    event_total = 0
    argument_total = 0
    cluster_total = 0

    for record in records:
        events = record.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue

            event_role = str(event.get("event_type") or "").strip()
            if event_role:
                event_roles[event_role] += 1
                event_total += 1
                cluster_roles[cluster_map.get(event_role, "unknown")] += 1
                cluster_total += 1

            arguments = event.get("arguments", [])
            if not isinstance(arguments, list):
                continue

            for argument in arguments:
                if not isinstance(argument, dict):
                    continue
                argument_role = str(argument.get("role") or "").strip()
                if argument_role:
                    argument_roles[argument_role] += 1
                    argument_total += 1

    return (
        event_roles,
        argument_roles,
        cluster_roles,
        event_total,
        argument_total,
        cluster_total,
    )


def print_distribution(title: str, counts: Counter[str], total: int) -> None:
    print(title)
    print(f"total: {total}")
    if total == 0:
        print("  (none)")
        return

    for label, count in counts.most_common():
        pct = (count / total) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")


def plot_distribution(title: str, counts: Counter[str], output_path: Path) -> None:
    if not counts:
        print(f"No data to plot for {title}")
        return

    labels, values = zip(*counts.most_common())

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(labels)), values, align="center")
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()  # Highest counts at the top
    plt.xlabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {title} plot to {output_path}")
    plt.close()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_path)
    cluster_map = load_cluster_map(args.cluster_path)
    (
        event_roles,
        argument_roles,
        cluster_roles,
        event_total,
        argument_total,
        cluster_total,
    ) = count_roles(records, cluster_map)

    articles_no_events = sum(1 for r in records if not r.get("events"))
    avg_events = event_total / len(records) if records else 0.0

    print(f"file: {args.input_path}")
    print(f"cluster map: {args.cluster_path}")
    print(f"records: {len(records)}")
    print(f"average events per article: {avg_events:.2f}")
    print(f"articles without events: {articles_no_events}")
    print()
    print_distribution("Event role distribution", event_roles, event_total)
    print()
    print_distribution("Cluster distribution", cluster_roles, cluster_total)
    print()
    print_distribution("Argument role distribution", argument_roles, argument_total)

    if args.plot_cluster:
        plot_distribution("Cluster distribution", cluster_roles, args.plot_cluster)

    if args.unknown_path:
        unknowns = [role for role in event_roles if role not in cluster_map]
        if unknowns:
            with args.unknown_path.open("w", encoding="utf-8") as f:
                for u in sorted(unknowns):
                    f.write(f"{u}\n")
            print(f"Saved {len(unknowns)} unknown event roles to {args.unknown_path}")
        else:
            print("No unknown event roles found.")


if __name__ == "__main__":
    main()
