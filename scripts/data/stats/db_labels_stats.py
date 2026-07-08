import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_science_clusters(path: Path) -> dict[str, str]:
    entries = json.loads(path.read_text())
    return {e["name"]: e["cluster"] for e in entries}


def print_table(title: str, counts: Counter, total: int, name_width: int = 40, top: int | None = None):
    print()
    print(title)
    print(f"  {'Name':<{name_width}} {'Count':>7}  {'%':>6}")
    print(f"  {'-'*name_width} {'-'*7}  {'-'*6}")
    items = counts.most_common(top)
    for name, count in items:
        pct = count / total * 100 if total else 0
        print(f"  {str(name):<{name_width}} {count:>7,}  {pct:>5.1f}%")


def print_stats(path: Path, clusters_path: Path):
    records = load_jsonl(path)
    total = len(records)
    name_to_cluster = load_science_clusters(clusters_path)

    label_counts = Counter(r.get("label", "unknown") for r in records)
    adm0_counts = Counter(r.get("adm0_code") or "unknown" for r in records)

    all_risk_factors = [rf for r in records for rf in (r.get("risk_factors") or [])]
    risk_factor_counts = Counter(all_risk_factors)

    cluster_counts = Counter(
        name_to_cluster.get(rf, "unmapped") for rf in all_risk_factors
    )

    n_with_risk_factors = sum(1 for r in records if r.get("risk_factors"))

    print(f"File: {path}")
    print(f"{'='*62}")
    print(f"Total records:            {total:>8,}")
    print(f"Records with risk_factors: {n_with_risk_factors:>7,}  ({n_with_risk_factors/total:.1%})" if total else "")
    print(f"Total risk_factor tags:   {len(all_risk_factors):>8,}")

    print_table("Label distribution:", label_counts, total)
    print_table("adm0_code distribution:", adm0_counts, total)
    print_table(
        "risk_factors distribution (raw):",
        risk_factor_counts,
        len(all_risk_factors),
        top=30,
    )
    print_table(
        "risk_factors distribution (science_clusters):",
        cluster_counts,
        len(all_risk_factors),
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Print label/adm0/risk_factor distribution stats for a db jsonl file."
    )
    arg_parser.add_argument("input_path", type=Path)
    arg_parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("ontologies/zhai/science_clusters.json"),
        help="Path to science_clusters.json mapping risk_factor name -> cluster.",
    )
    args = arg_parser.parse_args()

    print_stats(args.input_path, args.clusters)
