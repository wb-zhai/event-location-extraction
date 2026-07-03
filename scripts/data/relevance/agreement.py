"""Compute relevance-decision agreement between two labeled JSONL files.

Compares the `relevance.decision` field (as written by relevance_filter.py,
or human/argilla exports with the same shape) across two files, matched by
record `id`. Reports accuracy, Cohen's kappa, and a confusion matrix over the
ids common to both files.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[str(rec["id"])] = rec
    return records


def get_decision(rec: dict[str, Any]) -> str | None:
    rel = rec.get("relevance") or {}
    decision = rel.get("decision")
    if decision:
        return decision
    is_relevant = rel.get("is_relevant")
    if is_relevant is not None:
        return "relevant" if is_relevant else "irrelevant"
    return None


def get_model_name(records: dict[str, dict[str, Any]]) -> str | None:
    for rec in records.values():
        model = (rec.get("relevance") or {}).get("model") or (rec.get("llm") or {}).get("model")
        if model:
            return model
    return None


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    labels = sorted({v for pair in pairs for v in pair})
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    pe = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def record_title(rec: dict[str, Any]) -> str:
    return str(rec.get("title") or (rec.get("source") or {}).get("title", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare relevance decisions between two labeled JSONL files."
    )
    parser.add_argument("--a", required=True, type=Path, help="First JSONL file")
    parser.add_argument("--b", required=True, type=Path, help="Second JSONL file")
    parser.add_argument("--a-name", default=None, help="Label for --a in the report (default: model name)")
    parser.add_argument("--b-name", default=None, help="Label for --b in the report (default: model name)")
    parser.add_argument(
        "--disagreements", type=Path, default=None, help="Optional path to write disagreeing records as JSONL"
    )
    args = parser.parse_args()

    records_a = load_records(args.a)
    records_b = load_records(args.b)

    a_name = args.a_name or get_model_name(records_a) or args.a.name
    b_name = args.b_name or get_model_name(records_b) or args.b.name

    common_ids = sorted(set(records_a) & set(records_b))
    if not common_ids:
        raise SystemExit("No common ids between the two files.")

    pairs: list[tuple[str, str]] = []
    disagreements = []
    skipped = 0
    for rid in common_ids:
        rec_a, rec_b = records_a[rid], records_b[rid]
        dec_a, dec_b = get_decision(rec_a), get_decision(rec_b)
        if dec_a is None or dec_b is None:
            skipped += 1
            continue
        pairs.append((dec_a, dec_b))
        if dec_a != dec_b:
            disagreements.append(
                {
                    "id": rid,
                    "title": record_title(rec_a),
                    a_name: dec_a,
                    b_name: dec_b,
                    f"{a_name}_reason": (rec_a.get("relevance") or {}).get("reason"),
                    f"{b_name}_reason": (rec_b.get("relevance") or {}).get("reason"),
                }
            )

    n = len(pairs)
    agree = sum(1 for x, y in pairs if x == y)
    accuracy = agree / n if n else float("nan")
    kappa = cohen_kappa(pairs)

    labels = sorted({v for pair in pairs for v in pair})
    confusion: Counter[tuple[str, str]] = Counter(pairs)

    print(f"{a_name}: {len(records_a)} records ({args.a})")
    print(f"{b_name}: {len(records_b)} records ({args.b})")
    print(f"Common ids: {len(common_ids)} | comparable: {n} | skipped (missing decision): {skipped}")
    print()
    print(f"Accuracy (agreement rate): {accuracy:.3f} ({agree}/{n})")
    print(f"Cohen's kappa: {kappa:.3f}")
    print()
    header = f"{'':>14}" + "".join(f"{('b: ' + lbl):>16}" for lbl in labels)
    print(header)
    for a_label in labels:
        row = f"{('a: ' + a_label):>14}"
        for b_label in labels:
            row += f"{confusion[(a_label, b_label)]:>16}"
        print(row)

    if args.disagreements:
        with args.disagreements.open("w", encoding="utf-8") as f:
            for row in disagreements:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(disagreements)} disagreements to {args.disagreements}")


if __name__ == "__main__":
    main()
