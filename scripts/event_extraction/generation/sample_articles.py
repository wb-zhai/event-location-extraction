"""Random subsample split by relevance, with a configurable not-relevant ratio."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    import orjson as json_lib

    def loads(line: bytes):
        return json_lib.loads(line)

except ImportError:
    import json as json_lib

    def loads(line: bytes):
        return json_lib.loads(line)


from scripts.event_extraction.generation.io_utils import resolve_path, write_jsonl


def is_relevant(record: dict) -> bool | None:
    relevance = record.get("relevance")
    if not isinstance(relevance, dict):
        return None
    return relevance.get("is_relevant")


def reservoir_sample(
    input_path: Path, n_relevant: int, n_not_relevant: int, seed: int
) -> tuple[list[dict], list[dict], int]:
    rng = random.Random(seed)
    relevant_reservoir: list[dict] = []
    not_relevant_reservoir: list[dict] = []
    relevant_seen = 0
    not_relevant_seen = 0
    skipped = 0

    with input_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = loads(line)
            relevant = is_relevant(record)
            if relevant is None:
                skipped += 1
                continue
            if relevant:
                relevant_seen += 1
                reservoir, seen, limit = relevant_reservoir, relevant_seen, n_relevant
            else:
                not_relevant_seen += 1
                reservoir, seen, limit = (
                    not_relevant_reservoir,
                    not_relevant_seen,
                    n_not_relevant,
                )

            if len(reservoir) < limit:
                reservoir.append(record)
            else:
                j = rng.randint(0, seen - 1)
                if j < limit:
                    reservoir[j] = record

    return relevant_reservoir, not_relevant_reservoir, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample articles, capping the share that are not relevant."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("-n", "--n-rows", type=int, default=1000, help="Total rows to sample")
    parser.add_argument(
        "--not-relevant-ratio",
        type=float,
        default=0.15,
        help="Target share of sampled rows where relevance.is_relevant is false",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.not_relevant_ratio <= 1.0:
        print("--not-relevant-ratio must be between 0 and 1", file=sys.stderr)
        return 1

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    n_not_relevant = round(args.n_rows * args.not_relevant_ratio)
    n_relevant = args.n_rows - n_not_relevant

    relevant, not_relevant, skipped = reservoir_sample(
        input_path, n_relevant, n_not_relevant, args.seed
    )

    selected = relevant + not_relevant
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    write_jsonl(output_path, selected, overwrite=True)
    print(
        f"Wrote {len(selected)} records to {output_path} "
        f"({len(not_relevant)} not relevant, {len(relevant)} relevant, {skipped} skipped [no relevance label])"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
