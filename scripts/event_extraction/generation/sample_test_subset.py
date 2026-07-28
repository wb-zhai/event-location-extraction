"""Random subsample split by whether risk_factors is empty, for pipeline smoke tests."""

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


def reservoir_sample(
    input_path: Path, n_empty: int, n_nonempty: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    empty_reservoir: list[dict] = []
    nonempty_reservoir: list[dict] = []
    empty_seen = 0
    nonempty_seen = 0

    with input_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = loads(line)
            if record.get("risk_factors"):
                nonempty_seen += 1
                reservoir, seen, limit = nonempty_reservoir, nonempty_seen, n_nonempty
            else:
                empty_seen += 1
                reservoir, seen, limit = empty_reservoir, empty_seen, n_empty

            if len(reservoir) < limit:
                reservoir.append(record)
            else:
                j = rng.randint(0, seen - 1)
                if j < limit:
                    reservoir[j] = record

    return empty_reservoir + nonempty_reservoir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a balanced random subset for pipeline smoke tests."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--n-empty", type=int, default=100, help="Rows with empty risk_factors"
    )
    parser.add_argument(
        "--n-nonempty", type=int, default=100, help="Rows with non-empty risk_factors"
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    selected = reservoir_sample(input_path, args.n_empty, args.n_nonempty, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    write_jsonl(output_path, selected, overwrite=True)
    print(f"Wrote {len(selected)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
