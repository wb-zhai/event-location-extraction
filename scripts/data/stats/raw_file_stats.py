import argparse
from pathlib import Path

try:
    import orjson as json_lib

    def loads(line: bytes):
        return json_lib.loads(line)
except ImportError:
    import json as json_lib

    def loads(line: bytes):
        return json_lib.loads(line)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Print statistics about a raw jsonl dataset file."
    )
    arg_parser.add_argument("input_path", type=Path)
    args = arg_parser.parse_args()

    total_rows = 0
    empty_risk_factors = 0

    with args.input_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            record = loads(line)
            if not record.get("risk_factors"):
                empty_risk_factors += 1

    pct = empty_risk_factors / total_rows * 100 if total_rows else 0.0
    print(f"total rows: {total_rows}")
    print(f"rows with empty risk_factors: {empty_risk_factors} ({pct:.2f}%)")
