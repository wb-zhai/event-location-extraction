import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", type=str)
    arg_parser.add_argument("output_path", type=str)
    arg_parser.add_argument(
        "--label_key", type=str, required=True, choices=["name", "cluster"]
    )
    args = arg_parser.parse_args()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    labels = set()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for item in data:
        label = item[args.label_key]
        labels.add(label)

    with open(output_path, "w") as f:
        output_dict = {"events": sorted(list(labels))}
        json.dump(output_dict, f, indent=4)
