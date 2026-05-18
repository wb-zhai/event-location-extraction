import json
import argparse
from pathlib import Path


def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        mapping_data = json.load(f)

    # Create mapping from fine-grained name to cluster
    # Assuming mapping_data is a list of dicts: {"name": "...", "cluster": "..."}
    if isinstance(mapping_data, list):
        return {
            item["name"]: item["cluster"]
            for item in mapping_data
            if "name" in item and "cluster" in item
        }
    elif isinstance(mapping_data, dict):
        # Fallback if it's a dict
        return mapping_data
    return {}


def process_file(input_path, output_path, mapping):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            data = json.loads(line)

            # Map for sft format (usually contains "answer" or direct nested events)
            if "answer" in data:
                try:
                    # SFT answer is often a JSON string or dict
                    if isinstance(data["answer"], str):
                        answer_data = json.loads(data["answer"])
                        if "events" in answer_data:
                            for event in answer_data["events"]:
                                if "type" in event and event["type"] in mapping:
                                    event["type"] = mapping[event["type"]]
                        data["answer"] = json.dumps(answer_data)
                    elif isinstance(data["answer"], dict):
                        if "events" in data["answer"]:
                            for event in data["answer"]["events"]:
                                if "type" in event and event["type"] in mapping:
                                    event["type"] = mapping[event["type"]]
                except json.JSONDecodeError:
                    pass

            # Map for reader format
            if "events" in data:
                for event in data["events"]:
                    if "type" in event and event["type"] in mapping:
                        event["type"] = mapping[event["type"]]

            if "event_labels" in data:
                data["event_labels"] = [
                    mapping.get(label, label) for label in data["event_labels"]
                ]

            fout.write(json.dumps(data) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Map fine-grained labels to clusters in JSONL files."
    )
    parser.add_argument("input_files", nargs="+", help="Path to input JSONL files.")
    parser.add_argument(
        "--mapping",
        default="ontologies/risk-factors/risks.names.clusters.training.json",
        help="Path to the cluster mapping JSON file.",
    )
    parser.add_argument(
        "--suffix", default="-clustered", help="Suffix to add to output files."
    )

    args = parser.parse_args()
    mapping = load_mapping(args.mapping)

    for input_file in args.input_files:
        in_path = Path(input_file)
        out_path = in_path.with_name(f"{in_path.stem}{args.suffix}{in_path.suffix}")
        print(f"Processing {in_path} -> {out_path}")
        process_file(in_path, out_path, mapping)
        print("Done.")


if __name__ == "__main__":
    main()
