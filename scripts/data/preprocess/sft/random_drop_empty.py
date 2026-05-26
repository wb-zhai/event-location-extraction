import argparse
import json
import random

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", type=str)
    args = arg_parser.parse_args()

    with open(args.input_path, "r") as f:
        data = [json.loads(line) for line in f]

    # random seed for reproducibility
    random.seed(42)

    # randomly drop 70% of examples with empty window_labels
    new_data = []
    for example in data:
        if len(example["answer"]["events"]) == 0:
            if random.random() < 0.5:
                new_data.append(example)
        else:
            new_data.append(example)

    print(f"Original number of examples: {len(data)}")
    print(f"New number of examples after dropping empty window_labels: {len(new_data)}")
    print(f"Percentage of examples with empty window_labels before dropping: {sum(1 for example in data if len(example["answer"]["events"]) == 0) / len(data) * 100:.2f}%")
    print(
        f"Percentage of examples with empty window_labels after dropping: {sum(1 for example in new_data if len(example["answer"]["events"]) == 0) / len(new_data) * 100:.2f}%"
    )
    # save the new data to a new file
    output_path = args.input_path.replace(".jsonl", "_dropped.jsonl")
    with open(output_path, "w") as f:
        for example in new_data:
            f.write(json.dumps(example) + "\n")
    print(f"New data saved to {output_path}")
