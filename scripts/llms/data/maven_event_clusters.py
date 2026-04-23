import json
from pathlib import Path

if __name__ == "__main__":
    # open all the jsonl files in the input directory and read them into a list of dicts
    input_dir = Path("dataset/maven-arg/processed/sft")
    output_dir = Path("dataset/maven-arg/processed/clusters")
    output_dir.mkdir(exist_ok=True)

    for file_path in input_dir.glob("*.jsonl"):
        if file_path in [input_dir / "train.jsonl", input_dir / "valid.jsonl"]:
            print("Reading file:", file_path)
            with open(file_path, "r") as f:
                data = [json.loads(line) for line in f]

    # now iterate over the samples and create a dictionary from event label to
    # {"event_label": {"keywords": Counter(), "arguments": Counter()}}
    clusters = {}
    for sample in data:
        for event in sample["answer"]["events"]:
            event_label = event["event_type"]
            if event_label not in clusters:
                clusters[event_label] = {"keywords": {}, "arguments": {}}
            # for keyword in event["text"]:
            keyword = event["text"].lower()
            if keyword not in clusters[event_label]["keywords"]:
                clusters[event_label]["keywords"][keyword] = 0
            clusters[event_label]["keywords"][keyword] += 1
            for argument in event["arguments"]:
                argument_role = argument["role"]
                if argument_role not in clusters[event_label]["arguments"]:
                    clusters[event_label]["arguments"][argument_role] = 0
                clusters[event_label]["arguments"][argument_role] += 1

    # save the clusters to the output directory
    for event_label, event_data in clusters.items():
        with open(output_dir / f"{event_label}.json", "w") as f:
            json.dump(event_data, f)

    # statistics
    # average number of keywords per event label
    avg_keywords = sum(len(event_data["keywords"]) for event_data in clusters.values()) / len(clusters)
    print("Average number of keywords per event label:", avg_keywords)
    # average number of arguments per event label
    avg_arguments = sum(len(event_data["arguments"]) for event_data in clusters.values()) / len(clusters)
    print("Average number of arguments per event label:", avg_arguments)