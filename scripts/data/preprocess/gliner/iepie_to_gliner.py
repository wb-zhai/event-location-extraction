import argparse
import json
from pathlib import Path

# output example:
# Entities + Relations
# {"input": "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.", "output": {"entities": {"person": ["Elon Musk"], "organization": ["SpaceX"], "location": ["Hawthorne"], "date": ["2002"]}, "relations": [{"founded": {"head": "Elon Musk", "tail": "SpaceX", "year": "2002"}}, {"located_in": {"head": "SpaceX", "tail": "Hawthorne"}}]}}
# Multi-Task with Descriptions
# {"input": "Dr. Johnson prescribed medication X for condition Y. Patient shows improvement.", "output": {"entities": {"person": ["Dr. Johnson"], "medication": ["medication X"], "condition": ["condition Y"]}, "entity_descriptions": {"person": "Healthcare provider names", "medication": "Prescribed drugs", "condition": "Medical conditions"}, "classifications": [{"task": "patient_status", "labels": ["improving", "stable", "declining"], "true_label": ["improving"], "label_descriptions": {"improving": "Patient condition getting better", "stable": "No change in condition", "declining": "Patient condition worsening"}}], "json_structures": [{"prescription": {"doctor": "Dr. Johnson", "medication": "medication X", "condition": "condition Y"}}], "json_descriptions": {"prescription": {"doctor": "Prescribing physician", "medication": "Prescribed drug name", "condition": "Diagnosed condition"}}}}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Convert IEPIE dataset to GLINER format"
    )
    arg_parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input IEPIE dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output GLINER dataset file (JSONL format)",
    )
    args = arg_parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as infile, open(
        args.output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)

            task = data["task"]
            source = data["source"]
            output = data["output"]

            instruction = data["instruction"]["instruction"]
            schema = data["instruction"]["schema"]
            text = data["instruction"]["input"]

            new_sample = {
                "input": text,
                "metadata": {
                    "task": task,
                    "source": source,
                    "instruction": instruction,
                },
            }

            if task == "EEA":
                pass
