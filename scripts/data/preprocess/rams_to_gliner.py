import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# output example:
# Entities + Relations
# {"input": "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.", "output": {"entities": {"person": ["Elon Musk"], "organization": ["SpaceX"], "location": ["Hawthorne"], "date": ["2002"]}, "relations": [{"founded": {"head": "Elon Musk", "tail": "SpaceX", "year": "2002"}}, {"located_in": {"head": "SpaceX", "tail": "Hawthorne"}}]}}
# Multi-Task with Descriptions
# {"input": "Dr. Johnson prescribed medication X for condition Y. Patient shows improvement.", "output": {"entities": {"person": ["Dr. Johnson"], "medication": ["medication X"], "condition": ["condition Y"]}, "entity_descriptions": {"person": "Healthcare provider names", "medication": "Prescribed drugs", "condition": "Medical conditions"}, "classifications": [{"task": "patient_status", "labels": ["improving", "stable", "declining"], "true_label": ["improving"], "label_descriptions": {"improving": "Patient condition getting better", "stable": "No change in condition", "declining": "Patient condition worsening"}}], "json_structures": [{"prescription": {"doctor": "Dr. Johnson", "medication": "medication X", "condition": "condition Y"}}], "json_descriptions": {"prescription": {"doctor": "Prescribing physician", "medication": "Prescribed drug name", "condition": "Diagnosed condition"}}}}


def parse_ent_spans(spans, sentence):
    spans_dict = {}

    for span in spans:
        # span format: [start, end, [type, score]]
        # indices are word based in the sentence, and inclusive
        span_start = span[0]
        span_end = span[1]
        if len(span[2]) > 1:
            raise ValueError(f"Unexpected span format: {span}")
        span_type = span[2][0][0]

        # span_type as format like evt011arg02target, evt011arg04place
        # we want to extract the span_type_cleaned as target, place, etc.
        span_type_cleaned = span_type.split("arg", maxsplit=1)[-1][2:]

        if not span_type_cleaned:
            raise ValueError(f"Could not parse span type from: {span_type}")


        spans_dict[span_type_cleaned] = " ".join(sentence[span_start : span_end + 1])
    return spans_dict


def parse_trigger(triggers, sentence):
    if len(triggers) > 1:
        raise ValueError(f"Multiple triggers found: {triggers}")

    trigger = triggers[0]
    trigger_start = trigger[0]
    trigger_end = trigger[1]
    # conflict.attack.selfdirectedbattle
    # we keep everything for now
    trigger_type = trigger[2][0][0]
    # clean a bit
    trigger_type = trigger_type.replace(".n/a", "")

    return trigger_type, " ".join(sentence[trigger_start : trigger_end + 1])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Convert DWIE dataset to GLINER format"
    )
    arg_parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input DWIE dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output GLINER dataset file (JSONL format)",
    )
    args = arg_parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_samples = 0

    with open(args.input_file, "r", encoding="utf-8") as infile, open(
        args.output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in tqdm(infile, desc="Processing samples"):
            data = json.loads(line)

            tokens = data["sentences"]
            # flatten the list of sentences into a single list of tokens
            tokens = [token for sentence in tokens for token in sentence]
            text = " ".join(tokens)
            sample_id = data["doc_key"]

            spans = parse_ent_spans(data["ent_spans"], tokens)
            trigger_type, trigger_text = parse_trigger(data["evt_triggers"], tokens)

            # we now construct the output sample in GLINER format
            entities = spans
            relations = [
                {trigger_type: {"event": trigger_text, "argument": span}}
                for span in spans.values()
            ]

            new_sample = {
                "input": text,
                "output": {
                    "entities": entities,
                    "relations": relations,
                },
                "metadata": {
                    "dataset": "RAMS",
                    "id": sample_id,
                    "sentences": data["sentences"],
                },
            }
            outfile.write(json.dumps(new_sample) + "\n")
            processed_samples += 1

    print(
        f"Processed {processed_samples} samples. Output written to {args.output_file}"
    )
