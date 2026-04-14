import argparse
import json
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm


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
        "input_files",
        type=str,
        nargs="+",
        help="Path to the input DWIE dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Optional path to write the computed dataset statistics as JSON",
    )
    args = arg_parser.parse_args()

    processed_samples = 0

    data = []

    input_files = args.input_files
    for input_file in input_files:

        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Processing samples"):
                sample = json.loads(line)

                tokens = sample["sentences"]
                # flatten the list of sentences into a single list of tokens
                tokens = [token for sentence in tokens for token in sentence]
                text = " ".join(tokens)
                sample_id = sample["doc_key"]

                spans = parse_ent_spans(sample["ent_spans"], tokens)
                trigger_type, trigger_text = parse_trigger(
                    sample["evt_triggers"], tokens
                )

                # we now construct the output sample in GLINER format
                entities = spans
                relations = [
                    {trigger_type: {"event": trigger_text, "argument": span}}
                    for span in spans.values()
                ]

                data.append(
                    {
                        "id": sample_id,
                        "text": text,
                        "entities": entities,
                        "relations": relations,
                    }
                )

                processed_samples += 1

    # compute statistics about the dataset
    # we want to compute the distribution of entity types, relation types, and trigger types
    entity_type_counts = OrderedDict()
    relation_type_counts = OrderedDict()
    trigger_type_counts = OrderedDict()
    macro_trigger_type_counts = (
        OrderedDict()
    )  # counts of the main trigger type without subtypes

    for sample in data:
        for entity_type in sample["entities"].keys():
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        for relation in sample["relations"]:
            for relation_type in relation.keys():
                relation_type_counts[relation_type] = (
                    relation_type_counts.get(relation_type, 0) + 1
                )

        for relation in sample["relations"]:
            for relation_type, relation_info in relation.items():
                trigger_type = relation_type
                trigger_type_counts[trigger_type] = (
                    trigger_type_counts.get(trigger_type, 0) + 1
                )
                # also count the macro trigger type without subtypes
                macro_trigger_type = trigger_type.split(".")[0]
                macro_trigger_type_counts[macro_trigger_type] = (
                    macro_trigger_type_counts.get(macro_trigger_type, 0) + 1
                )

    entity_type_counts = OrderedDict(
        sorted(entity_type_counts.items(), key=lambda item: item[1], reverse=True)
    )
    relation_type_counts = OrderedDict(
        sorted(relation_type_counts.items(), key=lambda item: item[1], reverse=True)
    )
    trigger_type_counts = OrderedDict(
        sorted(trigger_type_counts.items(), key=lambda item: item[1], reverse=True)
    )
    macro_trigger_type_counts = OrderedDict(
        sorted(
            macro_trigger_type_counts.items(), key=lambda item: item[1], reverse=True
        )
    )

    print("Dataset statistics:")
    print(f"Total samples: {len(data)}")
    print(f"Unique entity types: {len(entity_type_counts)}")
    print(f"Unique relation types: {len(relation_type_counts)}")
    print(f"Unique trigger types: {len(trigger_type_counts)}")
    print(f"Unique macro trigger types: {len(macro_trigger_type_counts)}")

    print("\nEntity type distribution:")
    for entity_type, count in entity_type_counts.items():
        print(f"{entity_type}: {count}")
    print("\nRelation type distribution:")
    for relation_type, count in relation_type_counts.items():
        print(f"{relation_type}: {count}")
    print("\nTrigger type distribution:")
    for trigger_type, count in trigger_type_counts.items():
        print(f"{trigger_type}: {count}")

    if args.stats_output:
        stats_output_path = Path(args.stats_output)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_samples": len(data),
            "unique_entity_types": len(entity_type_counts),
            "unique_relation_types": len(relation_type_counts),
            "unique_trigger_types": len(trigger_type_counts),
            "unique_macro_trigger_types": len(macro_trigger_type_counts),
            "entity_type_distribution": entity_type_counts,
            "relation_type_distribution": relation_type_counts,
            "trigger_type_distribution": trigger_type_counts,
            "macro_trigger_type_distribution": macro_trigger_type_counts,
        }

        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
