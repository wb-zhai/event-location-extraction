import argparse
import json
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm


def normalize_label(label):
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def parse_focus_entity_types(entity_arg):
    if not entity_arg:
        return set()

    return {
        normalize_label(item)
        for item in entity_arg.split(",")
        if isinstance(item, str) and item.strip()
    }


def add_entity(entities_dict, entity_type, mention):
    if not mention:
        return
    if entity_type not in entities_dict:
        entities_dict[entity_type] = []
    if mention not in entities_dict[entity_type]:
        entities_dict[entity_type].append(mention)


def infer_macro_event_type(fine_event_type):
    if not fine_event_type:
        return "event"
    for separator in ("_", ".", "/", ":"):
        if separator in fine_event_type:
            return fine_event_type.split(separator, maxsplit=1)[0]
    return fine_event_type


def build_id_index(data):
    id_to_item = {}

    for event in data.get("events", []):
        event_type = normalize_label(event.get("type", "event"))
        mentions = event.get("mention", [])
        mention_text = ""
        if mentions:
            mention_text = mentions[0].get("trigger_word", "").strip()

        event_id = event.get("id")
        if event_id and mention_text:
            id_to_item[event_id] = {
                "text": mention_text,
                "entity_type": "event",
                "fine_type": event_type,
                "kind": "event",
            }

        for mention in mentions:
            mention_id = mention.get("id")
            trigger_word = mention.get("trigger_word", "").strip()
            if mention_id and trigger_word:
                id_to_item[mention_id] = {
                    "text": trigger_word,
                    "entity_type": "event",
                    "fine_type": event_type,
                    "kind": "event_mention",
                    "parent_event_id": event_id,
                }

    for timex in data.get("TIMEX", []):
        timex_id = timex.get("id")
        timex_text = timex.get("mention", "").strip()
        timex_type = normalize_label(timex.get("type", "timex"))
        if timex_id and timex_text:
            id_to_item[timex_id] = {
                "text": timex_text,
                "entity_type": timex_type,
                "fine_type": timex_type,
                "kind": "timex",
            }

    return id_to_item


def parse_entities(data, include_macro_event_labels=True):
    entities_dict = OrderedDict()

    for event in data.get("events", []):
        event_type = normalize_label(event.get("type", "event"))
        macro_event_type = infer_macro_event_type(event_type)
        for mention in event.get("mention", []):
            trigger_word = mention.get("trigger_word", "").strip()
            if not trigger_word:
                continue
            # add_entity(entities_dict, "event", trigger_word)
            if include_macro_event_labels:
                add_entity(entities_dict, macro_event_type, trigger_word)
            add_entity(entities_dict, event_type, trigger_word)

    for timex in data.get("TIMEX", []):
        timex_text = timex.get("mention", "").strip()
        timex_type = normalize_label(timex.get("type", "timex"))
        if not timex_text:
            continue
        add_entity(entities_dict, "timex", timex_text)
        add_entity(entities_dict, timex_type, timex_text)

    return dict(entities_dict)


def resolve_text_and_type(id_to_item, item_id):
    item = id_to_item.get(item_id)
    if not item:
        return None, None
    return item.get("text"), item.get("entity_type")


def append_relation(
    parsed_relations, seen_relations, relation_type, head_id, tail_id, id_to_item
):
    head_text, head_type = resolve_text_and_type(id_to_item, head_id)
    tail_text, tail_type = resolve_text_and_type(id_to_item, tail_id)
    if not head_text or not tail_text:
        return

    relation_key = (relation_type, head_text, tail_text)
    if relation_key in seen_relations:
        return

    seen_relations.add(relation_key)
    parsed_relations.append(
        {
            relation_type: {
                "head": head_text,
                "tail": tail_text,
                # "head_id": head_id,
                # "tail_id": tail_id,
                # "head_type": head_type,
                # "tail_type": tail_type,
            }
        }
    )


def parse_relations(data, id_to_item):
    parsed_relations = []
    seen_relations = set()

    temporal_relations = data.get("temporal_relations", {})
    for temporal_label, pairs in temporal_relations.items():
        relation_type = f"temporal_{normalize_label(temporal_label)}"
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            append_relation(
                parsed_relations,
                seen_relations,
                relation_type,
                pair[0],
                pair[1],
                id_to_item,
            )

    causal_relations = data.get("causal_relations", {})
    for causal_label, pairs in causal_relations.items():
        relation_type = f"causal_{normalize_label(causal_label)}"
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            append_relation(
                parsed_relations,
                seen_relations,
                relation_type,
                pair[0],
                pair[1],
                id_to_item,
            )

    for pair in data.get("subevent_relations", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        append_relation(
            parsed_relations,
            seen_relations,
            "subevent",
            pair[0],
            pair[1],
            id_to_item,
        )

    return parsed_relations


def filter_entities_and_relations(entities, relations, focus_entity_types):
    if not focus_entity_types:
        return entities, relations

    focus_mentions = set()
    for entity_type in focus_entity_types:
        focus_mentions.update(entities.get(entity_type, []))

    if not focus_mentions:
        return {}, []

    filtered_relations = []
    kept_mentions = set()
    for relation in relations:
        if not relation:
            continue

        relation_info = next(iter(relation.values()))
        head = relation_info.get("head")
        tail = relation_info.get("tail")

        if head in focus_mentions or tail in focus_mentions:
            filtered_relations.append(relation)
            if head:
                kept_mentions.add(head)
            if tail:
                kept_mentions.add(tail)

    filtered_entities = OrderedDict()
    for entity_type, mentions in entities.items():
        kept = [mention for mention in mentions if mention in kept_mentions]
        if kept:
            filtered_entities[entity_type] = kept

    return dict(filtered_entities), filtered_relations


def load_samples(input_path):
    if input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in tqdm(infile, desc=f"Processing {file_path.name}"):
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
        return

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc=f"Processing {input_path.name}"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_ontology_source_files(input_path, ontology_scope):
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))

    if ontology_scope == "input":
        return [input_path]

    # For single-split input (train/valid/test), infer full dataset coverage from sibling JSONL files.
    sibling_jsonl_files = sorted(input_path.parent.glob("*.jsonl"))
    if sibling_jsonl_files:
        return sibling_jsonl_files

    return [input_path]


def build_sample_input_text(data):
    # Use provided sentence text to preserve canonical detokenization.
    input_text = "\n".join(data.get("sentences", []))
    if not input_text.strip():
        token_sentences = data.get("tokens", [])
        input_text = "\n".join(" ".join(sent) for sent in token_sentences)
    return input_text


def build_ontology_from_sources(source_files, include_macro_event_labels):
    ontology_state = init_ontology_state()

    for source_file in source_files:
        for data in load_samples(source_file):
            id_to_item = build_id_index(data)
            entities = parse_entities(
                data,
                include_macro_event_labels=include_macro_event_labels,
            )
            relations = parse_relations(data, id_to_item)
            update_ontology_state(ontology_state, entities, relations)

    return build_ontology_payload(ontology_state)


def init_ontology_state():
    return {
        "entity_types": set(),
        "entity_type_counts": {},
        "relation_types": set(),
        "relation_type_counts": {},
        "timex_types": set(),
        "timex_type_counts": {},
    }


def update_ontology_state(ontology_state, entities, relations):
    for entity_type, mentions in entities.items():
        ontology_state["entity_types"].add(entity_type)
        ontology_state["entity_type_counts"][entity_type] = ontology_state[
            "entity_type_counts"
        ].get(entity_type, 0) + len(mentions)

    for relation in relations:
        for relation_type, relation_info in relation.items():
            ontology_state["relation_types"].add(relation_type)
            ontology_state["relation_type_counts"][relation_type] = (
                ontology_state["relation_type_counts"].get(relation_type, 0) + 1
            )

            head_type = relation_info.get("head_type")
            tail_type = relation_info.get("tail_type")
            for maybe_timex_type in (head_type, tail_type):
                if maybe_timex_type in {"timex", "date", "time", "duration"}:
                    ontology_state["timex_types"].add(maybe_timex_type)

    # Timex labels are tracked separately for convenience.
    for entity_type, mentions in entities.items():
        if entity_type in {"event", "timex"}:
            continue

        if entity_type in {"date", "time", "duration", "timex"}:
            ontology_state["timex_types"].add(entity_type)
            ontology_state["timex_type_counts"][entity_type] = ontology_state[
                "timex_type_counts"
            ].get(entity_type, 0) + len(mentions)


def build_ontology_payload(ontology_state):
    entity_types = sorted(ontology_state["entity_types"])
    entity_type_counts = dict(sorted(ontology_state["entity_type_counts"].items()))

    return {
        "entity_types": entity_types,
        "entity_type_counts": entity_type_counts,
        "relation_types": sorted(ontology_state["relation_types"]),
        "relation_type_counts": dict(
            sorted(ontology_state["relation_type_counts"].items())
        ),
        "timex_types": sorted(ontology_state["timex_types"]),
        "timex_type_counts": dict(sorted(ontology_state["timex_type_counts"].items())),
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Convert MAVEN-ERE dataset to GLINER format"
    )
    arg_parser.add_argument(
        "input_file",
        type=str,
        help="Path to MAVEN-ERE input (a JSONL file or a directory containing train/valid/test JSONL files)",
    )
    arg_parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output GLINER dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "--ontology_output",
        type=str,
        default=None,
        help="Optional ontology output path (JSON). Default: not written",
    )
    arg_parser.add_argument(
        "--ontology_scope",
        type=str,
        choices=["all", "input"],
        default="all",
        help="Scope for ontology stats when --ontology_output is provided: 'all' uses all sibling JSONL files (or all files in input directory), 'input' uses only the provided input.",
    )
    arg_parser.add_argument(
        "--macro_event_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit macro event labels in addition to fine-grained labels. Use --no-macro_event_labels to disable.",
    )
    arg_parser.add_argument(
        "--entities",
        "--entitites",
        dest="entities",
        type=str,
        default=None,
        help="Comma-separated entity types to focus on (e.g., 'event' or 'event,date'). Keeps only relations touching those entities and linked entity annotations.",
    )
    args = arg_parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ontology_output_path = None
    if args.ontology_output:
        ontology_output_path = Path(args.ontology_output)
        ontology_output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_samples = 0
    saved_samples = 0
    focus_entity_types = parse_focus_entity_types(args.entities)
    input_path = Path(args.input_file)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for data in load_samples(input_path):
            id_to_item = build_id_index(data)
            entities = parse_entities(
                data,
                include_macro_event_labels=args.macro_event_labels,
            )
            parsed_relations = parse_relations(data, id_to_item)

            entities, parsed_relations = filter_entities_and_relations(
                entities,
                parsed_relations,
                focus_entity_types,
            )

            if not entities:
                processed_samples += 1
                continue

            input_text = build_sample_input_text(data)

            new_sample = {
                "input": input_text,
                "output": {
                    "entities": entities,
                    "relations": parsed_relations,
                },
                "metadata": {
                    "dataset": "MAVEN-ERE",
                    "id": data.get("id"),
                    "title": data.get("title", ""),
                },
            }

            outfile.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            processed_samples += 1
            saved_samples += 1

    ontology_source_files = []
    if ontology_output_path is not None:
        ontology_source_files = get_ontology_source_files(
            input_path, args.ontology_scope
        )
        ontology_payload = build_ontology_from_sources(
            ontology_source_files,
            include_macro_event_labels=args.macro_event_labels,
        )
        with open(ontology_output_path, "w", encoding="utf-8") as ontology_outfile:
            json.dump(ontology_payload, ontology_outfile, ensure_ascii=False, indent=2)

    print(
        f"Processed {processed_samples} samples. Output written to {args.output_file}"
    )
    print(f"Saved {saved_samples} samples after filtering")
    if ontology_output_path is not None:
        print(
            f"Ontology scope '{args.ontology_scope}' used {len(ontology_source_files)} source file(s)"
        )
    print(f"Ontology written to {ontology_output_path}")
