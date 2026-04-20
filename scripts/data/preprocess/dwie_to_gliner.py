import argparse
import json
from collections import OrderedDict
from pathlib import Path

from tqdm import tqdm

# output example:
# Entities + Relations
# {"input": "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.", "output": {"entities": {"person": ["Elon Musk"], "organization": ["SpaceX"], "location": ["Hawthorne"], "date": ["2002"]}, "relations": [{"founded": {"head": "Elon Musk", "tail": "SpaceX", "year": "2002"}}, {"located_in": {"head": "SpaceX", "tail": "Hawthorne"}}]}}
# Multi-Task with Descriptions
# {"input": "Dr. Johnson prescribed medication X for condition Y. Patient shows improvement.", "output": {"entities": {"person": ["Dr. Johnson"], "medication": ["medication X"], "condition": ["condition Y"]}, "entity_descriptions": {"person": "Healthcare provider names", "medication": "Prescribed drugs", "condition": "Medical conditions"}, "classifications": [{"task": "patient_status", "labels": ["improving", "stable", "declining"], "true_label": ["improving"], "label_descriptions": {"improving": "Patient condition getting better", "stable": "No change in condition", "declining": "Patient condition worsening"}}], "json_structures": [{"prescription": {"doctor": "Dr. Johnson", "medication": "medication X", "condition": "condition Y"}}], "json_descriptions": {"prescription": {"doctor": "Prescribing physician", "medication": "Prescribed drug name", "condition": "Diagnosed condition"}}}}

TYPE_PRIORITY = [
    "person",
    "organization",
    "company",
    "agency",
    "gpe",
    "country",
    "city",
    "location",
    "loc",
    "event",
    "date",
    "time",
    "manager",
    "media",
    "technology",
    "product",
    "role",
]
GENERIC_TYPES = {"entity", "other", "misc", "value", "none"}


def normalize_label(label):
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def infer_entity_type(concept):
    if concept is None:
        return "entity"

    tags = concept.get("tags", [])
    raw_types = []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("type::"):
            raw_types.append(normalize_label(tag.split("::", maxsplit=1)[1]))

    if not raw_types:
        return "entity"

    for preferred in TYPE_PRIORITY:
        if preferred in raw_types:
            return preferred

    for raw_type in raw_types:
        if raw_type not in GENERIC_TYPES:
            return raw_type

    return raw_types[0]


def build_concept_index(concepts):
    return {concept["concept"]: concept for concept in concepts}


def build_mention_index(mentions):
    mention_by_concept = {}
    for mention in mentions:
        concept_id = mention.get("concept")
        mention_text = mention.get("text", "").strip()
        if concept_id is None or not mention_text:
            continue

        if concept_id not in mention_by_concept:
            mention_by_concept[concept_id] = []
        if mention_text not in mention_by_concept[concept_id]:
            mention_by_concept[concept_id].append(mention_text)

    return mention_by_concept


def parse_focus_entity_types(entity_arg):
    if not entity_arg:
        return set()

    return {
        normalize_label(item)
        for item in entity_arg.split(",")
        if isinstance(item, str) and item.strip()
    }


def parse_entities(mentions, concepts):
    entities_dict = OrderedDict()
    concept_index = build_concept_index(concepts)

    for mention in mentions:
        surface_form = mention.get("text", "").strip()
        concept_id = mention.get("concept")
        if not surface_form:
            continue

        concept = concept_index.get(concept_id)
        entity_type = infer_entity_type(concept)

        if entity_type not in entities_dict:
            entities_dict[entity_type] = []
        if surface_form not in entities_dict[entity_type]:
            entities_dict[entity_type].append(surface_form)

    return dict(entities_dict)


def parse_relations(relations, concepts, mention_by_concept, include_x_relations=False):
    concept_index = build_concept_index(concepts)
    parsed_relations = []
    seen_relations = set()

    for relation in relations:
        relation_type = relation.get("p")
        subject_id = relation.get("s")
        object_id = relation.get("o")

        if relation_type is None or subject_id is None or object_id is None:
            continue

        if relation_type.endswith("-x") and not include_x_relations:
            continue

        subject_mentions = mention_by_concept.get(subject_id, [])
        object_mentions = mention_by_concept.get(object_id, [])

        # Keep relations anchored in surface text for extraction training.
        if not subject_mentions or not object_mentions:
            continue

        head_text = subject_mentions[0]
        tail_text = object_mentions[0]

        relation_key = (relation_type, head_text, tail_text)
        if relation_key in seen_relations:
            continue

        seen_relations.add(relation_key)
        parsed_relations.append(
            {
                relation_type: {
                    "head": head_text,
                    "tail": tail_text,
                    "head_concept": concept_index.get(subject_id, {}).get("text"),
                    "tail_concept": concept_index.get(object_id, {}).get("text"),
                }
            }
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

    # Keep only entity mentions that participate in kept relations.
    filtered_entities = OrderedDict()
    for entity_type, mentions in entities.items():
        kept = [mention for mention in mentions if mention in kept_mentions]
        if kept:
            filtered_entities[entity_type] = kept

    return dict(filtered_entities), filtered_relations


def load_samples(input_path):
    if input_path.is_dir():
        for file_path in tqdm(
            sorted(input_path.glob("*.json")), desc="Processing DWIE JSON files"
        ):
            with open(file_path, "r", encoding="utf-8") as infile:
                yield json.load(infile)
        return

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_content = infile.read().strip()

    if not raw_content:
        return

    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            yield parsed
            return
        if isinstance(parsed, list):
            for sample in parsed:
                if isinstance(sample, dict):
                    yield sample
            return
    except json.JSONDecodeError:
        pass

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Processing DWIE JSONL samples"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def init_ontology_state():
    return {
        "entity_types": set(),
        "entity_type_counts": {},
        "relation_types": set(),
        "concept_types": {},
        "relation_type_counts": {},
        "relation_slot_evidence": {},
    }


def update_ontology_state(
    ontology_state,
    concepts,
    entities,
    parsed_relations,
    raw_relations,
):
    concept_index = build_concept_index(concepts)

    for entity_type in entities:
        ontology_state["entity_types"].add(entity_type)
        ontology_state["entity_type_counts"][entity_type] = ontology_state[
            "entity_type_counts"
        ].get(entity_type, 0) + len(entities.get(entity_type, []))

    for concept in concepts:
        concept_name = concept.get("text")
        if not concept_name:
            continue

        inferred_type = infer_entity_type(concept)
        concept_types = ontology_state["concept_types"].setdefault(concept_name, set())
        concept_types.add(inferred_type)

    for relation in parsed_relations:
        for relation_type in relation:
            ontology_state["relation_types"].add(relation_type)
            ontology_state["relation_type_counts"][relation_type] = (
                ontology_state["relation_type_counts"].get(relation_type, 0) + 1
            )

    for relation in raw_relations:
        relation_type = relation.get("p")
        subject_id = relation.get("s")
        object_id = relation.get("o")

        if relation_type is None or subject_id is None or object_id is None:
            continue

        subject_concept = concept_index.get(subject_id, {})
        object_concept = concept_index.get(object_id, {})
        subject_types = [
            normalize_label(tag.split("::", maxsplit=1)[1])
            for tag in subject_concept.get("tags", [])
            if isinstance(tag, str) and tag.startswith("type::")
        ]
        object_types = [
            normalize_label(tag.split("::", maxsplit=1)[1])
            for tag in object_concept.get("tags", [])
            if isinstance(tag, str) and tag.startswith("type::")
        ]

        relation_evidence = ontology_state["relation_slot_evidence"].setdefault(
            relation_type,
            {
                "subject_types": {},
                "object_types": {},
                "object_gpe_levels": {},
            },
        )

        for subject_type in subject_types:
            relation_evidence["subject_types"][subject_type] = (
                relation_evidence["subject_types"].get(subject_type, 0) + 1
            )

        for object_type in object_types:
            relation_evidence["object_types"][object_type] = (
                relation_evidence["object_types"].get(object_type, 0) + 1
            )
            if object_type.startswith("gpe") and object_type[3:].isdigit():
                relation_evidence["object_gpe_levels"][object_type] = (
                    relation_evidence["object_gpe_levels"].get(object_type, 0) + 1
                )


def infer_indexed_relation_semantics(ontology_state):
    indexed_families = {}
    for relation_type in ontology_state["relation_types"]:
        stripped = relation_type[:-2] if relation_type.endswith("-x") else relation_type
        suffix = stripped[-1:] if stripped else ""
        if not suffix.isdigit():
            continue
        family = stripped[:-1]
        variant = stripped
        indexed_families.setdefault(family, set()).add(variant)

    inferred = {}
    for family, variants in sorted(indexed_families.items()):
        if len(variants) < 2:
            continue

        family_payload = {
            "variants": {},
            "inferred_dimension": "hierarchical_location_level",
        }

        for variant in sorted(variants):
            evidence = ontology_state["relation_slot_evidence"].get(
                variant,
                {
                    "subject_types": {},
                    "object_types": {},
                    "object_gpe_levels": {},
                },
            )

            index = variant[-1]
            level_hint = {
                "0": "country_or_national_level",
                "1": "state_province_region_level",
                "2": "city_local_level",
            }.get(index, "other_indexed_level")

            family_payload["variants"][variant] = {
                "index": int(index),
                "likely_level": level_hint,
                "object_gpe_level_counts": dict(
                    sorted(evidence.get("object_gpe_levels", {}).items())
                ),
                "top_object_types": dict(
                    sorted(
                        evidence.get("object_types", {}).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:6]
                ),
            }

        inferred[family] = family_payload

    return inferred


def build_ontology_payload(ontology_state):
    concept_types = {
        concept: sorted(types)
        for concept, types in sorted(ontology_state["concept_types"].items())
    }
    relation_type_counts = dict(sorted(ontology_state["relation_type_counts"].items()))
    entity_type_counts = dict(sorted(ontology_state["entity_type_counts"].items()))
    indexed_relation_semantics = infer_indexed_relation_semantics(ontology_state)

    return {
        "entity_types": sorted(ontology_state["entity_types"]),
        "entity_type_counts": entity_type_counts,
        "relation_types": sorted(ontology_state["relation_types"]),
        "concept_types": concept_types,
        "relation_type_counts": relation_type_counts,
        "indexed_relation_semantics": indexed_relation_semantics,
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Convert DWIE dataset to GLINER format"
    )
    arg_parser.add_argument(
        "input_file",
        type=str,
        help="Path to DWIE input (directory of JSON files, a JSON file, or a JSONL file)",
    )
    arg_parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output GLINER dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "--include_x_relations",
        action="store_true",
        help="Include DWIE relations with '-x' suffix. Default: False",
    )
    arg_parser.add_argument(
        "--ontology_output",
        type=str,
        default=None,
        help="Optional ontology output path (JSON). Default: <output_file>.ontology.json",
    )
    arg_parser.add_argument(
        "--entities",
        "--entitites",
        dest="entities",
        type=str,
        default=None,
        help="Comma-separated entity types to focus on (e.g., 'event' or 'event,organization'). Keeps only relations touching those entities and linked entity annotations.",
    )
    args = arg_parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ontology_output_path = None
    if args.ontology_output:
        ontology_output_path = Path(args.ontology_output)
        ontology_output_path.parent.mkdir(parents=True, exist_ok=True)
    # else:
        # ontology_output_path = output_path.with_name(
        #     f"{output_path.stem}.ontology.json"
        # )


    processed_samples = 0
    saved_samples = 0
    ontology_state = init_ontology_state()
    focus_entity_types = parse_focus_entity_types(args.entities)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for data in load_samples(Path(args.input_file)):
            mentions = data.get("mentions", [])
            concepts = data.get("concepts", [])
            relations = data.get("relations", [])

            entities = parse_entities(mentions, concepts)
            mention_by_concept = build_mention_index(mentions)
            parsed_relations = parse_relations(
                relations,
                concepts,
                mention_by_concept,
                include_x_relations=args.include_x_relations,
            )
            entities, parsed_relations = filter_entities_and_relations(
                entities,
                parsed_relations,
                focus_entity_types,
            )

            # Skip documents without any remaining entities.
            if not entities:
                processed_samples += 1
                continue

            new_sample = {
                "input": data.get("content", ""),
                "output": {
                    "entities": entities,
                    "relations": parsed_relations,
                },
                "metadata": {
                    "dataset": "DWIE",
                    "id": data.get("id"),
                    "tags": data.get("tags", []),
                    "iptc": data.get("iptc", []),
                },
            }

            outfile.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            update_ontology_state(
                ontology_state,
                concepts,
                entities,
                parsed_relations,
                relations,
            )
            processed_samples += 1
            saved_samples += 1

    ontology_payload = build_ontology_payload(ontology_state)
    if ontology_output_path is not None:
        with open(ontology_output_path, "w", encoding="utf-8") as ontology_outfile:
            json.dump(ontology_payload, ontology_outfile, ensure_ascii=False, indent=2)

    print(
        f"Processed {processed_samples} samples. Output written to {args.output_file}"
    )
    print(f"Saved {saved_samples} samples after filtering")
    print(f"Ontology written to {ontology_output_path}")
