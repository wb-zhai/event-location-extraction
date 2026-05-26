import json


def parse_kb(kb_path: str):
    event_names = []
    with open(kb_path, "r") as f:
        kb = json.load(f)

        for obj in kb:
            event_names.append(obj["name"])

    return event_names


events = parse_kb(
    "/Users/ric/Projects/Job/event-location-extraction/ontologies/zhai/raw.json"
)
with open(
    "/Users/ric/Projects/Job/event-location-extraction/ontologies/zhai/ontology.json",
    "w",
) as f:
    json.dump({"events": events}, f, indent=2)
