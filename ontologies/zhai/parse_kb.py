import json
from pathlib import Path


def parse_kb(kb_path: Path):
    event_names = []
    with open(kb_path, "r") as f:
        kb = json.load(f)

        for obj in kb:
            event_names.append(obj["name"])

    return event_names


events = parse_kb(Path(__file__).with_name("raw_papers.json"))
with open(
    Path(__file__).with_name("ontology.papers.json"),
    "w",
) as f:
    json.dump({"events": events}, f, indent=2)
