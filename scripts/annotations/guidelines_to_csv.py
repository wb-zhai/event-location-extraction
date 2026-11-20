"""Write the events-annotation guidelines and allowed event-type labels to a
CSV/TSV for review in Google Sheets or Excel — no Argilla server needed.

Companion to events_to_csv.py: that script exports articles+events for
annotators who aren't using the Argilla UI; this one exports the guidelines
and label reference those annotators would otherwise only see inside Argilla.

See scripts/annotations/README.md for the companion Argilla workflow.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotations.events_argilla import (
    EVENT_TYPE_DESCRIPTIONS,
    load_allowed_event_types,
    load_guidelines,
)
from scripts.annotations.events_to_csv import SHEET_GUIDE

DELIMITERS = {"tab": "\t", "comma": ","}


def write_sheet(output_path: Path, delimiter: str) -> None:
    allowed_event_types = load_allowed_event_types()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["about_this_sheet"])
        writer.writerow([SHEET_GUIDE])
        writer.writerow([])
        writer.writerow(["guidelines"])
        writer.writerow([load_guidelines()])
        writer.writerow([])
        writer.writerow(["event_type", "description"])
        for event_type in allowed_event_types:
            writer.writerow([event_type, EVENT_TYPE_DESCRIPTIONS.get(event_type, "")])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delimiter", choices=list(DELIMITERS), default="tab")
    args = parser.parse_args()

    write_sheet(args.output, DELIMITERS[args.delimiter])
    print(f"[write] guidelines + {len(load_allowed_event_types())} labels -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
