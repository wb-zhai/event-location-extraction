from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sft_prompt import _build_system_prompt

DEFAULT_DATASET_INFO = Path(
    "dataset/risk-factor/run-15052025/llamafactory/dataset_info.json"
)


def _extract_passage_event_candidates(row: dict[str, Any]) -> dict[str, str]:
    passages = row.get("passages")
    if not isinstance(passages, list):
        raise ValueError("Row is missing a valid 'passages' list for event candidates")

    candidates: dict[str, str] = {}
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        document = passage.get("document")
        if not isinstance(document, dict):
            continue

        label = document.get("text")
        if not isinstance(label, str) or not label or label in candidates:
            continue

        description = ""
        metadata = document.get("metadata")
        if isinstance(metadata, dict):
            raw_description = metadata.get("description")
            if isinstance(raw_description, str):
                description = raw_description

        candidates[label] = description

    if not candidates:
        raise ValueError("Row passages did not contain any usable event candidates")
    return candidates


def _validate_answer(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("Row is missing an 'answer' object")

    events = answer.get("events")
    if not isinstance(events, list):
        raise ValueError("Row answer is missing an 'events' list")

    return {"events": events}


def _select_event_candidates(
    event_candidates: dict[str, str], *, include_descriptions: bool
) -> list[str] | dict[str, str]:
    if include_descriptions:
        return event_candidates
    return list(event_candidates.keys())


def _render_candidates(value: str | list[str] | dict[str, str]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " | ".join(
            f"{label}: {description}" if description else label
            for label, description in value.items()
        )
    return " | ".join(value)


def _build_instruction_and_input(
    document: str, event_candidates: list[str] | dict[str, str]
) -> tuple[str, str]:
    instruction = (
        "Extract all risk-factor events that clearly match the provided event labels."
    )
    input_text = (
        f"\nDocument:\n{document}\n\n"
        f"Select event labels from the following set: {_render_candidates(event_candidates)}\n"
        "Return valid JSON only. Use the exact output schema described above."
    )
    return instruction, input_text


def _convert_row(
    row: dict[str, Any], *, include_descriptions: bool, only_events: bool
) -> dict[str, str]:
    document = row.get("question")
    if not isinstance(document, str) or not document:
        raise ValueError("Row is missing a non-empty 'question' string")

    event_candidates = _extract_passage_event_candidates(row)
    answer_obj = _validate_answer(row)
    selected_event_candidates = _select_event_candidates(
        event_candidates,
        include_descriptions=include_descriptions,
    )
    instruction, input_text = _build_instruction_and_input(
        document, selected_event_candidates
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(answer_obj, ensure_ascii=False),
        "system": _build_system_prompt(
            events_only=only_events,
            omit_offsets=True,
            omit_context=True,
        ),
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    include_descriptions: bool,
    only_events: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                converted = _convert_row(
                    row,
                    include_descriptions=include_descriptions,
                    only_events=only_events,
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to convert row {line_number} from '{input_path}': {exc}"
                ) from exc

            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    return count


def write_dataset_info(
    dataset_info_path: Path, output_path: Path, dataset_name: str
) -> None:
    if dataset_info_path.exists():
        payload = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Existing dataset info at '{dataset_info_path}' must contain a JSON object"
            )
    else:
        payload = {}

    payload[dataset_name] = {
        "file_name": output_path.name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        },
    }

    dataset_info_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the event-only SFT candidates JSONL into a LlamaFactory Alpaca dataset."
        )
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--include-descriptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to include candidate descriptions in the rendered event-label list. "
            "Use --no-include-descriptions for bare labels only."
        ),
    )
    parser.add_argument(
        "--only-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Whether to include only event labels in the rendered candidate list."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_file(
        args.input,
        args.output,
        include_descriptions=args.include_descriptions,
        only_events=args.only_events,
    )
    write_dataset_info(args.dataset_info, args.output, args.dataset_name)
    print(
        f"Converted {rows} rows to '{args.output}' and updated '{args.dataset_info}' "
        f"with dataset name '{args.dataset_name}' "
        f"(include_descriptions={args.include_descriptions}, only_events={args.only_events})."
    )


if __name__ == "__main__":
    main()
