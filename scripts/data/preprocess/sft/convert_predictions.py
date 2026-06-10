#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from json_repair import repair_json



DEFAULT_GOLD_INPUT = Path(
    "dataset/risk-factor/run-15052025/sft/dev.v4.sft.events.384.candidates.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate prediction fields in a gold JSONL using LlamaFactory outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the LlamaFactory inference results JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the converted JSONL.",
    )
    parser.add_argument(
        "--gold-input",
        type=Path,
        default=DEFAULT_GOLD_INPUT,
        help="Path to the gold JSONL whose rows should be copied and updated.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_thinking_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _json_start_index(text: str) -> int:
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in (object_start, array_start) if index != -1]
    if not starts:
        raise ValueError("No JSON object or array found")
    return min(starts)


def repair_json_text(text: str) -> str:
    """Repair common model-output JSON issues without fabricating content."""
    text = _strip_markdown_fence(_strip_thinking_block(text))
    text = text[_json_start_index(text) :].strip()

    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    try:
        repaired = repair_json(text)
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        pass

    stack = []
    in_string = False
    escaped = False
    candidates: list[tuple[int, list[str]]] = []

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                break
            stack.pop()
            candidates.append((index + 1, stack.copy()))
            if not stack:
                break

    for end_index, remaining_stack in reversed(candidates):
        candidate = text[:end_index].rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        candidate += "".join(reversed(remaining_stack))
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate

    raise ValueError("Could not repair JSON")


def normalize_prediction(value: dict, row_index: int) -> dict:
    events = value.get("events")
    if events is None:
        value["events"] = []
        return value
    if not isinstance(events, list):
        raise ValueError(f"Expected 'events' to decode to a list in row {row_index}")

    normalized_events = []
    for event in events:
        if not isinstance(event, dict):
            normalized_events.append(event)
            continue

        event = dict(event)
        trigger = event.get("trigger")
        if isinstance(trigger, str):
            event["trigger"] = {"text": trigger}
        normalized_events.append(event)

    value = dict(value)
    value["events"] = normalized_events
    return value


def parse_json_field(row: dict, field_name: str, row_index: int) -> dict:
    try:
        value = json.loads(repair_json_text(row[field_name]))
    except KeyError as exc:
        raise ValueError(f"Missing '{field_name}' field in row {row_index}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in '{field_name}' field for row {row_index}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected '{field_name}' to decode to an object in row {row_index}")
    return normalize_prediction(value, row_index)


def convert_rows(gold_rows: list[dict], llamafactory_rows: list[dict]) -> list[dict]:
    if len(gold_rows) != len(llamafactory_rows):
        raise ValueError(
            "Row count mismatch: "
            f"{len(gold_rows)} gold rows vs {len(llamafactory_rows)} LlamaFactory rows"
        )

    converted = []
    for index, (gold_row, prediction_row) in enumerate(
        zip(gold_rows, llamafactory_rows, strict=True), start=1
    ):
        converted_row = dict(gold_row)
        converted_row["prediction"] = parse_json_field(prediction_row, "predict", index)
        converted.append(converted_row)

    return converted


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    gold_rows = load_jsonl(args.gold_input)
    llamafactory_rows = load_jsonl(args.input)
    converted = convert_rows(gold_rows, llamafactory_rows)
    write_jsonl(args.output, converted)
    print(f"Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()
