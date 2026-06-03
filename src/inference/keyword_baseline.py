from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.inference.text_anchor import TextAnchorResolver

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_LOCATION_RE = re.compile(
    r"\b(?:in|at|near|across|around|inside|outside|within|throughout|from|to|into|toward|towards)\s+"
    r"((?:[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,4}))"
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "with",
}
_SOURCE_PREPOSITIONS = {"from"}
_TARGET_PREPOSITIONS = {"to", "into", "toward", "towards"}
_ANCHOR_RESOLVER = TextAnchorResolver()


def _load_ontology_file(
    path: str, use_description: bool
) -> tuple[
    list[str] | dict[str, str], list[str] | dict[str, str], list[str] | dict[str, str]
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate ontology file '{path}' must contain a JSON object")

    def _extract_labels(field_name: str, alt_field_name: str = ""):
        if alt_field_name and alt_field_name in payload:
            labels_obj = payload[alt_field_name]
        elif field_name in payload:
            labels_obj = payload[field_name]
        else:
            return []

        if isinstance(labels_obj, dict):
            if use_description:
                return {str(k): str(v) for k, v in labels_obj.items()}
            return list(labels_obj.keys())
        if isinstance(labels_obj, list):
            if use_description:
                return {str(k): "" for k in labels_obj}
            return [str(k) for k in labels_obj]
        return []

    events = _extract_labels("events", "event_labels")
    argument_roles = _extract_labels("argument_roles")
    location_types = _extract_labels("location_types")

    if not events:
        raise ValueError(
            f"Candidate ontology file '{path}' must define 'event_labels' or 'events'"
        )

    return events, argument_roles, location_types


def _load_ontology_arguments(
    args: argparse.Namespace,
) -> tuple[
    list[str] | dict[str, str], list[str] | dict[str, str], list[str] | dict[str, str]
]:
    if args.event_labels:
        events = list(args.event_labels)
        if args.description:
            events = {label: "" for label in events}
        argument_roles = {} if args.description else []
        location_types = {} if args.description else []
    elif args.ontology_file:
        events, argument_roles, location_types = _load_ontology_file(
            args.ontology_file, args.description
        )
    else:
        raise ValueError("Either --ontology_file or --event_labels must be provided")

    event_keys = list(events.keys()) if isinstance(events, dict) else events
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("Event labels must not contain duplicates")
    return events, argument_roles, location_types


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError("Either --text, --text_file, or --interactive must be provided")


def _safe_substring(text: str, start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if start < 0 or end <= start or end > len(text):
        return ""
    return text[start:end]


def _normalize_span_prediction(
    document: str,
    span_like: dict[str, Any] | str | None,
    *,
    start: Any = None,
    end: Any = None,
    text: Any = None,
) -> dict[str, Any] | None:
    if isinstance(span_like, dict):
        start = span_like.get("start", start)
        end = span_like.get("end", end)
        text = span_like.get("text", text)
    elif isinstance(span_like, str):
        text = span_like

    span_text = _safe_substring(document, start, end)
    if span_text and (not isinstance(text, str) or not text or text == span_text):
        return {"start": start, "end": end, "text": span_text}

    if not isinstance(text, str) or not text.strip():
        return None

    match = _ANCHOR_RESOLVER.resolve(document, text)
    if match.start is None or match.end is None or match.matched_text is None:
        return None
    return {
        "start": match.start,
        "end": match.end,
        "text": match.matched_text,
    }


def _normalize_event(document: str, event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        return None

    normalized_trigger = _normalize_span_prediction(
        document,
        event.get("trigger"),
        start=event.get("start"),
        end=event.get("end"),
        text=event.get("text"),
    )
    if normalized_trigger is None:
        return None

    normalized_args = []
    if isinstance(event.get("arguments"), list):
        for arg in event["arguments"]:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if not isinstance(role, str) or not role:
                continue
            normalized_span = _normalize_span_prediction(
                document,
                arg.get("span"),
                start=arg.get("start"),
                end=arg.get("end"),
                text=arg.get("text"),
            )
            if normalized_span is None:
                continue
            normalized_arg = {"role": role, "span": normalized_span}
            if "location_type" in arg:
                normalized_arg["location_type"] = arg["location_type"]
            normalized_args.append(normalized_arg)

    normalized_event = {
        "event_type": event_type,
        "trigger": normalized_trigger,
    }
    if "arguments" in event:
        normalized_event["arguments"] = normalized_args
    return normalized_event


def _normalize_prediction(document: str, prediction: dict[str, Any]) -> dict[str, Any]:
    normalized_events: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for event in prediction.get("events", []):
        if not isinstance(event, dict):
            continue
        normalized_event = _normalize_event(document, event)
        if normalized_event is None:
            continue
        key = (
            normalized_event["event_type"],
            normalized_event["trigger"]["start"],
            normalized_event["trigger"]["end"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_events.append(normalized_event)

    normalized_events.sort(
        key=lambda item: (
            item["trigger"]["start"],
            item["trigger"]["end"],
            item["event_type"],
        )
    )
    return {"events": normalized_events}


def _extract_passage_event_candidates(row: dict[str, Any]) -> list[tuple[str, str]] | None:
    passages = row.get("passages")
    if not isinstance(passages, list):
        return None

    candidates: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        document = passage.get("document")
        if not isinstance(document, dict):
            continue
        text = document.get("text")
        if not isinstance(text, str) or not text or text in seen_labels:
            continue
        metadata = document.get("metadata")
        description = ""
        if isinstance(metadata, dict):
            raw_description = metadata.get("description")
            if isinstance(raw_description, str):
                description = raw_description
        seen_labels.add(text)
        candidates.append((text, description))

    return candidates or None


def _extract_document_text(row: dict[str, Any]) -> str:
    question = row.get("question")
    if isinstance(question, str):
        return question

    text = row.get("text")
    if isinstance(text, str):
        return text

    raise ValueError("Input row must contain a string 'question' or 'text' field")


def _interactive_should_stop(text: str) -> bool:
    return text.strip().lower() in {"exit", "quit"}


def _iter_sentences(text: str) -> list[tuple[str, int, int]]:
    sentences: list[tuple[str, int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        start = match.start()
        offset = match.group(0).find(sentence)
        sentence_start = start + offset
        sentence_end = sentence_start + len(sentence)
        sentences.append((sentence, sentence_start, sentence_end))
    if sentences:
        return sentences
    return [(text, 0, len(text))] if text else []


def _label_keywords(label: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(label)
        if len(token) >= 4 and token.lower() not in _STOPWORDS
    ]


def _match_trigger_in_sentence(
    sentence: str,
    sentence_start: int,
    label: str,
) -> dict[str, Any] | None:
    sentence_lower = sentence.lower()
    label_lower = label.lower()
    phrase_index = sentence_lower.find(label_lower)
    if phrase_index >= 0:
        start = sentence_start + phrase_index
        end = start + len(label)
        return {"start": start, "end": end, "text": sentence[phrase_index : phrase_index + len(label)]}

    keywords = _label_keywords(label)
    if not keywords:
        return None

    matched_keywords: list[tuple[int, str]] = []
    for keyword in keywords:
        keyword_index = sentence_lower.find(keyword)
        if keyword_index >= 0:
            matched_keywords.append((keyword_index, keyword))

    if not matched_keywords:
        return None

    if len(keywords) == 1:
        required_matches = 1
    else:
        required_matches = min(2, len(keywords))
    if len(matched_keywords) < required_matches:
        return None

    matched_keywords.sort()
    keyword_index, keyword = matched_keywords[0]
    start = sentence_start + keyword_index
    end = start + len(keyword)
    return {"start": start, "end": end, "text": sentence[keyword_index : keyword_index + len(keyword)]}


def _default_location_type(location_types: list[str] | dict[str, str]) -> str | None:
    if isinstance(location_types, dict):
        if "other" in location_types:
            return "other"
        return next(iter(location_types), None)
    if "other" in location_types:
        return "other"
    return location_types[0] if location_types else None


def _extract_location_arguments(
    sentence: str,
    sentence_start: int,
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
) -> list[dict[str, Any]]:
    if isinstance(argument_roles, dict):
        available_roles = set(argument_roles)
    else:
        available_roles = set(argument_roles)

    default_location_type = _default_location_type(location_types)
    arguments: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for match in _LOCATION_RE.finditer(sentence):
        preposition = match.group(0).split(maxsplit=1)[0].lower()
        location_text = match.group(1).rstrip(".,;:!?")
        location_start = sentence_start + match.start(1)
        location_end = location_start + len(location_text)
        if not location_text:
            continue

        role = "location"
        if preposition in _SOURCE_PREPOSITIONS and "source_location" in available_roles:
            role = "source_location"
        elif preposition in _TARGET_PREPOSITIONS and "target_location" in available_roles:
            role = "target_location"
        elif "location" not in available_roles:
            continue

        key = (role, location_start, location_end)
        if key in seen:
            continue
        seen.add(key)

        argument = {
            "role": role,
            "span": {
                "start": location_start,
                "end": location_end,
                "text": location_text,
            },
        }
        if default_location_type is not None:
            argument["location_type"] = default_location_type
        arguments.append(argument)

    return arguments


def predict_document(
    *,
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    events_only: bool = False,
) -> dict[str, Any]:
    labels = list(event_labels.keys()) if isinstance(event_labels, dict) else list(event_labels)
    raw_events: list[dict[str, Any]] = []

    for sentence, sentence_start, _ in _iter_sentences(document):
        for label in labels:
            trigger = _match_trigger_in_sentence(sentence, sentence_start, label)
            if trigger is None:
                continue
            event = {
                "event_type": label,
                "trigger": trigger,
            }
            if not events_only:
                arguments = _extract_location_arguments(
                    sentence,
                    sentence_start,
                    argument_roles,
                    location_types,
                )
                if arguments:
                    event["arguments"] = arguments
            raw_events.append(event)

    return _normalize_prediction(document, {"events": raw_events})


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    document = _read_text(args)
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)
    return predict_document(
        document=document,
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        events_only=getattr(args, "events_only", False),
    )


def run_inference_file(args: argparse.Namespace) -> None:
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)

    with open(args.input_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for row in rows:
            document = _extract_document_text(row)
            doc_event_labels = event_labels

            passage_candidates = _extract_passage_event_candidates(row)
            if passage_candidates is not None:
                if isinstance(doc_event_labels, dict):
                    doc_event_labels = {
                        label: description for label, description in passage_candidates
                    }
                else:
                    doc_event_labels = [label for label, _ in passage_candidates]

            row["prediction"] = predict_document(
                document=document,
                event_labels=doc_event_labels,
                argument_roles=argument_roles,
                location_types=location_types,
                events_only=getattr(args, "events_only", False),
            )
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_interactive(args: argparse.Namespace) -> None:
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)

    print("Interactive mode. Enter a document and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            document = input("> ")
        except EOFError:
            print()
            break
        if not document.strip():
            continue
        if _interactive_should_stop(document):
            break

        prediction = predict_document(
            document=document,
            event_labels=event_labels,
            argument_roles=argument_roles,
            location_types=location_types,
            events_only=getattr(args, "events_only", False),
        )
        print(json.dumps(prediction, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology_file", type=str, default=None)
    parser.add_argument("--event_labels", nargs="+", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--description",
        action="store_true",
        help="Accepted for CLI compatibility; descriptions are not used by this baseline.",
    )
    parser.add_argument(
        "--events_only",
        action="store_true",
        help="Keep only event predictions and omit location arguments.",
    )

    text_group = parser.add_mutually_exclusive_group(required=False)
    text_group.add_argument("--text", type=str, default=None)
    text_group.add_argument("--text_file", type=str, default=None)
    text_group.add_argument("--input_file", type=str, default=None)

    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args(argv)
    if (
        not args.interactive
        and args.text is None
        and args.text_file is None
        and args.input_file is None
    ):
        parser.error(
            "one of --text, --text_file, --input_file, or --interactive is required"
        )
    if args.input_file is not None and args.output_file is None:
        parser.error("--output_file is required when using --input_file")
    if args.output_file is not None:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.interactive:
        run_interactive(args)
        return
    if args.input_file:
        run_inference_file(args)
        return
    prediction = run_inference(args)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
