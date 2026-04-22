from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.inference.text_anchor import TextAnchorResolver
from src.sft_prompt import render_chat

_ANCHOR_RESOLVER = TextAnchorResolver()


def _load_event_labels_from_ontology_file(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate ontology file '{path}' must contain a JSON object")
    labels = payload.get("event_labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(
            f"Candidate ontology file '{path}' must define 'event_labels' as a list[str]"
        )
    return labels


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError("Either --text, --text_file, or --interactive must be provided")


def _load_event_labels(args: argparse.Namespace) -> list[str]:
    if args.event_labels:
        labels = list(args.event_labels)
    elif args.ontology_file:
        labels = _load_event_labels_from_ontology_file(args.ontology_file)
    else:
        raise ValueError("Either --ontology_file or --event_labels must be provided")

    if len(labels) != len(set(labels)):
        raise ValueError("Event labels must not contain duplicates")
    return labels


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_prediction_text(text: str) -> dict[str, Any]:
    parsed = _extract_first_json_object(text)
    if parsed is None:
        print(
            "Warning: failed to parse model output as JSON; returning empty events.",
            file=sys.stderr,
        )
        return {"events": []}
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return {"events": []}
    return {"events": events}


def _safe_substring(text: str, start: Any, end: Any) -> str:
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    if start < 0 or end <= start or end > len(text):
        return ""
    return text[start:end]


def _normalize_event(document: str, event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        return None

    start = event.get("start")
    end = event.get("end")
    text = event.get("text")
    span_text = _safe_substring(document, start, end)
    if span_text and (not isinstance(text, str) or not text or text == span_text):
        return {
            "event_type": event_type,
            "start": start,
            "end": end,
            "text": span_text,
        }

    if not isinstance(text, str) or not text.strip():
        return None

    match = _ANCHOR_RESOLVER.resolve(document, text)
    if match.start is None or match.end is None or match.matched_text is None:
        return None
    return {
        "event_type": event_type,
        "start": match.start,
        "end": match.end,
        "text": match.matched_text,
    }


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
            normalized_event["start"],
            normalized_event["end"],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_events.append(normalized_event)

    normalized_events.sort(
        key=lambda item: (item["start"], item["end"], item["event_type"])
    )
    return {"events": normalized_events}


def _generate_prediction_text(
    model,
    tokenizer,
    *,
    document: str,
    event_labels: list[str],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
) -> str:
    import torch

    prompt_text = render_chat(
        tokenizer,
        document,
        event_labels,
        add_generation_prompt=True,
    )
    # Some chat model stacks expose a processor-like interface where the first
    # positional argument is treated as `images`. Pass text explicitly so raw
    # chat templates are never interpreted as image paths or URLs.
    model_inputs = tokenizer(text=prompt_text, return_tensors="pt")
    model_inputs = {name: value.to(model.device) for name, value in model_inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "min_p": min_p,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = int(model_inputs["input_ids"].shape[-1])
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_inference_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_document(
    model,
    tokenizer,
    *,
    document: str,
    event_labels: list[str],
    max_new_tokens: int,
    temperature: float,
    min_p: float,
) -> dict[str, Any]:
    prediction_text = _generate_prediction_text(
        model,
        tokenizer,
        document=document,
        event_labels=event_labels,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
    )
    print("Raw model output:", prediction_text)
    parsed_prediction = _parse_prediction_text(prediction_text)
    return _normalize_prediction(document, parsed_prediction)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    document = _read_text(args)
    event_labels = _load_event_labels(args)
    model, tokenizer = load_inference_model(args)
    return predict_document(
        model,
        tokenizer,
        document=document,
        event_labels=event_labels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
    )


def _interactive_should_stop(text: str) -> bool:
    return text.strip().lower() in {"exit", "quit"}


def run_interactive(args: argparse.Namespace) -> None:
    event_labels = _load_event_labels(args)
    model, tokenizer = load_inference_model(args)

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
            model,
            tokenizer,
            document=document,
            event_labels=event_labels,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
        )
        print(json.dumps(prediction, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ontology_file", type=str, default=None)
    parser.add_argument("--event_labels", nargs="+", default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--interactive", action="store_true")

    text_group = parser.add_mutually_exclusive_group(required=False)
    text_group.add_argument("--text", type=str, default=None)
    text_group.add_argument("--text_file", type=str, default=None)
    args = parser.parse_args(argv)
    if not args.interactive and args.text is None and args.text_file is None:
        parser.error("one of --text, --text_file, or --interactive is required")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.interactive:
        run_interactive(args)
        return
    prediction = run_inference(args)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
