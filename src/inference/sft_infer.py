from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.inference.text_anchor import TextAnchorResolver
from src.sft_prompt import render_chat

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
        elif isinstance(labels_obj, list):
            if use_description:
                # Fallback to empty descriptions if not provided
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

    is_dict = isinstance(events, dict)
    events_keys = list(events.keys()) if is_dict else events
    if len(events_keys) != len(set(events_keys)):
        raise ValueError("Event labels must not contain duplicates")
    return events, argument_roles, location_types


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.text_file is not None:
        return Path(args.text_file).read_text(encoding="utf-8")
    raise ValueError("Either --text, --text_file, or --interactive must be provided")


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

    normalized_args = []
    if "arguments" in event and isinstance(event["arguments"], list):
        for arg in event["arguments"]:
            if not isinstance(arg, dict):
                continue
            arg_role = arg.get("role")
            if not isinstance(arg_role, str) or not arg_role:
                continue

            arg_start = arg.get("start")
            arg_end = arg.get("end")
            arg_text = arg.get("text")
            arg_span = _safe_substring(document, arg_start, arg_end)

            norm_arg = None
            if arg_span and (
                not isinstance(arg_text, str) or not arg_text or arg_text == arg_span
            ):
                norm_arg = {
                    "role": arg_role,
                    "start": arg_start,
                    "end": arg_end,
                    "text": arg_span,
                }
            elif isinstance(arg_text, str) and arg_text.strip():
                match = _ANCHOR_RESOLVER.resolve(document, arg_text)
                if (
                    match.start is not None
                    and match.end is not None
                    and match.matched_text is not None
                ):
                    norm_arg = {
                        "role": arg_role,
                        "start": match.start,
                        "end": match.end,
                        "text": match.matched_text,
                    }

            if norm_arg is not None:
                if "location_type" in arg:
                    norm_arg["location_type"] = arg["location_type"]
                normalized_args.append(norm_arg)

    if span_text and (not isinstance(text, str) or not text or text == span_text):
        return {
            "event_type": event_type,
            "start": start,
            "end": end,
            "text": span_text,
            "arguments": normalized_args,
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
        "arguments": normalized_args,
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


def _generate_prediction_texts(
    model,
    tokenizer,
    *,
    documents: list[str],
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    max_new_tokens: int,
    temperature: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    debug_prompt: bool = False,
) -> list[str]:
    import torch

    prompt_texts = [
        render_chat(
            tokenizer,
            doc,
            event_labels,
            argument_roles,
            location_types,
            add_generation_prompt=True,
        )
        for doc in documents
    ]

    if debug_prompt and prompt_texts:
        print("====== DEBUG PROMPT ======")
        print(prompt_texts[0])
        print("==========================")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(text=prompt_texts, return_tensors="pt", padding=True)
    model_inputs = {
        name: value.to(model.device) for name, value in model_inputs.items()
    }

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature is not None and temperature > 0.0,
    }
    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if min_p is not None:
        generation_kwargs["min_p"] = min_p
    if top_k is not None:
        generation_kwargs["top_k"] = top_k
    if top_p is not None:
        generation_kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = repetition_penalty

    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = int(model_inputs["input_ids"].shape[-1])
    generated_tokens = outputs[:, prompt_length:]

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def _generate_prediction_text(
    model,
    tokenizer,
    *,
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    max_new_tokens: int,
    temperature: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    debug_prompt: bool = False,
) -> str:
    return _generate_prediction_texts(
        model,
        tokenizer,
        documents=[document],
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        debug_prompt=debug_prompt,
    )[0]


def load_inference_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    if getattr(args, "adapter_path", None):
        model.load_adapter(args.adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def predict_document(
    model,
    tokenizer,
    *,
    document: str,
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    max_new_tokens: int,
    temperature: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    debug_prompt: bool = False,
) -> dict[str, Any]:
    return predict_batch_documents(
        model,
        tokenizer,
        documents=[document],
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        debug_prompt=debug_prompt,
    )[0]


def predict_batch_documents(
    model,
    tokenizer,
    *,
    documents: list[str],
    event_labels: list[str] | dict[str, str],
    argument_roles: list[str] | dict[str, str],
    location_types: list[str] | dict[str, str],
    max_new_tokens: int,
    temperature: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    debug_prompt: bool = False,
) -> list[dict[str, Any]]:
    prediction_texts = _generate_prediction_texts(
        model,
        tokenizer,
        documents=documents,
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        debug_prompt=debug_prompt,
    )

    results = []
    for doc, pred_text in zip(documents, prediction_texts):
        if debug_prompt:
            print("====== DEBUG RAW OUTPUT ======")
            print(pred_text)
            print("==============================")
            debug_prompt = False  # print only the first one

        parsed_prediction = _parse_prediction_text(pred_text)
        results.append(_normalize_prediction(doc, parsed_prediction))
    return results


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    document = _read_text(args)
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)
    model, tokenizer = load_inference_model(args)
    return predict_document(
        model,
        tokenizer,
        document=document,
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def _interactive_should_stop(text: str) -> bool:
    return text.strip().lower() in {"exit", "quit"}


def run_interactive(args: argparse.Namespace) -> None:
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)
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
            argument_roles=argument_roles,
            location_types=location_types,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(json.dumps(prediction, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--adapter_path", type=str, default=None, help="Optional path to a LoRA adapter"
    )
    parser.add_argument("--ontology_file", type=str, default=None)
    parser.add_argument("--event_labels", nargs="+", default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--description",
        action="store_true",
        help="Include ontology label descriptions in prompts in addition to label keys.",
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


def run_inference_file(args: argparse.Namespace) -> None:
    event_labels, argument_roles, location_types = _load_ontology_arguments(args)
    model, tokenizer = load_inference_model(args)

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
        total_lines = len(lines)

    is_first = True
    batch_size = args.batch_size

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(
            range(0, total_lines, batch_size), desc="Running inference (batched)"
        ):
            batch_lines = lines[i : i + batch_size]
            batch_data = [json.loads(line) for line in batch_lines]
            batch_documents = [
                d.get("text") or d.get("question") or d.get("document", "")
                for d in batch_data
            ]

            predictions = predict_batch_documents(
                model,
                tokenizer,
                documents=batch_documents,
                event_labels=event_labels,
                argument_roles=argument_roles,
                location_types=location_types,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                min_p=args.min_p,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                debug_prompt=is_first,
            )
            is_first = False

            for data, prediction in zip(batch_data, predictions):
                data["prediction"] = prediction
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")


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
