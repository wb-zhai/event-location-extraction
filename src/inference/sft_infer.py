from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional for file/batch progress only
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch
from torch.utils.data import DataLoader

from src.inference.text_anchor import TextAnchorResolver
from src.sft_prompt import render_chat

_ANCHOR_RESOLVER = TextAnchorResolver()

try:
    from json_repair import repair_json
except ModuleNotFoundError:  # pragma: no cover - optional dependency in tests
    repair_json = None

try:
    from unsloth import FastLanguageModel
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    FastLanguageModel = None


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


def _repair_first_json_object(text: str) -> dict[str, Any] | None:
    if repair_json is None:
        return None

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed = repair_json(text[index:], return_objects=True)
        except TypeError:
            try:
                parsed = repair_json(text[index:])
            except Exception:
                continue
        except Exception:
            continue

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                loaded = json.loads(parsed)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def _parse_prediction_text(text: str) -> dict[str, Any]:
    parsed = _extract_first_json_object(text)
    if parsed is None:
        parsed = _repair_first_json_object(text)
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


def _normalize_span_prediction(
    document: str,
    span_like: dict[str, Any] | None,
    *,
    start: Any = None,
    end: Any = None,
    text: Any = None,
) -> dict[str, Any] | None:
    left_context = None
    right_context = None
    if isinstance(span_like, dict):
        start = span_like.get("start", start)
        end = span_like.get("end", end)
        text = span_like.get("text", text)
        left_context = span_like.get("left_context")
        right_context = span_like.get("right_context")

    span_text = _safe_substring(document, start, end)
    has_context = bool(
        (isinstance(left_context, str) and left_context)
        or (isinstance(right_context, str) and right_context)
    )
    if span_text and not has_context and (
        not isinstance(text, str) or not text or text == span_text
    ):
        normalized = {
            "start": start,
            "end": end,
            "text": span_text,
        }
        if isinstance(span_like, dict):
            for field_name in ("left_context", "right_context"):
                value = span_like.get(field_name)
                if isinstance(value, str) and value:
                    normalized[field_name] = value
        return normalized

    if span_text and not isinstance(text, str):
        normalized = {
            "start": start,
            "end": end,
            "text": span_text,
        }
        if isinstance(span_like, dict):
            for field_name in ("left_context", "right_context"):
                value = span_like.get(field_name)
                if isinstance(value, str) and value:
                    normalized[field_name] = value
        return normalized

    if not isinstance(text, str) or not text.strip():
        if has_context:
            match = _ANCHOR_RESOLVER.resolve_with_context(
                document,
                None,
                left_context=left_context if isinstance(left_context, str) else None,
                right_context=right_context if isinstance(right_context, str) else None,
                start_hint=start if isinstance(start, int) else None,
                end_hint=end if isinstance(end, int) else None,
            )
            if (
                match.start is not None
                and match.end is not None
                and match.matched_text is not None
            ):
                normalized = {
                    "start": match.start,
                    "end": match.end,
                    "text": match.matched_text,
                }
                if isinstance(span_like, dict):
                    for field_name in ("left_context", "right_context"):
                        value = span_like.get(field_name)
                        if isinstance(value, str) and value:
                            normalized[field_name] = value
                return normalized
        return None

    match = _ANCHOR_RESOLVER.resolve_with_context(
        document,
        text,
        left_context=left_context if isinstance(left_context, str) else None,
        right_context=right_context if isinstance(right_context, str) else None,
        start_hint=start if isinstance(start, int) else None,
        end_hint=end if isinstance(end, int) else None,
    )
    if match.start is None or match.end is None or match.matched_text is None:
        return None

    normalized = {
        "start": match.start,
        "end": match.end,
        "text": match.matched_text,
    }
    if isinstance(span_like, dict):
        for field_name in ("left_context", "right_context"):
            value = span_like.get(field_name)
            if isinstance(value, str) and value:
                normalized[field_name] = value
    return normalized


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
    if "arguments" in event and isinstance(event["arguments"], list):
        for arg in event["arguments"]:
            if not isinstance(arg, dict):
                continue
            arg_role = arg.get("role")
            if not isinstance(arg_role, str) or not arg_role:
                continue

            normalized_span = _normalize_span_prediction(
                document,
                arg.get("span"),
                start=arg.get("start"),
                end=arg.get("end"),
                text=arg.get("text"),
            )
            if normalized_span is not None:
                norm_arg = {
                    "role": arg_role,
                    "span": normalized_span,
                }
                if "location_type" in arg:
                    norm_arg["location_type"] = arg["location_type"]
                normalized_args.append(norm_arg)

    return {
        "event_type": event_type,
        "trigger": normalized_trigger,
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
    max_input_length: int | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
) -> list[str]:

    prompt_texts = [
        render_chat(
            tokenizer,
            doc,
            event_labels,
            argument_roles,
            location_types,
            add_generation_prompt=True,
            events_only=events_only,
            omit_offsets=omit_offsets,
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

    model_inputs = tokenizer(
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
        max_length=max_input_length,
    )
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
    max_input_length: int | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
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
        max_input_length=max_input_length,
        events_only=events_only,
        omit_offsets=omit_offsets,
    )[0]


def load_inference_model(args: argparse.Namespace) -> tuple[Any, Any]:
    _require_unsloth()
    model_name = getattr(args, "model_name", None) or getattr(args, "model_path", None)
    if not model_name:
        raise ValueError("model_name is required.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        gpu_memory_utilization=0.9,
    )
    if getattr(args, "adapter_path", None):
        model.load_adapter(args.adapter_path)
        print("Loaded LoRA adapter from", args.adapter_path)

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
    max_input_length: int | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
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
        max_input_length=max_input_length,
        events_only=events_only,
        omit_offsets=omit_offsets,
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
    max_input_length: int | None = None,
    events_only: bool = False,
    omit_offsets: bool = False,
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
        max_input_length=max_input_length,
        events_only=events_only,
        omit_offsets=omit_offsets,
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
        max_input_length=args.max_seq_length,
        events_only=getattr(args, "events_only", False),
        omit_offsets=getattr(args, "omit_offsets", False),
    )


def _require_unsloth() -> None:
    if FastLanguageModel is None:
        raise ModuleNotFoundError(
            "unsloth is required for model loading in sft_infer.py"
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
            events_only=getattr(args, "events_only", False),
            omit_offsets=getattr(args, "omit_offsets", False),
            debug_prompt=True,
        )
        print(json.dumps(prediction, ensure_ascii=False, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None, help=argparse.SUPPRESS)
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
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--description",
        action="store_true",
        help="Include ontology label descriptions in prompts in addition to label keys.",
    )
    parser.add_argument(
        "--events_only",
        action="store_true",
        help="Keep only event labels in the prompt.",
    )
    parser.add_argument(
        "--omit_offsets",
        action="store_true",
        help="Omit character offsets from the generation.",
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

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
        total_lines = len(lines)
        print("Total documents to process:", total_lines)

    is_first = True
    batch_size = args.batch_size

    

    def collate_fn(batch):
        documents = [d["question"] for d in batch]
        prompt_texts = [
            render_chat(
                tokenizer,
                doc,
                event_labels,
                argument_roles,
                location_types,
                add_generation_prompt=True,
                events_only=getattr(args, "events_only", False),
                omit_offsets=getattr(args, "omit_offsets", False),
            )
            for doc in documents
        ]
        model_inputs = tokenizer(
            text=prompt_texts,
            return_tensors="pt",
            padding=True,
            max_length=args.max_seq_length,
        )
        return batch, prompt_texts, model_inputs

    dataloader = DataLoader(
        lines,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=getattr(args, "num_workers", 0)
    )

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature is not None and args.temperature > 0.0,
    }
    if args.temperature is not None: generation_kwargs["temperature"] = args.temperature
    if args.min_p is not None: generation_kwargs["min_p"] = args.min_p
    if args.top_k is not None: generation_kwargs["top_k"] = args.top_k
    if args.top_p is not None: generation_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None: generation_kwargs["repetition_penalty"] = args.repetition_penalty

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for batch_data, prompt_texts, model_inputs in tqdm(
            dataloader, desc="Running inference (batched)"
        ):
            if is_first and prompt_texts:
                print("====== DEBUG PROMPT ======")
                print(prompt_texts[0])
                print("==========================")

            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(**model_inputs, **generation_kwargs)

            prompt_length = int(model_inputs["input_ids"].shape[-1])
            generated_tokens = outputs[:, prompt_length:]
            prediction_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            for data, doc, pred_text in zip(batch_data, [d["question"] for d in batch_data], prediction_texts):
                if is_first:
                    print("====== DEBUG RAW OUTPUT ======")
                    print(pred_text)
                    print("==============================")
                    is_first = False

                parsed_prediction = _parse_prediction_text(pred_text)
                data["prediction"] = _normalize_prediction(doc, parsed_prediction)
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
