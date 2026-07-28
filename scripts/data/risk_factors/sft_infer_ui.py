from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from src.sft_prompt import render_chat  # noqa: E402

DEFAULT_ONTOLOGY = "ontologies/risk-factors/risk.label.description.training.json"


def get_sft_infer_module():
    return importlib.import_module("src.inference.sft_infer")


def get_gradio_module():
    return importlib.import_module("gradio")


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def list_ontology_files() -> list[str]:
    root = REPO_ROOT / "ontologies"
    if not root.exists():
        return [DEFAULT_ONTOLOGY]
    paths = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    ]
    if DEFAULT_ONTOLOGY not in paths:
        paths.insert(0, DEFAULT_ONTOLOGY)
    return paths


@dataclass
class SamplingConfig:
    max_new_tokens: int
    temperature: float | None
    min_p: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    max_seq_length: int
    events_only: bool
    omit_offsets: bool
    description: bool


class AppState:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.model_name: str | None = None
        self.adapter_path: str | None = None
        self.load_in_4bit: bool = True
        self.max_seq_length: int = 8192


def _coerce_optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _build_load_args(
    *,
    model_name: str,
    adapter_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
) -> argparse.Namespace:
    normalized_model_name = str(model_name or "").strip()
    if not normalized_model_name:
        raise ValueError("model_name is required.")

    normalized_adapter_path = str(adapter_path or "").strip() or None
    return argparse.Namespace(
        model_name=normalized_model_name,
        adapter_path=(
            str(resolve_local_path(normalized_adapter_path))
            if normalized_adapter_path
            else None
        ),
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(load_in_4bit),
    )


def _load_ontology_payload(
    ontology_file: str,
    *,
    description: bool,
) -> tuple[
    list[str] | dict[str, str],
    list[str] | dict[str, str],
    list[str] | dict[str, str],
]:
    sft_infer = get_sft_infer_module()
    return sft_infer._load_ontology_file(
        str(resolve_local_path(ontology_file)), description
    )


def _build_sampling_config(
    *,
    max_new_tokens: int,
    temperature: Any,
    min_p: Any,
    top_k: Any,
    top_p: Any,
    repetition_penalty: Any,
    max_seq_length: int,
    events_only: bool,
    omit_offsets: bool,
    description: bool,
) -> SamplingConfig:
    return SamplingConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=_coerce_optional_float(temperature),
        min_p=_coerce_optional_float(min_p),
        top_k=_coerce_optional_int(top_k),
        top_p=_coerce_optional_float(top_p),
        repetition_penalty=_coerce_optional_float(repetition_penalty),
        max_seq_length=int(max_seq_length),
        events_only=bool(events_only),
        omit_offsets=bool(omit_offsets),
        description=bool(description),
    )


def run_single_inference(
    *,
    model: Any,
    tokenizer: Any,
    document: str,
    ontology_file: str,
    sampling: SamplingConfig,
) -> dict[str, Any]:
    if not document.strip():
        raise ValueError("document is required.")

    event_labels, argument_roles, location_types = _load_ontology_payload(
        ontology_file,
        description=sampling.description,
    )

    prompt_text = render_chat(
        tokenizer,
        document,
        event_labels,
        argument_roles,
        location_types,
        add_generation_prompt=True,
        events_only=sampling.events_only,
        omit_offsets=sampling.omit_offsets,
    )
    sft_infer = get_sft_infer_module()
    raw_output = sft_infer._generate_prediction_text(
        model,
        tokenizer,
        document=document,
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=sampling.max_new_tokens,
        temperature=sampling.temperature,
        min_p=sampling.min_p,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        repetition_penalty=sampling.repetition_penalty,
        max_input_length=sampling.max_seq_length,
        events_only=sampling.events_only,
        omit_offsets=sampling.omit_offsets,
    )
    parsed = sft_infer._parse_prediction_text(raw_output)
    normalized = sft_infer._normalize_prediction(document, parsed)

    return {
        "prompt": prompt_text,
        "raw_output": raw_output,
        "parsed_prediction": parsed,
        "normalized_prediction": normalized,
    }


def format_model_status(state: AppState) -> str:
    if state.model is None:
        return "No model loaded."
    parts = [
        f"model `{state.model_name}`",
        f"adapter `{state.adapter_path}`" if state.adapter_path else "no adapter",
        "4-bit" if state.load_in_4bit else "16-bit",
        f"max_seq_length `{state.max_seq_length}`",
    ]
    return "Loaded: " + " · ".join(parts)


def load_model_action(
    state: AppState,
    model_name: str,
    adapter_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
):
    try:
        args = _build_load_args(
            model_name=model_name,
            adapter_path=adapter_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        sft_infer = get_sft_infer_module()
        model, tokenizer = sft_infer.load_inference_model(args)
        state.model = model
        state.tokenizer = tokenizer
        state.model_name = args.model_name
        state.adapter_path = args.adapter_path
        state.load_in_4bit = args.load_in_4bit
        state.max_seq_length = args.max_seq_length
        return state, format_model_status(state)
    except Exception as exc:
        return state, f"Error: {exc}"


def run_inference_action(
    state: AppState,
    document: str,
    ontology_choice: str,
    ontology_path_override: str,
    description: bool,
    events_only: bool,
    omit_offsets: bool,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: Any,
    min_p: Any,
    top_k: Any,
    top_p: Any,
    repetition_penalty: Any,
):
    empty_json = {}
    empty_text = ""

    if state.model is None or state.tokenizer is None:
        return (
            "Error: no model is loaded.",
            empty_text,
            empty_text,
            empty_json,
            empty_json,
        )

    ontology_path = str(ontology_path_override or ontology_choice or "").strip()
    if not ontology_path:
        return (
            "Error: ontology_file is required.",
            empty_text,
            empty_text,
            empty_json,
            empty_json,
        )

    try:
        sampling = _build_sampling_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_seq_length=max_seq_length,
            events_only=events_only,
            omit_offsets=omit_offsets,
            description=description,
        )
        result = run_single_inference(
            model=state.model,
            tokenizer=state.tokenizer,
            document=document,
            ontology_file=ontology_path,
            sampling=sampling,
        )
        return (
            "Inference complete.",
            result["prompt"],
            result["raw_output"],
            result["parsed_prediction"],
            result["normalized_prediction"],
        )
    except Exception as exc:
        return (f"Error: {exc}", empty_text, empty_text, empty_json, empty_json)


def build_app():
    gr = get_gradio_module()
    ontology_files = list_ontology_files()
    default_ontology = (
        DEFAULT_ONTOLOGY if DEFAULT_ONTOLOGY in ontology_files else ontology_files[0]
    )

    with gr.Blocks(title="SFT Inference UI") as demo:
        app_state = gr.State(AppState())

        gr.Markdown("# SFT Inference UI")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Model")
                model_name = gr.Textbox(
                    label="Base model name",
                    placeholder="Hugging Face model name, e.g. unsloth/Qwen3-4B",
                )
                adapter_path = gr.Textbox(
                    label="Adapter path",
                    placeholder="Optional local LoRA adapter path",
                )
                with gr.Row():
                    max_seq_length = gr.Number(
                        label="Max seq length",
                        value=8192,
                        precision=0,
                    )
                    load_in_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
                load_model_button = gr.Button("Load model", variant="primary")
                model_status = gr.Markdown("No model loaded.")

                gr.Markdown("## Prompt settings")
                ontology_choice = gr.Dropdown(
                    choices=ontology_files,
                    value=default_ontology,
                    label="Ontology",
                )
                ontology_path_override = gr.Textbox(
                    label="Ontology path override",
                    value=default_ontology,
                )
                description = gr.Checkbox(
                    label="Include ontology descriptions in prompt",
                    value=True,
                )
                events_only = gr.Checkbox(label="Events only", value=True)
                omit_offsets = gr.Checkbox(
                    label="Omit offsets during generation",
                    value=True,
                )

                gr.Markdown("## Sampling")
                max_new_tokens = gr.Number(
                    label="Max new tokens",
                    value=1024,
                    precision=0,
                )
                temperature = gr.Textbox(label="Temperature", value="")
                min_p = gr.Textbox(label="Min-p", value="")
                top_k = gr.Textbox(label="Top-k", value="")
                top_p = gr.Textbox(label="Top-p", value="")
                repetition_penalty = gr.Textbox(
                    label="Repetition penalty",
                    value="",
                )

            with gr.Column(scale=2):
                status = gr.Markdown("Ready.")
                document = gr.Textbox(
                    label="Document",
                    lines=18,
                    placeholder="Paste the document to run through the SFT model.",
                )
                run_button = gr.Button("Run inference", variant="primary")

                normalized_output = gr.JSON(
                    label="Normalized prediction",
                    value={},
                )
                with gr.Row():
                    parsed_output = gr.JSON(label="Parsed JSON", value={})
                    raw_output = gr.Textbox(
                        label="Raw model output",
                        lines=16,
                    )
                prompt_output = gr.Textbox(
                    label="Rendered prompt",
                    lines=18,
                )

        def sync_ontology_path(selected: str):
            return selected or ""

        ontology_choice.change(
            sync_ontology_path,
            inputs=[ontology_choice],
            outputs=[ontology_path_override],
        )

        load_model_button.click(
            load_model_action,
            inputs=[
                app_state,
                model_name,
                adapter_path,
                max_seq_length,
                load_in_4bit,
            ],
            outputs=[app_state, model_status],
        )

        run_button.click(
            run_inference_action,
            inputs=[
                app_state,
                document,
                ontology_choice,
                ontology_path_override,
                description,
                events_only,
                omit_offsets,
                max_seq_length,
                max_new_tokens,
                temperature,
                min_p,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[
                status,
                prompt_output,
                raw_output,
                parsed_output,
                normalized_output,
            ],
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        demo = build_app()
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency '{missing}'. Install gradio and the inference stack "
            "to use this UI."
        ) from exc

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
