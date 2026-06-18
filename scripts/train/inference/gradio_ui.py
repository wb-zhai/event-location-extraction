"""Gradio UI for single-article event location extraction using vLLM."""
from __future__ import annotations

import gc
import json
import pathlib
import sys

import gradio as gr

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MODELS_ROOT = REPO_ROOT / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v3.to_sft import load_ontology_labels  # noqa: E402
from scripts.train.inference.vllm_infer import (  # noqa: E402
    _build_windows,
    _parse_json_output,
    _resolve_events,
)

DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies" / "zhai" / "science.json"
DEFAULT_PROMPT_DIR = REPO_ROOT / "scripts" / "data" / "generation_v3" / "prompts" / "student"

_state: dict = {"llm": None, "tokenizer": None, "loaded_model": None}


def discover_models() -> list[str]:
    paths = sorted(
        str(d) for d in MODELS_ROOT.glob("*/*/merged") if (d / "config.json").exists()
    )
    return paths or [""]


def load_model(model_path: str, max_model_len_val: int, max_new_tokens: int) -> str:
    if not model_path:
        return "No model selected."

    if _state["llm"] is not None:
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del _state["llm"]
        _state["llm"] = None
        _state["tokenizer"] = None
        _state["loaded_model"] = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        from transformers import AutoTokenizer
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        engine_kwargs: dict = {
            "model": model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.95,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "language-model-only": True,
        }
        max_model_len = int(max_model_len_val) if max_model_len_val > 0 else None
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len + int(max_new_tokens)

        _state["llm"] = LLM(**engine_kwargs)
        _state["tokenizer"] = tokenizer
        _state["loaded_model"] = model_path
        return f"Loaded: {pathlib.Path(model_path).name}"
    except Exception as e:
        return f"Error loading model: {e}"


def load_from_json(json_text: str) -> tuple[str, str, str]:
    """Parse a dataset row JSON and return (article_text, publish_date, candidates)."""
    if not json_text.strip():
        return "", "", ""
    try:
        row = json.loads(json_text.strip())
    except json.JSONDecodeError as e:
        return "", "", f"JSON parse error: {e}"

    source = row.get("source") or {}
    article_text = source.get("text") or ""
    publish_date = source.get("publish_date") or ""

    # Extract candidates from row["candidates"] or row["candidate"] (list or comma-string)
    raw_cands = row.get("candidates") or row.get("candidate")
    if isinstance(raw_cands, list):
        candidates_text = "\n".join(str(c) for c in raw_cands)
    elif isinstance(raw_cands, str):
        candidates_text = raw_cands.replace(",", "\n")
    else:
        candidates_text = ""

    return article_text, publish_date, candidates_text


def run_inference(
    article_text: str,
    publish_date: str,
    candidates_text: str,
    top_k_candidates_raw: int,
    max_chars: int,
    min_chars: int,
    max_paras: int,
    overlap_paras: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    if _state["llm"] is None:
        return json.dumps({"error": "No model loaded — click 'Load Model' first."}, indent=2)
    if not article_text.strip():
        return json.dumps({"error": "Article text is empty."}, indent=2)

    from vllm import SamplingParams

    top_k_candidates = int(top_k_candidates_raw) if top_k_candidates_raw > 0 else None

    system_template = (DEFAULT_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
    user_template = (DEFAULT_PROMPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
    default_labels = load_ontology_labels(DEFAULT_ONTOLOGY)

    row: dict = {"source": {"text": article_text, "publish_date": publish_date or ""}}
    candidates = [c.strip() for c in candidates_text.splitlines() if c.strip()]
    if candidates:
        row["candidates"] = candidates

    windows = _build_windows(
        row, system_template, user_template, default_labels, _state["tokenizer"],
        max_chars=int(max_chars),
        max_paras=int(max_paras),
        overlap=int(overlap_paras),
        min_chars=int(min_chars),
        top_k_candidates=top_k_candidates,
    )

    if not windows:
        return json.dumps({"events": [], "note": "No windows built from this article."}, indent=2, ensure_ascii=False)

    sampling_params = SamplingParams(
        temperature=float(temperature),
        max_tokens=int(max_new_tokens),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        skip_special_tokens=True,
    )

    results = _state["llm"].generate([w["prompt"] for w in windows], sampling_params)

    window_preds = [
        {
            "window_start": w["window_start"],
            "window_end": w["window_end"],
            "window_text": w["window_text"],
            "prediction": _parse_json_output(result.outputs[0].text),
        }
        for w, result in zip(windows, results)
    ]

    return json.dumps(
        {
            "events": _resolve_events(window_preds),
            "window_count": len(windows),
            "window_predictions": window_preds,
        },
        indent=2,
        ensure_ascii=False,
    )


def build_ui() -> gr.Blocks:
    models = discover_models()

    with gr.Blocks(title="Event Location Extraction") as demo:
        gr.Markdown("## Event Location Extraction")

        # ── Model loading ────────────────────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(choices=models, value=models[0] if models else None, label="Model", scale=5)
            max_model_len_inp = gr.Number(value=0, label="max_model_len (0 = auto)", precision=0, scale=1)
            load_btn = gr.Button("Load Model", scale=1)
            status_txt = gr.Textbox(value="No model loaded.", label="Status", interactive=False, scale=2)

        # ── JSON row loader ──────────────────────────────────────────────────
        with gr.Accordion("Load from dataset JSON row", open=False):
            json_row_box = gr.Textbox(
                lines=6,
                label="Paste full JSON row",
                placeholder='{"id": "...", "source": {"text": "...", "publish_date": "..."}, "candidates": [...]}',
            )
            load_json_btn = gr.Button("Populate fields from JSON")

        # ── Article input ────────────────────────────────────────────────────
        with gr.Row():
            article_box = gr.Textbox(lines=14, label="News Article", placeholder="Paste article text here...", scale=4)
            with gr.Column(scale=1):
                date_box = gr.Textbox(value="", lines=1, label="Publish Date (optional)", placeholder="2024-01-15")
                candidates_box = gr.Textbox(
                    lines=10,
                    label="Candidates (one per line; leave empty to use full ontology)",
                    placeholder="displaced\nconflict\nfood assistance\n...",
                )

        # ── Inference parameters ─────────────────────────────────────────────
        with gr.Accordion("Inference Parameters", open=False):
            with gr.Row():
                top_k_cand = gr.Slider(0, 100, value=70, step=1, label="top_k_candidates (0 = all)")
                max_new_tokens_sl = gr.Slider(256, 16384, value=12288, step=256, label="max_new_tokens")
            with gr.Row():
                max_chars_sl = gr.Slider(500, 10000, value=3000, step=100, label="max_chars")
                min_chars_sl = gr.Slider(50, 1000, value=200, step=50, label="min_chars")
            with gr.Row():
                max_paras_sl = gr.Slider(1, 50, value=15, step=1, label="max_paras")
                overlap_paras_sl = gr.Slider(0, 5, value=1, step=1, label="overlap_paras")
            with gr.Row():
                temperature_sl = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="temperature")
                top_p_sl = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="top_p")
            with gr.Row():
                top_k_sl = gr.Slider(0, 100, value=0, step=1, label="top_k (0 = disabled)")
                rep_pen_sl = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="repetition_penalty")

        run_btn = gr.Button("Extract Events", variant="primary")
        output_box = gr.Code(language="json", label="Output")

        # ── Wiring ───────────────────────────────────────────────────────────
        load_btn.click(
            fn=load_model,
            inputs=[model_dd, max_model_len_inp, max_new_tokens_sl],
            outputs=[status_txt],
        )
        load_json_btn.click(
            fn=load_from_json,
            inputs=[json_row_box],
            outputs=[article_box, date_box, candidates_box],
        )
        run_btn.click(
            fn=run_inference,
            inputs=[
                article_box, date_box, candidates_box,
                top_k_cand, max_chars_sl, min_chars_sl, max_paras_sl, overlap_paras_sl,
                temperature_sl, max_new_tokens_sl, top_p_sl, top_k_sl, rep_pen_sl,
            ],
            outputs=[output_box],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=False)
