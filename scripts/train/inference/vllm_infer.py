"""vLLM batch inference for event extraction.

Input JSONL:  {"source": {"text": ..., "publish_date": ...}, ...}
Output JSONL: same fields + "predictions" (merged/deduplicated events)
                           + "window_predictions" (per-window raw output)

Recovery: re-running on an existing output file skips already-processed articles.
"""
from __future__ import annotations

import gc
import hashlib
import json
import pathlib
import re
import sys
from typing import Iterator

import fire
from json_repair import repair_json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies" / "zhai" / "science.json"
DEFAULT_PROMPT_DIR = REPO_ROOT / "scripts" / "data" / "generation_v3" / "prompts" / "student"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v3.to_sft import (  # noqa: E402
    build_paragraph_windows,
    build_user_message,
    coalesce_short_windows,
    load_ontology_labels,
    render_system_prompt,
    row_labels,
    split_oversized_paragraphs,
    split_paragraphs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _article_key(row: dict) -> str:
    text = (row.get("source") or {}).get("text") or json.dumps(row, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _load_processed_keys(output_path: pathlib.Path) -> set[str]:
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                source = row.get("source") or {}
                text = source.get("text") or json.dumps(row, sort_keys=True)
                keys.add(hashlib.sha256(text.encode()).hexdigest())
            except json.JSONDecodeError:
                pass
    return keys


def _parse_json_output(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    return {"events": [], "_parse_error": text[:300]}


def _resolve_events(window_preds: list[dict]) -> list[dict]:
    """Merge events from all windows, deduplicating by grounding_quote."""
    seen_quotes: set[str] = set()
    merged: list[dict] = []
    for wp in window_preds:
        prediction = wp.get("prediction") or {}
        for event in (prediction.get("events") or []):
            gq = (event.get("grounding_quote") or "").strip()
            if gq and gq in seen_quotes:
                continue
            if gq:
                seen_quotes.add(gq)
            merged.append(event)
    return merged


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _build_windows(
    row: dict,
    system_template: str,
    user_template: str,
    default_labels: list[str],
    tokenizer,
    *,
    max_chars: int,
    max_paras: int,
    overlap: int,
    min_chars: int,
    top_k_candidates: int | None,
) -> list[dict]:
    source = row.get("source") or {}
    text = source.get("text") or ""
    publish_date = source.get("publish_date") or ""

    if not text.strip():
        return []

    labels = row_labels(row, default_labels, top_k_candidates)
    system_prompt = render_system_prompt(system_template, labels)

    paras = split_paragraphs(text) or [(0, len(text))]
    paras = split_oversized_paragraphs(text, paras, max_chars=max_chars)
    if not paras:
        return []

    windows = build_paragraph_windows(paras, max_chars=max_chars, max_paras=max_paras, overlap=overlap)
    windows = coalesce_short_windows(
        paras, windows, min_chars=min_chars, max_chars=max_chars, max_paras=max_paras
    )

    result = []
    for lo, hi in windows:
        ws, we = paras[lo][0], paras[hi - 1][1]
        window_text = text[ws:we]
        user_msg = build_user_message(user_template, publish_date, window_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        result.append({
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "window_start": ws,
            "window_end": we,
            "window_text": window_text,
        })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def vllm_infer(
    model_name_or_path: str,
    input: str,
    output: str,
    adapter_name_or_path: str | None = None,
    prompt_dir: str | None = None,
    ontology: str | None = None,
    top_k_candidates: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    # Windowing
    max_chars: int = 3000,
    min_chars: int = 200,
    max_paras: int = 15,
    overlap_paras: int = 1,
    # Sampling
    temperature: float = 0.0,
    max_new_tokens: int = 4096,
    top_p: float = 0.8,
    top_k: int = 0,
    repetition_penalty: float = 1.05,
    skip_special_tokens: bool = True,
    seed: int | None = None,
    # Engine
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.95,
    tensor_parallel_size: int = 1,
    batch_size: int = 1_000,
    max_num_seqs: int | None = None,
):
    """Batch event extraction inference using vLLM."""
    ontology_path = pathlib.Path(ontology) if ontology else DEFAULT_ONTOLOGY
    prompt_dir_path = pathlib.Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR

    system_template = (prompt_dir_path / "system_prompt.txt").read_text(encoding="utf-8")
    user_template = (prompt_dir_path / "user_prompt.txt").read_text(encoding="utf-8")
    default_labels = load_ontology_labels(ontology_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    engine_kwargs: dict = {
        "model": model_name_or_path,
        "enable_lora": adapter_name_or_path is not None,
        "max_lora_rank": 64,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": True,
        "disable_log_stats": True,
        "language_model_only": True,
    }
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len + max_new_tokens
    if max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = max_num_seqs

    llm = LLM(**engine_kwargs)

    lora_request = None
    if adapter_name_or_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("default", 1, adapter_name_or_path)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )

    # Load input
    input_path = pathlib.Path(input)
    articles: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line {lineno}: {e}", file=sys.stderr)

    # Sharding (stride so load is balanced across shards)
    if num_shards > 1:
        articles = articles[shard_index::num_shards]
        print(f"Shard {shard_index}/{num_shards}: {len(articles)} articles")

    # Recovery
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = _load_processed_keys(output_path)
    if processed_keys:
        print(f"Skipping {len(processed_keys)} already-processed articles (found in {output_path})")

    pending = [row for row in articles if _article_key(row) not in processed_keys]
    print(f"Processing {len(pending)} / {len(articles)} articles")

    n_batches = (len(pending) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(pending) * 2, unit="article")
    _first_prompt_printed = False
    _first_generation_printed = False

    with open(output_path, "a", encoding="utf-8") as out_f:
        for b_idx, batch in enumerate(_batched(pending, batch_size)):

            # --- Window building ---
            pbar.set_description(f"Batch {b_idx + 1}/{n_batches} | Building")
            article_windows: list[list[dict]] = []
            vllm_inputs: list[dict] = []
            prompt_map: list[tuple[int, int]] = []

            for a_idx, row in enumerate(batch):
                src = row.get("source") or {}
                aid = (
                    row.get("id") or row.get("article_id")
                    or src.get("url") or src.get("id")
                    or _article_key(row)[:12]
                )
                try:
                    windows = _build_windows(
                        row, system_template, user_template, default_labels, tokenizer,
                        max_chars=max_chars, max_paras=max_paras, overlap=overlap_paras,
                        min_chars=min_chars, top_k_candidates=top_k_candidates,
                    )
                except Exception as e:
                    pbar.write(f"Window error for {aid}: {e}")
                    windows = []
                article_windows.append(windows)
                for w_idx, w in enumerate(windows):
                    vllm_inputs.append({"prompt_token_ids": w["prompt_token_ids"]})
                    prompt_map.append((a_idx, w_idx))
                pbar.update(1)

            pbar.write(
                f"Batch {b_idx + 1}/{n_batches}: {len(batch)} articles, "
                f"{len(vllm_inputs)} prompts (max_new_tokens={max_new_tokens})"
            )

            if vllm_inputs and not _first_prompt_printed:
                pbar.write("\n" + "=" * 80)
                pbar.write("DEBUG — first rendered prompt:")
                pbar.write("=" * 80)
                first_a_idx, first_w_idx = prompt_map[0]
                pbar.write(article_windows[first_a_idx][first_w_idx]["prompt"])
                pbar.write("=" * 80 + "\n")
                _first_prompt_printed = True

            # --- Inference ---
            pbar.set_description(f"Batch {b_idx + 1}/{n_batches} | Inference")
            article_preds: list[list[dict | None]] = [[None] * len(w) for w in article_windows]

            for a_idx, row in enumerate(batch):
                if not article_windows[a_idx]:
                    out_f.write(json.dumps({**row, "window_predictions": [], "predictions": []}, ensure_ascii=False) + "\n")
                    pbar.update(1)

            if vllm_inputs:
                # Hand all prompts to vLLM at once so its continuous-batching
                # scheduler can maximally parallelize across the GPU.
                results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)

                if results and not _first_generation_printed:
                    decoded = tokenizer.decode(results[0].outputs[0].token_ids, skip_special_tokens=False)
                    pbar.write("\n" + "=" * 80)
                    pbar.write("DEBUG — first generation (with special tokens):")
                    pbar.write("=" * 80)
                    pbar.write(decoded)
                    pbar.write("=" * 80 + "\n")
                    _first_generation_printed = True

                for i, result in enumerate(results):
                    a_idx, w_idx = prompt_map[i]
                    w = article_windows[a_idx][w_idx]
                    article_preds[a_idx][w_idx] = {
                        "window_start": w["window_start"],
                        "window_end": w["window_end"],
                        "window_text": w["window_text"],
                        "prediction": _parse_json_output(result.outputs[0].text),
                    }

                for a_idx, row in enumerate(batch):
                    if article_windows[a_idx]:
                        win_preds = article_preds[a_idx]
                        out_f.write(json.dumps(
                            {**row, "window_predictions": win_preds, "predictions": _resolve_events(win_preds)},
                            ensure_ascii=False,
                        ) + "\n")
                        pbar.update(1)

                out_f.flush()
                gc.collect()

    pbar.close()
    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(vllm_infer)
