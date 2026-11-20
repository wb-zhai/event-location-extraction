"""vLLM batch inference for event extraction.

Input JSONL:  {"source": {"text": ..., "publish_date": ...}, ...}
Output JSONL: same fields + "predictions" (merged/deduplicated events)
                           + "window_predictions" (per-window raw output)

With --gcs_input the input is instead a *manifest* (see
vertexai/inference/event-extraction/build_manifest.py): one
{"id", "gcs_path", "publish_date", "language"} per line, with the article text streamed
from GCS one object at a time. With --lean_output each result line is just
{"id", "predictions"} -- all the geocoding/ingest steps downstream need.

Both input modes are streamed, so memory is O(batch_size), not O(corpus).

Recovery: re-running on an existing output file skips already-processed articles (by "id"
when present, else by a hash of the article text).
"""

from __future__ import annotations

import copy
import gc
import hashlib
import itertools
import json
import logging
import pathlib
import re
import sys
from typing import Iterable, Iterator

import torch
import fire
from json_repair import repair_json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies" / "zhai" / "bona.v4.json"
DEFAULT_PROMPT_DIR = (
    REPO_ROOT / "scripts" / "event_extraction" / "generation" / "prompts" / "student_sft"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.event_extraction.generation.to_sft import (  # noqa: E402
    build_paragraph_windows,
    build_user_message,
    coalesce_short_windows,
    load_ontology_descriptions,
    load_ontology_labels,
    render_system_prompt,
    row_labels,
    row_language,
    split_oversized_paragraphs,
    split_paragraphs,
)

# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

_BASE_ANNOTATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "grounding_quote": {"type": "string"},
                    "event_location": {"type": "string"},
                    "event_location_admin_level": {
                        "type": "string",
                        "enum": ["country", "state", "county", "city", "district", "not_stated"],
                    },
                    "event_time": {"type": "string"},
                    "time_status": {
                        "type": "string",
                        "enum": ["past", "ongoing", "forecast", "not_stated"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "extreme", "not_stated"],
                    },
                },
                "required": [
                    "event_type",
                    "grounding_quote",
                    "event_location",
                    "event_location_admin_level",
                    "event_time",
                    "time_status",
                    "severity",
                ],
            },
        },
    },
    "required": ["events"],
}

_schema_cache: dict = {}


def _annotation_schema(labels: list[str] | None) -> dict:
    key = tuple(labels) if labels else None
    if key not in _schema_cache:
        schema = copy.deepcopy(_BASE_ANNOTATION_SCHEMA)
        if labels:
            schema["properties"]["events"]["items"]["properties"]["event_type"][
                "enum"
            ] = list(labels)
        _schema_cache[key] = schema
    return _schema_cache[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _article_key(row: dict) -> str:
    """Recovery key for a row: its id when it has one, else a hash of the article text.

    Preferring "id" is what lets --lean_output resume (its rows carry no text), and it
    also spares a restart from rehashing every article body in the output. Older
    full-echo outputs carry "id" too, so they keep resuming correctly.
    """
    article_id = row.get("id")
    if article_id:
        return str(article_id)
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
                keys.add(_article_key(json.loads(line)))
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
        for event in prediction.get("events") or []:
            if not isinstance(event, dict):
                continue
            gq = (event.get("grounding_quote") or "").strip()
            if gq and gq in seen_quotes:
                continue
            if gq:
                seen_quotes.add(gq)
            merged.append(event)
    return merged


def _batched(items: Iterable, size: int) -> Iterator[list]:
    """Group an iterable into lists of at most `size`, without materializing it."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _read_jsonl(path: pathlib.Path, shard_index: int, num_shards: int) -> Iterator[dict]:
    """Stream a local JSONL, keeping line k when k % num_shards == shard_index.

    Streaming (rather than json.loads-ing the whole file into a list up front) keeps
    memory O(batch) so the same code path works on a 33M-article corpus.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if num_shards > 1 and lineno % num_shards != shard_index:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {lineno + 1}: {e}", file=sys.stderr)


def _output_row(row: dict, win_preds: list, predictions: list, lean: bool) -> dict:
    """Build the output record. Lean mode keeps only what the geocoding + ingest steps
    read (scripts/geocoding/add_geotaxonomy.py, to_csv_ingest.py): the article id and the
    merged predictions. At corpus scale that is ~1 KB/article instead of ~20 KB."""
    if lean:
        return {"id": row.get("id"), "predictions": predictions}
    return {**row, "window_predictions": win_preds, "predictions": predictions}


def _apply_chat_template(tokenizer, messages: list[dict]) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(token_ids, (list, tuple)):
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        elif hasattr(token_ids, "__getitem__") and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        return token_ids.tolist()
    return [int(t) for t in token_ids]


_system_prompt_cache: dict = {}


def _render_system_prompt_cached(
    system_template: str, labels: list[str], descriptions: dict[str, str]
) -> str:
    """Memoized render_system_prompt.

    Without a `candidates` field every row gets the same default label list, so this
    otherwise re-renders an identical ~1.5k-token system prompt once per article.
    Keyed on the template text plus the labels; descriptions are fixed per run.
    """
    key = (system_template, tuple(labels))
    if key not in _system_prompt_cache:
        _system_prompt_cache[key] = render_system_prompt(
            system_template, labels, descriptions
        )
    return _system_prompt_cache[key]


def _build_windows(
    row: dict,
    templates: dict[str, tuple[str, str]],
    default_labels: list[str],
    descriptions: dict[str, str],
    tokenizer,
    *,
    max_chars: int,
    max_paras: int,
    overlap: int,
    min_chars: int,
    top_k_candidates: int | None,
    render_prompt: bool = True,
) -> list[dict]:
    source = row.get("source") or {}
    text = source.get("text") or ""
    publish_date = source.get("publish_date") or ""

    if not text.strip():
        return []

    # In per-window retrieval mode the caller re-renders the prompt per window
    # after retrieval, so skip the (throwaway) prompt build here.
    system_prompt = None
    user_template = None
    if render_prompt:
        lang = row_language(row)
        system_template, user_template = templates.get(lang, templates["eng"])
        labels = row_labels(row, default_labels, top_k_candidates)
        system_prompt = _render_system_prompt_cached(system_template, labels, descriptions)

    paras = split_paragraphs(text) or [(0, len(text))]
    paras = split_oversized_paragraphs(text, paras, max_chars=max_chars)
    if not paras:
        return []

    windows = build_paragraph_windows(
        paras, max_chars=max_chars, max_paras=max_paras, overlap=overlap
    )
    windows = coalesce_short_windows(
        paras, windows, min_chars=min_chars, max_chars=max_chars, max_paras=max_paras
    )

    result = []
    for lo, hi in windows:
        ws, we = paras[lo][0], paras[hi - 1][1]
        window_text = text[ws:we]
        win: dict = {
            "window_start": ws,
            "window_end": we,
            "window_text": window_text,
        }
        if render_prompt:
            user_msg = build_user_message(user_template, publish_date, window_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            prompt_token_ids = _apply_chat_template(tokenizer, messages)
            win["prompt_token_ids"] = prompt_token_ids
            win["prompt"] = tokenizer.decode(
                prompt_token_ids, skip_special_tokens=False
            )
            win["labels"] = labels
        result.append(win)

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
    ontology_descriptions: str = "none",
    top_k_candidates: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    # Input / output form
    gcs_input: bool = False,
    gcs_read_concurrency: int = 64,
    lean_output: bool = False,
    limit: int | None = None,
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
    quantization: str | None = None,
    # Retriever (optional)
    retriever_model_name: str | None = None,
    retriever_index: str | None = None,
    index_device: str = "cpu",
    retriever_gpu_memory_utilization: float = 0.1,
    retriever_max_model_len: int | None = None,
    retriever_query_mode: str = "full_doc",
    # Structured output
    use_guided_decoding: bool = False,
    guided_decoding_backend: str = "xgrammar",
):
    """Batch event extraction inference using vLLM."""
    if ontology_descriptions not in ("none", "all"):
        raise ValueError("ontology_descriptions must be 'none' or 'all'")

    # Validate retriever config up front so misconfiguration fails fast, before
    # the (slow) main model load below.
    if retriever_model_name is not None:
        if retriever_index is None:
            raise ValueError(
                "retriever_index is required when retriever_model_name is set"
            )
        if gpu_memory_utilization + retriever_gpu_memory_utilization > 1.0:
            raise ValueError(
                f"gpu_memory_utilization ({gpu_memory_utilization}) + "
                f"retriever_gpu_memory_utilization ({retriever_gpu_memory_utilization}) "
                f"exceeds 1.0"
            )

    ontology_path = pathlib.Path(ontology) if ontology else DEFAULT_ONTOLOGY
    prompt_dir_path = pathlib.Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR

    templates: dict[str, tuple[str, str]] = {
        "eng": (
            (prompt_dir_path / "system_prompt.txt").read_text(encoding="utf-8"),
            (prompt_dir_path / "user_prompt.txt").read_text(encoding="utf-8"),
        ),
    }
    system_prompt_fr = prompt_dir_path / "system_prompt.fr.txt"
    user_prompt_fr = prompt_dir_path / "user_prompt.fr.txt"
    if system_prompt_fr.exists() and user_prompt_fr.exists():
        templates["fra"] = (
            system_prompt_fr.read_text(encoding="utf-8"),
            user_prompt_fr.read_text(encoding="utf-8"),
        )

    default_labels = load_ontology_labels(ontology_path)

    descriptions: dict[str, str] = {}
    if ontology_descriptions == "all":
        descriptions = load_ontology_descriptions(ontology_path)
        if not descriptions:
            raise ValueError(
                f"ontology_descriptions='all': {ontology_path} carries no descriptions "
                '(expected {"events": {label: description}})'
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

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
    if quantization is not None:
        engine_kwargs["quantization"] = quantization
    if use_guided_decoding:
        logger.info(
            f"Using guided decoding with backend '{guided_decoding_backend}' "
            f"and structured output schema: {_annotation_schema(default_labels)}"
        )
        engine_kwargs["structured_outputs_config"] = {
            "backend": guided_decoding_backend
        }

    llm = LLM(**engine_kwargs)

    retriever_llm = None
    retriever_indexer = None
    if retriever_model_name is not None:
        from src.index.inmemory import InMemoryIndexer

        retriever_indexer = InMemoryIndexer.from_pretrained(
            retriever_index, device=index_device
        )
        _retriever_kwargs: dict = {
            "model": retriever_model_name,
            "runner": "pooling",
            "trust_remote_code": True,
            "gpu_memory_utilization": retriever_gpu_memory_utilization,
        }
        if retriever_max_model_len is not None:
            _retriever_kwargs["max_model_len"] = retriever_max_model_len
        retriever_llm = LLM(**_retriever_kwargs)

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

    # Recovery
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = _load_processed_keys(output_path)
    if processed_keys:
        print(
            f"Skipping {len(processed_keys)} already-processed articles (found in {output_path})"
        )

    # Input. Both modes are lazy generators, so memory stays O(batch_size) and the corpus
    # is never held in RAM. Sharding strides over lines so load is balanced; in Cloud
    # Batch num_shards is 1 because each task is handed its own manifest file.
    input_path = pathlib.Path(input)
    if num_shards > 1:
        print(f"Shard {shard_index}/{num_shards} (striding input lines)")

    if gcs_input:
        from scripts.event_extraction.inference.gcs_stream import (
            ArticleFetcher,
            read_manifest,
            stream_fetched,
        )

        fetcher = ArticleFetcher(pool_size=gcs_read_concurrency)
        manifest_rows = read_manifest(input_path, shard_index, num_shards, limit)
        # Skip already-done ids before fetching, so a resumed run doesn't re-download
        # every article it is about to discard.
        manifest_rows = (
            r for r in manifest_rows if str(r.get("id") or "") not in processed_keys
        )
        articles: Iterable[dict] = stream_fetched(
            manifest_rows,
            fetcher,
            gcs_read_concurrency,
            max_inflight=max(batch_size * 2, gcs_read_concurrency * 2),
        )
        pending = articles
    else:
        articles = _read_jsonl(input_path, shard_index, num_shards)
        if limit is not None:
            articles = itertools.islice(articles, limit)
        pending = (row for row in articles if _article_key(row) not in processed_keys)

    pbar = tqdm(unit="article")
    _first_prompt_printed = False
    _first_generation_printed = False

    # top_k_candidates doubles as both the prompt candidate filter and the number
    # of passages pulled from the index. When unset, retrieve the whole index so
    # nothing is dropped (row_labels then keeps all of them too).
    if top_k_candidates is not None:
        _effective_retriever_k = top_k_candidates
    elif retriever_indexer is not None and retriever_indexer.embeddings is not None:
        _effective_retriever_k = retriever_indexer.embeddings.shape[0]
    else:
        _effective_retriever_k = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for b_idx, batch in enumerate(_batched(pending, batch_size)):

            # --- Full-doc retrieval (before windowing) ---
            if retriever_llm is not None and retriever_query_mode == "full_doc":
                pbar.set_description(f"Batch {b_idx + 1} | Retrieving")
                texts = [(row.get("source") or {}).get("text") or "" for row in batch]
                pooling_outputs = retriever_llm.encode(texts, pooling_task="embed")
                query_embeddings = torch.stack(
                    [
                        out.outputs.data.to(torch.float32).cpu()
                        for out in pooling_outputs
                    ]
                )
                retrieval_results = retriever_indexer.search(
                    query_embeddings, _effective_retriever_k
                )
                for row, passages in zip(batch, retrieval_results):
                    row["candidates"] = [p["document"]["text"] for p in passages]

            # --- Window building ---
            # Skip prompt rendering when per-window retrieval will re-render each
            # window's prompt after this pass (avoids a wasted render per window).
            _render_prompt = not (
                retriever_llm is not None and retriever_query_mode == "per_window"
            )
            pbar.set_description(f"Batch {b_idx + 1} | Building")
            article_windows: list[list[dict]] = []

            for a_idx, row in enumerate(batch):
                src = row.get("source") or {}
                aid = (
                    row.get("id")
                    or row.get("article_id")
                    or src.get("url")
                    or src.get("id")
                    or _article_key(row)[:12]
                )
                try:
                    windows = _build_windows(
                        row,
                        templates,
                        default_labels,
                        descriptions,
                        tokenizer,
                        max_chars=max_chars,
                        max_paras=max_paras,
                        overlap=overlap_paras,
                        min_chars=min_chars,
                        top_k_candidates=top_k_candidates,
                        render_prompt=_render_prompt,
                    )
                except Exception as e:
                    pbar.write(f"Window error for {aid}: {e}")
                    windows = []
                article_windows.append(windows)

            # --- Per-window retrieval (after windowing) ---
            if retriever_llm is not None and retriever_query_mode == "per_window":
                pbar.set_description(
                    f"Batch {b_idx + 1} | Retrieving per window"
                )
                _win_texts: list[str] = []
                _win_index: list[tuple[int, int]] = []
                for a_idx, windows in enumerate(article_windows):
                    for w_idx, w in enumerate(windows):
                        _win_texts.append(w["window_text"])
                        _win_index.append((a_idx, w_idx))
                if _win_texts:
                    pooling_outputs = retriever_llm.encode(
                        _win_texts, pooling_task="embed"
                    )
                    window_embeddings = torch.stack(
                        [
                            out.outputs.data.to(torch.float32).cpu()
                            for out in pooling_outputs
                        ]
                    )
                    window_retrieval = retriever_indexer.search(
                        window_embeddings, _effective_retriever_k
                    )
                    for (a_idx, w_idx), passages in zip(_win_index, window_retrieval):
                        w = article_windows[a_idx][w_idx]
                        row = batch[a_idx]
                        src = row.get("source") or {}
                        publish_date = src.get("publish_date") or ""
                        row_with_cands = {
                            **row,
                            "candidates": [p["document"]["text"] for p in passages],
                        }
                        labels = row_labels(
                            row_with_cands, default_labels, top_k_candidates
                        )
                        lang = row_language(row)
                        sys_tmpl, usr_tmpl = templates.get(lang, templates["eng"])
                        system_prompt = render_system_prompt(sys_tmpl, labels, descriptions)
                        w["labels"] = labels
                        user_msg = build_user_message(
                            usr_tmpl, publish_date, w["window_text"]
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ]
                        prompt_token_ids = _apply_chat_template(tokenizer, messages)
                        w["prompt_token_ids"] = prompt_token_ids
                        w["prompt"] = tokenizer.decode(
                            prompt_token_ids, skip_special_tokens=False
                        )

            # --- Build vLLM inputs ---
            vllm_inputs: list[dict] = []
            per_window_params: list[SamplingParams] = []
            prompt_map: list[tuple[int, int]] = []
            for a_idx, windows in enumerate(article_windows):
                for w_idx, w in enumerate(windows):
                    vllm_inputs.append({"prompt_token_ids": w["prompt_token_ids"]})
                    prompt_map.append((a_idx, w_idx))
                    if use_guided_decoding:
                        guided = StructuredOutputsParams(
                            json=_annotation_schema(w.get("labels")),
                        )
                        per_window_params.append(
                            SamplingParams(
                                temperature=temperature,
                                max_tokens=max_new_tokens,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                                skip_special_tokens=skip_special_tokens,
                                seed=seed,
                                structured_outputs=guided,
                            )
                        )

            pbar.write(
                f"Batch {b_idx + 1}: {len(batch)} articles, "
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
            pbar.set_description(f"Batch {b_idx + 1} | Inference")
            article_preds: list[list[dict | None]] = [
                [None] * len(w) for w in article_windows
            ]

            for a_idx, row in enumerate(batch):
                if not article_windows[a_idx]:
                    out_f.write(
                        json.dumps(
                            _output_row(row, [], [], lean_output),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    pbar.update(1)

            if vllm_inputs:
                # Hand all prompts to vLLM at once so its continuous-batching
                # scheduler can maximally parallelize across the GPU.
                _params = per_window_params if use_guided_decoding else sampling_params
                results = llm.generate(vllm_inputs, _params, lora_request=lora_request)

                if results and not _first_generation_printed:
                    decoded = tokenizer.decode(
                        results[0].outputs[0].token_ids, skip_special_tokens=False
                    )
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
                        out_f.write(
                            json.dumps(
                                _output_row(
                                    row,
                                    win_preds,
                                    _resolve_events(win_preds),
                                    lean_output,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        pbar.update(1)

            # Flush per batch, not per generate(), so a batch that produced no windows
            # still lands on disk for the entrypoint's 90 s sync loop to pick up.
            out_f.flush()
            gc.collect()

    pbar.close()
    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    fire.Fire(vllm_infer)
