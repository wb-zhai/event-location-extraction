# retrieve — event-type candidate retrieval

Builds a dense vector index over the event-type ontology (label + description) and
retrieves the top-K most relevant event types for a query document. Used to shrink a
large ontology down to a small `candidates` list per document (e.g. for
`generation_v3`'s `--top-k-candidates` teacher-generation flag), and to evaluate how
well a given embedding model ranks the correct event types near the top.

---

## Files

| File | Purpose |
| --- | --- |
| `generate_index.py` | Step 1 — embed every event type (label + description) from an ontology JSON into a vector index |
| `retrieve.py` | Step 2 — retrieve top-K candidate event types per query using a Sentence Transformers model (CPU/GPU, in-process) |
| `retrieve_vllm.py` | Step 2 (alt) — same as above but encodes queries with vLLM in pooling mode (faster on GPU for large query sets) |
| `eval_recall_at_k.py` | Step 3 — compute Recall@K of the retrieved candidates against gold `event_type` annotations |

---

## Setup

From the repo root, with the project's virtualenv active and dependencies installed
(`pip install -r requirements.txt`):

- **Local Sentence Transformers models** (`retrieve.py`, `generate_index.py` without
  `gemini` in the model name): no extra config needed. Pass `--device cuda:0` to use a
  GPU.
- **`retrieve_vllm.py`**: requires the `vllm` package (already in `requirements.txt`)
  and a CUDA GPU — vLLM's pooling runner does not support CPU-only inference.
- **Gemini embeddings** (`generate_index.py` with a model name containing `gemini`,
  e.g. `gemini-embedding-001`): requires `.env` at the repo root with either
  `GEMINI_API_KEY`, or both `PROJECT_ID` and `VERTEX_LOCATION` set for Vertex AI. See
  the repo-root `.env` for the expected keys.

All scripts are run as modules from the **repo root** so the `src.*` imports resolve:

```bash
python scripts/data/retrieve/generate_index.py ...
```

---

## Step 1 — Build the index

Embeds every `(event_type, description)` pair from an ontology JSON into a vector
index saved to disk.

Input ontology format:

```json
{
  "events": {
    "flooding": "The overflow of water onto normally dry land...",
    "drought conditions": "A prolonged period of abnormally low rainfall..."
  }
}
```

(matches the files under `ontologies/`, e.g. `ontologies/zhai/science.json`).

```bash
# Local Sentence Transformers model
python scripts/data/retrieve/generate_index.py \
  ontologies/zhai/science.json \
  dataset/zhai/v3/index/science-bge-m3 \
  BAAI/bge-m3 \
  --sentence-transformers \
  --device cuda:0 \
  --normalize-embeddings

# Local Hugging Face model (plain AutoModel, mean/CLS pooling handled by HuggingFaceRetriever)
python scripts/data/retrieve/generate_index.py \
  ontologies/zhai/science.json \
  dataset/zhai/v3/index/science-e5 \
  intfloat/multilingual-e5-large \
  --device cuda:0

# Gemini embeddings (requires GEMINI_API_KEY or PROJECT_ID/VERTEX_LOCATION in .env)
python scripts/data/retrieve/generate_index.py \
  ontologies/zhai/science.json \
  dataset/zhai/v3/index/science-gemini \
  gemini-embedding-001 \
  --output-dimensionality 1536
```

Each document's indexed text is just the event-type label (`text`); the description is
stored as `metadata.description` but is not concatenated into the embedded text
(`--add_metadata_keys_to_text` is off by default in this script).

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `model_name` | — | HF/Sentence-Transformers repo id, or any name containing `gemini` to route to `GeminiRetriever` |
| `--sentence-transformers` | off | Use `SentenceTransformersRetriever` instead of the raw `HuggingFaceRetriever` (needed for most modern embedding models) |
| `--device` | `cpu` | `cuda:0` etc. for GPU encoding |
| `--precision` | `32` | Model compute precision (HF path only); `16` requires GPU |
| `--batch_size` | `128` | Passed through but indexing always batches at 100 internally (see `generate_index.py`) |
| `--num-workers` | `4` | DataLoader / concurrent-encode workers |
| `--encode-concurrency` | `--num-workers` | Concurrent request count for API-based retrievers (Gemini) |
| `--normalize-embeddings` | off | L2-normalize passage embeddings (Sentence Transformers path) |
| `--prompt-name` | None | Sentence-Transformers passage prompt template key (`passage_prompt_name`) |
| `--output-dimensionality` | `1536` | Gemini embedding output size |

Output is written to `<output_folder>/` as `documents.jsonl` + a memory-mapped
`embeddings.mmap` / `embeddings.meta.json` sidecar (`InMemoryIndexer.save_pretrained`,
always called with `mmap=True` here). Pass that folder as `--index` in Step 2.

On MacOS, keep `--num-workers 0` for the Gemini path — `DataLoader` with
`num_workers > 0` raises on Mac (see `GeminiRetriever.retrieve`).

---

## Step 2 — Retrieve candidates for queries

Both scripts read queries, encode them, search the index built in Step 1, and write
one JSONL line per query with a `candidates` field appended.

Query input can be:
- a `.txt` file (one query per line),
- a `.json` file (list of strings/objects, or `{"queries": [...]}`),
- a `.jsonl` file (one object per line),
- a `.csv` file (header row required), or
- one or more literal strings passed directly to `--queries`.

Each object/row is searched for one of `query`, `question`, `prompt`, `input`, `text`
(or a nested `source.text`) to find the query string; everything else on the record is
kept and written back out to the output alongside the new `candidates` field.

### `retrieve.py` — CPU/GPU, in-process Sentence Transformers

```bash
python scripts/data/retrieve/retrieve.py \
  --queries dataset/zhai/v3/dev.jsonl \
  --index   dataset/zhai/v3/index/science-bge-m3 \
  --output  dataset/zhai/v3/dev.candidates.jsonl \
  --model-name BAAI/bge-m3 \
  --top-k 50 \
  --device cuda:0 \
  --normalize-embeddings
```

### `retrieve_vllm.py` — vLLM pooling model (GPU only, faster for large query sets)

```bash
python scripts/data/retrieve/retrieve_vllm.py \
  --queries dataset/zhai/v3/dev.jsonl \
  --index   dataset/zhai/v3/index/science-bge-m3 \
  --output  dataset/zhai/v3/dev.candidates.jsonl \
  --model-name BAAI/bge-m3 \
  --top-k 50 \
  --normalize-embeddings \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

`retrieve_vllm.py` auto-detects a query prompt template from the model's
`config_sentence_transformers.json` (local HF cache or a local model directory) via
`--query-prompt-name <key>`, or accepts a literal `--query-prompt` string (mutually
exclusive; a `{query}` placeholder is formatted, otherwise treated as a prefix).

Common flags (both scripts):

| Flag | Default | Notes |
| --- | --- | --- |
| `--top-k` | `5` | Number of candidates per query |
| `--batch-size` | `32` | Queries per batch (vLLM still batches encoding internally) |
| `--device` | `cpu` | Device for the index/search (`retrieve.py` also uses it for encoding) |
| `--normalize-embeddings` | off | L2-normalize query embeddings before search |
| `--save-ids-only` | off | Write `metadata.estimate_id` per candidate instead of the passage text — only use this if your index documents carry that metadata field |

`retrieve_vllm.py`-only flags: `--tensor-parallel-size`, `--gpu-memory-utilization`,
`--max-model-len`, `--dtype`, `--trust-remote-code`, `--query-prompt-name`,
`--query-prompt`.

Output JSONL: the original query record plus `"candidates": [...]` (event-type
strings, ranked highest-similarity first, unless `--save-ids-only` is set).

---

## Step 3 — Evaluate Recall@K

Scores a candidates file produced by Step 2 against gold annotations.

```bash
python scripts/data/retrieve/eval_recall_at_k.py \
  dataset/zhai/v3/dev.candidates.jsonl \
  --k 1 3 5 10 20 50 70 100
```

Expects each line to have:

```json
{
  "annotation": {"events": [{"event_type": "flooding"}, ...]},
  "candidates": ["flooding", "drought conditions", ...]
}
```

For each document, Recall@K is the fraction of gold `event_type` values present in the
top-K candidates; documents with no gold events are skipped. The printed score is the
macro-average over all evaluated documents. Requires `--save-ids-only` to **not** have
been used in Step 2, since gold `event_type` values are compared directly against
candidate strings (not ids).
