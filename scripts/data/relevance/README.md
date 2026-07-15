# relevance — article relevance filtering

Pre-filters a raw article corpus before running the expensive teacher generation step
(see [`generation_v3`](../generation_v3/README.md)). Three filtering modes can be
composed: a cheap first pass (keyword or regex) followed by an optional LLM second
pass on the survivors.

## Files

| File | Purpose |
| --- | --- |
| `relevance_filter.py` | Step 0 — pre-filter articles before sending to the teacher |
| `train.py` | Fine-tune an encoder classifier on `relevance_filter.py` output |
| `local.py` | Test the relevance gate against a local OpenAI-compatible LLM server |
| `encoder.py` | Test the relevance gate against a local encoder text classifier (no LLM) |
| `view_relevance.py` | Gradio UI to browse a `relevance_filter.py` output JSONL |
| `agreement.py` | Compare relevance decisions between two labeled JSONL files (accuracy, kappa) |

---

## `relevance_filter.py`

```bash
# Keyword pre-filter (default keyword list, write all records + relevance metadata)
python scripts/data/relevance/relevance_filter.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output dataset/zhai/v3/articles.filtered.jsonl

# Regex pre-filter — faster than the keyword list
python scripts/data/relevance/relevance_filter.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output dataset/zhai/v3/articles.filtered.jsonl \
  --use-regex --filter-only

# LLM-only filter (skips keyword/regex pre-pass)
python scripts/data/relevance/relevance_filter.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output dataset/zhai/v3/articles.filtered.jsonl \
  --use-llm --filter-only

# Regex pre-pass then LLM on survivors
python scripts/data/relevance/relevance_filter.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output dataset/zhai/v3/articles.filtered.jsonl \
  --use-regex --use-llm --filter-only
```

**How the modes compose:**

1. If `--use-regex` is set, the compiled `food_insecurity_regex` is checked first. Articles that don't match are marked filtered immediately (no LLM call).
2. Otherwise, if `--keywords` is set (or the default keyword list is used), a case-insensitive substring check is run. Non-matching articles are marked filtered.
3. If `--use-llm` is set and the article was not already filtered by step 1/2, Gemini classifies the article with a structured `{is_relevant, confidence, reason}` response. An article is filtered only when `is_relevant=false` and `confidence ≥ --confidence-threshold`. With `--use-3label-prompt`, Gemini instead returns `{relevance_label, confidence, reason}` where `relevance_label` is one of `relevant` / `partially_relevant` / `not_relevant`; only `not_relevant` maps to `is_relevant=false` (and is thus filterable), while `partially_relevant` is treated as relevant but recorded in `relevance.decision` for downstream distinction.

Every output record gets a `relevance` field. With `--filter-only`, records marked filtered are omitted from the output entirely.

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--use-regex` | off | Use the built-in `food_insecurity_regex` as a fast first-pass filter |
| `--keywords` | built-in list | Space-separated keywords for substring matching; ignored when `--use-regex` is set |
| `--use-llm` | off | Run Gemini relevance classification on articles that survive the pre-pass |
| `--use-3label-prompt` | off | Use the 3-label (`relevant` / `partially_relevant` / `not_relevant`) prompt instead of the default 2-label (`is_relevant`) prompt. Only applies with `--use-llm` |
| `--model` | `gemini-2.5-flash` | Gemini model for LLM mode |
| `--max-chars` | `1000` | Article preview length sent to the LLM (title + first N chars) |
| `--confidence-threshold` | `0.0` | Minimum LLM confidence to act on an `is_relevant=false` decision |
| `--filter-only` | off | Omit filtered records from output (default: write all records with metadata) |
| `--concurrency` | `10` | Parallel async workers for LLM mode |
| `--verbose` | off | Log each LLM prompt and response |

The LLM filter favors recall: the system prompt instructs the model to mark borderline articles relevant. The `--confidence-threshold` flag lets you tighten this — at `0.8` the LLM must be quite confident before dropping an article.

---

## `local.py`

Runs the same relevance gate against a local OpenAI-compatible LLM server instead of Gemini —
useful for testing small local models (e.g. `LiquidAI/LFM2.5-350M`) as a cheap/offline stand-in.
Output has the same `relevance` shape as `relevance_filter.py`, so it can be diffed against a
Gemini-labeled file with `agreement.py`.

**1. Serve the model** with [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`,
which exposes an OpenAI-compatible endpoint identically on macOS and Linux/NVIDIA:

```bash
# macOS (Metal is used automatically)
brew install llama.cpp
llama-server -hf LiquidAI/LFM2.5-350M-GGUF --host 127.0.0.1 --port 8080 -c 4096

# Linux + NVIDIA (prebuilt CUDA binary from the llama.cpp releases page, or build with -DGGML_CUDA=ON)
./llama-server -hf LiquidAI/LFM2.5-350M-GGUF --host 127.0.0.1 --port 8080 -c 4096 -ngl 99
```

**2. Run the gate:**

```bash
python scripts/data/relevance/local.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output /tmp/articles.local.jsonl \
  --model LFM2.5-350M --base-url http://127.0.0.1:8080/v1 \
  --limit 50 --overwrite
```

**3. Compare against a Gemini-labeled file:**

```bash
python scripts/data/relevance/agreement.py \
  --a dataset/zhai/v3/articles.gemini.jsonl --b /tmp/articles.local.jsonl \
  --disagreements /tmp/local_vs_gemini.disagree.jsonl
```

Key flags (in addition to the ones shared with `relevance_filter.py` — `--max-chars`,
`--confidence-threshold`, `--concurrency`, `--verbose`, `--filter-only`, `--use-regex`,
`--limit`, `--resume`, `--overwrite`):

| Flag | Default | Notes |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible server endpoint |
| `--api-key` | none | Only needed if the server checks one |
| `--temperature` | `0.0` | Sampling temperature |
| `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--seed` | unset | Extra sampling knobs worth tuning for a small model — `--top-k`/`--min-p`/`--repeat-penalty` are forwarded via `extra_body` since they're llama.cpp/vLLM extensions, not part of the OpenAI API |
| `--max-tokens` | `256` | Max tokens generated for the `{reason, is_relevant, confidence}` response |

---

## `view_relevance.py`

Gradio table for eyeballing a `relevance_filter.py` output file — one row per record
with title, decision, confidence, filtered flag, and reason.

```bash
python scripts/data/relevance/view_relevance.py --input dataset/zhai/v3/articles.filtered.jsonl
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--input` | required | Path to `relevance_filter.py` output JSONL |
| `--port` | `7860` | Gradio server port |
| `--share` | off | Create a public Gradio share link |

---

## `encoder.py`

Runs the relevance gate against a local encoder text classifier instead of an LLM —
[`classla/multilingual-IPTC-news-topic-classifier`](https://huggingface.co/classla/multilingual-IPTC-news-topic-classifier)
(xlm-roberta-large, 94 languages). The classifier assigns each article one of 17
fixed IPTC top-level topics; `is_relevant` is then determined by a fixed
`RELEVANT_LABELS` set (`conflict, war and peace`, `disaster, accident and emergency
incident`, `economy, business and finance`, `environment`, `politics`, `weather`,
`health`, `society`). No prompt, no API cost, runs fully locally. `reason` is a templated
sentence, e.g. `"The news is classified as disaster, accident and emergency
incident therefore it is relevant."` Output has the same `relevance` shape as
`relevance_filter.py`, so it's diffable against a Gemini-labeled file with
`agreement.py`.

```bash
python scripts/data/relevance/encoder.py \
  --input  dataset/zhai/v3/articles.jsonl \
  --output /tmp/articles.encoder.jsonl \
  --limit 50 --overwrite

python scripts/data/relevance/agreement.py \
  --a dataset/zhai/v3/articles.gemini.jsonl --b /tmp/articles.encoder.jsonl
```

Key flags (in addition to the ones shared with the other scripts — `--max-chars`,
`--confidence-threshold`, `--filter-only`, `--use-regex`, `--limit`, `--resume`,
`--overwrite`):

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | `classla/multilingual-IPTC-news-topic-classifier` | Any HF `text-classification` model |
| `--device` | auto-detect | `cuda` / `mps` / `cpu`; auto-picks the best available |
| `--max-length` | `512` | Tokenizer max sequence length (the model's native limit) |
| `--batch-size` | `32` | Pipeline's internal inference batch size |
| `--chunk-size` | `500` | Records per progress/flush chunk |

---

## `train.py`

Fine-tunes a binary (`relevant` / `irrelevant`) sequence classifier on the labeled JSONL produced
by `relevance_filter.py`. Default backbone is `answerdotai/ModernBERT-base`. Self-contained — no
imports from other repo scripts.

**Input format** — each JSONL record must have:
- `relevance.is_relevant` — `true` / `false` label (records missing this field are skipped)
- `title` or `source.title` and `text` or `source.text` — concatenated to form the input text
- `id` or `url` — used for deduplication (optional but recommended)

**Outputs** saved to `--output-dir`:
- `final/` — best checkpoint (model + tokenizer), loadable with `AutoModelForSequenceClassification.from_pretrained`
- `eval_metrics.json` — final evaluation metrics (accuracy, precision, recall, F1, confusion matrix counts)

```bash
# Auto train/eval split (85/15 stratified)
python scripts/data/relevance/train.py \
  --input  dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \
  --output-dir /tmp/relevance-modernbert

# Separate eval file + class balancing (useful for skewed datasets)
python scripts/data/relevance/train.py \
  --input     data/train.jsonl \
  --eval-file data/eval.jsonl \
  --output-dir /tmp/relevance-modernbert \
  --balance-classes

# Smoke test — 1 epoch, small batch, CPU-safe fp32
python scripts/data/relevance/train.py \
  --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \
  --output-dir /tmp/relevance-modernbert-smoke \
  --num-epochs 1 --batch-size 8 --precision fp32
```

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--input` | required | Training JSONL file |
| `--output-dir` | required | Directory for checkpoints and final model |
| `--eval-file` | none | Separate eval JSONL; if omitted, a stratified split of `--input` is used |
| `--model-name` | `answerdotai/ModernBERT-base` | Any HF sequence-classification model |
| `--max-length` | `512` | Max token length; longer inputs are truncated |
| `--max-chars` | `2000` | Max characters taken from raw text before tokenization |
| `--batch-size` | `16` | Per-device train and eval batch size |
| `--learning-rate` | `2e-5` | AdamW learning rate |
| `--num-epochs` | `3` | Number of training epochs |
| `--eval-frac` | `0.15` | Fraction of `--input` held out for eval when no `--eval-file` is given |
| `--precision` | `auto` | `auto` picks bf16 > fp16 > fp32 based on hardware; `fp32` is safe on CPU/MPS |
| `--balance-classes` | off | Applies inverse-frequency class weights to the loss (helps with skewed label distributions) |
| `--metric-for-best` | `f1` | Metric used to select the best checkpoint (`accuracy` / `precision` / `recall` / `f1`) |
| `--wandb-project` | none | WandB project name; omit to disable WandB logging |
| `--resume-from-checkpoint` | none | Path to a checkpoint directory to resume training from |

---

## Human annotation (Argilla)

For scoring the Gemini relevance gate against human labels, see
[`scripts/annotations/README.md`](../../annotations/README.md).
