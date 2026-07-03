# relevance — article relevance filtering

Pre-filters a raw article corpus before running the expensive teacher generation step
(see [`generation_v3`](../generation_v3/README.md)). Three filtering modes can be
composed: a cheap first pass (keyword or regex) followed by an optional LLM second
pass on the survivors.

## Files

| File | Purpose |
| --- | --- |
| `relevance_filter.py` | Step 0 — pre-filter articles before sending to the teacher |
| `view_relevance.py` | Gradio UI to browse a `relevance_filter.py` output JSONL |

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
3. If `--use-llm` is set and the article was not already filtered by step 1/2, Gemini classifies the article with a structured `{is_relevant, confidence, reason}` response. An article is filtered only when `is_relevant=false` and `confidence ≥ --confidence-threshold`.

Every output record gets a `relevance` field. With `--filter-only`, records marked filtered are omitted from the output entirely.

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--use-regex` | off | Use the built-in `food_insecurity_regex` as a fast first-pass filter |
| `--keywords` | built-in list | Space-separated keywords for substring matching; ignored when `--use-regex` is set |
| `--use-llm` | off | Run Gemini relevance classification on articles that survive the pre-pass |
| `--model` | `gemini-2.5-flash` | Gemini model for LLM mode |
| `--max-chars` | `1000` | Article preview length sent to the LLM (title + first N chars) |
| `--confidence-threshold` | `0.0` | Minimum LLM confidence to act on an `is_relevant=false` decision |
| `--filter-only` | off | Omit filtered records from output (default: write all records with metadata) |
| `--concurrency` | `10` | Parallel async workers for LLM mode |
| `--verbose` | off | Log each LLM prompt and response |

The LLM filter favors recall: the system prompt instructs the model to mark borderline articles relevant. The `--confidence-threshold` flag lets you tighten this — at `0.8` the LLM must be quite confident before dropping an article.

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

## Human annotation (Argilla)

For scoring the Gemini relevance gate against human labels, see
[`scripts/annotations/README.md`](../../annotations/README.md).
