# Inference & Evaluation Guide

## Running inference

Batch inference is handled by [vllm_infer.py](vllm_infer.py). It uses [vLLM](https://github.com/vllm-project/vllm) for high-throughput generation, reads raw JSONL directly (no LlamaFactory dataset registration required), and writes predictions back to the same format with `predictions` and `window_predictions` fields appended. Re-running on an existing output file automatically skips already-processed articles.

### Input format

Each line of the input JSONL must contain a `source` object with at least a `text` field:

```json
{"source": {"text": "...", "publish_date": "2024-01-15"}, ...}
```

### Basic usage

```bash
python scripts/train/inference/vllm_infer.py \
  --model_name_or_path <hf-repo-or-local-path> \
  --input data/articles.jsonl \
  --output outputs/predictions.jsonl \
  --temperature 0 \
  --top_k_candidates 70 \
  --max_model_len 8192 \
  --max_chars 3000 \
  --max_new_tokens 2048
```

With a LoRA adapter:

```bash
python scripts/train/inference/vllm_infer.py \
  --model_name_or_path <base-model> \
  --adapter_name_or_path <adapter-path> \
  --input data/articles.jsonl \
  --output outputs/predictions.jsonl \
  --temperature 0 \
  --top_k_candidates 70 \
  --max_model_len 8192 \
  --max_chars 3000 \
  --max_new_tokens 2048
```

### Multi-GPU tensor parallelism

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train/inference/vllm_infer.py \
  --model_name_or_path <model> \
  --input data/articles.jsonl \
  --output outputs/predictions.jsonl \
  --tensor_parallel_size 4
```

### Sharding across machines

To split work across multiple nodes, pass `--num_shards` and `--shard_index` (stride-based, so load is balanced):

```bash
# Node 0
python scripts/train/inference/vllm_infer.py ... --num_shards 4 --shard_index 0

# Node 1
python scripts/train/inference/vllm_infer.py ... --num_shards 4 --shard_index 1
```

### Key arguments

| Argument                 | Default  | Description                                                            |
| ------------------------ | -------- | ---------------------------------------------------------------------- |
| `model_name_or_path`     | required | HuggingFace repo or local path                                         |
| `input`                  | required | Input JSONL file                                                       |
| `output`                 | required | Output JSONL file (appended; processed articles are skipped on re-run) |
| `adapter_name_or_path`   | `None`   | Path to a LoRA adapter directory                                       |
| `ontology`               | built-in | Path to ontology JSON for event labels                                 |
| `prompt_dir`             | built-in | Directory containing `system_prompt.txt` and `user_prompt.txt`         |
| `top_k_candidates`       | `None`   | Limit candidate event labels shown in the prompt                       |
| `shard_index`            | `0`      | Index of this shard (0-based)                                          |
| `num_shards`             | `1`      | Total number of shards                                                 |
| **Windowing**            |          |                                                                        |
| `max_chars`              | `3000`   | Maximum characters per window                                          |
| `min_chars`              | `200`    | Minimum characters to keep a window                                    |
| `max_paras`              | `15`     | Maximum paragraphs per window                                          |
| `overlap_paras`          | `1`      | Overlap paragraphs between adjacent windows                            |
| **Sampling**             |          |                                                                        |
| `temperature`            | `0.0`    | Sampling temperature (0 = greedy)                                      |
| `max_new_tokens`         | `4096`   | Maximum tokens to generate                                             |
| `top_p`                  | `0.8`    | Top-p nucleus sampling                                                 |
| `top_k`                  | `0`      | Top-k sampling                                                         |
| `repetition_penalty`     | `1.05`   | Repetition penalty                                                     |
| `seed`                   | `None`   | Random seed for reproducibility                                        |
| **Engine**               |          |                                                                        |
| `max_model_len`          | `None`   | vLLM max sequence length (input + output tokens)                       |
| `gpu_memory_utilization` | `0.95`   | Fraction of GPU memory vLLM may use                                    |
| `tensor_parallel_size`   | `1`      | Number of GPUs for tensor parallelism                                  |
| `batch_size`             | `1000`   | Prompts handed to vLLM per call                                        |
| `max_num_seqs`           | `None`   | vLLM max concurrent sequences                                          |

### Output format

Each output line is the original input row with two fields added:

```json
{
  "source": {"text": "...", "publish_date": "..."},
  "predictions": [{"event_type": "...", "grounding_quote": "...", ...}],
  "window_predictions": [
    {
      "window_start": 0, "window_end": 1500,
      "window_text": "...",
      "prediction": {"events": [...]}
    }
  ]
}
```

`predictions` is the deduplicated merge across all windows (keyed on `grounding_quote`). `window_predictions` contains each window's raw model output.

---

## Windowing strategy

Long articles are split into overlapping paragraph-based windows inside `vllm_infer.py` itself (via helpers in `scripts/data/generation/to_sft.py`). No external windowing script is needed.

### How it works

1. **Paragraph splitting** — The article is split on blank lines into paragraphs. Paragraphs longer than `max_chars` are further split on whitespace.

2. **Window packing** — Paragraphs are greedily packed into windows so each window spans at most `max_chars` characters and at most `max_paras` paragraphs.

3. **Overlap** — Adjacent windows share `overlap_paras` paragraphs of context.

4. **Short-window coalescing** — Windows shorter than `min_chars` are merged into a neighbour.

5. **Deduplication** — After inference, events are merged across windows and deduplicated by `grounding_quote`.

---

# eval_v3_sft.py — Evaluation Guide

Evaluates a distilled event-location extraction model against annotated gold data.
Each line of the input JSONL contains both the ground-truth annotation and the model
predictions for one document.

## Usage

```bash
python scripts/train/inference/eval_v3_sft.py \
  --pred-jsonl dataset/risk-factor/run-15062025/predictions/quick_dev.jsonl \
  [--report-json /tmp/report.json] \
  [--errors-jsonl /tmp/errors.jsonl] \
  [--cluster ontologies/risk-factors/clusters.json]
```

| Flag             | Description                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--pred-jsonl`   | Predictions JSONL file (required). Each record must contain both `annotation` and `predictions` fields.                                                 |
| `--report-json`  | Write the full nested metrics dict as JSON.                                                                                                             |
| `--errors-jsonl` | Write one row per document that had unmatched events; useful for error analysis.                                                                        |
| `--cluster`      | Studio results JSON mapping event names to cluster labels. When provided, adds four additional cluster-level metric families to the report (see below). |

---

## Output format

The report prints P/R/F1 tables plus conditional accuracy tables:

- **MICRO** — pooled TP/FP/FN across all documents (standard NLP micro-average).
- **MACRO** — P/R/F1 computed per document then averaged; gives equal weight to each document regardless of how many events it contains.
- **MICRO CONDITIONAL ACCURACY** — pooled correctness among already matched events.
- **MACRO CONDITIONAL ACCURACY** — conditional accuracy computed per document and then averaged.

For linked fields such as `event_location` and `event_time`, the conditional
accuracy rows are usually the clearest performance view. The end-to-end P/R/F1
rows are stricter diagnostics because they include event detection and grounding
errors in addition to linked-field errors.

Each table has two column groups:

| Group       | Meaning                                                               |
| ----------- | --------------------------------------------------------------------- |
| **RELAXED** | Lenient matching (headline number)                                    |
| **EXACT**   | Strict matching; shows how much the model deviates from verbatim gold |

---

## Core metric families (always computed)

### 1. Event extraction

Measures whether the model finds the right events at all.

Each model prediction is matched one-to-one to a gold event using:
- `event_type` must agree (case-insensitive).
- `grounding_quote` token overlap (SequenceMatcher ratio) ≥ 0.5.

Matched pairs = **TP**; unmatched gold events = **FN**; unmatched predictions = **FP**.

> **Why quote overlap and not exact match?** The model frequently appends location
> context to the grounding quote (e.g. gold: *"5,000 IDP households"* → pred:
> *"5,000 IDP households in Maiduguri"*). Requiring verbatim equality would
> misclassify these as wrong events.

Exact tier additionally requires identical normalized quotes (ratio = 1.0).

---

### 2. Event type (multiset)

Measures whether the model identifies the correct categories of events, independent
of how well it grounds them.

Computed as a **multiset** comparison of `event_type` values per document.
If gold has two "displaced" events and the model predicts one, that counts as
TP=1, FN=1.

Event type uses the same exact/relaxed labels as other families, but the scores
are identical in both tiers because type matching is always a simple string
equality (no fuzzy component).

---

### 3. Event type (doc-level set)

Same as family 2 but **deduplicates** event types per document before comparing.
If gold has two "displaced" events and the model predicts one, that counts as
TP=1, FN=0 (the type was covered). This answers the question *"did the model
identify all distinct event categories present in this document?"* without
penalizing it for not finding every individual instance.

As with family 2, exact and relaxed scores are identical (type matching is always
string equality).

---

### 4. Location extraction

Measures whether the model extracts the correct place names, independent of which
event they are attached to. This is useful as a diagnostic, but it does not answer
whether locations are attached to the right event.

`event_location` values may contain multiple locations separated by `";"` (e.g.
`"Rome; Milan"`). Each semicolon-separated part is split out and treated as an
independent entry in the per-document bag before comparison.

A predicted location matches a gold location if:

| Tier        | Matching rule                                                                                                                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Relaxed** | After lowercasing and stripping directional prefixes (*northern*, *southern*, *eastern*, *western*, *central*): exact equality **or** one is a substring of the other **or** SequenceMatcher ratio ≥ 0.85 |
| **Exact**   | Normalized string equality only                                                                                                                                                                           |

> **Why strip directional prefixes?** Annotations often normalize *"northern
> Bangladesh"* to *"Bangladesh"*. The model may or may not strip the prefix; both
> are correct.

---

### 5. Event-location linking

Measures whether the model correctly associates a location **with the right event**.
For reporting linked-location quality, prefer the conditional accuracy rows:
they only evaluate event pairs that were already matched by event type and quote.

Among matched event pairs, the evaluator checks whether the predicted
`event_location` matches the gold `event_location` using the same location rules
as family 4. Multi-location values (semicolon-separated) are handled as sets: for
the exact tier both sets must be equal; for the relaxed tier every location in each
set must fuzzy-match some location in the other set.

The P/R/F1 row named `End-to-end event-location` is stricter: unmatched events
still contribute to FP/FN. That makes it an end-to-end score covering event
detection, quote grounding, and location linking together.

The conditional accuracy rows isolate location linking:

| Metric                                       | Meaning                                                                                                                                                        |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location on matched events**               | Among event pairs already matched by family 1, how often does `event_location` also match? Includes `not_stated` ↔ `not_stated`.                               |
| **Location on matched events (gold stated)** | Same, but only for matched gold events with an actual stated location. This focuses on geographic extraction/linking and excludes correct absence predictions. |

---

### 6. Event-time linking

Measures whether the model correctly associates a time expression **with the right
event**. As with location, prefer the conditional accuracy rows for reporting
linked-time quality. The end-to-end P/R/F1 row is stricter because event detection,
quote grounding, and time linking all affect the final number.

`event_time` values are ISO 8601 strings (`"2018-01/2018-10"`, `"2015"`, `"not_stated"`).

| Tier        | Match rule                                                                                                                              |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Exact**   | Normalized string equality                                                                                                              |
| **Relaxed** | Exact equality **or** the two values share at least one year in common (e.g. `"2018-01"` and `"2018"` both start with `"2018"` → match) |

> The relaxed rule avoids penalizing the model for predicting the correct year but
> getting the month/day granularity wrong, which is often ambiguous from the source text.

The conditional accuracy rows isolate time linking:

| Metric                                   | Meaning                                                                                                                                                                       |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Time on matched events**               | Among event pairs already matched by family 1, how often does `event_time` also match? Includes `not_stated` ↔ `not_stated`.                                                  |
| **Time on matched events (gold stated)** | Same, but only for matched gold events with an actual stated time. This focuses on extracting/linking concrete temporal expressions and excludes correct absence predictions. |

---

## Cluster metric families (require `--cluster`)

When `--cluster` is passed, each event's `event_type` is mapped to a coarser cluster
label (e.g. `"displaced"` → `"forced displacement"`) before comparison. This adds
four extra rows to the report.

### 7. Cluster event extraction

Like family 1, but the bipartite matching uses **cluster labels** instead of
fine-grained types. Two events that differ in fine-grained type but share the same
cluster (e.g. gold `"drought"` vs pred `"floods"`, both `"weather shocks"`) can now
match, provided their grounding quotes also overlap.

---

### 8. Cluster type (multiset)

Like family 2 (event type multiset) but each `event_type` is replaced by its cluster
label before the bag comparison. Tells you whether the model covers the right broad
categories regardless of exact type naming.

---

### 9. Cluster type (doc-level set)

Like family 3 (event type set) but using cluster labels. Answers *"did the model
cover all distinct risk-factor clusters mentioned in this document?"*

As with families 2–3, exact and relaxed scores are identical.

---

### 10. Cluster event-location linking

Like family 5, but the alignment step uses **cluster-level event matching** from
family 7. The conditional cluster rows ask: once events are aligned by cluster and
quote, how often does the linked location match?

### 11. Cluster event-time linking

Like family 6, but the alignment step uses cluster-level event matching from family
7. The conditional cluster rows ask: once events are aligned by cluster and quote,
how often does the linked time match?

The conditional accuracy table also includes cluster versions of the location and
time on matched events metrics. These use cluster-level event matching as the
alignment step before checking the linked field.

---

## Interpreting the numbers

**Relaxed ≫ exact on event extraction** is expected. The model often extends
the grounding quote to include location or time context, which is semantically
correct but doesn't match verbatim.

**Event type F1 > event extraction F1** is expected. The model may identify
the right category of event but anchor it to a slightly different span, or
merge two gold events into one.

**Event type (set) ≥ event type (multiset)** is expected. The set metric gives
credit for covering a category at least once, so documents with repeated event
types (e.g. multiple displacement events) will score higher on the set metric
even if the model underestimates the count.

**Cluster metrics ≥ fine-grained equivalents** is expected. Collapsing types to
clusters removes synonymy penalties (e.g. `"drought"` vs `"failed rains"`), so
scores are always at least as high as their fine-grained counterparts.

**Standalone location extraction and end-to-end event-location are diagnostics.**
Standalone location extraction ignores event links, while end-to-end
event-location includes event detection and grounding errors. Neither is the
cleanest linked-location number.

**Location on matched events** is the cleaner location-linking number. It removes
missed/spurious events from the denominator and asks: once the event alignment is
already correct, did the model attach the right location?

**Time on matched events** plays the same role for time. It is the cleaner
time-linking number because missed/spurious events are removed from the
denominator.

**Cluster location/time on matched events** uses looser event alignment before
checking the linked field. These rows are useful when fine-grained event labels
are noisy or synonym-heavy.

**Gold stated** rows exclude `not_stated` gold fields. They are the better view
when you want to measure extraction/linking of concrete places or times, rather
than correct absence predictions.

**Macro vs micro divergence** tells you about consistency. If macro F1 is much
higher than micro, the model is strong on short documents but struggles on
long ones with many events. The reverse suggests the model handles dense
documents better than sparse ones.
