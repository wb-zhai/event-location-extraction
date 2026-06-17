# generation_v3 — Gemini teacher pipeline + SFT data prep

Distills Gemini silver labels for food-insecurity risk-factor event extraction into
an SFT dataset for fine-tuning a small open model (Qwen3-4B).

The approach is **context/prompt distillation**: the teacher keeps its full elaborate
prompt to emit good labels; the student is trained on the teacher's outputs, learning
to reproduce them from a simpler prompt. Rules and few-shot behaviour get internalized
into the student's weights.

---

## Ontology

Event labels are defined in a JSON file with the schema:

```json
{
  "events": {
    "flooding": "The overflow of water onto normally dry land...",
    "drought conditions": "A prolonged period of abnormally low rainfall..."
  }
}
```

Each script uses a different ontology file and placeholder format:

| Script | Default ontology path | Placeholder format |
| --- | --- | --- |
| `generate.py` | baked into teacher system prompt | `<allowed_event_types>...</allowed_event_types>` XML block replaced when a per-record `candidates` list is present |
| `validate.py` | `ontologies/zhai/science.json` | — (loads labels for validation checks) |
| `fix_events.py` | `ontologies/zhai/science.json` (hardcoded) | — (loads labels for re-validation) |
| `to_sft.py` | `ontologies/zhai/science.json` | `{{ALLOWED_EVENT_TYPES}}` in student system prompt |

`validate.py` and `to_sft.py` accept `--ontology <path>` to swap the label file.
For `generate.py`, per-record candidate label restriction is done via a `candidates` field
in the input records (see `--top-k-candidates`).

---

## Files

| File | Purpose |
| --- | --- |
| `generate.py` | Step 1 — run the Gemini teacher to produce silver JSONL |
| `validate.py` | Step 2 — validate and filter silver output |
| `fix_events.py` | Step 2b — repair invalid events with a stronger model |
| `merge.py` | Step 2c — assemble clean + fixed events into a single final JSONL |
| `retrieve.py` | Utility — retrieve articles from the corpus for input to generation |
| `costs.py` | Utility — report token usage and estimated cost for any pipeline JSONL |
| `to_sft.py` | Step 3 — convert final JSONL to LlamaFactory Alpaca format |
| `prompts/teacher/` | Teacher system + user prompt templates |
| `prompts/fixer/` | Fixer system + user prompt templates |
| `prompts/student/` | Student system + user prompt templates (compact, no few-shot) |

---

## Pipeline

### Step 1 — Teacher generation

```bash
python scripts/data/generation_v3/generate.py \
  --input  dataset/zhai/v3/<stratified-articles>.jsonl \
  --output dataset/zhai/v3/silver.gemini.jsonl \
  --model  gemini-2.5-flash \
  --temperature 1.0 \
  --reasoning-effort low \
  --batch-api \
  --batch-size 100
```

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | `gemini-2.5-flash` | Use `gemini-3.1-pro-preview` for higher quality |
| `--temperature` | `0.0` | **Override to `1.0`** for thinking models (Gemini 3 guidance) |
| `--reasoning-effort` | `None` | `low` is sufficient for extraction; `medium` if recall is weak |
| `--batch-api` | off | Recommended for large runs; polls until complete |
| `--batch-size` | 100 | Requests per batch chunk |
| `--batch-poll-interval-seconds` | 30 | Seconds between batch job status polls |
| `--workers` | 4 | Parallel async workers for sync mode |
| `--limit` | None | Cap number of records (useful for smoke tests) |
| `--random` | off | Random sample when `--limit` is set |
| `--stratified` | None | `url` — stratified sample by source URL when `--limit` is set |
| `--top-k-candidates` | None | Restrict allowed event types to the first N entries in each record's `candidates` list |
| `--include-thoughts` | off | Store model thinking traces in output metadata |
| `--interactive` | off | Single-record interactive mode for debugging (reads article text from stdin) |

Output is appended JSONL with one record per article. Each record has `source`
(title, text, publish_date, source_url) and `annotation` (document_relevance +
events array). Events that fail grounding verification are moved to
`annotation.unverified_events` (each tagged with `_unverified_field`) rather than
being silently dropped — they survive in the output for later inspection, but
`validate.py` will re-reject them.

The script resumes automatically: records whose `id` is already present with
`status=ok` in the output file are skipped.

---

### Step 2 — Validation and filtering

```bash
python scripts/data/generation_v3/validate.py \
  --input       dataset/zhai/v3/silver.gemini.jsonl \
  --output-stem dataset/zhai/v3/silver.validated
```

Emits two files:

- `silver.validated.clean.jsonl` — records with invalid events removed
- `silver.validated.invalid.jsonl` — one line per rejected event with error details

Each event is checked for:

- **Ontology**: `event_type` must be in the ontology (default: `ontologies/zhai/science.json`)
- **Grounding**: `grounding_quote`, `event_location_text`, `event_time_text` must be
  exact substrings of `source.text`
- **Enum validity**: `time_status`, `severity`, `modality` must be allowed values
- **Deduplication**: same `(event_type, grounding_quote)` within a document is dropped

A summary is printed to stderr. Expect drop rates < ~15% on good teacher output;
investigate if grounding failures are high.

Optional flags:

```bash
--ontology     <path>   # default: ontologies/zhai/science.json
--system-prompt <path>  # cross-checks ontology against <allowed_event_types> in teacher prompt
```

---

### Step 2b — Fix invalid events

Sends the rejected events from `*.invalid.jsonl` back to Gemini 3.1 Pro, which
either remaps each event to the best-fitting ontology label (re-grounding all
fields) or drops it when no label fits. Every decision is saved with the
original event and the model's reason, so off-ontology types that can't be
remapped can be analysed (which real-world hazards the ontology is missing).

```bash
python scripts/data/generation_v3/fix_events.py \
  --input  dataset/zhai/v3/silver.validated.invalid.jsonl \
  --output dataset/zhai/v3/silver.fixed.jsonl \
  --model  gemini-3.1-pro-preview \
  --temperature 1.0 \
  --reasoning-effort low
```

The fixer re-validates fixed events against
`ontologies/zhai/science.json` (hardcoded). The ontology
path cannot currently be changed without editing the script.

Key flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--mode` | `per-article` | `per-article` sends one call per document (all its invalid events together); `per-event` sends one call per event |
| `--model` | `gemini-3.1-pro-preview` | Use any Gemini model |
| `--temperature` | `0.0` | Override to `1.0` for Gemini 3 thinking models |
| `--reasoning-effort` | `None` | `low` / `medium` / `high` |
| `--batch-api` | off | Use Gemini Batch API for large runs |
| `--batch-size` | 100 | Tasks per batch chunk |
| `--batch-poll-interval-seconds` | 30 | Seconds between batch job status polls |
| `--workers` | 4 | Parallel async workers (sync mode) |
| `--limit` | None | Cap number of input rows (smoke tests) |

Output is JSONL with one record per input invalid event:

```json
{
  "row": 0,
  "doc_id": "...",
  "publish_date": "...",
  "original_event": { ... },
  "errors": [ ... ],
  "decision": "fixed" | "dropped",
  "fixed_event": { ... } | null,
  "revalidation": {"valid": true, "errors": []},
  "reason": "why this label / why it can't be mapped",
  "llm": {"model": "...", "metadata": { ... }}
}
```

`revalidation.valid=true` means the fixed event passed all validation checks
(ontology, grounding, enums). `dropped` events carry a non-empty `reason`
explaining why no ontology label fits.

---

### Step 2c — Merge clean and fixed events

Combines `validate.py`'s clean output with the successfully fixed events from
`fix_events.py` into a single JSONL ready for downstream use. The output format
is identical to `generate.py`'s output.

```bash
python scripts/data/generation_v3/merge.py \
  --clean  dataset/zhai/v3/silver.validated.clean.jsonl \
  --fixed  dataset/zhai/v3/silver.fixed.jsonl \
  --output dataset/zhai/v3/silver.final.jsonl
```

Only fixed events that passed re-validation (`revalidation.valid=true`) are merged.
Dropped events and failed API calls are silently skipped.

The `--fixed` flag is optional — omit it if there are no invalid events to merge.

`merge.py` also prints a token + cost report for both the generation and fix steps
(see [Cost reporting](#cost-reporting) below).

**Intermediate files to inspect:**

| File | What to look for |
| --- | --- |
| `silver.validated.invalid.jsonl` | Off-ontology types, hallucinated grounding quotes |
| `silver.fixed.jsonl` | Dropped events + `reason` field reveal ontology gaps |

---

## Cost reporting

`costs.py` reads any pipeline JSONL and prints a token breakdown + estimated cost
grouped by model. Token counts are stored in every record under `llm.metadata`.

```bash
# Generation cost
python scripts/data/generation_v3/costs.py dataset/zhai/v3/silver.jsonl

# Fix cost (dedup by doc_id because per-article mode writes one record per
# invalid event, but they all share the same LLM call per article)
python scripts/data/generation_v3/costs.py \
  dataset/zhai/v3/silver.fixed.jsonl --dedup-key doc_id

# Both files at once
python scripts/data/generation_v3/costs.py \
  dataset/zhai/v3/silver.jsonl \
  dataset/zhai/v3/silver.fixed.jsonl --dedup-key doc_id

# Use Batch API pricing (~50% off input/output) instead of standard pricing
python scripts/data/generation_v3/costs.py dataset/zhai/v3/silver.jsonl --batch
```

Model pricing is a plain dict at the top of `costs.py` — edit it when rates change.
Models not in the dict are reported as "pricing unknown".

---

### Step 3 — Convert to LlamaFactory SFT format

```bash
python scripts/data/generation_v3/to_sft.py \
  dataset/zhai/v3/silver.final.jsonl \
  dataset/zhai/v3/silver.sft.json
```

Produces a JSON array in LlamaFactory **Alpaca** format:

```json
[
  {
    "system":      "<student system prompt>",
    "instruction": "",
    "input":       "<rendered window + date>",
    "output":      "<filtered annotation JSON>"
  }
]
```

The output is the `annotation` JSON with intermediate/verbose fields stripped.
Defaults strip: `document_relevance`, `event_location_text`, `event_time_text`,
`affected_group`, `affected_entity`, `severity`.

---

#### Paragraph windowing (default)

News articles are split into overlapping paragraph windows so each training
example fits a small model's context. Each window's events are determined by
whether the event's grounding spans (`grounding_quote`, `event_location_text`,
`event_time_text`) lie physically inside that window's text.

**Key behaviours:**

- Windows are built greedily: paragraphs are accumulated until the next one
  would exceed `--max-chars` or `--max-paras` is already reached. Oversized
  paragraphs are split into complete sentence spans before windowing; a single
  sentence longer than `--max-chars` remains intact rather than being cut.
- Short windows are merged into a neighboring window when the merged window
  still fits. Remaining no-event windows shorter than `--min-chars` are dropped.
- Consecutive windows share `--overlap-paras` paragraphs.
- An event appearing in the overlap region is emitted in **every** window that
  contains it (mild duplication; correct by construction).
- An event whose location/time text falls outside the grounding-quote's paragraph
  causes the window to expand minimally only if the expanded window still fits
  `--max-chars` and `--max-paras`.
- Windows with no events are kept as negative examples (`"events": []`).

Window character and event statistics are printed after conversion. Pass
`--tokenizer` to also see token-count statistics.

**Windowing flags:**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-chars` | `3000` | Target char cap per window; complete sentences are not cut |
| `--min-chars` | `200` | Drop no-event windows shorter than this after merge attempts; `0` disables |
| `--max-paras` | `15` | Hard paragraph cap per window |
| `--overlap-paras` | `1` | Paragraphs shared between consecutive windows |
| `--no-window` | off | Disable windowing — one record per whole article |
| `--prompt-dir` | `prompts/student` | Directory containing `system_prompt.txt` and `user_prompt.txt` |
| `--ontology` | `ontologies/zhai/science.json` | Label list JSON; used to fill `{{ALLOWED_EVENT_TYPES}}` in the student system prompt |
| `--top-k-candidates` | None | Limit per-record candidate label list to the first N entries |

**Examples:**

```bash
# Default: 3000-char windows, 1-para overlap, student prompts
python scripts/data/generation_v3/to_sft.py \
  dataset/zhai/v3/silver.final.jsonl \
  dataset/zhai/v3/silver.sft.json

# Larger windows for a model with more context
python scripts/data/generation_v3/to_sft.py \
  dataset/zhai/v3/silver.final.jsonl \
  dataset/zhai/v3/silver.sft.json \
  --max-chars 6000 --overlap-paras 2

# Whole-article mode (no windowing) with teacher prompts
python scripts/data/generation_v3/to_sft.py \
  dataset/zhai/v3/silver.final.jsonl \
  dataset/zhai/v3/silver.sft.json \
  --no-window --prompt-dir scripts/data/generation_v3/prompts/teacher

# Strip only grounding-quote helper fields, keep document_relevance
python scripts/data/generation_v3/to_sft.py input.jsonl output.json \
  --ignore event_location_text event_time_text

# Report token counts (requires transformers)
python scripts/data/generation_v3/to_sft.py \
  dataset/zhai/v3/silver.final.jsonl \
  dataset/zhai/v3/silver.sft.json \
  --tokenizer Qwen/Qwen3-4B
```

---

Register the output in LlamaFactory's `dataset_info.json`:

```json
"zhai_v3_sft": {
  "file_name": "silver.sft.json",
  "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
}
```

---

## Steps not yet implemented

The following steps from the distillation plan are pending:

| Step | Description |
| --- | --- |
| 4 | Train/dev/test split by document id (80/10/10) |
| 6 | LlamaFactory LoRA training — Qwen3-4B, `lora_target all`, rank 32, lr 2e-4, 2 epochs |
| 7 | vLLM inference + eval (event-type P/R/F1, grounding rate, JSON-valid rate) |

The student prompt (`prompts/student/`) is implemented. It keeps the label list
(closed label set = genuine conditioning) but drops the verbose location/time/field
rules and few-shot examples — those become internalized from the training examples.
The student must use the **same prompt byte-for-byte at train and inference**.

---

## Smoke test

```bash
# Step 2 — validate
python scripts/data/generation_v3/validate.py \
  --input dataset/zhai/v3/sample.json \
  --output-stem /tmp/sample_validated

# Step 2b — fix (skip if no invalid events)
python scripts/data/generation_v3/fix_events.py \
  --input  /tmp/sample_validated.invalid.jsonl \
  --output /tmp/sample_fixed.jsonl \
  --model  gemini-3.1-pro-preview

# Step 2c — merge
python scripts/data/generation_v3/merge.py \
  --clean  /tmp/sample_validated.clean.jsonl \
  --fixed  /tmp/sample_fixed.jsonl \
  --output /tmp/sample_final.jsonl

# Step 3 — convert
python scripts/data/generation_v3/to_sft.py \
  /tmp/sample_final.jsonl \
  /tmp/sample.sft.json

# Inspect final output
python -c "
import json
data = json.load(open('/tmp/sample.sft.json'))
ev = json.loads(data[0]['output'])
print('records:', len(data))
print('events:', len(ev['events']))
print('keys:', list(ev['events'][0].keys()) if ev['events'] else [])
"
```
