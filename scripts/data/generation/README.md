# Event IE Data Generation

This directory contains the Gemini-based data generation pipeline for event
information extraction. It labels real news articles with event triggers,
argument spans, exact character offsets, and run metadata.

The main entrypoint is:

```bash
uv run python scripts/data/generation/gemini_event_gen.py INPUT_JSONL OUTPUT_JSONL
```

## How It Works

The pipeline is intentionally small and deterministic around the LLM call.

1. Load input records from JSON or JSONL.
2. Normalize each record to `id`, `title`, `text`, `source_url`, and
   `publish_date`.
3. Load the ontology from `ontologies/risk-factors/risk.cluster.description.json`.
4. For long articles, split text into sentence-aware windows with overlap.
5. Ask Gemini for structured annotations.
6. Validate labels, argument roles, copied text, and character offsets.
7. Optionally run self-consistency and keep majority-supported annotations.
8. Optionally run a verifier pass for `events-with-args`.
9. Write accepted annotations to JSONL.
10. Write a JSON report with counts, token usage, and synthetic gap requests.

The accepted event schema stays stable:

```json
{
  "event_type": "weather shocks",
  "trigger_text": "drought crisis",
  "start_char": 120,
  "end_char": 134,
  "arguments": [
    {
      "role": "location",
      "text": "Somalia",
      "start_char": 142,
      "end_char": 149
    }
  ]
}
```

Offsets always index the article `text`, not the title.

## Files

- `gemini_event_gen.py`: CLI orchestration and Gemini calls.
- `windowing.py`: sentence-aware windows with exact offset preservation.
- `validation.py`: deterministic label, role, duplicate, and offset checks.
- `adjudication.py`: self-consistency merge and verifier filtering.
- `reports.py`: output summaries and synthetic gap request reporting.
- `pipeline_config.py`: small mode/config loader.
- `ui.py`: local browser UI for inspecting generated JSONL files.

## Input Format

JSONL input should contain one object per line. The loader accepts `text` or
`body`, and uses `title` only as context.

```json
{"id":"a1","title":"Drought worsens","text":"Drought hit Somalia after poor rains.","source_url":"https://example.test/a1","publish_date":"2025-01-10"}
```

For JSON files, the top-level value may be a list of records or an object with
`records`, `samples`, `articles`, or `data`.

## Environment

Gemini credentials are loaded through the existing LLM client. In normal use,
put credentials in `.env` or export them in the shell:

```bash
export GEMINI_API_KEY=...
```

The script loads `.env` by default. Use `--env-file PATH` to override it.

## Modes

Modes set practical defaults before explicit CLI flags are applied.

### `quality_first`

Default mode.

- `events-with-args`
- strict offsets
- long-document windowing
- 5 self-consistency samples
- verifier enabled
- synthetic gap requests in the report

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl
```

### `balanced`

Lower cost, still majority-voted.

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --mode balanced
```

### `low_cost`

Single teacher pass, no verifier, no synthetic gap report.

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --mode low_cost
```

## Common Runs

Process only the first 10 records:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.first10.jsonl \
  --limit 10
```

Retry only failed records from an existing output:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --retry-failed
```

Overwrite an output file:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --overwrite
```

Disable verifier while keeping quality-first self-consistency:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --no-enable-verifier
```

Generate flat spans instead of linked events:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/spans.generated.jsonl \
  --output-mode spans
```

## Windowing

Long-document mode uses sentence-aware windows instead of splitting only on
blank lines. The LLM sees one window at a time and returns offsets relative to
that window. The code projects those offsets back to article-level offsets.

Useful knobs:

```bash
--long-document-threshold-chars 1000
--window-target-chars 6000
--window-max-chars 9000
--window-overlap-sentences 2
```

Use smaller windows for debugging:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/debug.windows.jsonl \
  --limit 3 \
  --window-target-chars 1200 \
  --window-max-chars 1800
```

## Config File

You can pass a small JSON file or simple `key: value` config file:

```yaml
mode: balanced
model: gemini-2.5-flash
teacher_samples: 3
teacher_temperature: 0.5
enable_verifier: false
window_target_chars: 5000
window_max_chars: 8000
window_overlap_sentences: 2
```

Run with:

```bash
uv run python scripts/data/generation/gemini_event_gen.py \
  dataset/input.jsonl \
  dataset/events.generated.jsonl \
  --config scripts/data/generation/generation_pipeline.yaml
```

Explicit CLI flags override mode/config defaults.

## Output Files

The main output is JSONL. Each line contains:

- `id`
- `status`
- `source`
- `events` or `spans`
- `llm` metadata

A report is written next to the output by default:

```text
dataset/events.generated.report.json
```

The report includes:

- total records
- status counts
- event counts
- role counts
- token usage
- synthetic gap requests when enabled

Use `--report PATH` to choose a different report path.

## Inspecting Results

Use the local UI to inspect spans/events and compare files:

```bash
uv run python scripts/data/generation/ui.py
```

Then open the printed local URL in a browser.

## Tests

Run the focused generation tests:

```bash
uv run pytest tests/test_gemini_event_gen.py
```

The full test suite may require optional training dependencies such as
`unsloth`.
