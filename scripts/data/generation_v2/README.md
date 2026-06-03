# Event Extraction Data Generation V2

This directory defines a fresh, quality-first data generation pipeline for
token classification and event-location extraction from long news articles.

The goal is not to maximize cheap labels. The goal is a smaller set of labels
with consistent spans, exact offsets, clear negatives, and reviewable evidence.

## Current Recommendation

Use a staged LLM-assisted annotation pipeline:

1. Build a clean article pool.
2. Select balanced event-positive, hard-negative, and random-negative articles.
3. Split long articles into exact-offset windows.
4. Ask Gemini 3.1 Pro for structured JSON spans using ontology-derived
   guidelines and a small set of reviewed exemplars.
5. Validate every span deterministically against article text.
6. Run offset alignment and an independent verifier/adjudicator pass.
7. Recover and visualize accepted spans against the article text for review.
8. Review only uncertain/high-impact cases, then use the reviewed set to drive
   active sampling.

This is intentionally not a fully synthetic generation pipeline. For event
extraction, real article text plus LLM labels is safer than generated article
text because trigger wording, local context, and negative cases matter.

## Evidence Base

Recent work supports several design choices used here:

- Google LangExtract emphasizes exact source grounding, structured outputs,
  long-document chunking, parallel passes, and review visualization:
  https://github.com/google/langextract
- Gemini structured outputs support Pydantic/JSON Schema and Gemini 3.1 Pro
  structured output mode:
  https://ai.google.dev/gemini-api/docs/structured-output
- ACL 2025 work on event extraction with annotation guidelines shows that
  event type and argument role descriptions help LLM event extraction,
  especially for low-frequency event types:
  https://aclanthology.org/2025.findings-acl.677.pdf
- EMNLP 2025 work on large event type sets highlights two failure modes that
  matter here: role understanding and span-offset bias. It mitigates them with
  guidelines, manual examples, offset alignment, and voting:
  https://aclanthology.org/2025.emnlp-main.1743.pdf
- ACL 2024 NER dataset generation work reports malformed spans, unseen labels,
  missing spans, overlaps, and conflicting annotations as common LLM annotation
  failures. V2 treats those as deterministic rejection criteria:
  https://aclanthology.org/2024.findings-acl.947.pdf
- LREC-COLING 2024 active-learning work frames token classification annotation
  as a selection problem, not just a labeling problem:
  https://aclanthology.org/2024.lrec-main.30.pdf

## Data Contract

Each output record should keep article-level offsets:

```json
{
  "id": "article-id",
  "text": "Shelling damaged homes in northern Gaza.",
  "events": [
    {
      "event_type": "artillery bombing",
      "start": 0,
      "end": 8,
      "text": "Shelling",
      "arguments": [
        {
          "role": "location",
          "start": 35,
          "end": 39,
          "text": "Gaza",
          "location_type": "other"
        }
      ]
    }
  ],
  "locations": [
    {
      "start": 35,
      "end": 39,
      "text": "Gaza",
      "location_type": "other"
    }
  ],
  "negatives": {
    "has_target_event": true,
    "negative_reason": null
  },
  "metadata": {
    "annotation_model": "gemini-3.1-pro-preview",
    "pipeline_version": "generation_v2"
  }
}
```

`events[*].arguments` contains only locations linked to that event. `locations`
contains all location mentions, including mentions not linked to an event.

The current Zhai ontology does not include `air attack`. If `drone attacks`
must be a first-class event, add it to `ontologies/zhai/ontology.json` before
generation. Otherwise v2 should map only to an existing ontology label when the
definition is genuinely supported, or omit the event.

## Sampling Strategy

Start with a curated article pool:

- Keep English news articles with title, date, source URL, and at least 500
  characters of article text.
- Remove boilerplate, duplicate syndicated articles, near duplicates, live-blog
  fragments, stock tickers, and articles dominated by tables.
- Prefer reputable sources or known source lists if available; otherwise rank
  by text quality signals: paragraph count, sentence count, low boilerplate
  ratio, low all-caps ratio, and few malformed characters.

Sample four default buckets:

- `event_seeded_positive`: articles retrieved by ontology terms, synonyms,
  locations, and event-specific trigger lexicons.
- `hard_negative`: articles near an event domain but with no target event, such
  as political analysis mentioning a country but no violence/disaster outcome.
- `sibling_negative`: articles containing a nearby ontology event but not the
  candidate label, such as "heavy rainfall" vs "flooding".
- `random_negative`: clean news articles sampled without event retrieval.

Default target mix:

- 50% event-seeded positives.
- 20% hard negatives.
- 15% sibling negatives.
- 15% random negatives.

For food-insecurity focused collection, pass `--keyword`. This adds a
`keyword_risk_factor` bucket, a `keyword_quality_score` field, and a weighted
lexicon derived from the ontology plus direct food-insecurity terms such as
`food insecurity`, `food shortages`, `famine`, `malnutrition`, `food aid`,
`crop failure`, `drought`, `food prices`, and related access/market shocks.
Keyword records are sorted by `keyword_quality_score` before selection.

Keyword target mix:

- 30% keyword risk-factor articles.
- 35% event-seeded positives.
- 15% hard negatives.
- 10% sibling negatives.
- 10% random negatives.

After the first model is trained, switch to active sampling:

- Add high-disagreement Gemini cases.
- Add cases where the trained tagger has high entropy.
- Add documents with predicted rare event types.
- Add documents where event trigger and location are far apart.
- Add clean negatives that the model incorrectly predicts as positive.

## Long Article Strategy

Do not ask the LLM to annotate a long article in one pass unless it fits with
large margin. Use exact text windows:

- Segment by paragraph/sentence boundaries without normalizing text.
- Target 4k-8k characters per window with 1-2 sentence overlap.
- Ask for window-relative offsets, then project to article offsets.
- Keep only spans whose trigger is inside the non-overlap core window.
- Allow arguments outside the core window when they are visible in overlap.
- Merge duplicate events by `(event_type, start, end)`.

For very long articles, add a cheap routing pass before annotation:

- Identify candidate windows with possible event mentions.
- Always include a small random subset of apparently negative windows.
- Always run all-location extraction over every window, because locations can
  be relevant even when no event is present.

## Annotation Passes

Use three passes:

1. `extract`: Gemini 3.1 Pro extracts events and all locations with structured
   JSON.
2. `align`: deterministic code verifies copied text and repairs offsets only
   when there is exactly one nearby match.
3. `verify`: a separate Gemini call receives the original text window plus
   candidate JSON and returns keep/drop/fix decisions.

For expensive batches, use a cheaper model only for retrieval/routing. Do not
use a weaker model for final span annotation until measured against reviewed
data.

## Span Rules

See `annotation_guidelines.md` for the rules that should be put directly in
the LLM prompt and used by human reviewers.

High-priority rules:

- Trigger spans must be the shortest natural phrase that names the event.
- Location spans must be exact text copies from the article.
- Offsets must index the article body text, not title or prompt text.
- Do not infer a location that is not textually mentioned.
- Do not annotate generic background facts unless the article asserts the event
  as occurring, worsening, expected, threatened, or affecting people/assets.
- For token classification, nested or overlapping spans are rejected unless a
  later modeling format explicitly supports them.

## Review Policy

Treat the existing 30 reviewed samples as calibration examples, not gold.

Human review should prioritize:

- All new event types before scale-up.
- All rare event types with fewer than 20 accepted examples.
- Disagreements between extract and verify passes.
- Offset repairs.
- Sibling-label confusion.
- Negative records where the verifier says an event might exist.

Stop scaling until reviewed precision on a 100-record audit is acceptable:

- Event trigger exact-match precision >= 0.90.
- Location exact-match precision >= 0.95.
- Event-argument link precision >= 0.85.
- Negative false-positive rate <= 0.05.

## Implementation Plan

1. `schema.py`: Pydantic models for records, events, locations, and metadata.
2. `sample_articles.py`: score and bucket article candidates.
3. `window_articles.py`: exact-offset windowing.
4. `annotate_gemini.py`: structured Gemini extraction with worker and Batch API
   modes.
5. `verify_gemini.py`: independent candidate verification with worker and Batch
   API modes.
6. `recover_annotations.py`: re-ground accepted spans and generate review
   JSONL/HTML highlights.
7. `audit_report.py`: label coverage, status, API-mode, and token-usage
   reporting.

## How to Run

Run commands from the repository root.

Set Gemini credentials before annotation or verification:

```bash
export GEMINI_API_KEY="..."
```

Or put `GEMINI_API_KEY=...` in `.env`; the Gemini CLIs load `.env` by default.

Input should be JSONL or JSON records with at least:

```json
{"id": "article-1", "title": "Optional title", "text": "Full article body..."}
```

The loader also accepts `body` instead of `text`, plus common metadata fields
such as `source_url`, `url`, `publish_date`, and `published_at`.

### One-Command Pipeline

Use the wrapper script for a complete small run:

```bash
bash scripts/data/generation_v2/run_pipeline.sh \
  dataset/gdelt/raw/food_security_2020_2025_100.jsonl \
  output/generation_v2/example_run \
  --limit 20 \
  --keyword \
  --workers 4
```

Use Batch API for larger, non-urgent runs:

```bash
bash scripts/data/generation_v2/run_pipeline.sh \
  dataset/gdelt/raw/food_security_2020_2025_100.jsonl \
  output/generation_v2/batch_run \
  --limit 1000 \
  --batch-api \
  --batch-size 500
```

The output directory contains:

- `sampled.jsonl`: selected article records with `source_bucket`,
  `quality_score`, and, when `--keyword` is used, `keyword_quality_score`.
- `windows.jsonl`: exact-offset article windows.
- `raw.jsonl`: extracted annotations.
- `verified.jsonl`: verifier-filtered annotations.
- `recovered.jsonl`: source-grounded recovered spans.
- `review.html`: human-review highlighting page.
- `report/summary.json`: label/status/API-mode/token usage report.
- `runs/`: Batch API request files, job metadata, and raw response files.

The second argument is an output directory, not a single output JSONL file. If
it ends in `.jsonl`, `run_pipeline.sh` strips that suffix and uses the remaining
path as the directory.

### Step-by-Step Run

```bash
uv run python scripts/data/generation_v2/sample_articles.py INPUT_JSONL sampled.jsonl
uv run python scripts/data/generation_v2/sample_articles.py INPUT_JSONL sampled.food.jsonl --keyword
uv run python scripts/data/generation_v2/window_articles.py sampled.jsonl windows.jsonl
uv run python scripts/data/generation_v2/annotate_gemini.py windows.jsonl raw.jsonl --workers 8
uv run python scripts/data/generation_v2/annotate_gemini.py windows.jsonl raw.jsonl --batch-api --batch-size 1000
uv run python scripts/data/generation_v2/verify_gemini.py raw.jsonl verified.jsonl --workers 8
uv run python scripts/data/generation_v2/recover_annotations.py verified.jsonl recovered.jsonl --html review.html
uv run python scripts/data/generation_v2/audit_report.py verified.jsonl report_dir
```

The sampler exposes `--ontology`, `--limit`, `--seed`, `--keyword`, and
`--overwrite`.

Gemini scripts expose `--temperature`, `--reasoning-effort`, `--max-tokens`,
`--workers`, `--batch-api`, `--batch-size`, `--batch-display-name`,
`--batch-poll-interval-seconds`, `--run-dir`, `--overwrite`, and
`--retry-failed`. Token usage is normalized to `input_tokens`, `output_tokens`,
`cached_input_tokens`, and `thoughts_tokens`.

### Recommended Settings

For a small quality-check run:

```bash
bash scripts/data/generation_v2/run_pipeline.sh INPUT_JSONL output/generation_v2/quality_check \
  --limit 20 \
  --keyword \
  --workers 2 \
  --temperature 0.0 \
  --reasoning-effort low
```

For a larger non-urgent run where cost matters:

```bash
bash scripts/data/generation_v2/run_pipeline.sh INPUT_JSONL output/generation_v2/bulk \
  --limit 1000 \
  --batch-api \
  --batch-size 500 \
  --batch-poll-interval-seconds 60
```

Use `--skip-verify` only for debugging. The verifier is part of the intended
quality gate.

### Resume and Overwrite

The wrapper is resume-first. Without `--overwrite`, existing deterministic
outputs (`sampled.jsonl`, `windows.jsonl`, `recovered.jsonl`, and
`review.html`) are reused. Annotation and verification append missing rows and
the wrapper passes `--retry-failed` so rows with `status="error"` are retried.

Use `--overwrite` to recreate pipeline outputs in the output directory.

```bash
bash scripts/data/generation_v2/run_pipeline.sh INPUT_JSONL output/generation_v2/example --overwrite
```

## Success Criteria

Before a full run:

- `schema.py` rejects malformed labels, offsets, and overlapping token spans.
- A 20-article dry run produces JSON that round-trips through validation.
- A human audit of 100 accepted records meets the review policy thresholds.
- The dataset contains explicit negative records and rare-label coverage.
- Every accepted span can be highlighted exactly in the original article text.
