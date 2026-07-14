# Annotation

Three annotation workflows, all on **Argilla**, sharing a similar `push`/`export`
CLI shape:

- **Relevance** ([`relevance_argilla.py`](relevance_argilla.py)) — is an
  article relevant / irrelevant to food-security risk-event extraction, to
  build a gold set for scoring the Gemini relevance gate in
  [`relevance_filter.py`](../data/relevance/relevance_filter.py). A single
  label per document, which Argilla's `LabelQuestion` handles well.
- **Events** ([`events_argilla.py`](events_argilla.py)) — validate/correct/add
  to the structured `annotation.events` list extracted for each article. Each
  event's 8 fields are edited as a single hand-edited JSON blob
  (`event_details_json`) rather than per-span highlighting — the simplest UI
  for annotators: one field to read, one field to edit.
- **Relevance comparison** ([`relevance_comparison_argilla.py`](relevance_comparison_argilla.py))
  — given two relevance-labeled JSONL files covering the same articles (e.g.
  two prompt versions), show both predictions side by side and record which
  one a human prefers. Defaults to only pushing articles where the two
  predictions disagree.

## 1. Run a local Argilla server (Docker)

```bash
docker run -d --name argilla -p 6900:6900 \
  --restart unless-stopped \
  -v argilla-data:/var/lib/argilla \
  -v argilla-es-data:/usr/share/elasticsearch/data \
  argilla/argilla-quickstart:latest
```

The `-v` flags mount named volumes for Argilla's own database (`/var/lib/argilla`)
and the bundled Elasticsearch index (`/usr/share/elasticsearch/data`). Without
them, all pushed datasets and annotations are stored only in the container's
writable layer and are lost as soon as the container is removed (`docker rm`) —
`docker stop`/`docker start` alone is fine either way. With the volumes, you can
freely `docker rm` and recreate the container and your data survives.

`--restart unless-stopped` makes the container come back up automatically
whenever the Docker daemon/VM restarts (e.g. Docker Desktop relaunching or the
host rebooting), without needing to manually re-run `docker run`. To apply it
to an already-running container instead of recreating it:

```bash
docker update --restart unless-stopped argilla
```

On macOS, Docker only runs inside Docker Desktop's own VM, so that VM has to
be up too — enable *Docker Desktop → Settings → General → "Start Docker
Desktop when you sign in to your computer"* if you also want it to launch on
login rather than manually.

Open http://localhost:6900 and log in (default user `argilla`, default password
`12345678`). Grab an API key from the user settings page.

Add to `.env`:

```
ARGILLA_API_URL=http://localhost:6900
ARGILLA_API_KEY=<your api key>
```

## 2. Relevance: push articles for annotation

Input is the same JSONL format used by `relevance_filter.py` (records with
`title`/`text` or `source.title`/`source.text`, optionally an existing
`relevance` block). When a record already has a Gemini `relevance` prediction,
it is attached as a suggestion so annotators review/correct it rather than
labeling from scratch.

```bash
python scripts/annotations/relevance_argilla.py push \
  --input dataset/db/matrix_5M.test_sample_200.relevance.jsonl \
  --dataset-name relevance-review \
  --limit 50
```

The first `push` to a given `--dataset-name` creates the Argilla dataset, with
a single `relevance` label question (see `build_settings`) and human-readable
guidelines rewritten from `DEFAULT_RELEVANCE_SYSTEM_PROMPT_3LABEL` (see
`load_guidelines` in `relevance_argilla.py`) so annotators apply the same
criteria as the Gemini gate.

Annotate in the browser at http://localhost:6900. Re-running `push` on the
same `--dataset-name` upserts records (keyed by article id/url), so it's safe
to run repeatedly as new data arrives.

## 3. Relevance: export human labels

```bash
python scripts/annotations/relevance_argilla.py export \
  --dataset-name relevance-review \
  --output annotations.jsonl \
  --only-submitted
```

Each output row has `human_relevance`, `gemini_relevance`, and an `agreement`
flag, ready for scoring the Gemini gate's precision/recall against human
judgment.

## 4. Events: push articles for annotation

Input is JSONL with records shaped like
`dataset/db/relevance/matrix_5M.sample_1000.3.1pro.extracted.jsonl`: each
record has `title`/`text` or `source.title`/`source.text`, an existing
`relevance` block (only `relevant`/`partially_relevant` records are pushed —
see `keep_by_relevance`), and an `annotation` block
(`{"document_relevance": ..., "events": [...]}`) produced by
[`generate.py`](../data/generation_v3/generate.py).

```bash
python scripts/annotations/events_argilla.py push \
  --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.extracted.jsonl \
  --dataset-name events-review \
  --limit 50
```

The first `push` to a given `--dataset-name` creates the Argilla dataset, with
a single `event_details_json` text question (see `build_settings`) and
condensed guidelines drawn from the teacher system prompt (see
`load_guidelines`). Pass `--update-settings` to push a schema/guidelines
update to an existing dataset.

Each article's model-extracted events are pre-filled as a JSON **suggestion**
in `event_details_json` (see `build_record`), ready to review/correct rather
than annotate from scratch.

Re-running `push` on the same `--dataset-name` upserts records (keyed by id —
see `_record_key`), so it's safe to run repeatedly as new data arrives.

Text is **not truncated by default** (`--max-chars 0`).

In the browser at http://localhost:6900: open the dataset, read the article,
and edit the `event_details_json` field directly — fix values on existing
entries, delete entries that aren't valid events, or add new entries for
events the model missed. Submit `[]` if the article has no valid events.

## 5. Events: export human-corrected events

```bash
python scripts/annotations/events_argilla.py export \
  --dataset-name events-review \
  --output events_annotations.jsonl \
  --only-submitted
```

Each output row has the human-corrected `annotation.events` list (parsed from
the submitted `event_details_json`) and the original `model_annotation` for
diffing. Export is strict: only rows whose events match the schema, use
ontology event types, have valid enum values, and keep
`grounding_quote`/`event_location_text`/`event_time_text` verbatim in the
article are written to `--output`. Invalid rows are written to
`<output stem>.invalid<suffix>` by default, or to `--invalid-output`, with
`events_validation_errors` explaining what needs review.

## 6. Relevance comparison: push two predictions for the same articles

Input is two JSONL files in the same format as `relevance_argilla.py push`
(records keyed by `id`, with a `relevance.decision` block), covering the same
set of articles — e.g. the same sample scored under two different prompt
versions. Only articles present in both files are considered, and by default
only the ones where the two `relevance.decision` values **disagree** are
pushed (pass `--include-agreements` to push everything).

```bash
python scripts/annotations/relevance_comparison_argilla.py push \
  --input-a dataset/db/relevance/matrix_5M.sample_1000.3.1pro.v3.jsonl \
  --input-b dataset/db/relevance/matrix_5M.sample_1000.3.1pro.old_prompt.jsonl \
  --name-a v3 \
  --name-b old_prompt \
  --dataset-name relevance-v3-vs-old-prompt
```

`--name-a`/`--name-b` (defaulting to each file's stem) are shown as field
titles (`Prediction A: v3`, `Prediction B: old_prompt`) and in the
`preference` question's label text, so annotators know which article
preview/prediction came from which source. The first `push` to a given
`--dataset-name` creates the dataset with a `title`/`text`/`prediction_a`/
`prediction_b` field layout and a single `preference` label question
(`prefer_a` / `prefer_b` / `tie` / `neither`). Pass `--update-settings` to
push a schema/guidelines update (e.g. after changing `--name-a`/`--name-b`)
to an existing dataset.

Annotate in the browser at http://localhost:6900. To filter to only the
disagreement cases already pushed, use Argilla's metadata filter on
`agreement` (`disagree`); if you pushed with `--include-agreements`, that
filter also lets you drill into agreement cases.

## 7. Relevance comparison: export preferences

```bash
python scripts/annotations/relevance_comparison_argilla.py export \
  --dataset-name relevance-v3-vs-old-prompt \
  --output relevance_preferences.jsonl \
  --only-submitted
```

Each output row has the submitted `preference` (`prefer_a`/`prefer_b`/`tie`/
`neither`) plus `decision_a`/`confidence_a`/`decision_b`/`confidence_b` from
each input file's prediction, for tallying which prompt version humans
preferred.
