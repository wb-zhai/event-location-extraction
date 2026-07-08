# Annotation

Two annotation workflows, on two different tools, sharing a similar `push`/`export`
CLI shape:

- **Relevance** ([`relevance_argilla.py`](relevance_argilla.py), on **Argilla**) —
  is an article relevant / irrelevant to food-security risk-event extraction, to
  build a gold set for scoring the Gemini relevance gate in
  [`relevance_filter.py`](../data/relevance/relevance_filter.py). A single
  label per document, which Argilla's `LabelQuestion` handles well.
- **Events** ([`events_label_studio.py`](events_label_studio.py), on **Label
  Studio**) — validate/correct/add to the structured `annotation.events` list
  extracted for each article. Each event is a text span (`grounding_quote`)
  plus 9 structured fields, which needs a per-span structured-detail UI —
  see [below](#why-label-studio-for-events) for why this moved off Argilla.
  The older Argilla-based version, [`events_argilla.py`](events_argilla.py),
  still works but is superseded by the Label Studio one.

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

## Why Label Studio for events

Argilla has no question type for "one span, with N structured fields attached
to it" — so `events_argilla.py` has to split an event's `event_type` +
`grounding_quote` into a `SpanQuestion`, and its other 9 fields into a
hand-edited JSON blob (`event_details_json`) matched back to the span
*positionally* (by article order). That's fragile: annotators can misalign
entries, and the export step has to detect and flag span/detail count
mismatches (`events_merge_warnings`).

Label Studio's `Labels` control tag supports `perRegion` sibling controls
(`TextArea`, `Choices`) that only apply to whichever span is currently
selected, and are stored keyed to that span's own id — no JSON typing, no
positional matching, and add/remove-event follows directly from
highlight/delete-span. `events_label_studio.py` uses this instead.

## 4. Set up Label Studio

**Run the server (Docker):**

```bash
docker run -d --name label-studio -p 8080:8080 \
  --restart unless-stopped \
  -v label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest
```

The `-v` flag persists projects/tasks/annotations in the `label-studio-data`
volume, so (as with the Argilla container above) you can `docker rm` and
recreate the container without losing data; `--restart unless-stopped` brings
it back up after a Docker/host restart.

Open http://localhost:8080 and create an account (this just creates a local
user in your own container — nothing external). Then:

1. Click your user icon (top right) → **Account & Settings**.
2. Open **API Tokens Settings**. If neither token type is listed yet, enable
   **Legacy Tokens** for your organization here first.
3. Under **Legacy Token**, copy the token. (Legacy tokens are static and don't
   expire, unlike Personal Access Tokens which need a refresh step — simpler
   for scripted use here.)

Add to `.env`:

```
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<your legacy token>
```

## 5. Events: push articles for annotation

Input is JSONL with records shaped like
`dataset/db/relevance/matrix_5M.sample_1000.3.1pro.extracted.jsonl`: each
record has `title`/`text` or `source.title`/`source.text`, an existing
`relevance` block (only `relevant`/`partially_relevant` records are pushed —
see `keep_by_relevance`), and an `annotation` block
(`{"document_relevance": ..., "events": [...]}`) produced by
[`generate.py`](../data/generation_v3/generate.py).

```bash
python scripts/annotations/events_label_studio.py push \
  --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.extracted.jsonl \
  --project-name events-review \
  --limit 50
```

The first `push` to a given `--project-name` creates the Label Studio project,
with a labeling config generated from `ontologies/zhai/science.json` by default
(`--ontology` can point to another ontology shaped as `{"events": {...}}`).
The event types are searchable via a filter box — see `build_label_config` —
and a condensed version of the teacher system prompt is used as the project's
instructions (the "?" help icon in the labeling UI — see `load_guidelines`).

Each article's model-extracted events are pushed as **predictions**: wherever
an event's `grounding_quote` is found verbatim in the article text and its
`event_type` is present in the ontology, it's pre-filled as a highlighted span
plus its 9 detail fields, ready to review/correct rather than label from
scratch. Events whose quote isn't found verbatim, or whose model label is not
in the ontology, are listed in a small note above the article text so the
annotator can add them manually if still valid.

Re-running `push` on the same `--project-name` skips records already present
in the project (matched by id — see `_record_key`); pass `--force` to push
them again anyway (Label Studio has no upsert, so this adds duplicate tasks).

Text is **not truncated by default** (`--max-chars 0`) — span offsets are
computed against whatever text is actually shown, so truncating can push
later events' quotes out of reach (about half the articles in the sample
dataset exceed the old 4000-char default).

In the browser at http://localhost:8080: open the project, click a task,
highlight each event mention in the text and tag it with `event_type` from the
list on the left (type to filter), then click the highlighted span to open its
detail panel below the text and fill in the 9 remaining fields. Delete a span
to remove that event. Set `document_relevance` at the bottom and submit.

## 6. Events: export human-corrected events

```bash
python scripts/annotations/events_label_studio.py export \
  --project-name events-review \
  --output events_annotations.jsonl \
  --only-submitted
```

Each output row has a human-corrected `annotation.events` list (reconstructed
by grouping each span's result with its per-region detail fields by their
shared Label Studio region id — see `parse_annotation_result`) and the
original `model_annotation` for diffing. Export is strict: only rows whose
events match the schema, use ontology event types, have valid enum values, and
keep `grounding_quote`/`event_location_text`/`event_time_text` verbatim in the
article are written to `--output`. Invalid rows are written to
`<output stem>.invalid<suffix>` by default, or to `--invalid-output`, with
`events_validation_errors` explaining what needs review.
