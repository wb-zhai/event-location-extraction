# event-location-extraction

Extracts food-insecurity risk-factor events and their locations from news articles.
Articles are sampled from the article database, filtered for relevance, run through
an LLM-distilled extraction model, and geocoded into administrative regions —
producing `article_uri` ↔ `risk_factor` and `article_uri` ↔ `adm_code` tables ready
for DB ingestion.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv venv --python 3.13
source .venv/bin/activate

uv pip install -r requirements.txt

cp .env.example .env   # fill in SQL_* (DB), GEMINI_API_KEY/PROJECT_ID, ARGILLA_*
```

`requirements.txt` pins `vllm` with a `sys_platform != "darwin"` marker, so `uv pip install`
installs it everywhere except macOS (no CUDA there) automatically — nothing to toggle by hand.

Re-run `uv pip install -r requirements.txt` any time `requirements.txt` changes to keep the venv
in sync.

## Pipeline

This repo builds and runs **two separately-trained models** that share the same DB
sample and Gemini-teacher pattern but are otherwise independent pipelines.
Each needs its own data-generation and training step before it's usable;
they only meet at the very end, where the relevance filter gates what the
event-extraction model sees:

- **Relevance filter** — a small ModernBERT classifier (`scripts/relevance/`) that
  cheaply flags whether an article can contain a food-insecurity event at all.
  - *Data generation* (`scripts/relevance/relevance_filter.py`) — Gemini labels each
    sampled article `relevant`/`irrelevant`, producing the classifier's training set.
  - *Training* (`scripts/relevance/train.py`) — fine-tunes the ModernBERT classifier on
    those Gemini-labeled judgments.
- **Event extraction** — the small student model (Qwen3.5-4B) that pulls out
  risk-factor events and their locations.
  - *Data generation* (`scripts/event_extraction/generation/`) — a one-time, offline
    step that runs a Gemini teacher over a small stratified DB sample to *produce the
    SFT training dataset*. That sample is **not relevance-filtered** and deliberately
    includes negatives (articles with no risk-factor tag) alongside positives, so the
    teacher (and thus the training set) also covers the *no events* case, not just
    articles known to contain an event.
  - *Training* (`scripts/train/llamafactory/train.sh`, or as a Vertex AI custom job via
    `vertexai/train/event-extraction/`) — fine-tunes Qwen3.5-4B on that generated
    dataset.

```text
Sample from DB (download/)
   │
   ├─→ RELEVANCE FILTER
   │     Data gen (relevance/relevance_filter.py)      [Gemini relevant/irrelevant labels]
   │       └─→ Training (relevance/train.py)           [→ ModernBERT classifier]
   │             └─→ Inference at scale                [gates the full ~130M-article DB]
   │                 (vertexai/inference/relevance/)                  │
   │                                                                  │
   └─→ EVENT EXTRACTION                                               │
         Data gen (event_extraction/generation/)       [Gemini teacher, pos/neg sample]
           └─→ Training (train/llamafactory/, or       [→ Qwen3.5-4B student]
                 vertexai/train/event-extraction/)
                 └─→ Inference at scale  ◄──────────────────────────────┘
                     (vertexai/inference/event-extraction/)
                     [runs the trained model on every article that passes the filter]
                             │
                             ▼
                        Geocoding (geocoding/)
```

1. **Sample from the DB** — pull a stratified article sample, incl. a positive/negative
   split for training data (`scripts/download/`).
2. **Relevance filter — data generation** — label the sampled articles
   `relevant`/`irrelevant` with Gemini (`scripts/relevance/relevance_filter.py`).
3. **Relevance filter — training** — fine-tune the ModernBERT classifier on those
   labels (`scripts/relevance/train.py`).
4. **Event extraction — data generation** *(offline — produces the training dataset,
   not the model)* — distill Gemini teacher labels over the sampled positives **and**
   negatives into an SFT dataset (`scripts/event_extraction/generation/`).
5. **Event extraction — training** — fine-tune the small extraction model on that SFT
   dataset, either locally (`scripts/train/llamafactory/`) or as a Vertex AI custom job
   (`vertexai/train/event-extraction/`).
6. **Relevance filter — inference at scale** — cheaply drop articles that can't contain
   a food-insecurity event before spending inference budget on them; gates the
   production run only, not data generation (`vertexai/inference/relevance/`).
7. **Event extraction — inference at scale** — run the already-trained model over the
   entire filtered DB (`vertexai/inference/event-extraction/`).
8. **Geocoding** — resolve each extracted `event_location` string to coordinates and
   an administrative region, ready for DB ingestion (`scripts/geocoding/`).

---

## 1. Sampling from the DB

Two stratified samplers in `scripts/download/`, both reading connection params
from the repo-root `.env` (`SQL_HOST`/`SQL_PORT`/`SQL_DATABASE`/`SQL_USERNAME`/`SQL_PASSWORD`)
and requiring `psycopg2`. Both avoid ever scanning `article_concept_association`
(2.3B rows / 470 GB) directly — candidate discovery goes through chunked, indexed
probes instead, run in parallel across `--workers` connections, with clean Ctrl+C
cancellation of in-flight server-side queries.

There are two separate scripts because French articles have no risk-factor tags in the
DB (`article_risk_factor_tags` is English-only), so the positive/negative balancing that
`from_db_matrix_v2.py` does isn't possible for French — `sample_french_by_country.py`
drops that split and just samples/stratifies by country.

### `from_db_matrix_v2.py` — positive/negative risk-factor sample

Downloads a country-stratified sample split into **positives** (risk-factor-tagged,
`article_risk_factor_tags`) and **negatives** (geo-tagged but untagged), for training
data. Country stratification is round-robin, rarest-country-first; positives are
additionally prioritized by risk-factor diversity.

```bash
PYTHONPATH=. python scripts/download/from_db_matrix_v2.py \
    --n 50000 --pos-ratio 0.7 --output dataset/matrix_sample.jsonl
```

Key flags: `--pos-ratio` (default 0.7), `--language` (default `eng`), `--start-month`/
`--end-month` (default: full history to today), `--tag-method-id`, `--workers`, `--seed`.

### `sample_french_by_country.py` — single-language country sample

Same discovery/stratification approach without the positive/negative split — samples
N articles in one language (default `fra`), stratified by country. Supports `--resume`
for interrupted runs (content-fetch chunks are flushed to disk as they land
and matched by article id on restart).

```bash
PYTHONPATH=. python scripts/download/sample_french_by_country.py \
    --n 500 --output dataset/french_sample.jsonl
```

Key flags: `--language` (default `fra`), `--start-month`/`--end-month`, `--workers`, `--seed`, `--resume`.

Both write JSONL locally or to `gs://bucket/path`, and support a small `--start-month`-bounded
run for smoke-testing output shape before a large download.

---

## 2. Relevance filter

Pre-filters articles before the expensive extraction step. Composes a cheap
keyword/regex first pass with an optional LLM (Gemini) second pass; the trained
encoder classifier is the cheap, high-throughput option once labeled data exists.
See [`scripts/relevance/README.md`](scripts/relevance/README.md) for full detail.

This filter gates **inference at scale** (§3, the whole-DB production run) only. It is
not applied ahead of data generation — that training pipeline uses its own DB sample
and needs both relevant and irrelevant articles (see §3 below).

Some notes from Notion:

- https://app.notion.com/p/Relevance-Filter-3964fc796fae8004b69efd501788a830 

### Data generation — `scripts/relevance/`

`relevance_filter.py` labels articles `relevant`/`not relevant`, optionally cascading
a cheap model's positive calls to a stronger one for cheaper high-quality labels:

```bash
PYTHONPATH=. python scripts/relevance/relevance_filter.py \
    --input dataset/zhai/v3/articles.jsonl \
    --output dataset/zhai/v3/articles.filtered.jsonl \
    --use-llm --cascade --model gemini-2.5-flash --cascade-model gemini-3.1-pro-preview \
    --filter-only
```

Related tools in the same directory: `local_llm.py` (test a local llama.cpp-served
model as a cheap stand-in), `encoder.py` (local encoder classifier, no LLM cost),
`view_relevance.py` (Gradio browser for labeled output), `agreement.py` (compare two
labeled files — accuracy/kappa).

### Train the relevance model — `scripts/relevance/train.py`

Fine-tunes a binary sequence classifier (default `answerdotai/ModernBERT-base`) on
`relevance_filter.py`'s labeled output:

```bash
PYTHONPATH=. python scripts/relevance/train.py \
    --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \
    --output-dir /tmp/relevance-modernbert
```

`inference.py` (HF or vLLM backend) and `eval.py` (accuracy/precision/recall/F1
against labeled data) round out local iteration on a checkpoint.

### Inference at scale — `vertexai/inference/relevance/`

Classifies the **entire** `article_downloads` table (~130M articles) with a trained
checkpoint via a GCP Cloud Batch job running vLLM's pooling `classify()`, reading
articles from GCS (not the DB) so it scales horizontally with zero DB load. Fully
self-contained — no imports from the rest of the repo. Output is `id,label` CSV
shards, no merge step (read downstream with a wildcard).

```bash
cd vertexai/inference/relevance
cp .env.example .env   # fill in GCP_PROJECT, MANIFEST_GCS_PREFIX, MODEL_GCS, ...
python build_manifest.py --output-prefix "${MANIFEST_GCS_PREFIX}" --num-shards "${SHARDS}"
bash setup.sh           # build + push image (one-time)
./submit_batch.sh       # submit the Cloud Batch job
```

Measured ~110.5 articles/s per L4 GPU end-to-end; full detail (throughput, cost,
shard-count tuning, spot-preemption recovery) in
[`vertexai/inference/relevance/README.md`](vertexai/inference/relevance/README.md).

---

## 3. Event extraction

Two separate pipelines below, sharing the same ontology and model but **not** the
same data source — see the note at the top of this README's Pipeline section: data
generation produces the SFT training dataset offline from a pos/neg DB sample (not
relevance-filtered), which a separate training step then fine-tunes the small model
on; inference at scale runs that already-trained model on the whole DB downstream of
the relevance filter.

### Ontology — `ontologies/zhai/`

Event labels come from a flat `{event_name: description}` ontology JSON, and this
pipeline only cares about two of them:

- `bona.v4.json` (48 event types) — **the current default.** ZHAI-specific, authored
  by Bonaventure; this is what data generation, training, and inference all use now.
- `science.json` (167 event types) — the original ontology, taken from the food-
  insecurity science paper this project started from. Used at the beginning of the
  project and still supported (e.g. by `retrieve/`), but superseded by `bona.v4.json`
  for anything current.

Each is paired with a `*_clusters.json` (`bona.v4.clusters.json`,
`science_clusters.json`) mapping every flat event type to a higher-level cluster —
`bona.v4`'s 48 events roll up into 10 clusters (e.g. "conflict and security",
"weather and natural hazards"), `science`'s 167 into 12. `scripts/download/sample.py
--stratify-by-cluster` and the `scripts/data/stats/` tools read these for
cluster-stratified sampling/reporting.

The other ontology files under `ontologies/` (`dwie/`, `maven/`, `maven-arg/`, `rams/`)
are leftovers from past experiments and are not used anywhere in this pipeline, you can ignore them.

### Data generation — `scripts/event_extraction/generation/`

Context/prompt-distillation pipeline: a Gemini teacher (elaborate prompt, few-shot)
labels articles with events, producing an SFT dataset that a small student model
(Qwen3.5-4B) is later trained on to reproduce those labels from a simpler prompt (the
training step itself lives in `scripts/train/llamafactory/` or `vertexai/train/event-extraction/`,
not here). Runs on the
`from_db_matrix_v2.py` positive/negative sample from §1 — **not** on relevance-filtered
input — so the teacher (and thus the resulting training set) also covers negatives,
teaching the student to emit no events on them, rather than only ever seeing articles
known to contain an event. Full detail in
[`scripts/event_extraction/generation/README.md`](scripts/event_extraction/generation/README.md).

| Step | Script                    | Purpose                                                                    |
| ---- | ------------------------- | -------------------------------------------------------------------------- |
| 0    | (DB sample, above)        | pos/neg sample, incl. negatives on purpose — no relevance filtering        |
| 1    | `generate.py`             | run the Gemini teacher, emit silver JSONL                                  |
| 2    | `validate.py`             | check ontology/grounding/enums, split clean vs. invalid                    |
| 2b   | `fix_events.py`           | send invalid events to a stronger model for re-grounding or drop           |
| 2c   | `merge.py`                | combine clean + fixed events into final JSONL                              |
| 3    | `to_sft.py`               | window articles into paragraph chunks, emit LlamaFactory Alpaca SFT format |

```bash
PYTHONPATH=. python scripts/event_extraction/generation/generate.py \
    --input  dataset/zhai/v3/<stratified-articles>.jsonl \
    --output dataset/zhai/v3/silver.gemini.jsonl \
    --model  gemini-2.5-flash --temperature 1.0 --reasoning-effort low \
    --batch-api --batch-size 100

PYTHONPATH=. python scripts/event_extraction/generation/validate.py \
    --input dataset/zhai/v3/silver.gemini.jsonl \
    --output-stem dataset/zhai/v3/silver.validated

PYTHONPATH=. python scripts/event_extraction/generation/to_sft.py \
    dataset/zhai/v3/silver.final.jsonl \
    dataset/zhai/v3/silver.sft.json
```

`costs.py` reports token usage/cost per model from any pipeline JSONL. Training does
not deduplicate against the same closed label set used at inference time — student
and teacher must use byte-identical prompts.

### Training — `scripts/train/llamafactory/` (local) / `vertexai/train/event-extraction/` (cloud)

Fine-tunes the small student model (Qwen3.5-4B, LoRA) on the SFT dataset produced by
data generation (above), via LlamaFactory. `scripts/train/llamafactory/train.sh` runs
this directly on a local/on-prem GPU. `vertexai/train/event-extraction/` packages the
same LoRA SFT configs as a Vertex AI custom job — a self-contained image that clones
upstream [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) at a pinned
commit, with no dependency on the rest of this repo. Full detail in
[`vertexai/train/event-extraction/README.md`](vertexai/train/event-extraction/README.md).

```bash
cd vertexai/train/event-extraction
cp .env.example .env    # fill in GCP_PROJECT, GCS_BUCKET, WANDB_API_KEY, ...

bash build_push.sh --cloud latest    # build + push image (one-time; rebuild only on Dockerfile changes)
gsutil cp config/qwen3_5_4b_base_new_schema.yaml "gs://${GCS_BUCKET}/${GCS_TRAIN_CONFIG}"
bash submit_job.sh                   # submit the Vertex AI custom job (A100 40GB by default)
```

### Candidate retrieval — `scripts/event_extraction/retrieve/`

Shrinks the full event-type ontology down to a small per-document `candidates` list,
for both teacher generation (`generate.py --top-k-candidates`) and scaled inference
(`vertexai/inference/event-extraction`'s `--top-k-candidates`). Two reasons to use it:

- **Hard limit** — Gemini structured generation caps out at 100 candidate event
  types per call (beyond that, hallucination/schema errors rise sharply), so any
  ontology larger than 100 labels *must* be pruned per-document before it can be
  passed to the teacher at all. `ontologies/zhai/science.json` has 167 event types,
  so retrieval is required for it; `ontologies/zhai/bona.v4.json` has only 48, so
  it's optional there.
- **Ranking** — even under the 100-label cap, retrieval re-ranks candidates by
  relevance to the document instead of leaving them in arbitrary ontology order,
  which improves what ends up near the top of a truncated `--top-k-candidates` list.

Builds a dense vector index over the ontology, retrieves top-K nearest event types
per query, and can score Recall@K against gold annotations. Full detail in
[`scripts/event_extraction/retrieve/README.md`](scripts/event_extraction/retrieve/README.md).

```bash
# science.json (167 event types) needs retrieval to fit Gemini's 100-candidate cap
PYTHONPATH=. python scripts/event_extraction/retrieve/generate_index.py \
    ontologies/zhai/science.json dataset/zhai/v3/index/science-harrier \
    microsoft/harrier-oss-v1-0.6b --sentence-transformers --device cuda:0 --normalize-embeddings

# bona.v4.json (48 event types) — optional, mainly for ranking quality
PYTHONPATH=. python scripts/event_extraction/retrieve/generate_index.py \
    ontologies/zhai/bona.v4.json dataset/zhai/v3/index/bona-v4-harrier \
    microsoft/harrier-oss-v1-0.6b --sentence-transformers --device cuda:0 --normalize-embeddings

PYTHONPATH=. python scripts/event_extraction/retrieve/retrieve.py \
    --queries dataset/zhai/v3/dev.jsonl --index dataset/zhai/v3/index/science-harrier \
    --output  dataset/zhai/v3/dev.candidates.jsonl \
    --model-name microsoft/harrier-oss-v1-0.6b --top-k 50 --device cuda:0 --normalize-embeddings \
    --query-prompt-name sts_query

PYTHONPATH=. python scripts/event_extraction/retrieve/eval_recall_at_k.py \
    dataset/zhai/v3/dev.candidates.jsonl --k 1 3 5 10 20 50 70 100
```

`retrieve_vllm.py` is a GPU-only, faster alternative to `retrieve.py` for large query
sets (vLLM pooling-mode encoding instead of in-process Sentence Transformers).

### Inference at scale — `vertexai/inference/event-extraction/`

The production run: takes the model already trained on the data-generation dataset
(above) and runs it over the **whole** article DB, on input that has already been
through the relevance filter (§2) — the inverse of data generation's unfiltered
pos/neg sample. Runs the fine-tuned model (`scripts/train/inference/vllm_infer.py`) as an N-shard GCP
Cloud Batch array job, one GPU VM per shard, merged after completion. Full detail —
GPU tier costs, shard-count sizing, spot recovery — in
[`vertexai/inference/event-extraction/README.md`](vertexai/inference/event-extraction/README.md).

```bash
bash vertexai/inference/event-extraction/setup.sh    # build + push image (one-time)

cd vertexai/inference/event-extraction
./submit_batch.sh \
    --tier spot-a100 --shards 20 \
    --input  gs://BUCKET/data/input.jsonl \
    --output gs://BUCKET/output/run-001 \
    --model  gs://BUCKET/models/qwen3.5-4b-merged \
    --top-k-candidates 90 --quantization fp8

./merge_shards.sh --output gs://BUCKET/output/run-001 --dest combined.jsonl
```

Supports an optional retriever (e.g. `microsoft/harrier-oss-v1-0.6b`, see
[Candidate retrieval](#candidate-retrieval--scriptsevent_extractionretrieve) above) to restrict
and/or rank the ontology candidates sent per article/window — required to fit
`ontologies/zhai/science.json`'s 167 event types under the 100-candidate cap, optional
for `ontologies/zhai/bona.v4.json`'s 48. Spot preemption loses at most ~90s of work
(entrypoint syncs partial output to GCS; `vllm_infer.py` skips already-processed
articles by text hash on retry).

---

## 4. Geocoding — `scripts/geocoding/`

Resolves each extracted `event_location` string to coordinates and an administrative
region, then produces DB-ready ingest CSVs. Full detail in
[`scripts/geocoding/README.md`](scripts/geocoding/README.md).

### `add_geotaxonomy.py` — Photon + zhai taxonomy

Two phases:

1. **Geocode** — each `event_location` string is queried against a [Photon](https://photon.komoot.io/)
   instance (local by default, `http://localhost:2322`); Photon's up-to-10 candidates
   are re-ranked by a composite `reranker_factor × importance_proxy` score (favors
   both a strong name match and geographic significance — neither alone is enough)
   and the best one is written back as a `geotaxonomy` entry, including a `zhai`
   place-type (`country`/`province`/`district`/`city`/`village`/`suburb`) mapped from
   Photon's `admin_level`/`type`.
2. **Spatial join** — resolved coordinates are joined against World Bank admin
   boundary GeoJSONs (`geotaxonomy_prewb_{0,1,2}.geojson`, auto-downloaded from
   `gs://zhai-data-geotaxonomy` if missing locally) to attach `adm_code`/`adm_level`.

```bash
PYTHONPATH=. python scripts/geocoding/add_geotaxonomy.py predictions.jsonl -o predictions.geo.jsonl

# Batch a whole directory
./scripts/geocoding/run_geotaxonomy.sh --workers 10 --parallel 4 predictions/ predictions-geo/
```

### `to_csv_ingest.py` — DB ingest CSVs

Converts `.geo.jsonl` files into `risk_matches.csv` (`article_uri, risk_id`) and
`locations.csv` (`article_uri, adm_code`), validating every `event_type` and
`adm_code` against the `res/risk_factors.csv` / `res/geo_taxonomy.csv` reference
tables (hard error on any unmapped value).

```bash
PYTHONPATH=. python scripts/geocoding/to_csv_ingest.py predictions_dir/   # merges all *.geo.jsonl shards
```

---

## Reproducibility

Exact commands used to produce the current (latest) version of each trained artifact.
Kept up to date as the pipeline evolves — if a step's commands change, update its entry
here rather than adding a new one.

### Relevance filter model

**1. Sample from the DB** (§1) — one run per language, English via the pos/neg matrix
sampler, French via the country sampler (no risk-factor tags exist for French):

```bash
PYTHONPATH=. python scripts/download/from_db_matrix_v2.py --n 20000 --output dataset/en_20k.jsonl
PYTHONPATH=. python scripts/download/sample_french_by_country.py --n 20000 --output dataset/fr_20k.jsonl
```

**2. Gemini data annotation** (§2, `relevance_filter.py`) — label each sample
`relevant`/`irrelevant` with Gemini 3.1 Pro via the Batch API, one run per language
(`--french` switches the prompt for the French sample):

```bash
PYTHONPATH=. python scripts/relevance/relevance_filter.py \
    --input  dataset/en_20k.jsonl \
    --output dataset/relevance/en_20k.relevance.jsonl \
    --use-llm --model gemini-3.1-pro-preview --batch-api

PYTHONPATH=. python scripts/relevance/relevance_filter.py \
    --input  dataset/fr_20k.jsonl \
    --output dataset/relevance/fr_20k.relevance.jsonl \
    --use-llm --model gemini-3.1-pro-preview --batch-api --french
```

**3. Train the classifier** (§2, `train.py`) — fine-tunes `jhu-clsp/mmBERT-small` on
the combined English + French Gemini labels:

```bash
PYTHONPATH=. python scripts/relevance/train.py \
    --input dataset/relevance/en_20k.relevance.jsonl dataset/relevance/fr_20k.relevance.jsonl \
    --output-dir outputs/relevance/relevance-mmbert-small --wandb-project zhai-relevance \
    --num-epochs 10 --model-name jhu-clsp/mmBERT-small
```

`train.py` writes each run to a timestamped subdirectory of `--output-dir`
(`outputs/relevance/relevance-mmbert-small/<timestamp>/`), with the checkpoint under
`final/` and the stratified train/dev split saved alongside it as
`train_<source>.jsonl`/`dev_<source>.jsonl` (one pair per input file) — no separate
data-splitting step is needed before inference/eval below.

**4. Inference + eval on the held-out dev split** — one run per language, reusing the
`dev_*.jsonl` files `train.py` saved into the run directory:

```bash
# English
PYTHONPATH=. python scripts/relevance/inference.py \
    --checkpoint outputs/relevance/relevance-mmbert-small/20260721_195341/final \
    --input  outputs/relevance/relevance-mmbert-small/20260721_195341/dev_en_20k.relevance.jsonl \
    --output outputs/relevance/relevance-mmbert-small/20260721_195341/predictions/dev_en_20k.relevance.jsonl \
    --backend vllm

PYTHONPATH=. python scripts/relevance/eval.py \
    --predictions outputs/relevance/relevance-mmbert-small/20260721_195341/predictions/dev_en_20k.relevance.jsonl \
    --labels outputs/relevance/relevance-mmbert-small/20260721_195341/dev_en_20k.relevance.jsonl

# French
PYTHONPATH=. python scripts/relevance/inference.py \
    --checkpoint outputs/relevance/relevance-mmbert-small/20260721_195341/final \
    --input  outputs/relevance/relevance-mmbert-small/20260721_195341/dev_fr_20k.relevance.jsonl \
    --output outputs/relevance/relevance-mmbert-small/20260721_195341/predictions/dev_fr_20k.relevance.jsonl \
    --backend vllm

PYTHONPATH=. python scripts/relevance/eval.py \
    --predictions outputs/relevance/relevance-mmbert-small/20260721_195341/predictions/dev_fr_20k.relevance.jsonl \
    --labels outputs/relevance/relevance-mmbert-small/20260721_195341/dev_fr_20k.relevance.jsonl
```

**5. Inference at scale** (§2, `vertexai/inference/relevance/`) — run the trained classifier over the entire ~130M-article DB.

```bash
cd vertexai/inference/relevance
cp .env.example .env   # fill in GCP_PROJECT, MANIFEST_GCS_PREFIX, MODEL_GCS, ...
# SHARDS=100 and MANIFEST_GCS_PREFIX=gs://zhai-risk-factor-extraction/relevance/manifests
python build_manifest.py --output-prefix "${MANIFEST_GCS_PREFIX}" --num-shards "${SHARDS}"
bash setup.sh           # build + push image (one-time)
./submit_batch.sh       # submit the Cloud Batch job

```

#### Paths

- Data: gs://zhai-risk-factor-extraction/relevance/data
- Model: zhai-risk-factor-extraction/relevance/models/experiments/relevance-mmbert-small/20260721_195341/final
- Inference output: gs://zhai-risk-factor-extraction/relevance/output/run-001

**6. DB Ingestion**  — TODO

### Event Extraction model

**1. Sample from the DB or use the samples from the previous step** (§1) — For this experiment, we reuse the same 20k English and 20k French samples from the relevance filter step. We randomly sample 5k articles from each language for the event extraction data generation step. We want to have both relevant and irrelevant articles in the sample to ensure the model learns to distinguish between them. We use a relevant/irrelevant ratio of 0.85 for both languages.

```bash
PYTHONPATH=. python scripts/event_extraction/generation/sample_articles.py \
    dataset/relevance/en_20k.relevance.jsonl \
    dataset/extraction/en_5k.relevance.jsonl \
    -n 5000

PYTHONPATH=. python scripts/event_extraction/generation/sample_articles.py \
    dataset/relevance/fr_20k.relevance.jsonl \
    dataset/extraction/fr_5k.relevance.jsonl \
    -n 5000
```

**2. Gemini data annotation** (§2) - We are using the latest zhai ontology (`ontologies/zhai/bona.v4.json`) for event extraction. The ontology has 48 event types, which is below the 100-candidate cap for Gemini structured generation, so we do not need to use the candidate retrieval step here. We will run the Gemini teacher on the sampled articles to produce silver labels for training.

```bash
PYTHONPATH=. python scripts/event_extraction/generation/generate.py \
  --input  dataset/extraction/en_5k.relevance.jsonl \
  --output dataset/extraction/en_5k.relevance.annotated.jsonl \
  --model  gemini-3.1-pro-preview \
  --temperature 0.3 \
  --workers 8 --batch-api --reasoning medium --stratified url

PYTHONPATH=. python scripts/event_extraction/generation/generate.py \
  --input  dataset/extraction/fr_5k.relevance.jsonl \
  --output dataset/extraction/fr_5k.relevance.annotated.jsonl \
  --model  gemini-3.1-pro-preview \
  --temperature 0.3 \
  --workers 8 --batch-api --reasoning medium --stratified url
```

**3. Train the event extraction model** (§3)

**4. Inference and eval on the held-out dev split** (§4)

**5. Inference at scale** (§5)

**6. Geocoding** (§6)

**7. DB Ingestion** (§7)

#### Paths

- Data: gs://zhai-risk-factor-extraction/extraction/data
- Model: zhai-risk-factor-extraction/extraction/models/
- Inference output: gs://zhai-risk-factor-extraction/extraction/output/run-001
