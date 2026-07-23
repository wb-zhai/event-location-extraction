# event-location-extraction

Extracts food-insecurity risk-factor events and their locations from news articles.
Articles are sampled from the article database, filtered for relevance, run through
an LLM-distilled extraction model, and geocoded into administrative regions —
producing `article_uri` ↔ `risk_factor` and `article_uri` ↔ `adm_code` tables ready
for DB ingestion.

## Pipeline

```text
Sample from DB  →  Relevance filter  →  Event extraction  →  Geocoding
(download/)        (relevance/)         (generation/,      (geocoding/)
                                          vertexai/inference)
```

1. **Sample from the DB** — pull a stratified article sample (`scripts/data/download/`).
2. **Relevance filter** — cheaply drop articles that can't contain a food-insecurity
   event before spending LLM budget on them (`scripts/data/relevance/`, `vertexai/relevance/`).
3. **Event extraction** — distill Gemini silver labels into an SFT dataset, then run
   the fine-tuned model at scale (`scripts/data/generation/`, `vertexai/inference/`).
4. **Geocoding** — resolve each extracted `event_location` string to coordinates and
   an administrative region, ready for DB ingestion (`scripts/geocoding/`).

---

## 1. Sampling from the DB

Two stratified samplers in `scripts/data/download/`, both reading connection params
from the repo-root `.env` (`SQL_HOST`/`SQL_PORT`/`SQL_DATABASE`/`SQL_USERNAME`/`SQL_PASSWORD`)
and requiring `psycopg2`. Both avoid ever scanning `article_concept_association`
(2.3B rows / 470 GB) directly — candidate discovery goes through chunked, indexed
probes instead, run in parallel across `--workers` connections, with clean Ctrl+C
cancellation of in-flight server-side queries.

### `from_db_matrix_v2.py` — positive/negative risk-factor sample

Downloads a country-stratified sample split into **positives** (risk-factor-tagged,
`article_risk_factor_tags`) and **negatives** (geo-tagged but untagged), for training
data. Country stratification is round-robin, rarest-country-first; positives are
additionally prioritized by risk-factor diversity.

```bash
python scripts/data/download/from_db_matrix_v2.py \
    --n 50000 --pos-ratio 0.7 --output dataset/matrix_sample.jsonl
```

Key flags: `--pos-ratio` (default 0.7), `--language` (default `eng`), `--start-month`/
`--end-month` (default: full history to today), `--tag-method-id`, `--workers`, `--seed`.

### `sample_french_by_country.py` — single-language country sample

Same discovery/stratification approach without the positive/negative split — samples
N articles in one language (default `fra`), stratified by country. Supports `--resume`
for interrupted multi-hour runs (content-fetch chunks are flushed to disk as they land
and matched by article id on restart).

```bash
python scripts/data/download/sample_french_by_country.py \
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
See [`scripts/data/relevance/README.md`](scripts/data/relevance/README.md) for full detail.

### Data generation — `scripts/data/relevance/`

`relevance_filter.py` labels articles `relevant`/`not relevant`, optionally cascading
a cheap model's positive calls to a stronger one for cheaper high-quality labels:

```bash
python scripts/data/relevance/relevance_filter.py \
    --input dataset/zhai/v3/articles.jsonl \
    --output dataset/zhai/v3/articles.filtered.jsonl \
    --use-llm --cascade --model gemini-2.5-flash --cascade-model gemini-3.1-pro-preview \
    --filter-only
```

Related tools in the same directory: `local_llm.py` (test a local llama.cpp-served
model as a cheap stand-in), `encoder.py` (local encoder classifier, no LLM cost),
`view_relevance.py` (Gradio browser for labeled output), `agreement.py` (compare two
labeled files — accuracy/kappa).

### Train the relevance model — `scripts/data/relevance/train.py`

Fine-tunes a binary sequence classifier (default `answerdotai/ModernBERT-base`) on
`relevance_filter.py`'s labeled output:

```bash
python scripts/data/relevance/train.py \
    --input dataset/db/relevance/matrix_5M.sample_1000.3.1pro.2label_prompt.jsonl \
    --output-dir /tmp/relevance-modernbert
```

`inference.py` (HF or vLLM backend) and `eval.py` (accuracy/precision/recall/F1
against labeled data) round out local iteration on a checkpoint.

### Inference at scale — `vertexai/relevance/`

Classifies the **entire** `article_downloads` table (~130M articles) with a trained
checkpoint via a GCP Cloud Batch job running vLLM's pooling `classify()`, reading
articles from GCS (not the DB) so it scales horizontally with zero DB load. Fully
self-contained — no imports from the rest of the repo. Output is `id,label` CSV
shards, no merge step (read downstream with a wildcard).

```bash
cd vertexai/relevance
cp .env.example .env   # fill in GCP_PROJECT, MANIFEST_GCS_PREFIX, MODEL_GCS, ...
python build_manifest.py --output-prefix "${MANIFEST_GCS_PREFIX}" --num-shards "${SHARDS}"
bash setup.sh           # build + push image (one-time)
./submit_batch.sh       # submit the Cloud Batch job
```

Measured ~110.5 articles/s per L4 GPU end-to-end; full detail (throughput, cost,
shard-count tuning, spot-preemption recovery) in
[`vertexai/relevance/README.md`](vertexai/relevance/README.md).

---

## 3. Event extraction

### Data generation — `scripts/data/generation/`

Context/prompt-distillation pipeline: a Gemini teacher (elaborate prompt, few-shot)
labels articles with events, and a small student model (Qwen3-4B) is SFT-trained to
reproduce those labels from a simpler prompt. Full detail in
[`scripts/data/generation/README.md`](scripts/data/generation/README.md).

| Step | Script                    | Purpose                                                                    |
| ---- | ------------------------- | -------------------------------------------------------------------------- |
| 0    | (relevance filter, above) | drop irrelevant articles before the teacher call                           |
| 1    | `generate.py`             | run the Gemini teacher, emit silver JSONL                                  |
| 2    | `validate.py`             | check ontology/grounding/enums, split clean vs. invalid                    |
| 2b   | `fix_events.py`           | send invalid events to a stronger model for re-grounding or drop           |
| 2c   | `merge.py`                | combine clean + fixed events into final JSONL                              |
| 3    | `to_sft.py`               | window articles into paragraph chunks, emit LlamaFactory Alpaca SFT format |

```bash
python scripts/data/generation/generate.py \
    --input  dataset/zhai/v3/<stratified-articles>.jsonl \
    --output dataset/zhai/v3/silver.gemini.jsonl \
    --model  gemini-2.5-flash --temperature 1.0 --reasoning-effort low \
    --batch-api --batch-size 100

python scripts/data/generation/validate.py \
    --input dataset/zhai/v3/silver.gemini.jsonl \
    --output-stem dataset/zhai/v3/silver.validated

python scripts/data/generation/to_sft.py \
    dataset/zhai/v3/silver.final.jsonl \
    dataset/zhai/v3/silver.sft.json
```

`costs.py` reports token usage/cost per model from any pipeline JSONL. Event labels
come from an ontology JSON — the current default is `ontologies/zhai/bona.v4.json`
(48 event types); the older, larger `ontologies/zhai/science.json` (167 event types)
is still supported and referenced throughout this pipeline. Training does not
deduplicate against the same closed label set used at inference time — student and
teacher must use byte-identical prompts.

### Candidate retrieval — `scripts/data/retrieve/`

Shrinks the full event-type ontology down to a small per-document `candidates` list,
for both teacher generation (`generate.py --top-k-candidates`) and scaled inference
(`vertexai/inference`'s `--top-k-candidates`). Two reasons to use it:

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
[`scripts/data/retrieve/README.md`](scripts/data/retrieve/README.md).

```bash
# science.json (167 event types) needs retrieval to fit Gemini's 100-candidate cap
python scripts/data/retrieve/generate_index.py \
    ontologies/zhai/science.json dataset/zhai/v3/index/science-bge-m3 \
    BAAI/bge-m3 --sentence-transformers --device cuda:0 --normalize-embeddings

# bona.v4.json (48 event types) — optional, mainly for ranking quality
python scripts/data/retrieve/generate_index.py \
    ontologies/zhai/bona.v4.json dataset/zhai/v3/index/bona-v4-bge-m3 \
    BAAI/bge-m3 --sentence-transformers --device cuda:0 --normalize-embeddings

python scripts/data/retrieve/retrieve.py \
    --queries dataset/zhai/v3/dev.jsonl --index dataset/zhai/v3/index/science-bge-m3 \
    --output  dataset/zhai/v3/dev.candidates.jsonl \
    --model-name BAAI/bge-m3 --top-k 50 --device cuda:0 --normalize-embeddings

python scripts/data/retrieve/eval_recall_at_k.py \
    dataset/zhai/v3/dev.candidates.jsonl --k 1 3 5 10 20 50 70 100
```

`retrieve_vllm.py` is a GPU-only, faster alternative to `retrieve.py` for large query
sets (vLLM pooling-mode encoding instead of in-process Sentence Transformers).

### Inference at scale — `vertexai/inference/`

Runs the fine-tuned model (`scripts/train/inference/vllm_infer.py`) as an N-shard GCP
Cloud Batch array job, one GPU VM per shard, merged after completion. Full detail —
GPU tier costs, shard-count sizing, spot recovery — in
[`vertexai/inference/README.md`](vertexai/inference/README.md).

```bash
bash vertexai/inference/setup.sh    # build + push image (one-time)

cd vertexai/inference
./submit_batch.sh \
    --tier spot-a100 --shards 20 \
    --input  gs://BUCKET/data/input.jsonl \
    --output gs://BUCKET/output/run-001 \
    --model  gs://BUCKET/models/qwen3-4b-merged \
    --top-k-candidates 90 --quantization fp8

./merge_shards.sh --output gs://BUCKET/output/run-001 --dest combined.jsonl
```

Supports an optional retriever (e.g. `microsoft/harrier-oss-v1-0.6b`, see
[Candidate retrieval](#candidate-retrieval--scriptsdataretrieve) above) to restrict
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
python scripts/geocoding/add_geotaxonomy.py predictions.jsonl -o predictions.geo.jsonl

# Batch a whole directory
./scripts/geocoding/run_geotaxonomy.sh --workers 10 --parallel 4 predictions/ predictions-geo/
```

### `to_csv_ingest.py` — DB ingest CSVs

Converts `.geo.jsonl` files into `risk_matches.csv` (`article_uri, risk_id`) and
`locations.csv` (`article_uri, adm_code`), validating every `event_type` and
`adm_code` against the `res/risk_factors.csv` / `res/geo_taxonomy.csv` reference
tables (hard error on any unmapped value).

```bash
python scripts/geocoding/to_csv_ingest.py predictions_dir/   # merges all *.geo.jsonl shards
```
