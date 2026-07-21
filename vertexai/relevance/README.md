# Relevance filter — GCP Cloud Batch runbook

Classifies **every** article in `article_downloads` (~140M) as `relevant` / `not relevant`
using the trained relevance encoder (ModernBERT, served via vLLM's pooling `classify`), and
writes `id,label` CSV shards to GCS.

This pipeline is **self-contained**: every file lives in `vertexai/relevance/` and imports
nothing from the rest of the repo.

## How it works

1. **`build_manifest.py`** streams `SELECT uri, cloud_uri FROM article_downloads` (no body —
   fast, minimal DB load) into `--num-shards` pre-sharded manifest files on GCS
   (`manifest-000.jsonl …`), each line `{"id": <uri>, "gcs_path": "gs://…/uri.json"}`.
2. **Cloud Batch** runs one task per shard. Each task downloads **only its** `manifest-NNN.jsonl`,
   then streams it through a bounded *GCS-fetch → vLLM-classify → CSV-write* pipeline
   (`relevance_vllm_infer.py`) — memory-flat regardless of shard size — and writes
   `shard-N.csv` (`id,label`) to the output prefix.
3. **No merge.** Output stays as `shard-*.csv`; read it downstream with a wildcard
   (e.g. BigQuery wildcard load / external table).

Ingestion is from GCS, not the DB: each article is an individual object
(`cloud_uri = gs://newsapi-news-data/<uri>.json`), and object storage bulk-read scales
horizontally with zero load on production cloudSQL. The run is **GPU-bound** at ~120 art/s
on an L4, so total GPU-time is fixed at ≈ 324 GPU-hours for 140M and wall-clock = 324 / shards.

## Prerequisites

- `gcloud` CLI authenticated (`gcloud auth login`)
- DB creds in the **repo-root** `.env` (`SQL_HOST/SQL_PORT/SQL_DATABASE/SQL_USERNAME/SQL_PASSWORD`)
  for `build_manifest.py`
- GPU quota in your region for the chosen tier
- `pip install psycopg2-binary google-cloud-storage` to run `build_manifest.py` locally

---

## Step 0 — Configure `.env`

```bash
cd vertexai/relevance
cp .env.example .env
# edit .env: GCP_PROJECT, GCP_REGION, IMAGE, MANIFEST_GCS_PREFIX, OUTPUT_GCS_PREFIX, MODEL_GCS, ...
```

`setup.sh` and `submit_batch.sh` **auto-source this `.env`** — every value in it becomes the
default for the matching flag, so once it's filled in you generally don't need to pass flags at
all (see Steps 3–4 below). `build_manifest.py` does **not** read this file (it only reads the
repo-root `.env`, and only for DB creds), so for Step 1 below, source it into your own shell too:

```bash
set -a && source .env && set +a
```

Now `$MANIFEST_GCS_PREFIX`, `$MODEL_GCS`, `$SHARDS`, etc. are available in your shell for the
commands below, and stay consistent with what `setup.sh`/`submit_batch.sh` will use internally.

---

## Step 1 — Build the manifest

Pick your shard count first (it fixes the number of manifest files, which must match `--shards`
in `.env`/Step 4):

```bash
python vertexai/relevance/build_manifest.py \
    --output-prefix "${MANIFEST_GCS_PREFIX}" \
    --num-shards    "${SHARDS}"
# optional: --language eng   --limit N (testing)
```

## Step 2 — Stage the model

```bash
gsutil -m cp -r outputs/relevance/relevance-modernbert/…/checkpoint-XXXX/ \
    "${MODEL_GCS}/"
```

## Step 3 — Build and push the image (one-time, idempotent)

```bash
bash vertexai/relevance/setup.sh          # add --local to build with local Docker
```

Reads `.env` for `GCP_PROJECT`/`GCP_REGION`/`IMAGE`; override with `--project`, `--region`, `--tag`
only if you want to deviate from `.env` for this invocation.

## Step 4 — Submit the job

```bash
cd vertexai/relevance
chmod +x submit_batch.sh

./submit_batch.sh
```

That's it — with `.env` filled in, every setting (`TIER`, `SHARDS`, `IMAGE`, `MANIFEST_GCS_PREFIX`,
`OUTPUT_GCS_PREFIX`, `MODEL_GCS`, `GCP_PROJECT`, `GCP_REGION`, …) is picked up automatically.
Pass a flag only to **override** a single `.env` value for this run, e.g. a 1-shard smoke test:

```bash
./submit_batch.sh --shards 1 --output "${OUTPUT_GCS_PREFIX}-smoke"
```

> Don't pass `--project "${PROJECT}"` / `--image "${IMAGE}"` etc. unless you've sourced `.env`
> into *your own shell* first (Step 0) — otherwise those variables are empty in your shell, and
> e.g. `--project ""` will override the correct value `submit_batch.sh` would have picked up on
> its own, causing `ERROR: --project required`.

`submit_batch.sh` refuses to submit if `--shards` ≠ the number of `manifest-*.jsonl` files present.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--max-chars` | `4000` | Article chars used (title + body[:N]) |
| `--max-length` | `2048` | vLLM max sequence length |
| `--batch-size` | `2000` | Texts per `classify()` call |
| `--gcs-read-concurrency` | `64` | Concurrent GCS GETs per worker |
| `--gpu-memory-util` | `0.9` | Fraction of GPU VRAM for vLLM |
| `--max-run-duration` | `86400s` | Per-task wall-clock cap (required for flex tiers) |

### Choosing `--shards` (at ~120 art/s on L4, 140M articles)

Total GPU-time is fixed (~324 GPU-h); more shards just finish faster.

| shards | articles/worker | wall-clock |
|-------:|----------------:|-----------:|
|     20 |            7.0M |     ~16 h  |
|     50 |            2.8M |    ~6.5 h  |
|    100 |            1.4M |    ~3.2 h  |
|    140 |            1.0M |    ~2.3 h  |

**Verify throughput first** with the timing test below before committing to a shard count.

---

## Step 5 — Monitor

`submit_batch.sh` prints the exact `JOB_ID` it used — copy it from there. Note `.env` defines
`GCP_PROJECT`/`GCP_REGION` (not `PROJECT`/`REGION` — those are internal to the scripts):

```bash
gcloud batch jobs describe relevance-YYYYMMDD-HHMMSS \
    --project="${GCP_PROJECT}" --location="${GCP_REGION}"
gcloud logging read "resource.labels.job_uid=relevance-YYYYMMDD-HHMMSS" \
    --project="${GCP_PROJECT}" --limit=200 --format="value(textPayload)"
```

## Step 6 — Consume the output (no merge)

```bash
gsutil ls "${OUTPUT_GCS_PREFIX}/shard-*.csv"
# e.g. BigQuery: bq load --source_format=CSV --autodetect \
#   dataset.relevance "${OUTPUT_GCS_PREFIX}/shard-*.csv" id:STRING,label:STRING
```

---

## Local timing test (single process, no sharding)

Run 10K articles on one L4, measure throughput, extrapolate:

```bash
python vertexai/relevance/relevance_vllm_infer.py \
    --model_name_or_path /path/to/checkpoint \
    --input   manifest-000.jsonl \
    --output  /tmp/relevance.csv \
    --num_shards 1 --shard_index 0 \
    --limit 10000
# rate = 10000 / elapsed_seconds;  full run ≈ 324 GPU-h / shards
```

Re-running resumes: already-labeled ids in the output CSV are skipped.

---

## Spot preemption / recovery

`entrypoint.sh` syncs the partial `shard-N.csv` to GCS every 90 s and restores it on retry;
`relevance_vllm_infer.py` skips already-labeled ids on resume. A preempted task loses at most
~90 s (`maxRetryCount` 3 for spot tiers).

## Troubleshooting

- **Quota exceeded** — request `NVIDIA_L4_GPUS` (or A100 / preemptible variants) in your region.
- **Shard/manifest mismatch** — `--shards` must equal the number of `manifest-*.jsonl` files;
  rebuild with the right `--num-shards` or adjust `--shards`.
- **Missing article objects** — `relevance_vllm_infer.py` skips unreadable/missing GCS objects
  (they simply won't appear in the output); expect `total ≈ manifest rows − missing`.
- **Region ≠ us-central1** — the batch templates `docker login` to `us-central1-docker.pkg.dev`;
  edit the tier JSON if your Artifact Registry is in another region.
