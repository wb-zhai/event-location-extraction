# Relevance filter — GCP Cloud Batch runbook

Classifies **every** article in `article_downloads` (**130,490,360** — 118,746,759 English +
11,743,601 French) as `relevant` / `not relevant` using the trained relevance encoder
(ModernBERT, served via vLLM's pooling `classify`), and writes `id,label` CSV shards to GCS.

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
horizontally with zero load on production cloudSQL. The run is **GPU-bound** at a **measured**
~110.5 art/s on an L4 (real end-to-end Cloud Batch run, see "Local timing test" below — GCS
fetch + tokenize/truncate + vLLM classify + CSV write), so total GPU-time is fixed at
≈ 328 GPU-hours for 130.49M articles and wall-clock = 328 / shards.

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
# optional: --language {eng,fra,both}   --limit N (testing)
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

### Choosing `--shards` (at measured ~110.5 art/s on L4, 130,490,360 articles)

Total GPU-time is fixed (~328 GPU-h); more shards just finish faster.

| shards | articles/worker | wall-clock |
|-------:|----------------:|-----------:|
|     20 |            6.5M |    ~16.4 h |
|     50 |            2.6M |     ~6.6 h |
|    100 |            1.3M |     ~3.3 h |
|    140 |            0.9M |     ~2.3 h |

### Cost by `TIER`

GPU-hours are fixed at the **measured** L4 rate (328 GPU-h, see "Local timing test" below) — A100
rows assume the *same* 328 GPU-h, since classify throughput hasn't been benchmarked on A100
(unlike `vertexai/inference`, where L4 vs A100 was measured separately for the generative
workload). A100 is very likely faster at this encoder-classify workload too, so treat the A100
rows below as a **conservative ceiling**, not a verified number — re-run the "Local timing test"
on an `a100`-tier task before trusting it for a real run. **Total cost** below = GPU-h × $/GPU-h,
plus a fixed **$52** for ~130.49M GCS Class B reads (~$0.004/10k) added to every row.

| `TIER` | GPU | GPU-h | $/GPU-h | **Total cost** |
|---|---|---|---|---|
| `spot-l4` | L4 | 328 | $0.30 | **~$150** |
| `flex-l4` | L4 | 328 | $0.54 | **~$229** |
| `l4` | L4 | 328 | $0.85 | **~$331** |
| `spot-a100` | A100 | 328† | $1.10 | **~$413†** |
| `flex-a100` | A100 | 328† | $2.20 | **~$774†** |
| `a100` | A100 | 328† | $3.50 | **~$1200†** |

† unverified — assumes A100 throughput equals the measured L4 rate; likely an overestimate.

### GPU tiers

| `TIER` | Machine | GPU | Spot? | ~$/GPU-h | Notes |
|---|---|---|---|---|---|
| `spot-l4` | g2-standard-8 | 1× L4 24GB | Yes | ~$0.30 | Cheapest; measured/verified baseline; `.env` default |
| `flex-l4` | g2-standard-8 | 1× L4 24GB | Flex | ~$0.54 | Lower interruption risk than spot; GCP queues until capacity available |
| `l4` | g2-standard-8 | 1× L4 24GB | No | ~$0.85 | No interruptions |
| `spot-a100` | a2-highgpu-1g | 1× A100 40GB | Yes | ~$1.10 | Throughput unverified for this classify workload |
| `flex-a100` | a2-highgpu-1g | 1× A100 40GB | Flex | ~$2.20 | Throughput unverified for this classify workload |
| `a100` | a2-highgpu-1g | 1× A100 40GB | No | ~$3.50 | Throughput unverified for this classify workload |

`spot-l4` is cheapest and matches the measured, verified numbers above. `flex-*` trades the spot
discount for lower interruption risk (GCP queues for capacity instead of preempting) — see "Spot
preemption / recovery" below. Re-verify GPU-hours if you change `--max-chars`/`--max-length`/the
model, since those directly affect articles/sec.

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

Run a batch of articles on one L4, measure throughput, extrapolate:

```bash
python vertexai/relevance/relevance_vllm_infer.py \
    --model_name_or_path /path/to/checkpoint \
    --input   manifest-000.jsonl \
    --output  /tmp/relevance.csv \
    --num_shards 1 --shard_index 0 \
    --limit 10000
# rate = 10000 / elapsed_seconds;  full run ≈ 328 GPU-h / shards
```

Re-running resumes: already-labeled ids in the output CSV are skipped.

**Verified result** (2026-07-21, real Cloud Batch run on `l4` tier, 1000-article test manifest,
`relevance-mmbert-small` checkpoint, `MAX_LENGTH=4096`):

- **1000 articles in 9.05s → 110.5 articles/sec** end-to-end (GCS fetch + tokenize/truncate +
  vLLM classify + CSV write).
- Output: `shard-0.csv`, 1000 rows, `id,label`, only `relevant`/`not relevant` present
  (381 relevant / 619 not relevant).
- One-time overhead per worker, separate from the steady-state rate above: ~73s for vLLM engine
  init + CUDA graph compile between `[entrypoint] starting inference` and the first classified
  batch. Negligible on a 1M+-article shard; noticeable on tiny test runs.
- This run also caught a real bug (now fixed): char-based `--max-chars` truncation doesn't bound
  token count, so an outlier article could exceed `--max-length` tokens and crash the whole
  `vLLM.classify()` call with `VLLMValidationError`. `RelevanceClassifier` now tokenizes and
  truncates to `--max-length` itself before calling `classify()`, so this can't recur regardless
  of shard content.

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
