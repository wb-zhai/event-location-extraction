# GCP Cloud Batch — batch inference runbook

Runs `scripts/event_extraction/inference/vllm_infer.py` as an N-task array job on GCP Cloud Batch.
Each task is an independent GPU VM and owns exactly one **manifest shard**.

The input is not a materialized JSONL of articles. `build_manifest.py` does one cheap metadata
scan of the DB (`uri`, `cloud_uri`, `published_at`, `language` — never `body`) and writes
`manifest-000.jsonl … manifest-NNN.jsonl` to GCS. At inference each task downloads only its own
manifest and streams article bodies out of GCS, one object per article. This is what makes the
job scale to the full ~34.4M relevant-article corpus: no per-task copy of the corpus, no
`article_downloads.body` export, and memory that stays O(batch) instead of O(corpus).

Same shape as `vertexai/inference/relevance/`.

## Prerequisites

- `gcloud` CLI authenticated — run `gcloud auth login`
- DB credentials in the **repo-root** `.env` (`SQL_HOST`, `SQL_DATABASE`, `SQL_USERNAME`, `SQL_PASSWORD`)
- GCS bucket for model weights, manifests, and outputs
- GPU quota in your target region for your chosen tier (check before large jobs)

---

## Step 1 — Stage the model and build the manifest

```bash
# Upload the merged model (only once per model version).
# Verified present: gs://zhai-risk-factor-extraction/extraction/models/<run>/merged/
gsutil -m cp -r outputs/extraction/qwen3_5-4b-lora-en-fr-a100-40gb-20260803-120523/merged/ \
    "${MODEL_GCS}/"

# Build the pre-sharded manifest. --num-shards MUST equal the --shards you submit with.
python vertexai/inference/event-extraction/build_manifest.py \
    --output-prefix gs://BUCKET/extraction/manifests \
    --num-shards    200
```

`build_manifest.py` selects the articles the relevance pipeline marked relevant:

```sql
SELECT d.uri, d.cloud_uri, d.published_at, d.language
FROM article_event_extraction.article_relevance r
JOIN article_downloads d ON d.uri = r.article_uri
WHERE r.relevance_version = 'mmbert-small-title-only-v1'
  AND r.is_relevant = true
  AND d.cloud_uri IS NOT NULL
```

Articles with no row for that version are excluded by the inner join. Override with
`--relevance-version`. Two versions exist: `mmbert-small-title-only-v1` (the default,
~34.4M relevant) and the older title+body `mmbert-small-v1` (~50.0M relevant).

Verified plan for the default version: **Parallel Hash Join**, 2 workers — one sequential
pass over `article_downloads` hashed against the relevant uris, which is what you want.

**Check the query plan before a full run.** `article_downloads` is range-partitioned on
`published_at` across ~300 partitions, so a nested-loop probe by `uri` alone can fan out
across every one of them. You want a hash or merge join — a single sequential pass. If the
planner picks badly, `--join-mode memory` loads the relevant uris into a set first (~2 GB at
34.4M rows) and filters `article_downloads` in Python instead.

Dry-run it locally first:

```bash
python vertexai/inference/event-extraction/build_manifest.py \
    --output-prefix /tmp/manifests --num-shards 4 --limit 1000
head -1 /tmp/manifests/manifest-000.jsonl
# {"id":"578612029","gcs_path":"gs://newsapi-news-data/578612029.json",
#  "publish_date":"2017-01-07 19:57:00","language":"eng"}
```

`publish_date` must come out as `YYYY-MM-DD HH:MM:SS` — that is the format the student model
was trained on, and `vllm_infer.py` reads `source.publish_date` only.

---

## Step 2 — Build and push the Docker image

Run the setup script from the **repo root** (one-time; idempotent):

```bash
bash vertexai/inference/event-extraction/setup.sh
```

This creates the Artifact Registry repo if needed, then submits a Cloud Build job to build and push the image.
Reads `vertexai/inference/event-extraction/.env` automatically; override with `--project`, `--region`, or `--tag`.

Add `--local` to build with local Docker instead of Cloud Build.

---

## Step 3 — Submit the job

```bash
cd vertexai/inference/event-extraction
chmod +x submit_batch.sh merge_shards.sh

./submit_batch.sh \
    --tier     spot-a100 \        # spot-l4 | spot-a100 | flex-l4 | flex-a100 | l4 | a100
    --shards   200 \              # MUST equal the number of manifest files
    --image    "${IMAGE}" \
    --manifest gs://BUCKET/extraction/manifests \
    --output   gs://BUCKET/extraction/output/run-001 \
    --model    "${MODEL_GCS}" \
    --project  "${PROJECT}" \
    --region   "${REGION}"
```

`submit_batch.sh` refuses to submit if `--shards` does not match the number of
`manifest-*.jsonl` objects under `--manifest`. Task N reads `manifest-NNN.jsonl`, so a
mismatch either kills those tasks on the manifest download or silently leaves shards
unprocessed. Bypass with `--skip-manifest-check` only if you know why.

### Optional inference flags

| Flag | Default | Description |
|---|---|---|
| `--max-new-tokens` | `4096` | Max tokens to generate per window |
| `--max-model-len` | vLLM auto | Total context length (prompt + output); set to limit VRAM |
| `--max-chars` | `3000` | Max characters per text window |
| `--top-k-candidates` | all | Ontology labels per article. Only bites when rows carry a `candidates` field (i.e. with the retriever) — with plain `bona.v4` every row gets all 48 |
| `--quantization` | none | vLLM quantization mode (e.g. `fp8`) |
| `--batch-size` | `500` | Articles per vLLM batch |
| `--gpu-memory-util` | `0.95` | Fraction of GPU VRAM reserved for the main model |
| `--gcs-read-concurrency` | `64` | Concurrent GCS GETs for streaming article bodies |
| `--min-chars` | `200` | Minimum characters to keep a window |
| `--max-paras` | `15` | Maximum paragraphs per window |
| `--overlap-paras` | `1` | Overlap paragraphs between adjacent windows |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--use-guided-decoding` | off | Constrain `event_type` to the ontology enum (see above) |
| `--guided-decoding-backend` | `xgrammar` | Backend passed to vLLM |
| `--ontology` | `bona.v4.json` | Ontology JSON to draw event labels from |
| `--ontology-descriptions` | `none` | `all` inlines label descriptions — required by `*-desc` models |
| `--lean-output` | `true` | `true` → `{"id","predictions"}` per line; `false` → full echo + `window_predictions` |
| `--retriever-model-name` | none | HF model id for the retriever (e.g. `microsoft/harrier-oss-v1-0.6b`) |
| `--retriever-index` | none | Path to the retriever index **on the container** |
| `--retriever-gpu-mem-util` | none | GPU memory fraction for the retriever |
| `--retriever-query-mode` | none | Retriever query mode (e.g. `per_window`) |
| `--retriever-max-model-len` | none | Max context length for the retriever |

> Every flag in this table reaches the container. Until 2026-08-10, `TEMPERATURE`, `MIN_CHARS`,
> `MAX_PARAS` and `OVERLAP_PARAS` were read by `entrypoint.sh` but never forwarded by the job
> templates, so setting them in `.env` silently did nothing; `--use-guided-decoding` and the
> `--ontology*` flags had no env var at all. To reproduce a local `vllm_infer.py` run in Cloud
> Batch, for example:
>
> ```bash
> # local
> python scripts/event_extraction/inference/vllm_infer.py ... \
>     --max_model_len 12688 --max_new_tokens 4096 --max_chars 3000 \
>     --use_guided_decoding --temperature 0.3
>
> # same settings on Cloud Batch
> ./submit_batch.sh --tier spot-a100 --shards N --manifest ... --output ... \
>     --max-model-len 12688 --max-new-tokens 4096 --max-chars 3000 \
>     --use-guided-decoding --temperature 0.3
> ```

> **The retriever flags do not currently work in this container.** `vllm_infer.py` imports
> `src.index.inmemory` for that path but the Dockerfile does not copy `src/`; and
> `InMemoryIndexer.from_pretrained` takes a local directory, while nothing in `entrypoint.sh`
> downloads a `gs://` index. The default 48-label `bona.v4` ontology needs no retrieval
> (retrieving 90 candidates out of 48 labels is a no-op), so this pipeline runs without it.
> Wiring it back up means copying `src/`, `gsutil cp`-ing the index and retriever model to
> `/local`, and passing local paths.

### Output format

With `--lean-output true` (the default) each line is:

```json
{"id":"578612029","predictions":[{"event_type":"civilian casualties","grounding_quote":"...",
  "event_location":"Kaduna","event_location_admin_level":"state","event_time":"not_stated",
  "time_status":"past","severity":"not_stated"}]}
```

That is exactly what `scripts/geocoding/add_geotaxonomy.py` (mutates `predictions` in place)
and `scripts/geocoding/to_csv_ingest.py` (reads `id` + `predictions`) consume. At ~1 KB/article
the full corpus is ~35 GB; the full-echo form is ~20 KB/article, i.e. ~700 GB. Use
`--lean-output false` when you need `window_predictions` for eval or debugging.

### Out-of-ontology event types

Guided decoding is **off** by default (`use_guided_decoding=False`), so the model can emit an
`event_type` outside the 48-label `bona.v4` ontology. Measured rate on the smoke run: **1 event
in 8,542 (0.012%)**, one affected article in 2000 — a `"food waste"` label. Extrapolated to the
full corpus that is roughly **17,000 bad events**.

This matters because `scripts/geocoding/to_csv_ingest.py` calls `sys.exit(1)` on the first
unknown `event_type`, so a single hallucinated label aborts ingest for the entire run. Pick one
before ingesting a full-scale run:

- filter unknown `event_type`s out of the predictions before ingest (cheapest; keeps inference
  throughput unchanged);
- make `to_csv_ingest.py` skip-and-count unknown labels instead of exiting;
- run inference with `--use_guided_decoding`, which constrains generation to the enum at some
  throughput cost — note the 0.172 s/article figure below was measured *without* it.

### Choosing `--shards`

Cost is flat in shard count — you pay total GPU-hours either way. Shards only buy wall clock.

**Measured** on a 2 x 1000-article `spot-a100` run (2026-08-10, bf16, `--batch-size 200`,
`--max-model-len 12238`, no retriever):

| Phase | Cost |
|---|---|
| Model download (8.5 GB) | ~48 s per task |
| vLLM engine init (incl. 50 s compilation) | ~169 s per task |
| **Steady-state generation** | **0.172 s/article** (1.67 windows/article) |

Startup is ~4.4 min of fixed cost per task, which is 20% of a 1000-article shard and under
1% of a 172k-article one.

For the full 34.39M corpus at 0.172 s/article:

| Tier | Shards | GPU-h | Cost | Wall clock |
|---|---|---|---|---|
| `spot-a100` | 100 | 1,650 | ~$1,815 | ~16.5 h |
| `spot-a100` | **200** | 1,658 | **~$1,823** | **~8.3 h** |
| `spot-a100` | 300 | 1,665 | ~$1,831 | ~5.5 h |
| `spot-l4` | 200 | 12,433 | ~$3,730 | ~62 h |
| `spot-l4` | 400 | 12,448 | ~$3,734 | ~31 h |

Plus ~$14 for 34.4M GCS Class B reads and ~41 GB of lean output. Model pull is 8.5 GB x N
tasks, same-region and free.

> The 0.172 s/article figure comes from 2000 contiguous manifest rows, which may not match
> the corpus-wide average article length. Treat it as +/-30% until a larger calibration run.
> It supersedes the 0.25 s/article that used to be quoted here — that number predates this
> pipeline.

**A100 spot quota is the binding constraint.** Check `PREEMPTIBLE_NVIDIA_A100_GPUS` in your
region before planning around 200 shards — the increase request takes days. `spot-l4` quota
is far easier but costs ~2x and runs ~4x longer.

### GPU tiers

| `--tier` | Machine | GPU | Spot? | ~$/GPU-h | Notes |
|---|---|---|---|---|---|
| `spot-l4` | g2-standard-8 | 1× L4 24GB | Yes | ~$0.30 | Cheapest per hour; auto-retries on preemption |
| `spot-a100` | a2-highgpu-1g | 1× A100 40GB | Yes | ~$1.10 | Cheapest per article and fastest; preemptible |
| `flex-l4` | g2-standard-8 | 1× L4 24GB | Flex | ~$0.54 | Lower interruption risk than spot; GCP queues until capacity available |
| `flex-a100` | a2-highgpu-1g | 1× A100 40GB | Flex | ~$2.20 | Best A100 availability; lower interruption risk than spot |
| `l4` | g2-standard-8 | 1× L4 24GB | No | ~$0.85 | No interruptions |
| `a100` | a2-highgpu-1g | 1× A100 40GB | No | ~$3.50 | Faster per-GPU, fewer nodes needed |

---

## Step 4 — Monitor

```bash
gcloud batch jobs describe infer-YYYYMMDD-HHMMSS \
    --project="${PROJECT}" --location="${REGION}"

# Live logs from all tasks
gcloud logging read \
    "resource.labels.job_uid=infer-YYYYMMDD-HHMMSS" \
    --project="${PROJECT}" --limit=200 --format="value(textPayload)"
```

---

## Step 5 — Collect the output

Outputs stay as `shard-*.jsonl` under `--output` and are meant to be read by wildcard, the way
the relevance pipeline consumes its CSVs. Sanity-check with:

```bash
gsutil ls -l "gs://BUCKET/extraction/output/run-001/shard-*.jsonl" | tail -1
```

Total line count should be close to total manifest rows, short by however many articles had a
missing or unreadable GCS object (those are dropped silently).

`merge_shards.sh` still exists for small runs:

```bash
./merge_shards.sh --output gs://BUCKET/extraction/output/run-001 --dest combined.jsonl --shards 20
```

**Do not use it above 32 shards** — that is the `gsutil compose` limit, past which it falls back
to downloading every shard locally and `cat`-ing them.

---

## Spot preemption / recovery

Three independent layers:

1. The entrypoint uploads partial output to GCS every 90 s, so a preemption loses ≤ ~90 s.
2. On retry the entrypoint pulls that partial output back down before starting.
3. `vllm_infer.py` skips already-processed articles — by `id` when rows carry one (both lean and
   full output do), falling back to a SHA-256 of the article text otherwise. In manifest mode
   the skip happens *before* the GCS fetch, so a resumed task doesn't re-download what it is
   about to discard.

**Verified 2026-08-10**: re-submitting an identical 2-shard job against an output prefix that
already held 2000 completed articles produced byte-identical shard files (same SHA-256, still
1000 rows each, no duplicate ids). Because `vllm_infer.py` opens the output in append mode, any
reprocessing would have appended rows — zero appended means every article was correctly skipped.

Spot tiers set `maxRetryCount: 10`, since a multi-hour shard can be preempted more than a
handful of times. **10 is the hard ceiling** — Cloud Batch rejects the job outright with
`OUT_OF_RANGE: max_retry_count with value N is not in between 0 and 10` for anything higher.
(`vertexai/inference/relevance/` still carries 50 in its templates and would be rejected the
same way.)

---

## Troubleshooting

**"Quota exceeded"** — request quota for `NVIDIA_L4_GPUS` (or `NVIDIA_A100_GPUS` / `PREEMPTIBLE_NVIDIA_A100_GPUS` for spot-a100) in your region via IAM & Admin → Quotas.

**"no manifest-*.jsonl found" / shard-count mismatch at submit** — the guard in `submit_batch.sh`.
Rebuild the manifest with the right `--num-shards`, or submit with `--shards` equal to the file count.

**Image not found** — confirm the Artifact Registry repo exists and the image was pushed successfully.

**Model load fails** — verify the GCS path contains the full merged model (should have `config.json`, `tokenizer.json`, `model-*.safetensors`).

**Output much shorter than the manifest** — articles whose GCS object is missing, unreadable, or
has empty text are dropped. Compare `wc -l` of a manifest shard against its `shard-N.jsonl`.

**`CODE_GCE_BAD_REQUEST` on a flex tier** — **the `flex-l4` and `flex-a100` tiers do not
currently work.** GCE rejects a `FLEX_START` VM carrying a `maxRunDuration` unless the instance
also has an *instance termination action*, and Cloud Batch's `AllocationPolicy.InstancePolicy`
has no field to express one (verified against the v1 discovery document: it accepts only
`machineType`, `minCpuPlatform`, `provisioningModel`, `accelerators`, `bootDisk`, `disks`,
`reservation`). Setting `taskSpec.maxRunDuration` is not sufficient — Batch still propagates its
own 7-day default to instance scheduling, and the job fails during scheduling with no task logs:

```
Batch Error: code - CODE_GCE_BAD_REQUEST, description - googleapi: Error 400:
Invalid value for field 'resource.properties.scheduling.maxRunDuration': '{ "seconds": "604800"}'.
max-run-duration for given provisioning model is not supported without an instance
termination action
```

Use `spot-a100` (or `spot-l4`) instead. Reproduced 2026-08-10 on `flex-a100`.

**Shard count mismatch after merge** — check if any tasks failed via `gcloud batch jobs describe`; re-run failed tasks or resubmit with the same `--output` prefix (recovery will skip completed articles).

---

## Example invocations

### Verified smoke test (2 shards x 1000 articles)

Exactly what was run on 2026-08-10, end to end and green:

```bash
# 1. Build a 2-shard manifest. ADC was broken locally, so this writes to a local dir and
#    uploads with gsutil; with working ADC you can pass the gs:// prefix directly.
python vertexai/inference/event-extraction/build_manifest.py \
    --output-prefix /tmp/manifests-smoke \
    --num-shards    2 \
    --limit         2000

gsutil -m cp /tmp/manifests-smoke/manifest-*.jsonl \
    gs://zhai-risk-factor-extraction/extraction/manifests-smoke/

# 2. Build and push the image (~4.5 min on Cloud Build)
bash vertexai/inference/event-extraction/setup.sh

# 3. Submit
cd vertexai/inference/event-extraction
./submit_batch.sh \
    --tier            spot-a100 \
    --shards          2 \
    --job-id          "extraction-smoke-$(date -u +%Y%m%d-%H%M%S)" \
    --manifest        gs://zhai-risk-factor-extraction/extraction/manifests-smoke \
    --output          gs://zhai-risk-factor-extraction/extraction/output/smoke \
    --max-model-len   12238 \
    --batch-size      200

# 4. Watch
gcloud batch jobs describe JOB_ID --project=zerohungerai --location=us-central1 \
    --format='value(status.state)'
```

Result: `SUCCEEDED`, both tasks, ~11.5 min wall clock. 2000/2000 rows out (no article dropped),
8,542 events, 4.27 events/article, 1,185 B/article, 0 JSON parse errors. 327 French articles
produced 1,413 events through the FR prompt path.

`--quantization fp8` is deliberately absent: A100 is SM80 and fp8 W8A8 needs Hopper/Ada. Older
docs paired `spot-a100` with `fp8`, but that template dropped the flag before it reached the
container, so fp8 was never actually exercised on A100 here.

### Full-corpus run (~34.4M relevant articles)

```bash
python vertexai/inference/event-extraction/build_manifest.py \
    --output-prefix gs://zhai-risk-factor-extraction/extraction/manifests \
    --num-shards    200

bash vertexai/inference/event-extraction/setup.sh

cd vertexai/inference/event-extraction
./submit_batch.sh \
    --tier          spot-a100 \
    --shards        200 \
    --job-id        "extraction-full-$(date -u +%Y%m%d-%H%M%S)" \
    --manifest      gs://zhai-risk-factor-extraction/extraction/manifests \
    --output        gs://zhai-risk-factor-extraction/extraction/output/run-001 \
    --max-model-len 12238 \
    --batch-size    500
```

### Local run against a manifest, no Cloud Batch

```bash
python scripts/event_extraction/inference/vllm_infer.py \
    --model_name_or_path outputs/extraction/qwen3_5-4b-lora-en-fr-a100-40gb-20260803-120523/merged \
    --input  /tmp/manifests-smoke/manifest-000.jsonl \
    --output /tmp/preds.jsonl \
    --gcs_input --lean_output \
    --max_model_len 12238 --max_chars 3000 --limit 200
```
