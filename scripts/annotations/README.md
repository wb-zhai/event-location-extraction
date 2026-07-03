# Relevance annotation (Argilla)

Human annotation of article relevance (relevant / irrelevant to food-security
risk-event extraction), to build a gold set for scoring the Gemini relevance gate
in [`relevance_filter.py`](../data/relevance/relevance_filter.py).

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

## 2. Push articles for annotation

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

## 3. Export human labels

```bash
python scripts/annotations/relevance_argilla.py export \
  --dataset-name relevance-review \
  --output annotations.jsonl \
  --only-submitted
```

Each output row has `human_relevance`, `gemini_relevance`, and an `agreement`
flag, ready for scoring the Gemini gate's precision/recall against human
judgment.
