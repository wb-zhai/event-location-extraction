# scripts/location — Location Extraction Inference

Extracts location mentions (toponym strings) from news article text.
Output is fed downstream to Photon for georeferencing.

## Backends

| Backend | Speed | Label set | Notes |
|---|---|---|---|
| `spacy` | **Fastest** — CPU-friendly, `nlp.pipe` batching | `GPE`, `LOC`, `FAC` | No per-entity confidence (fixed 1.0). Default model: `en_core_web_lg`. |
| `gliner2` | Slower (encoder, GPU recommended) | `city`, `country`, `region`, `facility`, `landmark` | Zero-shot custom schema; handles ambiguous/novel toponyms well. Default model: `fastino/gliner2-large-v1`. |

Use **spaCy** when throughput is the priority (e.g., 140M articles). Use **GLiNER2** when label flexibility or zero-shot accuracy matters more than speed.

## Usage

```zsh
# spaCy (fastest)
uv run scripts/location/run_inference.py \
    --input_dir data/in \
    --output_dir data/out \
    --backend spacy

# GLiNER2
uv run scripts/location/run_inference.py \
    --input_dir data/in \
    --output_dir data/out \
    --backend gliner2 \
    --threshold 0.5

# Collapse all labels to "location" (simplifies downstream Photon queries)
uv run scripts/location/run_inference.py \
    --input_dir data/in \
    --output_dir data/out \
    --backend spacy \
    --coarse

# Override model, adjust batch size
uv run scripts/location/run_inference.py \
    --input_dir data/in \
    --output_dir data/out \
    --backend spacy \
    --model en_core_web_trf \
    --batch_size 16

# Dry-run on first 100 rows
uv run scripts/location/run_inference.py \
    --input_dir data/in \
    --output_dir data/out \
    --backend gliner2 \
    --limit 100
```

## Input / Output format

**Input** — each line in a `*.jsonl` file must be a JSON object with a `"text"` field:
```json
{"id": "abc", "text": "Floods in northern Bangladesh displaced thousands."}
```

**Output** — same object with a `"locations"` list appended:
```json
{
  "id": "abc",
  "text": "Floods in northern Bangladesh displaced thousands.",
  "locations": [
    {"text": "Bangladesh", "label": "GPE", "start": 18, "end": 28, "confidence": 1.0, "source": "spacy"}
  ]
}
```

## Resumability

Re-running on an existing `output_dir` skips rows whose `text` hash is already present. Safe to interrupt and restart. Pass `--overwrite` to reprocess everything.

## Notes

- spaCy `confidence` is always `1.0` — the CNN pipeline does not expose per-entity scores.
- GLiNER2 chunks long texts with a sliding window (`~1500 chars`) to stay within the model's token limit; spans are re-mapped to absolute offsets.
- `--coarse` collapses all fine-grained labels to `"location"` at output time, which can simplify Photon query logic downstream.
