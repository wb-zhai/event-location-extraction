# add_geotaxonomy.py

Enriches event location strings in a JSONL predictions file with structured geo taxonomy data by resolving each location through the [Photon geocoding API](https://photon.komoot.io/).

## What it does

For every line in the input JSONL, the script reads `annotation.events` and `predictions` event dicts, extracts each `event_location` string, geocodes it, and writes a `geotaxonomy` key back onto the event. The output is a new JSONL file with the same structure but with `geotaxonomy` fields added.

### Geotaxonomy schema

Each resolved location in the `geotaxonomy` list looks like:

```json
{
  "query": "Berlin",
  "resolved_name": "Berlin",
  "type": "city",
  "photon_type": "city",
  "lat": 52.5170365,
  "lon": 13.3888599,
  "osm_id": 62422,
  "osm_type": "R",
  "country": "Germany",
  "countrycode": "DE",
  "province": "Berlin",
  "district": "Berlin"
}
```

The `type` field is mapped from Photon's OSM place type to the project taxonomy:

| Photon type | Our type |
| --- | --- |
| `country` | `country` |
| `state` | `province` |
| `county`, `borough`, `district`, `municipality` | `district` |
| `city`, `town` | `city` |
| `village` | `village` |
| `suburb` | `suburb` |

Semicolon-separated location strings (e.g. `"Paris; Lyon"`) are split and each part is resolved independently. Values of `not_stated` are skipped.

## Requirements

- Python 3.10+
- `requests`, `tqdm`
- A running **Photon** instance at `http://localhost:2322` (see note below)

Install Python dependencies:

```bash
pip install requests tqdm
```

### Photon server

The script defaults to a **local** Photon instance (`http://localhost:2322`). To use the public Photon API instead, change `PHOTON_URL` at the top of the script to `https://photon.komoot.io/api/` and respect their rate-limit (use `--delay 1.0` and `--workers 1`).

## Usage

### Single file

```bash
python add_geotaxonomy.py input.jsonl
# Output: input.geo.jsonl

python add_geotaxonomy.py input.jsonl -o output.jsonl
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `input` | — | Input JSONL file (required) |
| `-o`, `--output` | `<input>.geo.jsonl` | Output file path |
| `--workers` | `1` | Parallel geocoding threads |
| `--delay` | `0.1` | Seconds between requests per thread |

### Batch processing a directory

`run_geotaxonomy.sh` runs the script over all `.jsonl` files in a directory:

```bash
./run_geotaxonomy.sh [options] <input_dir> <output_dir>

# Options:
#   --workers N    Parallel threads per file (default: 10)
#   --delay N      Seconds between requests per thread (default: 0.0)
#   --pattern P    Glob pattern for input files (default: *.jsonl)
```

Example:

```bash
./run_geotaxonomy.sh --workers 10 data/predictions/ data/predictions-geo/
```

## Ranking

Photon returns up to 10 candidates per query. The script picks the best one using a composite score that replicates Photon's own importance × reranker pipeline:

```text
score = reranker_factor × importance_proxy
```

Each signal guards against a different failure mode that arises with free-text event location strings:

- **Importance proxy alone** would pick the most geographically significant entity regardless of name. A query for "Berlin" could resolve to a large rural relation called "Berlin Township, Ohio" simply because it is an OSM relation with broad coverage.
- **Reranker factor alone** would pick the best string match regardless of significance. A query for "Germany" could resolve to a minor OSM node called "Germany" in Pennsylvania because the name is an exact match.

The multiplicative combination means a candidate needs both a reasonable name match and geographic significance to rank first — neither signal can fully override the other. The minimum reranker factor of 0.5 (no match) ensures that even a completely unrelated name only halves the importance score rather than zeroing it out, which mirrors how Photon balances OpenSearch BM25 scores against importance weights internally.

### Importance proxy

A weighted blend of two signals, both derived from OSM properties in the API response:

```text
importance_proxy = 0.7 × osm_type_score + 0.3 × geo_scope_score
```

**OSM object type** (`osm_type`) — a proxy for geographic area covered:

| osm_type | Score |
| --- | --- |
| `R` (relation) | 1.0 |
| `W` (way) | 0.6 |
| `N` (node) | 0.3 |

**Geographic scope** (`type`) — how broad the place is:

| type | Score |
| --- | --- |
| `continent` | 1.0 |
| `country` | 0.9 |
| `state` | 0.7 |
| `county`, `borough`, `municipality`, `district` | 0.55 |
| `city` | 0.5 |
| `town` | 0.4 |
| `village` | 0.3 |
| `suburb` | 0.25 |

For example, Germany as an OSM relation (`R`) with type `country` scores `0.7×1.0 + 0.3×0.9 = 0.97`.

### Reranker factor

A string-match multiplier applied to the importance proxy, matching the tiers Photon uses internally:

| Condition | Factor |
| --- | --- |
| `name == query` (exact, case-insensitive) | 1.0 |
| `name` starts with `"query "` (word boundary) | 0.9 |
| `name` starts with `query` (no boundary) | 0.8 |
| A word in `name` contains the full `query` | `0.8 × min(matched_chars / len(query), 1.0)` |
| No match | 0.5 |

A high-importance result with no name match (factor 0.5) can still outscore a low-importance exact match, which mirrors Photon's behaviour.

The top-scoring candidate is written to `geotaxonomy`.

## Caching

Resolved locations are cached in-memory for the duration of a run. Duplicate location strings within a file are only geocoded once.

---

# to_csv_ingest.py

Converts `.geo.jsonl` prediction files (produced by `add_geotaxonomy.py`) into two flat CSVs ready for database ingestion.

## Output

| File | Columns | Description |
| --- | --- | --- |
| `risk_matches.csv` | `article_uri, risk_id` | One row per unique (article, risk factor) pair |
| `locations.csv` | `article_uri, adm_code` | One row per unique (article, administrative region) pair |

`article_uri` is the `source.cloud_uri` value from the JSONL record. `risk_id` and `adm_code` are foreign keys into the `risk_factors` and `geo_taxonomy` reference tables respectively.

## Validation

- Every `event_type` in the input must match a `name` in `res/risk_factors.csv` — the script exits with an error if one is missing.
- Every resolved `adm_code` in a geotaxonomy entry must appear in `res/geo_taxonomy.csv` — the script exits with an error if one is missing.
- Geotaxonomy entries without an `adm_code` (unresolvable locations) are silently skipped.

## Usage

### Single file

```bash
python to_csv_ingest.py path/to/shard.geo.jsonl
# Output: risk_matches.csv and locations.csv written next to the input file
```

### Folder (merges all shards)

```bash
python to_csv_ingest.py path/to/predictions_dir/
# Reads all *.geo.jsonl files in the folder and writes merged CSVs into the same folder
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `input` | — | Path to a `.geo.jsonl` file or a folder of them (required) |
| `--out-dir` | input file/folder directory | Directory where output CSVs are written |
| `--risk-factors` | `res/risk_factors.csv` | Override risk factors reference file |
| `--geo-taxonomy` | `res/geo_taxonomy.csv` | Override geo taxonomy reference file |
