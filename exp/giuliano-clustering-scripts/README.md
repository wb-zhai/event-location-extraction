# Compact 3-day Table 3 builder

`3d.py` is the consolidated replacement for the older interval-3d and
continuous-7d builders. `build_table3.py` is an equivalent, discoverable
entry point for workflows that expect a `build_table3` script.

## What it does

- Reads every `data/table2_wb_raw_mentions/shard-*.jsonl.gz` mention.
- Preserves every source row in an assignments Parquet file.
- Clusters only valid exact-day start dates with `adm_code`, `adm_level`, and
  `event_type` present.
- Uses interval-aware connected components: a mention joins the current
  component when its start is no more than three days after the running
  maximum end date for that component.
- Keeps singleton clusters and records normalized dates, severity labels,
  precision, and assignment status for auditing.
- Validates row conservation, complete eligible assignment, unique IDs, and
  cluster boundaries before reporting success.

Month-only, year-only, unresolved, open-ended, and invalid dates are retained
in assignments but are not given an invented day for clustering.

## Install

From the repository root:

```bash
python -m pip install duckdb
```

## Run

From the repository root:

```bash
python event-location-extraction/exp/giuliano-clustering-scripts/3d.py
```


Useful options:

```bash
python event-location-extraction/exp/giuliano-clustering-scripts/3d.py \
  --gap-days 3 \
  --threads 6 \
  --memory-limit 8GB
```

The default outputs are:

- `data/handoff/table3_wb_3d.parquet`: one row per cluster.
- `data/handoff/table3_wb_3d_assignments.parquet`: one row per source mention.
- `data/handoff/table3_wb_3d_summary.json`: counts, audit categories, and
  validation results.
- `data/_3d.duckdb`: the temporary/queryable build database.

The cluster Parquet can be passed to the existing `evaluate_clustering.py`
script for CSV summaries and plots:

```bash
python event-location-extraction/exp/giuliano-clustering-scripts/evaluate_clustering.py \
  data/handoff/table3_wb_3d.parquet \
  --output-dir analysis_output
```

The input glob must be quoted only when the shell would otherwise expand it
incorrectly; DuckDB accepts the glob directly.
