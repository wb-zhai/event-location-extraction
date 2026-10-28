#!/usr/bin/env bash
set -euo pipefail

i="$1"
s=$(printf "%03d" "$i")

ROOT="global_v4_audit"
IN="$ROOT/postprocessing_v1/shard-$s/candidate_geocoding_input.jsonl"
OUT="$ROOT/final_geocoded/shard-$s.geo.jsonl"
AUDIT="$ROOT/final_geocoded/audits/shard-$s.audit.json"
LOG="$ROOT/final_geocoded/logs/shard-$s.log"

python3 "$ROOT/materialize_final_geotaxonomy.py" \
  --input "$IN" \
  --output "$OUT" \
  --audit "$AUDIT" \
  --lookup-db /home/bonadossou/global_v4_work/global_v4_geotaxonomy_lookup.sqlite \
  --overwrite \
  > "$LOG" 2>&1

echo "DONE shard-$s"
