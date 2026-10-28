#!/usr/bin/env bash
set -euo pipefail

s="$1"

ROOT="global_v4_audit"

IN="$ROOT/postprocessing_v1/$s/candidate_geocoding_input.jsonl"
OUT="$ROOT/final_geocoded/${s}.geo.jsonl"
AUDIT="$ROOT/final_geocoded/audits/${s}.audit.json"
LOG="$ROOT/final_geocoded/logs/${s}.log"

python3 "$ROOT/materialize_final_geotaxonomy.py" \
  --input "$IN" \
  --output "$OUT" \
  --audit "$AUDIT" \
  --lookup-db /home/bonadossou/global_v4_work/global_v4_geotaxonomy_lookup.PATCHED.sqlite \
  --overwrite \
  > "$LOG" 2>&1

echo "DONE $s"
