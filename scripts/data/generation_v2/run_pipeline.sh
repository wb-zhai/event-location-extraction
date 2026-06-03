#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/data/generation_v2/run_pipeline.sh INPUT_JSONL OUTPUT_DIR [options]

Options:
  --ontology PATH                         Default: ontologies/zhai/ontology.json
  --limit N                               Sample at most N articles
  --seed N                                Sampling seed. Default: 13
  --keyword                               Boost likely food-insecurity risk-factor articles
  --model NAME                            Annotator model. Default: gemini-3.1-pro-preview
  --verifier-model NAME                   Verifier model. Default: gemini-3.1-pro-preview
  --temperature FLOAT                     Annotator temperature. Default: 0.0
  --verifier-temperature FLOAT            Verifier temperature. Default: 0.0
  --reasoning-effort VALUE                Annotator reasoning effort. Default: low
  --verifier-reasoning-effort VALUE       Verifier reasoning effort. Default: low
  --max-tokens N                          Default: 8192
  --workers N                             Default: 4
  --batch-api                             Use Gemini Batch API for annotation and verification
  --batch-size N                          Default: 1000
  --batch-poll-interval-seconds N         Default: 30
  --target-chars N                        Window target chars. Default: 6000
  --max-chars N                           Window max chars. Default: 9000
  --overlap-sentences N                   Default: 2
  --skip-verify                           Skip Gemini verification
  --overwrite                             Recreate outputs
  --env-file PATH                         Default: .env
  -h, --help                              Show this help

Outputs:
  sampled.jsonl, windows.jsonl, raw.jsonl, verified.jsonl, recovered.jsonl,
  review.html, report/summary.json, and runs/ under OUTPUT_DIR.
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

INPUT="$1"
OUTPUT_DIR="$2"
shift 2

if [[ "$OUTPUT_DIR" == *.jsonl ]]; then
  NORMALIZED_OUTPUT_DIR="${OUTPUT_DIR%.jsonl}"
  echo "Warning: OUTPUT_DIR ended with .jsonl; using directory '$NORMALIZED_OUTPUT_DIR' instead." >&2
  OUTPUT_DIR="$NORMALIZED_OUTPUT_DIR"
fi

ONTOLOGY="ontologies/zhai/ontology.json"
LIMIT=""
SEED="13"
KEYWORD="0"
MODEL="gemini-3.1-pro-preview"
VERIFIER_MODEL="gemini-3.1-pro-preview"
TEMPERATURE="0.0"
VERIFIER_TEMPERATURE="0.0"
REASONING_EFFORT="low"
VERIFIER_REASONING_EFFORT="low"
MAX_TOKENS="8192"
WORKERS="4"
BATCH_API="0"
BATCH_SIZE="1000"
BATCH_POLL_INTERVAL_SECONDS="30"
TARGET_CHARS="6000"
MAX_CHARS="9000"
OVERLAP_SENTENCES="2"
SKIP_VERIFY="0"
OVERWRITE="0"
ENV_FILE=".env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ontology) ONTOLOGY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --keyword) KEYWORD="1"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --verifier-model) VERIFIER_MODEL="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --verifier-temperature) VERIFIER_TEMPERATURE="$2"; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
    --verifier-reasoning-effort) VERIFIER_REASONING_EFFORT="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --batch-api) BATCH_API="1"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --batch-poll-interval-seconds) BATCH_POLL_INTERVAL_SECONDS="$2"; shift 2 ;;
    --target-chars) TARGET_CHARS="$2"; shift 2 ;;
    --max-chars) MAX_CHARS="$2"; shift 2 ;;
    --overlap-sentences) OVERLAP_SENTENCES="$2"; shift 2 ;;
    --skip-verify) SKIP_VERIFY="1"; shift ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

SAMPLED="$OUTPUT_DIR/sampled.jsonl"
WINDOWS="$OUTPUT_DIR/windows.jsonl"
RAW="$OUTPUT_DIR/raw.jsonl"
VERIFIED="$OUTPUT_DIR/verified.jsonl"
RECOVERED="$OUTPUT_DIR/recovered.jsonl"
REVIEW_HTML="$OUTPUT_DIR/review.html"
REPORT_DIR="$OUTPUT_DIR/report"
RUN_DIR="$OUTPUT_DIR/runs"

COMMON_GEMINI_ARGS=(
  --model "$MODEL"
  --verifier-model "$VERIFIER_MODEL"
  --temperature "$TEMPERATURE"
  --verifier-temperature "$VERIFIER_TEMPERATURE"
  --reasoning-effort "$REASONING_EFFORT"
  --verifier-reasoning-effort "$VERIFIER_REASONING_EFFORT"
  --max-tokens "$MAX_TOKENS"
  --workers "$WORKERS"
  --run-dir "$RUN_DIR"
  --env-file "$ENV_FILE"
  --retry-failed
)

if [[ "$BATCH_API" == "1" ]]; then
  COMMON_GEMINI_ARGS+=(--batch-api --batch-size "$BATCH_SIZE" --batch-poll-interval-seconds "$BATCH_POLL_INTERVAL_SECONDS")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_GEMINI_ARGS+=(--overwrite)
fi

SAMPLE_ARGS=(--ontology "$ONTOLOGY" --seed "$SEED")
if [[ -n "$LIMIT" ]]; then
  SAMPLE_ARGS+=(--limit "$LIMIT")
fi
if [[ "$KEYWORD" == "1" ]]; then
  SAMPLE_ARGS+=(--keyword)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  SAMPLE_ARGS+=(--overwrite)
fi

echo "[1/6] Sampling articles -> $SAMPLED"
uv run python scripts/data/generation_v2/sample_articles.py "$INPUT" "$SAMPLED" "${SAMPLE_ARGS[@]}"

WINDOW_ARGS=(
  --target-chars "$TARGET_CHARS"
  --max-chars "$MAX_CHARS"
  --overlap-sentences "$OVERLAP_SENTENCES"
)
if [[ "$OVERWRITE" == "1" ]]; then
  WINDOW_ARGS+=(--overwrite)
fi

echo "[2/6] Windowing articles -> $WINDOWS"
uv run python scripts/data/generation_v2/window_articles.py "$SAMPLED" "$WINDOWS" "${WINDOW_ARGS[@]}"

echo "[3/6] Annotating with Gemini -> $RAW"
uv run python scripts/data/generation_v2/annotate_gemini.py "$WINDOWS" "$RAW" \
  --ontology "$ONTOLOGY" \
  "${COMMON_GEMINI_ARGS[@]}"

if [[ "$SKIP_VERIFY" == "1" ]]; then
  echo "[4/6] Verification skipped; using raw annotations"
  if [[ -f "$VERIFIED" && "$OVERWRITE" != "1" ]]; then
    echo "Verified output already exists, skipping: $VERIFIED"
  else
    cp "$RAW" "$VERIFIED"
  fi
else
  echo "[4/6] Verifying with Gemini -> $VERIFIED"
  uv run python scripts/data/generation_v2/verify_gemini.py "$RAW" "$VERIFIED" \
    --ontology "$ONTOLOGY" \
    "${COMMON_GEMINI_ARGS[@]}"
fi

RECOVER_ARGS=(--html "$REVIEW_HTML")
if [[ "$OVERWRITE" == "1" ]]; then
  RECOVER_ARGS+=(--overwrite)
fi

echo "[5/6] Recovering annotations and writing review HTML -> $RECOVERED"
uv run python scripts/data/generation_v2/recover_annotations.py "$VERIFIED" "$RECOVERED" "${RECOVER_ARGS[@]}"

PRICING_MODE="standard"
if [[ "$BATCH_API" == "1" ]]; then
  PRICING_MODE="batch"
fi

echo "[6/6] Writing audit report -> $REPORT_DIR"
uv run python scripts/data/generation_v2/audit_report.py "$VERIFIED" "$REPORT_DIR" --pricing-mode "$PRICING_MODE"

echo "Pipeline complete."
echo "Verified annotations: $VERIFIED"
echo "Recovered annotations: $RECOVERED"
echo "Review HTML: $REVIEW_HTML"
echo "Report: $REPORT_DIR/summary.json"
