#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [options] <input_dir> <output_dir>"
    echo ""
    echo "Options:"
    echo "  --workers N    Parallel geocoding threads per file (default: 10)"
    echo "  --parallel N   Number of files to process in parallel (default: 4)"
    echo "  --delay N      Seconds between requests per thread (default: 0.0)"
    echo "  --pattern P    Glob pattern for input files (default: *.jsonl)"
    echo "  -h, --help     Show this help"
    exit 1
}

WORKERS=10
PARALLEL=4
DELAY=0.0
PATTERN="*.jsonl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)  WORKERS="$2";  shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --delay)    DELAY="$2";    shift 2 ;;
        --pattern)  PATTERN="$2";  shift 2 ;;
        -h|--help) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) break ;;
    esac
done

[[ $# -lt 2 ]] && usage

INPUT_DIR="$1"
OUTPUT_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEO_SCRIPT="$SCRIPT_DIR/add_geotaxonomy.py"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory '$INPUT_DIR' does not exist" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

files=("$INPUT_DIR"/$PATTERN)
if [[ ! -e "${files[0]}" ]]; then
    echo "No files matching '$PATTERN' in '$INPUT_DIR'" >&2
    exit 1
fi

total=${#files[@]}
count=0

echo "Processing $total file(s) with --parallel $PARALLEL (workers=$WORKERS per file, delay=$DELAY)" >&2

for input_file in "${files[@]}"; do
    count=$((count + 1))
    filename="$(basename "$input_file")"
    base="${filename%.jsonl}"
    output_file="$OUTPUT_DIR/${base}.geo.jsonl"

    echo "[$count/$total] starting $filename → $(basename "$output_file")" >&2

    python "$GEO_SCRIPT" "$input_file" -o "$output_file" \
        --workers "$WORKERS" \
        --delay "$DELAY" &

    # Keep at most $PARALLEL files running at once
    while [[ $(jobs -rp | wc -l) -ge $PARALLEL ]]; do
        wait -n 2>/dev/null || wait
    done
done

wait
echo "Done. Processed $total file(s) → $OUTPUT_DIR" >&2
