#!/usr/bin/env bash
# Merge per-shard output files from GCS into a single local JSONL.
#
# Usage:
#   ./merge_shards.sh \
#     --output  gs://BUCKET/output/run-001   # GCS prefix used in submit_batch.sh
#     --dest    combined.jsonl               # local output path
#     [--shards N]                           # optional: validate exactly N shards present

set -euo pipefail

OUTPUT_GCS=""
DEST="combined.jsonl"
EXPECTED_SHARDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)  OUTPUT_GCS="$2";      shift 2 ;;
        --dest)    DEST="$2";            shift 2 ;;
        --shards)  EXPECTED_SHARDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${OUTPUT_GCS}" ]] && { echo "ERROR: --output is required" >&2; exit 1; }

echo "Listing shards at ${OUTPUT_GCS}/shard-*.jsonl ..."
SHARD_LIST="$(gsutil ls "${OUTPUT_GCS}/shard-*.jsonl" 2>/dev/null | sort -t'-' -k2 -n)"
SHARD_COUNT="$(echo "${SHARD_LIST}" | grep -c 'shard-' || true)"

echo "Found ${SHARD_COUNT} shard(s)"

if [[ -n "${EXPECTED_SHARDS}" && "${SHARD_COUNT}" -ne "${EXPECTED_SHARDS}" ]]; then
    echo "WARNING: expected ${EXPECTED_SHARDS} shards but found ${SHARD_COUNT}" >&2
fi

if [[ "${SHARD_COUNT}" -eq 0 ]]; then
    echo "ERROR: no shards found at ${OUTPUT_GCS}" >&2
    exit 1
fi

# gsutil compose is limited to 32 objects; for larger counts use cat
if [[ "${SHARD_COUNT}" -le 32 ]]; then
    echo "Composing shards in GCS ..."
    COMPOSED="${OUTPUT_GCS}/combined_tmp.jsonl"
    # shellcheck disable=SC2086
    gsutil compose ${SHARD_LIST} "${COMPOSED}"
    echo "Downloading to ${DEST} ..."
    gsutil cp "${COMPOSED}" "${DEST}"
    gsutil rm "${COMPOSED}"
else
    echo "Downloading ${SHARD_COUNT} shards locally then concatenating ..."
    TMP_DIR="$(mktemp -d)"
    trap "rm -rf ${TMP_DIR}" EXIT
    gsutil -m cp "${OUTPUT_GCS}/shard-*.jsonl" "${TMP_DIR}/"
    # Sort numerically by shard index before concatenating
    ls "${TMP_DIR}"/shard-*.jsonl | sort -t'-' -k2 -n | xargs cat > "${DEST}"
fi

LINES="$(wc -l < "${DEST}")"
echo "Done. ${DEST}: ${LINES} articles"
