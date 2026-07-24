#!/usr/bin/env bash
# Build and push the training image to Artifact Registry.
#
# Usage:
#   ./build_push.sh [TAG] [--cloud]
#
# Reads vertexai/train/event-extraction/.env if present.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${HERE}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a && source "${HERE}/.env" && set +a
fi

: "${GCP_PROJECT:?GCP_PROJECT must be set}"
: "${GCP_REGION:?GCP_REGION must be set}"

TAG="${1:-latest}"
IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/llamafactory/trainer:${TAG}"

if [[ "${2:-}" == "--cloud" || "${1:-}" == "--cloud" ]]; then
    # Build and push entirely in GCP — no local pull required
    TAG="${2:-latest}"
    [[ "${1}" == "--cloud" ]] && TAG="latest"
    IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/llamafactory/trainer:${TAG}"
    echo "Submitting Cloud Build: ${IMAGE}"
    gcloud beta builds submit "${HERE}" \
        --config="${HERE}/cloudbuild.yaml" \
        --substitutions="_REGION=${GCP_REGION},_TAG=${TAG}" \
        --project="${GCP_PROJECT}"
else
    # Local build — pulls the NGC base image (~20 GB) on first run
    echo "Building locally: ${IMAGE}"
    docker build \
        -f "${HERE}/Dockerfile" \
        -t "${IMAGE}" \
        "${HERE}"
    echo "Pushing: ${IMAGE}"
    docker push "${IMAGE}"
fi

echo ""
echo "IMAGE=${IMAGE}"
echo "Set this in vertexai/train/event-extraction/.env before running submit_job.sh"
