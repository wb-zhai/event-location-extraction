#!/usr/bin/env bash
# One-time setup: create Artifact Registry repo, build and push the inference image.
# Run from the repo root:
#   bash vertexai/inference/setup.sh [--local] [--tag TAG]
#
# Reads vertexai/inference/.env if present; CLI flags override.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${HERE}/.env" ]]; then
    set -a && source "${HERE}/.env" && set +a
fi

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
TAG="latest"
LOCAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;
        --region)  REGION="$2";  shift 2 ;;
        --tag)     TAG="$2";     shift 2 ;;
        --local)   LOCAL=true;   shift   ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${PROJECT}" ]] && { echo "ERROR: set GCP_PROJECT in .env or pass --project" >&2; exit 1; }

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/event-extraction/event-infer:${TAG}"
REPO="event-extraction"

echo "=== Config ==="
echo "  project : ${PROJECT}"
echo "  region  : ${REGION}"
echo "  image   : ${IMAGE}"
echo ""

# 1 — Create Artifact Registry repo (idempotent)
echo "--- Step 1: Artifact Registry repo ---"
if gcloud artifacts repositories describe "${REPO}" \
        --location="${REGION}" --project="${PROJECT}" &>/dev/null; then
    echo "Repo '${REPO}' already exists, skipping."
else
    gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT}"
    echo "Repo '${REPO}' created."
fi

# 2 — Build and push
echo ""
if [[ "${LOCAL}" == true ]]; then
    echo "--- Step 2: Local build + push ---"
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
    docker build -t "${IMAGE}" -f "${HERE}/Dockerfile" "${REPO_ROOT}"
    docker push "${IMAGE}"
else
    echo "--- Step 2: Cloud Build ---"
    gcloud builds submit . \
        --config="${HERE}/cloudbuild.yaml" \
        --substitutions="_REGION=${REGION},_TAG=${TAG}" \
        --project="${PROJECT}"
fi

echo ""
echo "Done. Image available at: ${IMAGE}"
echo "You can now run: vertexai/inference/submit_batch.sh ..."
