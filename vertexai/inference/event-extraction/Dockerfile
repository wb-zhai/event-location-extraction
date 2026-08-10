# Build context must be the repo root:
#   docker build -t IMAGE -f vertexai/inference/event-extraction/Dockerfile .
#
# Base image ships CUDA + torch + vLLM. Verify this tag exists:
#   docker pull vllm/vllm-openai:v0.23.0
FROM vllm/vllm-openai:v0.23.0

WORKDIR /app

# Runtime deps not bundled with vLLM
# gsutil: entrypoint.sh moves model/manifest/output to and from GCS; the vLLM base
# image has no Cloud SDK. Standalone (pip) gsutil does NOT auto-detect the VM
# service account — without config it runs as an anonymous caller. The boto
# config below tells it to authenticate with the GCE metadata service account.
# google-cloud-storage: gcs_stream.py streams article objects at inference time.
RUN pip install --no-cache-dir fire json-repair gsutil google-cloud-storage && \
    printf '[GoogleCompute]\nservice_account = default\n' > /etc/boto.cfg

# Repo code needed at inference time.
# inference/  → vllm_infer.py + gcs_stream.py
# generation/ → to_sft.py windowing helpers and the student_sft prompt templates
# src/ is deliberately NOT copied: this pipeline runs without the retriever.
COPY scripts/event_extraction/inference/  scripts/event_extraction/inference/
COPY scripts/event_extraction/generation/ scripts/event_extraction/generation/
COPY ontologies/ ontologies/

# Make scripts importable from repo root
ENV PYTHONPATH=/app

COPY vertexai/inference/event-extraction/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
