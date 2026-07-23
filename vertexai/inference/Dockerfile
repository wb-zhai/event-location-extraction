# Build context must be the repo root:
#   docker build -t IMAGE -f vertexai/inference/Dockerfile .
#
# Base image ships CUDA + torch + vLLM. Verify this tag exists:
#   docker pull vllm/vllm-openai:v0.23.0
FROM vllm/vllm-openai:v0.23.0

WORKDIR /app

# Runtime deps not bundled with vLLM
# gsutil: entrypoint.sh moves model/input/output to and from GCS; the vLLM base
# image has no Cloud SDK. Standalone (pip) gsutil does NOT auto-detect the VM
# service account — without config it runs as an anonymous caller. The boto
# config below tells it to authenticate with the GCE metadata service account.
RUN pip install --no-cache-dir fire json-repair gsutil && \
    printf '[GoogleCompute]\nservice_account = default\n' > /etc/boto.cfg

# Repo code needed at inference time
COPY scripts/train/inference/vllm_infer.py scripts/train/inference/vllm_infer.py
COPY scripts/data/generation/ scripts/data/generation/
COPY ontologies/ ontologies/

# Make scripts importable from repo root
ENV PYTHONPATH=/app

COPY vertexai/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
