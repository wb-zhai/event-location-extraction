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
COPY scripts/event_extraction/inference/historical/v1_production_21055/vllm_infer.py scripts/event_extraction/inference/historical/v1_production_21055/vllm_infer.py
COPY scripts/event_extraction/generation/ scripts/event_extraction/generation/
COPY ontologies/ ontologies/

# Make scripts importable from repo root
ENV PYTHONPATH=/app

COPY scripts/event_extraction/inference/historical/v1_production_21055/entrypoint.sh /entrypoint.sh
COPY scripts/event_extraction/inference/historical/v1_production_21055/system_prompt.txt scripts/event_extraction/inference/historical/v1_production_21055/system_prompt.txt
COPY scripts/event_extraction/inference/historical/v1_production_21055/user_prompt.txt scripts/event_extraction/inference/historical/v1_production_21055/user_prompt.txt
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
