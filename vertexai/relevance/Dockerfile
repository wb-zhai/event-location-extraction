# Build context must be this folder:
#   docker build -t IMAGE -f vertexai/relevance/Dockerfile vertexai/relevance
# (the pipeline is self-contained — nothing outside vertexai/relevance/ is needed).
#
# Base image ships CUDA + torch + vLLM. Verify this tag exists:
#   docker pull vllm/vllm-openai:v0.23.0
FROM vllm/vllm-openai:v0.23.0

WORKDIR /app

# Runtime deps not bundled with vLLM.
# gsutil: entrypoint.sh moves model/manifest/output to and from GCS; the vLLM base
# image has no Cloud SDK. Standalone (pip) gsutil does NOT auto-detect the VM service
# account — the boto config below tells it to authenticate with the GCE metadata SA.
# google-cloud-storage: relevance_vllm_infer.py reads article objects from GCS.
RUN pip install --no-cache-dir fire google-cloud-storage gsutil && \
    printf '[GoogleCompute]\nservice_account = default\n' > /etc/boto.cfg

# The self-contained inference script (no repo-wide COPY).
COPY relevance_vllm_infer.py /app/relevance_vllm_infer.py

ENV PYTHONPATH=/app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
