# Build context must be the repo root:
#   docker build -t IMAGE -f vertexai/inference/Dockerfile .
#
# Base image ships CUDA + torch + vLLM. Verify this tag exists:
#   docker pull vllm/vllm-openai:v0.23.0
FROM vllm/vllm-openai:v0.23.0

WORKDIR /app

# Runtime deps not bundled with vLLM
RUN pip install --no-cache-dir fire json-repair

# Repo code needed at inference time
COPY scripts/train/inference/vllm_infer.py scripts/train/inference/vllm_infer.py
COPY scripts/data/generation_v3/ scripts/data/generation_v3/
COPY ontologies/ ontologies/

# Make scripts importable from repo root
ENV PYTHONPATH=/app

COPY vertexai/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
