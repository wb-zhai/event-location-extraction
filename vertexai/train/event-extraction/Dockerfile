# nvcr.io/nvidia/pytorch tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# 26.05-py3 → PyTorch 2.12.0 | CUDA 12.9 | Flash Attention pre-built | Python 3.12
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

ARG PIP_INDEX=https://pypi.org/simple

# Pinned commit of https://github.com/hiyouga/LLaMA-Factory — verified to include the
# qwen3_nothink / qwen3_5_nothink chat templates the configs in this folder use.
ARG LF_COMMIT=2ebe7be6116e13ccb5be38f468dcd8cd8a07a051

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

SHELL ["/bin/bash", "-c"]
WORKDIR /app

RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"

RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /app && \
    cd /app && git checkout "${LF_COMMIT}"

COPY requirements-extra.txt /tmp/requirements-extra.txt

# Freeze torch/torchaudio/torchvision at the NGC-provided versions so that
# installing LlamaFactory's dependencies doesn't pull mismatched CUDA builds from PyPI.
# pip freeze emits file:// URLs for NGC packages (the .whl is gone after install),
# so we use pip list --format=json to get clean name==version constraints instead.
RUN python3 -c "import subprocess,json; pkgs=json.loads(subprocess.check_output(['pip','list','--format=json'])); print('\n'.join(f\"{p['name']}=={p['version']}\" for p in pkgs if p['name'].lower().startswith('torch')))" > /tmp/torch_constraints.txt && \
    pip install --no-cache-dir --no-build-isolation \
        -c /tmp/torch_constraints.txt \
        -e . && \
    pip install --no-cache-dir --no-build-isolation \
        -c /tmp/torch_constraints.txt \
        -r requirements/metrics.txt -r requirements/deepspeed.txt -r /tmp/requirements-extra.txt

# torchaudio is a LlamaFactory dependency only needed for audio inputs. The PyPI build
# ships a CUDA version that mismatches the NGC base image's PyTorch, which crashes
# `import torchaudio` at startup. The import is guarded in data/mm_plugin.py, so it is
# safe to drop here for text/vision training.
RUN pip uninstall -y torchaudio

# Install Google Cloud SDK for gsutil
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
