# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# HPC-LLM-Forge — CUDA 12.4 / PyTorch 2.4 / DeepSpeed training image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NCCL_DEBUG=WARN \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0"

# ---- System layer (cache-friendly) ----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
        openssh-client pkg-config && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Python wheels (heavier, less volatile) -------------------------------
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install \
        deepspeed==0.15.0 \
        numpy==1.26.4 \
        pyyaml==6.0.1 \
        rich==13.7.1 \
        tqdm==4.66.4 \
        wandb==0.17.0

# ---- Application layer (most volatile) ------------------------------------
WORKDIR /workspace
COPY . /workspace
RUN pip install -e ".[deepspeed,cloud]"

ENTRYPOINT ["bash", "/workspace/docker/entrypoint.sh"]
CMD ["bash"]
