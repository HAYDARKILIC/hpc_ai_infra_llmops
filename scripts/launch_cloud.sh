#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command cloud launcher — RunPod / Lambda Labs / Vast.ai.
# Required env: WANDB_API_KEY, plus the provider's API key.
# ---------------------------------------------------------------------------
set -euo pipefail

PROVIDER="${PROVIDER:-runpod}"
GPU="${GPU:-H100}"
NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-configs/training/gpt_1b3.yaml}"
IMAGE="${IMAGE:-ghcr.io/HAYDARKILIC/hpc-llm-forge:latest}"

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

case "${PROVIDER}" in
    runpod) : "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set}" ;;
    lambda) : "${LAMBDA_API_KEY:?LAMBDA_API_KEY must be set}" ;;
    vastai) : "${VASTAI_API_KEY:?VASTAI_API_KEY must be set}" ;;
    *) echo "Unknown PROVIDER=${PROVIDER}"; exit 1 ;;
esac

echo "[launch_cloud] provider=${PROVIDER} gpu=${GPU} num_gpus=${NUM_GPUS}"
echo "[launch_cloud] image=${IMAGE} config=${CONFIG}"

python -m hpcllmforge.orchestration.cloud_launcher \
    --provider "${PROVIDER}" \
    --gpu-type "${GPU}" \
    --num-gpus "${NUM_GPUS}" \
    --image "${IMAGE}" \
    --command "make launch-deepspeed CONFIG=${CONFIG} NPROC=${NUM_GPUS}"
