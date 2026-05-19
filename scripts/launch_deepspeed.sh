#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch DeepSpeed ZeRO training on the local node.
# ---------------------------------------------------------------------------
set -euo pipefail

NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
CONFIG="${CONFIG:-configs/training/gpt_1b3.yaml}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-configs/deepspeed/zero3_offload.json}"

echo "[launch_deepspeed] CONFIG=${CONFIG} DS_CFG=${DEEPSPEED_CFG} NPROC=${NPROC}"

deepspeed --num_gpus="${NPROC}" \
    examples/train_gpt_capstone.py \
        --config "${CONFIG}" \
        --parallelism zero \
        --deepspeed "${DEEPSPEED_CFG}"
