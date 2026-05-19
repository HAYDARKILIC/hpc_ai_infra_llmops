#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch from-scratch DDP training on the local node.
# ---------------------------------------------------------------------------
set -euo pipefail

NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
CONFIG="${CONFIG:-configs/training/gpt_350m.yaml}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[launch_ddp] CONFIG=${CONFIG} NPROC=${NPROC}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    examples/train_gpt_capstone.py \
        --config "${CONFIG}" \
        --parallelism ddp
