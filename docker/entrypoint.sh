#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Entrypoint for hpc-llm-forge containers.
# ---------------------------------------------------------------------------
set -euo pipefail

# Echo a one-line environment fingerprint so the WandB log records the hardware.
echo "[entrypoint] $(date -Iseconds)"
echo "[entrypoint] python: $(python --version 2>&1)"
echo "[entrypoint] torch:  $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Authenticate WandB if a key is present (do NOT log it).
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin --quiet || echo "[entrypoint] wandb login failed (non-fatal)"
fi

exec "$@"
