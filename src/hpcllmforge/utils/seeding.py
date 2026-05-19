"""Deterministic seed orchestration across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, and (CUDA) PyTorch RNGs.

    Parameters
    ----------
    seed
        Master seed.
    deterministic_algorithms
        If True, also call ``torch.use_deterministic_algorithms(True)`` and set
        ``CUBLAS_WORKSPACE_CONFIG`` to enforce bit-exact reproducibility
        (at a throughput cost of ~5–15%).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
