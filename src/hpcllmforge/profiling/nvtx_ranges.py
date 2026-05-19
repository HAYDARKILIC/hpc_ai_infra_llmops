"""Convenience wrappers around ``torch.cuda.nvtx`` for Nsight Systems profiling."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str):
    """Push an NVTX range for the duration of the ``with`` block."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
