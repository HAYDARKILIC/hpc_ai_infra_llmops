"""Integration tests — exercised under torchrun in CI.

These tests are marked as ``distributed`` and skipped unless a multi-GPU
environment is detected.
"""

from __future__ import annotations

import os

import pytest


distributed = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="Requires WORLD_SIZE >= 2 (run under torchrun).",
)


@distributed
def test_ring_allreduce_matches_nccl():
    """Skeleton: the from-scratch ring should match dist.all_reduce bit-exactly."""
    # TODO: import ring_allreduce, run on a known input, assert torch.allclose.
    pytest.skip("ring_allreduce implementation is a TODO")
