"""ZeRO Stage 2 — Stage 1 + gradient partitioning.

In addition to partitioning the optimizer state, Stage 2 partitions the
gradient tensors. Each rank keeps only the slice of the gradient that
corresponds to its share of the optimizer state.

The implementation relies on a *reduce-scatter* during the backward pass
instead of an AllReduce. After reduce-scatter, each rank holds the summed
gradient for its own shard only, exactly matching the memory layout of the
optimizer state.

Memory per rank:  M_param + (M_grad + M_opt) / N_GPU.
"""

from __future__ import annotations

from .stage1 import ZeroStage1Optimizer


class ZeroStage2Optimizer(ZeroStage1Optimizer):
    """Stage 1 with gradient reduce-scatter instead of all-reduce."""

    def __init__(self, params, base_optimizer_cls, **base_optimizer_kwargs) -> None:
        super().__init__(params, base_optimizer_cls, **base_optimizer_kwargs)
        # TODO: install per-parameter backward hooks that:
        #   1. Bucket gradients by their owning rank.
        #   2. On bucket completion, fire dist.reduce_scatter so that the owner
        #      rank receives the summed gradient and all other ranks free their
        #      copies.
        # TODO: track per-shard gradient buffers.

    def step(self) -> None:
        # The gradients are already partitioned by reduce-scatter.
        # The local optimizer step proceeds exactly as in Stage 1.
        super().step()
