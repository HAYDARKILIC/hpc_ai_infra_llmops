"""ZeRO Stage 1 — optimizer-state partitioning.

The simplest stage: each rank holds the *full* parameter and gradient tensors,
but only the slice of the optimizer state corresponding to its rank. After the
local optimizer step, parameter updates are broadcast (via AllGather) back to
every rank.

Memory per rank:  M_param + M_grad + M_opt / N_GPU.
"""

from __future__ import annotations

import torch


class ZeroStage1Optimizer:
    """A pedagogical Stage-1 optimizer wrapper.

    Parameters
    ----------
    params
        Iterable of model parameters (already replicated on every rank).
    base_optimizer_cls
        Underlying optimizer class (e.g. ``torch.optim.AdamW``).
    base_optimizer_kwargs
        Keyword arguments forwarded to the base optimizer constructor.
    """

    def __init__(self, params, base_optimizer_cls, **base_optimizer_kwargs) -> None:
        self.params = list(params)
        # TODO: partition self.params into N_GPU contiguous shards by element count.
        # TODO: construct the base optimizer only over the local shard.
        # TODO: store rank, world_size, and per-rank ownership maps.
        self._local_optimizer = None  # to be initialised in step()

    def step(self) -> None:
        # TODO:
        # 1. Run base_optimizer.step() on the local shard.
        # 2. AllGather the updated local shard across ranks.
        # 3. Copy gathered values back into self.params.
        raise NotImplementedError

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def state_dict(self) -> dict:
        # TODO: include the local-shard state plus the global partitioning plan.
        raise NotImplementedError
