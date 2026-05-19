"""Gradient bucketing for overlap of allreduce with backward computation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GradientBucket:
    """A contiguous buffer holding the flattened gradients of several parameters.

    A bucket fires its allreduce as soon as every parameter assigned to it has
    produced a gradient. This is what enables communication-compute overlap
    during the backward pass.
    """

    params: list[torch.nn.Parameter] = field(default_factory=list)
    buffer: torch.Tensor | None = None
    pending: int = 0
    work: object | None = None

    @property
    def size_mb(self) -> float:
        """Cumulative parameter size in MiB."""
        return sum(p.numel() * p.element_size() for p in self.params) / 1024 / 1024

    def mark_ready(self, param: torch.nn.Parameter) -> bool:
        """Mark one parameter's gradient as ready; return True iff bucket full."""
        # TODO: copy param.grad into the bucket buffer at its assigned offset.
        # TODO: decrement self.pending and return self.pending == 0.
        raise NotImplementedError

    def launch_allreduce(self) -> None:
        """Fire an async all_reduce on the bucket buffer."""
        # TODO: self.work = dist.all_reduce(self.buffer, async_op=True)
        raise NotImplementedError
