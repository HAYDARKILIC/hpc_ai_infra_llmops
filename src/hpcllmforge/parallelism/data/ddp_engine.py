"""A from-scratch DistributedDataParallel wrapper.

The reference implementation in ``torch.nn.parallel.DistributedDataParallel``
hides several engineering decisions that are critical for understanding why
DDP scales linearly while ``nn.DataParallel`` does not:

* **Gradient bucketing.** Gradients of consecutive parameters are concatenated
  into fixed-size buckets (default 25 MiB) so that allreduce launches are
  amortised across many tensors.
* **Overlap with backward.** Each bucket fires an asynchronous allreduce as
  soon as the last gradient in the bucket is produced, so communication is
  hidden behind compute.
* **No GIL contention.** Unlike ``DataParallel``, each rank runs its own Python
  process, eliminating the GIL bottleneck on the dispatcher thread.

This module replicates these behaviours with as little magic as possible.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .gradient_bucket import GradientBucket


class DDPEngine(nn.Module):
    """A pedagogical re-implementation of DistributedDataParallel.

    Parameters
    ----------
    module
        The model to wrap. Must already be moved to the local device.
    bucket_size_mb
        Target gradient-bucket size in MiB (default 25, matching the PyTorch
        reference).
    broadcast_buffers
        If ``True``, non-parameter buffers (e.g. BatchNorm running stats) are
        broadcast from rank 0 at the start of every forward pass.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        bucket_size_mb: float = 25.0,
        broadcast_buffers: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.broadcast_buffers = broadcast_buffers
        self._buckets: list[GradientBucket] = []

        # TODO: broadcast module.state_dict() from rank 0 to ensure identical
        # initialisation across workers.
        # TODO: register a per-parameter post-accumulate-grad hook that pushes
        # the gradient into its assigned bucket and fires allreduce on full buckets.
        # TODO: partition parameters into buckets respecting bucket_size_mb.

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if self.broadcast_buffers and dist.is_initialized():
            # TODO: broadcast buffers from rank 0.
            pass
        return self.module(*args, **kwargs)

    def finalize_backward(self) -> None:
        """Wait for all outstanding allreduces and write averaged grads back."""
        # TODO: for each bucket, wait on its work handle and copy reduced grads
        # back into the original parameter .grad tensors.
        raise NotImplementedError
