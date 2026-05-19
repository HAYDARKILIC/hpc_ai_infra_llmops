"""From-scratch Ring-AllReduce.

Theory
------
For ``N`` workers each holding a tensor of size ``S`` bytes, the bandwidth-
optimal AllReduce moves a total of

    C(N, S) = 2 * (N - 1) * (S / N)   bytes per worker.

The algorithm proceeds in two phases:

1. **Reduce-scatter** — over ``N - 1`` steps each worker sends a chunk to its
   right neighbour and receives a chunk from its left neighbour, summing the
   received chunk into its local copy. After this phase, each worker holds the
   final reduced value for one chunk.
2. **All-gather** — over ``N - 1`` steps each worker rotates its reduced chunk
   around the ring so that every worker ends up with the full reduced tensor.

The asymptotic cost is independent of the number of workers for large ``S``,
which is what makes Ring-AllReduce bandwidth-optimal.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def ring_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """Perform a bandwidth-optimal Ring-AllReduce sum across the default group.

    Parameters
    ----------
    tensor
        A contiguous tensor on the calling rank. It is divided into ``world_size``
        chunks for the reduce-scatter phase. The function is in-place but also
        returns the tensor for ergonomics.

    Returns
    -------
    torch.Tensor
        The tensor with the global sum across all ranks.

    Notes
    -----
    This implementation is for *pedagogical* purposes — it uses the slower
    point-to-point primitives (``dist.send`` / ``dist.recv``) so that the
    algorithm is visible. For production, use ``dist.all_reduce`` which calls
    NCCL's tuned implementation.
    """
    # TODO:
    # 1. Determine world_size, rank, left_neighbour, right_neighbour.
    # 2. Split the flattened tensor into world_size contiguous chunks.
    # 3. Reduce-scatter phase: for step in range(world_size - 1):
    #        send chunk[(rank - step) % N] right; recv into chunk[(rank - step - 1) % N]; sum.
    # 4. All-gather phase: for step in range(world_size - 1):
    #        send chunk[(rank - step + 1) % N] right; recv into chunk[(rank - step) % N].
    # 5. Return the reassembled tensor.
    raise NotImplementedError


def benchmark_against_nccl(tensor_size_mb: int, iters: int = 100) -> dict[str, float]:
    """Benchmark this implementation vs. ``dist.all_reduce`` (NCCL) for correctness and speed."""
    # TODO: warm up both, time with CUDA events, assert torch.allclose for correctness.
    raise NotImplementedError
