"""CPU and NVMe offload for ZeRO partitioned state.

When even Stage-3 sharding does not fit a model on the available GPUs, the
partitioned state can be **offloaded** to host (CPU) or NVMe storage, and
streamed back to the GPU on demand.

Bandwidth budget
----------------
* HBM3e (H100)          : ~3.3 TB/s
* NVLink 4              : ~900 GB/s per direction
* PCIe Gen5 x16         : ~64 GB/s (CPU offload)
* NVMe PCIe Gen4 x4     : ~7 GB/s  (NVMe offload)

Offload is therefore viable only when prefetching can hide the host-device
transfer behind GPU compute. This module implements double-buffered
asynchronous transfers with pinned-memory staging buffers.
"""

from __future__ import annotations

from enum import Enum

import torch


class OffloadTier(str, Enum):
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


class OffloadEngine:
    """Stream partitioned state between GPU, CPU, and NVMe with double buffering."""

    def __init__(self, tier: OffloadTier, *, nvme_path: str | None = None) -> None:
        self.tier = tier
        self.nvme_path = nvme_path
        # TODO: allocate pinned-memory staging buffers and CUDA streams for
        # prefetch / writeback.

    def offload(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move ``tensor`` to the offload tier and return a handle."""
        raise NotImplementedError

    def prefetch(self, handle: torch.Tensor) -> torch.Tensor:
        """Asynchronously stage ``handle`` back to GPU."""
        raise NotImplementedError
