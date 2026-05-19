"""Bandwidth probes for the HBM, L2, and shared-memory tiers of the GPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BandwidthProbe:
    """Measured effective bandwidth (GB/s) at each tier of the GPU hierarchy."""

    hbm_gbs: float
    l2_gbs: float
    smem_gbs: float


def probe_device(device_idx: int = 0) -> BandwidthProbe:
    """Run a streaming-copy microbenchmark at each memory tier.

    The HBM tier is measured with a buffer larger than the L2 cache
    (≥ 100 MiB on H100); the L2 tier with a buffer that fits within L2;
    the SMEM tier with a small in-kernel shared-memory copy.
    """
    # TODO: implement using torch.empty + torch.cuda.Event timing or a custom CUDA kernel.
    raise NotImplementedError
