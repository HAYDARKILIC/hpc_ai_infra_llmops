"""Roofline analyzer and CUDA memory tracer.

Theory
------
For a kernel with arithmetic intensity I = FLOPs / Bytes, the attainable
performance under the roofline model is

    P_attainable = min(P_peak, B_peak * I),

where P_peak is the device peak FLOPS and B_peak is the peak memory bandwidth.
A kernel is *memory-bound* iff I < P_peak / B_peak (the so-called "machine
balance"), and *compute-bound* otherwise.

This module measures per-layer FLOPs and byte traffic, computes the ridge point
of the host device, and classifies kernels accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RooflinePoint:
    """A single (intensity, throughput) sample on the roofline plot."""

    layer_name: str
    flops: int
    bytes_read: int
    bytes_written: int
    measured_ms: float

    @property
    def arithmetic_intensity(self) -> float:
        """Compute arithmetic intensity I = FLOPs / total bytes moved."""
        total_bytes = self.bytes_read + self.bytes_written
        return self.flops / max(total_bytes, 1)

    @property
    def measured_tflops(self) -> float:
        """Achieved throughput in TFLOPS."""
        return self.flops / (self.measured_ms * 1e-3) / 1e12


def measure_layer(
    layer: nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 50,
) -> RooflinePoint:
    """Time a single layer with CUDA events and estimate FLOPs / byte traffic.

    Parameters
    ----------
    layer
        The PyTorch module under test (already moved to the target device).
    sample_input
        A representative input tensor on the same device.
    warmup
        Number of warm-up iterations before measurement begins.
    iters
        Number of measured forward passes; the median is reported.

    Returns
    -------
    RooflinePoint
        A frozen dataclass with FLOPs, byte traffic, and measured time.
    """
    # TODO: use torch.cuda.Event(enable_timing=True) for accurate ms timing.
    # TODO: count FLOPs via torch.utils.flop_counter.FlopCounterMode.
    # TODO: estimate bytes from parameter & activation sizes in the chosen dtype.
    raise NotImplementedError


def classify(point: RooflinePoint, peak_tflops: float, peak_bw_gbs: float) -> str:
    """Classify a measurement as ``memory-bound`` or ``compute-bound``."""
    ridge_intensity = (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)
    return "memory-bound" if point.arithmetic_intensity < ridge_intensity else "compute-bound"
