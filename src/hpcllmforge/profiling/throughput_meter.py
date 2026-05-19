"""Throughput / MFU (Model FLOPs Utilization) tracker.

MFU
---
Given a model with ``F_step`` FLOPs per training step (forward + backward, ~6N
per token for a Transformer with N parameters), measured step time ``t_step``,
and device peak FLOPs ``F_peak``,

    MFU = (F_step / t_step) / F_peak.

A well-tuned H100 BF16 run typically reaches 40-55% MFU. This module computes
MFU online and logs it to WandB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ThroughputMeter:
    """Online throughput / MFU estimator."""

    flops_per_step: float
    peak_flops: float
    window: int = 50

    _times: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._times = []

    def tick(self) -> None:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self.window + 1:
            self._times.pop(0)

    def mfu(self) -> float | None:
        if len(self._times) < 2:
            return None
        step_time = (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        return (self.flops_per_step / step_time) / self.peak_flops
