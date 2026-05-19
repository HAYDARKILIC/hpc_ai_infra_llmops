"""Spot-resilient checkpoint manager.

Optimal checkpoint frequency
----------------------------
Let ``λ`` denote the preemption rate (events per hour), ``C_ckpt`` the cost
of writing a checkpoint (seconds), and ``T_recover`` the cost of restarting
from the most recent checkpoint. Minimising expected lost work yields

    f* = sqrt(λ / C_ckpt),

i.e. the optimal checkpoint interval is the geometric mean of the preemption
time and the checkpoint cost. This module records empirical preemption events
and self-tunes the interval over time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class CheckpointManager:
    """A checkpoint scheduler with optimal-interval self-tuning."""

    out_dir: str
    initial_interval_sec: float = 600.0
    min_interval_sec: float = 60.0
    max_interval_sec: float = 3600.0

    _last_save: float = 0.0
    _preempt_times: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._preempt_times = []
        self._last_save = time.time()

    def should_checkpoint(self) -> bool:
        return (time.time() - self._last_save) >= self._current_interval()

    def record_preemption(self, ts: float) -> None:
        self._preempt_times.append(ts)

    def _current_interval(self) -> float:
        if len(self._preempt_times) < 2:
            return self.initial_interval_sec
        # Estimate λ as 1 / (mean gap between preemptions).
        gaps = [b - a for a, b in zip(self._preempt_times, self._preempt_times[1:], strict=False)]
        lam = 1.0 / (sum(gaps) / len(gaps))
        # Assume a fixed checkpoint cost C_ckpt (could be measured).
        c_ckpt = 30.0
        return min(self.max_interval_sec, max(self.min_interval_sec, math.sqrt(lam / c_ckpt) * 3600))
