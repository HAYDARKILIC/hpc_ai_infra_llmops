"""One-Forward-One-Backward (1F1B) pipeline scheduler.

Theory
------
Let ``p`` denote the number of pipeline stages and ``m`` the number of
microbatches. A naive (all-forward-then-all-backward) GPipe schedule has a
peak activation memory proportional to ``m``, because every forward activation
must be kept alive until its backward begins.

The **1F1B** schedule alternates forward and backward microbatches once the
pipeline is full, so that the in-flight activation count at each stage is
bounded by ``p`` (independent of ``m``). The pipeline bubble fraction is

    ρ_bubble = (p - 1) / m,

which vanishes as ``m → ∞``. This is the schedule used by Megatron-LM,
PipeDream, and the DeepSpeed pipeline engine.

References
----------
* Huang et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*.
* Narayanan et al., *PipeDream*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .microbatch_queue import MicrobatchQueue


@dataclass
class PipelineStage:
    """A single stage in the pipeline — a submodule + its rank assignment."""

    module: nn.Module
    device: str
    stage_id: int


class OneForwardOneBackwardScheduler:
    """1F1B pipeline scheduler.

    Parameters
    ----------
    stages
        The ordered list of pipeline stages (one per device).
    num_microbatches
        The number of microbatches per macrobatch. Larger values shrink the
        pipeline bubble at the cost of more activation memory per stage.
    """

    def __init__(self, stages: list[PipelineStage], num_microbatches: int) -> None:
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.queue = MicrobatchQueue()

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def bubble_fraction(self) -> float:
        """Theoretical 1F1B bubble fraction (p - 1) / m."""
        return (self.num_stages - 1) / self.num_microbatches

    def step(self, microbatches: list) -> list:
        """Execute one optimizer step across all microbatches.

        The schedule emitted on each rank is conceptually::

            warmup        : F F F ... F          (p - rank - 1 microbatches)
            steady-state  : F B F B F B ...      (m - (p - rank - 1) microbatches)
            cooldown      : B B B ... B          (p - rank - 1 microbatches)
        """
        # TODO:
        # 1. For rank r, run (p - r - 1) warm-up forwards.
        # 2. Enter steady state: alternate one forward and one backward.
        # 3. After all forwards are consumed, drain remaining backwards.
        # 4. Use point-to-point send/recv between consecutive stages.
        raise NotImplementedError
