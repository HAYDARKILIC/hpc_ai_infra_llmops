"""Pipeline bubble visualization and theoretical/empirical comparison."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BubbleReport:
    """Numeric report on a pipeline schedule's idle fraction."""

    num_stages: int
    num_microbatches: int
    theoretical_bubble: float
    measured_bubble: float

    @property
    def utilization(self) -> float:
        """1 - measured_bubble."""
        return 1.0 - self.measured_bubble


def render_schedule(num_stages: int, num_microbatches: int) -> str:
    """Render an ASCII timeline of the 1F1B schedule.

    Each row represents one stage; each column one time step. ``F<id>`` denotes
    a forward pass of microbatch ``id``; ``B<id>`` denotes its backward.
    Empty cells correspond to bubble time.
    """
    # TODO: emit a deterministic ASCII chart suitable for inclusion in notebooks.
    raise NotImplementedError


def analyze(num_stages: int, num_microbatches: int, measured_step_ms: float, ideal_step_ms: float) -> BubbleReport:
    """Compare the measured wallclock to the bubble-free ideal."""
    theoretical = (num_stages - 1) / num_microbatches
    measured = 1.0 - (ideal_step_ms / measured_step_ms)
    return BubbleReport(num_stages, num_microbatches, theoretical, measured)
