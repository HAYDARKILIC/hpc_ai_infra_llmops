"""Per-stage microbatch queue with FIFO ordering and activation pinning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Microbatch:
    """A single microbatch flowing through the pipeline."""

    microbatch_id: int
    activations: torch.Tensor | None = None
    saved_for_backward: tuple | None = None


class MicrobatchQueue:
    """A FIFO queue that pins activations until the matching backward fires."""

    def __init__(self) -> None:
        self._inflight: deque[Microbatch] = deque()

    def push(self, mb: Microbatch) -> None:
        self._inflight.append(mb)

    def pop(self) -> Microbatch:
        return self._inflight.popleft()

    def __len__(self) -> int:  # noqa: D105
        return len(self._inflight)
