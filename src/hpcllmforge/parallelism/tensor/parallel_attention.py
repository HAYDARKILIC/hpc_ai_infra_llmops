"""Tensor-parallel multi-head attention.

The Megatron-LM partitioning strategy for a multi-head attention block is:

* The QKV projection is a *column-parallel* linear, so each TP rank holds a
  contiguous slice of attention heads. No communication is needed for the
  scaled dot-product attention itself — each rank operates on its local heads.
* The output projection is a *row-parallel* linear, which performs the
  AllReduce that re-assembles the contributions from all heads.

This pairing keeps the per-block communication cost down to a single AllReduce
on the activations, regardless of the number of heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .column_parallel import ColumnParallelLinear
from .row_parallel import RowParallelLinear


class TensorParallelSelfAttention(nn.Module):
    """A tensor-parallel multi-head self-attention block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        tp_world_size: int,
        tp_rank: int,
    ) -> None:
        super().__init__()
        if num_heads % tp_world_size != 0:
            raise ValueError("num_heads must be divisible by tp_world_size")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.head_dim = hidden_size // num_heads
        self.local_heads = num_heads // tp_world_size

        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=False,
        )
        self.out = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: project to local QKV, reshape into (B, H_local, T, head_dim),
        # apply scaled dot-product attention, then RowParallelLinear AllReduce.
        raise NotImplementedError
