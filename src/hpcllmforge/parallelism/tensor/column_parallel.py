"""Column-parallel ``Linear`` — Megatron-LM style.

Theory
------
Consider the linear layer ``Y = X A`` where ``A ∈ R^{d_in × d_out}``. In
column-parallel decomposition the weight matrix is split *along the output
dimension*:

    A = [A_1, A_2, ..., A_p],   each A_i ∈ R^{d_in × (d_out / p)}.

Each of the ``p`` workers computes a partial output

    Y_i = X A_i,

and the global output is the concatenation ``Y = [Y_1, ..., Y_p]``. No
communication is needed for the forward pass beyond a single broadcast of the
input ``X``. In the backward pass the gradient w.r.t. ``X`` is
``∂L/∂X = Σ_i (∂L/∂Y_i) A_iᵀ``, which requires an AllReduce.

This layer is typically paired with a row-parallel layer downstream so that the
AllReduce on the activations is amortised across two consecutive matmuls — the
canonical Megatron-LM MLP pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ColumnParallelLinear(nn.Module):
    """A linear layer whose weight matrix is partitioned along the output dim."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_world_size: int,
        tp_rank: int,
        bias: bool = True,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        if out_features % tp_world_size != 0:
            raise ValueError("out_features must be divisible by tp_world_size")
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output

        local_out = out_features // tp_world_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        self.bias = nn.Parameter(torch.empty(local_out)) if bias else None
        # TODO: initialise weight (e.g. xavier_uniform_ then slice deterministically
        # so that all ranks share the same global init).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            # TODO: all-gather across tp_world_size and concatenate along last dim.
            raise NotImplementedError
        return local_out
