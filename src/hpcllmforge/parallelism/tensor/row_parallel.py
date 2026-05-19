"""Row-parallel ``Linear`` — Megatron-LM style.

Theory
------
Row-parallel decomposition splits the weight matrix *along the input dimension*:

    A = [A_1; A_2; ...; A_p]ᵀ,   each A_i ∈ R^{(d_in / p) × d_out}.

The input ``X = [X_1, ..., X_p]`` is also split along the same axis, and each
worker computes ``Y_i = X_i A_i``. The global output is the *sum* of the partials:

    Y = Σ_i X_i A_i.

This sum is realised by an AllReduce across the tensor-parallel group. Row-
parallel layers are the natural downstream partner of column-parallel layers —
the column-parallel output ``[Y_1, ..., Y_p]`` is already partitioned along the
input dim of the row-parallel layer, so no re-shard is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RowParallelLinear(nn.Module):
    """A linear layer whose weight matrix is partitioned along the input dim."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_world_size: int,
        tp_rank: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ) -> None:
        super().__init__()
        if in_features % tp_world_size != 0:
            raise ValueError("in_features must be divisible by tp_world_size")
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel

        local_in = in_features // tp_world_size
        self.weight = nn.Parameter(torch.empty(out_features, local_in))
        # Bias is only added on one rank (or after the AllReduce) to avoid
        # summing it p times.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # TODO: scatter x along the input dim.
            raise NotImplementedError
        local_out = torch.nn.functional.linear(x, self.weight, None)
        # TODO: dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=tp_group)
        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out
