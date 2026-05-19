"""ZeRO Stage 3 — Stage 2 + parameter partitioning.

Stage 3 is the maximally memory-efficient setting: parameters are also
partitioned across ranks. During the forward pass, each layer's parameters
must first be **all-gathered** from their owning shards, used for the matmul,
and then immediately released. The same gather happens on the backward pass.

Memory per rank:  (M_param + M_grad + M_opt) / N_GPU + M_activations.

Trade-off
---------
Stage 3 multiplies forward-pass communication volume by ``2 × num_layers``
(one all-gather + one release per layer). The benefit is that arbitrarily
large models fit so long as the *largest single layer* fits on one GPU.
"""

from __future__ import annotations

from .stage2 import ZeroStage2Optimizer


class ZeroStage3Optimizer(ZeroStage2Optimizer):
    """Stage 2 with parameter sharding and per-layer all-gather."""

    def __init__(self, model, base_optimizer_cls, **base_optimizer_kwargs) -> None:
        # We need the full model object (not just params) so we can wrap each
        # submodule with all-gather / release hooks.
        super().__init__(model.parameters(), base_optimizer_cls, **base_optimizer_kwargs)
        self.model = model

        # TODO:
        # 1. Partition every parameter tensor into N_GPU shards; replace the
        #    .data of each parameter on non-owning ranks with a zero-size view.
        # 2. Register forward-pre and forward-post hooks on every nn.Module:
        #    * pre: all-gather the parameters needed by this module.
        #    * post: release the gathered parameters back to their shards.
        # 3. Mirror the same hooks for the backward pass.
        # 4. Optionally enable CPU/NVMe offload for the partitioned state.
