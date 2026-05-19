"""HPC-LLM-Forge — a from-scratch engineering codex for high-performance LLM training.

This package re-implements the core algorithms behind modern distributed training
(Ring-AllReduce, tensor parallelism, 1F1B pipelining, ZeRO partitioning) from
first principles, and packages them into a cloud-portable boilerplate.

Sub-packages
------------
- ``memory``        : GPU memory hierarchy probes, precision utilities, roofline analyzer.
- ``parallelism``   : Data / tensor / pipeline parallelism engines.
- ``zero``          : ZeRO Stage 1/2/3 partitioning with CPU/NVMe offload.
- ``orchestration`` : Docker builders, cloud launchers, spot-resilient checkpointing.
- ``profiling``     : Throughput meters, NVTX helpers, roofline classifier.
- ``utils``         : Logging, seeding, topology inspection.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Haydar Kılıç"

# TODO: re-export the most commonly used symbols here for ergonomics, e.g.
#   from .parallelism.data.ddp_engine import DDPEngine
#   from .zero.stage3 import ZeroStage3


def print_environment() -> None:
    """Print a one-shot environment fingerprint for reproducibility.

    Captures: Python version, PyTorch version, CUDA version, GPU count,
    GPU model, NVLink topology, NCCL version, and a hash thereof.
    """
    # TODO: implement using torch.utils.collect_env and topology inspection
    raise NotImplementedError
