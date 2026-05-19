"""Benchmark: from-scratch Ring-AllReduce vs NCCL all_reduce.

Launch with:
    torchrun --standalone --nproc_per_node=4 benchmarks/allreduce_throughput.py
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def time_op(op, iters: int = 50) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def benchmark(size_mb: int, iters: int = 50) -> dict:
    n_floats = size_mb * 1024 * 1024 // 4
    x = torch.randn(n_floats, device="cuda")

    # Warm-up.
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    ms_nccl = time_op(lambda: dist.all_reduce(x.clone()), iters=iters)

    # TODO: import and time the from-scratch ring_allreduce once implemented.
    bytes_moved = 2 * x.element_size() * x.numel() * (dist.get_world_size() - 1) / dist.get_world_size()
    bandwidth_gbs = bytes_moved / (ms_nccl * 1e-3) / 1e9

    return {
        "size_mb": size_mb,
        "world_size": dist.get_world_size(),
        "nccl_ms": ms_nccl,
        "effective_bw_gbs": bandwidth_gbs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", default="1,4,16,64,256")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default="reports/allreduce_bench.json")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    results = []
    for size in (int(s) for s in args.sizes_mb.split(",")):
        results.append(benchmark(size, iters=args.iters))
        if dist.get_rank() == 0:
            print(results[-1])

    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
