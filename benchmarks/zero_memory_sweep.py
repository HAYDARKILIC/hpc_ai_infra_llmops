"""Benchmark: per-rank memory under ZeRO Stage 0/1/2/3.

Computes the theoretical curve from ``docs/theory/04_zero_memory_equations.md``
and (optionally) compares it to measured DeepSpeed memory if invoked under a
distributed launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SIZE_MAP = {
    "350m": 350_000_000,
    "1.3b": 1_300_000_000,
    "2.7b": 2_700_000_000,
    "7b": 7_000_000_000,
    "13b": 13_000_000_000,
}


def theoretical_state_bytes(n_params: int, n_gpu: int, stage: int) -> dict[str, float]:
    if stage == 0:
        param, grad, opt = 2 * n_params, 2 * n_params, 12 * n_params
    elif stage == 1:
        param, grad, opt = 2 * n_params, 2 * n_params, 12 * n_params / n_gpu
    elif stage == 2:
        param, grad, opt = 2 * n_params, 2 * n_params / n_gpu, 12 * n_params / n_gpu
    elif stage == 3:
        param, grad, opt = 2 * n_params / n_gpu, 2 * n_params / n_gpu, 12 * n_params / n_gpu
    else:
        raise ValueError(stage)
    return {
        "param_GiB": param / 2**30,
        "grad_GiB": grad / 2**30,
        "opt_GiB": opt / 2**30,
        "total_GiB": (param + grad + opt) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-sizes", default="350m,1.3b,2.7b,7b")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--out", default="reports/zero_memory_sweep.json")
    args = parser.parse_args()

    sweep = []
    for label in args.model_sizes.split(","):
        n = SIZE_MAP[label.strip()]
        for stage in range(4):
            entry = theoretical_state_bytes(n, args.num_gpus, stage)
            entry.update({"model": label, "stage": stage, "n_gpu": args.num_gpus})
            sweep.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sweep, indent=2))
    print(f"wrote {out_path} with {len(sweep)} rows")


if __name__ == "__main__":
    main()
