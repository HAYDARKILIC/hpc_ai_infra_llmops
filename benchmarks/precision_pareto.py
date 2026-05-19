"""Benchmark: throughput / loss-degradation Pareto across FP32, BF16, FP16, FP8, FP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", default="fp32,bf16,fp16,fp8_e4m3,fp4")
    parser.add_argument("--matmul-size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="reports/precision_pareto.json")
    args = parser.parse_args()

    # TODO:
    # 1. For each dtype in args.dtypes, run an (NxN) matmul `iters` times,
    #    measure ms and compute TFLOPS.
    # 2. Run a 1-step forward+backward of a fixed model in that dtype and
    #    record final loss vs the FP32 baseline as a quality proxy.
    # 3. Emit a JSON list of {dtype, tflops, rel_loss_drift}.

    results = [
        {"dtype": d, "tflops": None, "rel_loss_drift": None}
        for d in args.dtypes.split(",")
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
