"""Command-line entrypoint for roofline profiling.

Usage
-----
    python -m hpcllmforge.profiling.roofline --model gpt2 --dtype bf16

The script measures FLOPs and byte traffic for every layer in the named model,
computes arithmetic intensity, and classifies each kernel as memory-bound or
compute-bound against the host device's measured peak bandwidth and FLOPS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (HF hub or local path).")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16", "fp8_e4m3"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # TODO:
    # 1. Load the model and a representative input.
    # 2. Iterate over named modules, call hpcllmforge.memory.profiler.measure_layer.
    # 3. Classify each measurement.
    # 4. Dump JSON to args.output.
    report = {"model": args.model, "dtype": args.dtype, "layers": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
