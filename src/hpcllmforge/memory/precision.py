"""Precision lattice utilities — FP32, TF32, BF16, FP16, FP8, FP4.

The dtype trade-off space is governed by three quantities:

* **Dynamic range** — the ratio between the largest and smallest representable
  positive numbers. Determines whether activations / gradients can be expressed
  without overflow or underflow.
* **Mantissa precision** — the number of significand bits; bounds the maximum
  relative quantization error |Δx / x| ≤ 2^{-(p+1)} for p mantissa bits.
* **Throughput multiplier** — Tensor Core throughput typically doubles per
  precision step (FP32 → BF16 → FP8 → FP4) on H100 and B200.

See ``docs/theory/01_roofline_derivation.md`` for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DtypeSpec:
    """Describes a numeric format."""

    name: str
    bits: int
    exponent_bits: int
    mantissa_bits: int

    @property
    def max_value(self) -> float:
        """Largest representable finite positive value."""
        # TODO: compute from exponent bias and mantissa width.
        raise NotImplementedError

    @property
    def relative_eps(self) -> float:
        """Worst-case relative quantization error 2^{-(mantissa_bits + 1)}."""
        return 2.0 ** -(self.mantissa_bits + 1)


# Precision catalogue used throughout the repository.
FP32 = DtypeSpec("fp32", 32, 8, 23)
TF32 = DtypeSpec("tf32", 19, 8, 10)
BF16 = DtypeSpec("bf16", 16, 8, 7)
FP16 = DtypeSpec("fp16", 16, 5, 10)
FP8_E4M3 = DtypeSpec("fp8_e4m3", 8, 4, 3)
FP8_E5M2 = DtypeSpec("fp8_e5m2", 8, 5, 2)
FP4 = DtypeSpec("fp4", 4, 2, 1)  # NF4 / FP4 — non-uniform quantization grid


def memory_footprint_bytes(num_params: int, dtype: DtypeSpec) -> int:
    """Compute the raw parameter memory for ``num_params`` weights in ``dtype``."""
    return num_params * dtype.bits // 8
