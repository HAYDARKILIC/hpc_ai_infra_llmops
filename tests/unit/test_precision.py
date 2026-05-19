"""Unit tests for hpcllmforge.memory.precision."""

from __future__ import annotations

import pytest

from hpcllmforge.memory.precision import (
    BF16,
    FP4,
    FP8_E4M3,
    FP16,
    FP32,
    memory_footprint_bytes,
)


@pytest.mark.parametrize(
    ("dtype", "n_params", "expected_bytes"),
    [
        (FP32, 1_000_000, 4_000_000),
        (BF16, 1_000_000, 2_000_000),
        (FP16, 1_000_000, 2_000_000),
        (FP8_E4M3, 1_000_000, 1_000_000),
        (FP4, 1_000_000, 500_000),
    ],
)
def test_memory_footprint(dtype, n_params, expected_bytes):
    assert memory_footprint_bytes(n_params, dtype) == expected_bytes


def test_relative_eps_monotone():
    # Lower mantissa precision must produce larger quantization error.
    assert FP32.relative_eps < BF16.relative_eps < FP8_E4M3.relative_eps < FP4.relative_eps
