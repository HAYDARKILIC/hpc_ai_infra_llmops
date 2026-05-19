"""Unit tests for the throughput meter."""

from __future__ import annotations

import time

from hpcllmforge.profiling.throughput_meter import ThroughputMeter


def test_mfu_none_until_two_ticks():
    meter = ThroughputMeter(flops_per_step=1e12, peak_flops=1e15)
    assert meter.mfu() is None
    meter.tick()
    assert meter.mfu() is None
    meter.tick()
    assert meter.mfu() is not None


def test_mfu_window_truncation():
    meter = ThroughputMeter(flops_per_step=1e12, peak_flops=1e15, window=4)
    for _ in range(10):
        meter.tick()
        time.sleep(0.001)
    # Window cap ensures at most window+1 timestamps retained.
    assert len(meter._times) <= meter.window + 1
