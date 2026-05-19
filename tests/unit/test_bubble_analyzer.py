"""Unit tests for the pipeline bubble analyzer."""

from __future__ import annotations

import pytest

from hpcllmforge.parallelism.pipeline.bubble_analyzer import analyze


@pytest.mark.parametrize(
    ("p", "m", "expected_bubble"),
    [
        (2, 4, 0.25),
        (4, 4, 0.75),
        (4, 16, 0.1875),
        (8, 64, 0.109375),
    ],
)
def test_theoretical_bubble(p, m, expected_bubble):
    report = analyze(num_stages=p, num_microbatches=m, measured_step_ms=1.0, ideal_step_ms=1.0)
    assert report.theoretical_bubble == pytest.approx(expected_bubble, rel=1e-6)


def test_utilization_complement():
    report = analyze(num_stages=4, num_microbatches=16, measured_step_ms=100.0, ideal_step_ms=80.0)
    assert report.measured_bubble == pytest.approx(0.2, rel=1e-6)
    assert report.utilization == pytest.approx(0.8, rel=1e-6)
