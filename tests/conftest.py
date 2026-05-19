"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def small_seed():
    """A deterministic seed for use in tests that involve RNG."""
    return 1234
