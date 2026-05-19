"""Programmatic launchers for RunPod, Lambda Labs, and Vast.ai.

Each provider exposes a slightly different CLI / REST surface. This module
wraps the three under a single ``CloudLauncher`` interface so that the
capstone pipeline can be retargeted with a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    RUNPOD = "runpod"
    LAMBDA = "lambda"
    VASTAI = "vastai"


@dataclass
class LaunchSpec:
    """Cloud job specification."""

    provider: Provider
    gpu_type: str           # e.g. "H100", "A100-80G", "B200"
    num_gpus: int
    image: str              # Docker image tag (must be pushed to a registry)
    command: list[str]
    spot: bool = True
    max_runtime_hours: int = 24


class CloudLauncher:
    """Provider-agnostic cloud launcher."""

    def launch(self, spec: LaunchSpec) -> str:
        """Submit the job and return a provider-specific instance ID."""
        if spec.provider is Provider.RUNPOD:
            return self._launch_runpod(spec)
        if spec.provider is Provider.LAMBDA:
            return self._launch_lambda(spec)
        if spec.provider is Provider.VASTAI:
            return self._launch_vastai(spec)
        raise ValueError(f"Unknown provider: {spec.provider}")

    # ---------- Provider-specific adapters ----------

    def _launch_runpod(self, spec: LaunchSpec) -> str:
        # TODO: call runpod.create_pod via the official Python SDK.
        raise NotImplementedError

    def _launch_lambda(self, spec: LaunchSpec) -> str:
        # TODO: shell out to `lambda-cloud launch ...`.
        raise NotImplementedError

    def _launch_vastai(self, spec: LaunchSpec) -> str:
        # TODO: REST POST to https://console.vast.ai/api/v0/asks/.
        raise NotImplementedError
