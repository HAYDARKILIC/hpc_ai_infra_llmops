"""Programmatic Dockerfile composition for reproducible training images."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DockerSpec:
    """Declarative spec for a CUDA training image."""

    cuda_version: str = "12.4.1"
    cudnn_variant: str = "cudnn-devel-ubuntu22.04"
    python_version: str = "3.11"
    pytorch_version: str = "2.4.0"
    extra_apt: list[str] = field(default_factory=list)
    extra_pip: list[str] = field(default_factory=list)
    entrypoint: str = "/workspace/docker/entrypoint.sh"

    def render(self) -> str:
        """Return the Dockerfile text for this spec."""
        # TODO: emit a multi-stage build with a slim runtime layer.
        raise NotImplementedError
