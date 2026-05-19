"""NVLink / PCIe topology inspection."""

from __future__ import annotations

import hashlib
import json
import subprocess


def fingerprint() -> str:
    """Return a SHA-256 fingerprint of the current hardware topology.

    Used for reproducibility — two runs with identical fingerprints are
    eligible for bit-exact comparison; runs with different fingerprints are
    not directly comparable.
    """
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], stderr=subprocess.DEVNULL
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        nvidia_smi = ""

    info = {
        "nvidia_smi_topo": nvidia_smi,
    }
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return digest[:16]
