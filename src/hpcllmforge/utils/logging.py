"""Structured logging with WandB integration."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout in JSON-friendly form."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
