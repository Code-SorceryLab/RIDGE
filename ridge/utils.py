"""Utility functions: config loading, seeding, device setup."""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str, defaults_path: str | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally merging over a defaults file.

    Args:
        path: Path to the primary YAML config file.
        defaults_path: Optional path to a base defaults YAML. Primary config
            values take precedence over defaults.

    Returns:
        Merged config dictionary.
    """
    config: dict[str, Any] = {}

    if defaults_path is not None:
        with open(defaults_path, "r") as f:
            config.update(yaml.safe_load(f) or {})

    with open(path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    config.update(overrides)
    logger.debug("Loaded config from %s: %s", path, config)
    return config


def load_default_config(config_path: str) -> dict[str, Any]:
    """Load a config file, merging with default.yaml if not already the default.

    Args:
        config_path: Path to the config YAML to load.

    Returns:
        Config dict with defaults filled in.
    """
    this_dir = Path(__file__).parent.parent
    default_path = this_dir / "configs" / "default.yaml"

    if Path(config_path).resolve() == default_path.resolve():
        return load_config(config_path)

    return load_config(config_path, defaults_path=str(default_path))


def set_seeds(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Set all seeds to %d", seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Returns:
        torch.device instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Using device: %s", device)
    return device


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a standard format.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
