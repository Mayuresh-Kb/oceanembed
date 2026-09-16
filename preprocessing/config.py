"""Load configuration YAML for OceanEmbed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load default configuration, optionally merged with a small override."""
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if path is not None and cfg_path.resolve() != DEFAULT_CONFIG.resolve():
        with cfg_path.open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle)
        if not isinstance(override, dict):
            raise ValueError(f"Config override at {cfg_path} must be a mapping, got {type(override)}")
        data = _deep_merge(data, override)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping, got {type(data)}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply an experiment override without duplicating defaults."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
