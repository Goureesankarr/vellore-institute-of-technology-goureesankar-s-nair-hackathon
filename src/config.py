"""Configuration loading. All paths in settings.yaml are relative to the repository root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    path = Path(os.environ.get("RISK_ENGINE_SETTINGS", ROOT / "config" / "settings.yaml"))
    return _read_yaml(path)


def path(key: str) -> Path:
    """Resolve a path from settings.paths relative to the repo root."""
    p = Path(settings()["paths"][key])
    return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def universe() -> dict[str, Any]:
    return _read_yaml(path("universe"))


@lru_cache(maxsize=1)
def taxonomy() -> dict[str, Any]:
    return _read_yaml(path("taxonomy"))


@lru_cache(maxsize=1)
def shocks() -> dict[str, Any]:
    return _read_yaml(path("shocks"))
