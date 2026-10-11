"""Config loading."""
from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    path = root / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
