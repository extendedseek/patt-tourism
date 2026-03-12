from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    if "inherits" in cfg:
        parent_path = Path(cfg["inherits"])
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = load_config(parent_path)
        cfg = deep_update(parent, {k: v for k, v in cfg.items() if k != "inherits"})
    return cfg
