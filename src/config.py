from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    include = cfg.pop("include", None)
    if include is None:
        return cfg
    base_path = path.parent / Path(include).name if not str(include).startswith("/") else Path(include)
    if not base_path.exists():
        base_path = path.parent / include
    base = load_config(base_path)
    return _deep_update(copy.deepcopy(base), cfg)


def _coerce_scalar(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        if value.startswith("[") and value.endswith("]"):
            parsed = yaml.safe_load(value)
            return parsed
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        cursor = out
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], dict):
                cursor[k] = {}
            cursor = cursor[k]
        cursor[keys[-1]] = _coerce_scalar(value)
    return out
