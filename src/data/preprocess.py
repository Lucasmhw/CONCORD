from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from concord.data.io import load_time_series
from concord.data.scalers import IdentityScaler, StandardScaler


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_boundaries(length: int, split_cfg: dict[str, Any]) -> tuple[int, int, int]:
    if "train_end" in split_cfg:
        train_end = int(split_cfg["train_end"])
        val_end = int(split_cfg["val_end"])
        test_end = int(split_cfg.get("test_end", length))
    else:
        train_end = int(length * float(split_cfg["train"]))
        val_end = train_end + int(length * float(split_cfg["val"]))
        test_end = length
    if not 0 < train_end < val_end < test_end <= length:
        raise ValueError(
            f"Invalid chronological boundaries {(train_end, val_end, test_end)} for length={length}"
        )
    return train_end, val_end, test_end


def split_array(x: np.ndarray, split_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    train_end, val_end, test_end = split_boundaries(len(x), split_cfg)
    return {
        "train": x[:train_end],
        "val": x[train_end:val_end],
        "test": x[val_end:test_end],
    }


def _causal_fill(x: np.ndarray, train_end: int) -> np.ndarray:
    filled = np.asarray(x, dtype=np.float32).copy()
    train_means = np.nanmean(filled[:train_end], axis=0, keepdims=True)
    train_means = np.where(np.isfinite(train_means), train_means, 0.0)
    for time_index in range(1, len(filled)):
        missing = ~np.isfinite(filled[time_index])
        filled[time_index, missing] = filled[time_index - 1, missing]
    initial_missing = ~np.isfinite(filled)
    if initial_missing.any():
        rows, columns = np.where(initial_missing)
        filled[rows, columns] = train_means[0, columns]
    return filled


def preprocess_dataset(cfg: dict[str, Any]) -> Path:
    data_cfg = cfg["data"]
    raw = load_time_series(data_cfg["raw_path"], data_cfg.get("date_column", "date"))
    expected_channels = data_cfg.get("expected_channels")
    if expected_channels is not None and raw.shape[1] != int(expected_channels):
        raise ValueError(
            f"{data_cfg.get('dataset_name', 'dataset')} has {raw.shape[1]} channels; "
            f"expected {expected_channels}. Refusing to run a non-comparable benchmark."
        )

    train_end, val_end, test_end = split_boundaries(len(raw), data_cfg["split"])
    raw = _causal_fill(raw[:test_end], train_end=train_end)
    scaler_name = data_cfg.get("scaler", "standard")
    scaler = StandardScaler() if scaler_name == "standard" else IdentityScaler()
    scaler.fit(raw[:train_end])
    normalized = scaler.transform(raw).astype(np.float32)

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    mean = getattr(scaler, "mean_", np.zeros((1, raw.shape[1]), dtype=np.float32))
    std = getattr(scaler, "std_", np.ones((1, raw.shape[1]), dtype=np.float32))
    np.savez(processed_dir / "scaler_stats.npz", mean=mean, std=std)

    lookback = int(data_cfg["lookback"])
    ranges = {
        "train": (0, train_end, 0),
        "val": (max(0, train_end - lookback), val_end, min(lookback, train_end)),
        "test": (max(0, val_end - lookback), test_end, min(lookback, val_end)),
    }
    metadata = {
        "dataset_name": data_cfg.get("dataset_name"),
        "raw_path": str(data_cfg["raw_path"]),
        "raw_sha256": _sha256(data_cfg["raw_path"]),
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "split_boundaries": {
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
        },
        "lookback_context": lookback,
        "splits": {},
        "scaler": scaler_name,
    }
    for split, (start, end, target_offset) in ranges.items():
        np.save(processed_dir / f"{split}.npy", normalized[start:end])
        np.save(processed_dir / f"{split}_raw.npy", raw[start:end])
        metadata["splits"][split] = {
            "source_start": start,
            "source_end": end,
            "target_offset": target_offset,
        }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return processed_dir
