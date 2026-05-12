from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from concord.data.io import load_time_series
from concord.data.scalers import IdentityScaler, StandardScaler


def split_array(x: np.ndarray, split_cfg: dict[str, float]) -> dict[str, np.ndarray]:
    n = len(x)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    train = x[:n_train]
    val = x[n_train:n_train + n_val]
    test = x[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def preprocess_dataset(cfg: dict[str, Any]) -> Path:
    x = load_time_series(cfg["data"]["raw_path"], cfg["data"].get("date_column", "date"))
    splits = split_array(x, cfg["data"]["split"])
    scaler_name = cfg["data"].get("scaler", "standard")
    scaler = StandardScaler() if scaler_name == "standard" else IdentityScaler()
    scaler.fit(splits["train"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "mean": getattr(scaler, "mean_", np.zeros((1, x.shape[1]), dtype=np.float32)),
        "std": getattr(scaler, "std_", np.ones((1, x.shape[1]), dtype=np.float32)),
    }
    np.savez(processed_dir / "scaler_stats.npz", **stats)
    for split, arr in splits.items():
        np.save(processed_dir / f"{split}.npy", scaler.transform(arr).astype(np.float32))
    return processed_dir
