from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_time_series(path: str | Path, date_column: str | None = "date") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if date_column is not None and date_column in df.columns:
            df = df.drop(columns=[date_column])
        return df.to_numpy(dtype=np.float32)
    if suffix in {".npy"}:
        arr = np.load(path)
        return arr.astype(np.float32)
    if suffix in {".npz"}:
        data = np.load(path)
        if "data" in data:
            arr = data["data"]
        elif "x" in data:
            arr = data["x"]
        else:
            first = list(data.keys())[0]
            arr = data[first]
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported file type: {path}")
