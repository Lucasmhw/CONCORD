from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMN_PATTERN = re.compile(r"^(date|datetime|timestamp|time)$", re.IGNORECASE)


def _text_has_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as stream:
        first_line = stream.readline().strip()
    if not first_line:
        return False
    for token in first_line.split(","):
        try:
            float(token)
        except ValueError:
            return True
    return False


def load_time_series(path: str | Path, date_column: str | None = "date") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        header = "infer" if suffix == ".csv" or _text_has_header(path) else None
        frame = pd.read_csv(path, header=header)
        drop_columns = [
            column
            for column in frame.columns
            if (date_column is not None and column == date_column)
            or DATE_COLUMN_PATTERN.fullmatch(str(column).strip())
        ]
        if drop_columns:
            frame = frame.drop(columns=drop_columns)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(axis=1, how="all")
        if frame.shape[1] == 0:
            raise ValueError(f"No numeric time-series columns found in {path}")
        return frame.to_numpy(dtype=np.float32)
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        archive = np.load(path)
        if "data" in archive:
            array = archive["data"]
        elif "x" in archive:
            array = archive["x"]
        else:
            array = archive[list(archive.keys())[0]]
        if array.ndim == 3:
            array = array[..., 0]
        return array.astype(np.float32)
    raise ValueError(f"Unsupported file type: {path}")
