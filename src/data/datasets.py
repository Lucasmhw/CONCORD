from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSpec:
    lookback: int
    horizon: int
    stride: int = 1


class ForecastingDataset(Dataset):
    def __init__(
        self,
        array: np.ndarray,
        spec: WindowSpec,
        raw_array: np.ndarray | None = None,
        target_offset: int = 0,
    ) -> None:
        self.x = torch.from_numpy(array).float()
        self.raw = torch.from_numpy(raw_array).float() if raw_array is not None else None
        self.spec = spec
        first_origin = max(spec.lookback, int(target_offset))
        last_origin = len(self.x) - spec.horizon
        self.origins = list(range(first_origin, last_origin + 1, spec.stride))

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        origin = self.origins[idx]
        start = origin - self.spec.lookback
        future_end = origin + self.spec.horizon
        batch = {
            "x_hist": self.x[start:origin],
            "y": self.x[origin:future_end],
        }
        if self.raw is not None:
            batch["x_hist_raw"] = self.raw[start:origin]
            batch["y_raw"] = self.raw[origin:future_end]
        return batch


class ImputationDataset(Dataset):
    def __init__(
        self,
        array: np.ndarray,
        seq_len: int,
        stride: int,
        mask_ratios: Iterable[float],
        seed: int = 0,
        raw_array: np.ndarray | None = None,
        target_offset: int = 0,
    ) -> None:
        self.x = torch.from_numpy(array).float()
        self.raw = torch.from_numpy(raw_array).float() if raw_array is not None else None
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.mask_ratios = list(mask_ratios)
        self.seed = int(seed)
        first_start = max(0, int(target_offset) - self.seq_len)
        self.indices = list(range(first_start, len(self.x) - self.seq_len + 1, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        sequence = self.x[start : start + self.seq_len]
        ratio = self.mask_ratios[idx % len(self.mask_ratios)]
        generator = torch.Generator().manual_seed(self.seed + idx)
        mask = (torch.rand(sequence.shape, generator=generator) > ratio).float()
        observed = sequence * mask
        batch = {
            "sequence": sequence,
            "observed": observed,
            "mask": mask,
            "ratio": torch.tensor(ratio),
        }
        if self.raw is not None:
            batch["sequence_raw"] = self.raw[start : start + self.seq_len]
        return batch


def load_processed_split(processed_dir: str | Path, split: str, raw: bool = False) -> np.ndarray:
    suffix = "_raw" if raw else ""
    return np.load(Path(processed_dir) / f"{split}{suffix}.npy")


def load_processed_metadata(processed_dir: str | Path) -> dict:
    path = Path(processed_dir) / "metadata.json"
    if not path.exists():
        return {"splits": {name: {"target_offset": 0} for name in ("train", "val", "test")}}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
