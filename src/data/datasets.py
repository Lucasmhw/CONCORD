from __future__ import annotations

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
    def __init__(self, array: np.ndarray, spec: WindowSpec) -> None:
        self.x = torch.from_numpy(array).float()
        self.spec = spec
        self.indices = list(range(0, len(self.x) - spec.lookback - spec.horizon + 1, spec.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.spec.lookback
        fut = end + self.spec.horizon
        x_hist = self.x[start:end]
        y = self.x[end:fut]
        return {"x_hist": x_hist, "y": y}


class ImputationDataset(Dataset):
    def __init__(self, array: np.ndarray, seq_len: int, stride: int, mask_ratios: Iterable[float]) -> None:
        self.x = torch.from_numpy(array).float()
        self.seq_len = seq_len
        self.stride = stride
        self.mask_ratios = list(mask_ratios)
        self.indices = list(range(0, len(self.x) - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        seq = self.x[start:start + self.seq_len]
        ratio = self.mask_ratios[idx % len(self.mask_ratios)]
        mask = (torch.rand_like(seq) > ratio).float()
        observed = seq * mask
        return {"sequence": seq, "observed": observed, "mask": mask, "ratio": torch.tensor(ratio)}


def load_processed_split(processed_dir: str | Path, split: str) -> np.ndarray:
    return np.load(Path(processed_dir) / f"{split}.npy")
