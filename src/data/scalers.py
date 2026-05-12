from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return x * self.std_ + self.mean_


@dataclass
class IdentityScaler:
    def fit(self, x: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x
