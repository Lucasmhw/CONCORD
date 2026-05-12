from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(pred, target) + 1e-12)


def mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = target.abs().clamp_min(1e-6)
    return ((pred - target).abs() / denom).mean() * 100.0


METRICS = {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}
