from __future__ import annotations

import math
from typing import Iterable

import torch


def _last_tau(x: torch.Tensor, tau: int) -> torch.Tensor:
    return x[:, -tau:, :]


def level(x: torch.Tensor, tau: int) -> torch.Tensor:
    return _last_tau(x, tau).mean(dim=1)


def velocity(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1, :] - x[:, -2, :]


def power(x: torch.Tensor, tau: int) -> torch.Tensor:
    return x[:, -1, :] * velocity(x)


def volatility(x: torch.Tensor, tau: int) -> torch.Tensor:
    buf = _last_tau(x, tau)
    mu = buf.mean(dim=1, keepdim=True)
    return ((buf - mu).pow(2).mean(dim=1) + 1e-8).sqrt()


def amplitude(x: torch.Tensor, tau: int) -> torch.Tensor:
    buf = _last_tau(x, tau)
    mu = buf.mean(dim=1, keepdim=True)
    centered = buf - mu
    idx = torch.arange(tau, device=x.device, dtype=x.dtype)
    cos = torch.cos(2 * math.pi * idx / tau).view(1, tau, 1)
    sin = torch.sin(2 * math.pi * idx / tau).view(1, tau, 1)
    a1 = (centered * cos).mean(dim=1)
    b1 = (centered * sin).mean(dim=1)
    return 2.0 * torch.sqrt(a1.pow(2) + b1.pow(2) + 1e-8)


def compute_concept_targets(x_hist: torch.Tensor, scales: Iterable[int]) -> torch.Tensor:
    """Return [B, N, 5*S]."""
    parts = []
    for tau in scales:
        parts.extend([
            level(x_hist, tau),
            velocity(_last_tau(x_hist, max(2, tau))),
            power(x_hist, tau),
            amplitude(x_hist, tau),
            volatility(x_hist, tau),
        ])
    stacked = torch.stack(parts, dim=-1)  # [B, N, 5*S]
    return stacked


def build_mixed_buffer(x_hist: torch.Tensor, preds: torch.Tensor, h: int, tau: int) -> torch.Tensor:
    """Return last tau values available at lead h; h=0 means ending at history end."""
    if h == 0:
        seq = x_hist
    else:
        seq = torch.cat([x_hist, preds[:, :h, :]], dim=1)
    return seq[:, -tau:, :]


def compute_rollout_concepts(x_hist: torch.Tensor, preds: torch.Tensor, h: int, scales: Iterable[int]) -> torch.Tensor:
    parts = []
    for tau in scales:
        buf = build_mixed_buffer(x_hist, preds, h, tau)
        parts.extend([
            buf.mean(dim=1),
            buf[:, -1, :] - buf[:, -2, :],
            buf[:, -1, :] * (buf[:, -1, :] - buf[:, -2, :]),
            amplitude(buf, tau),
            volatility(buf, tau),
        ])
    return torch.stack(parts, dim=-1)
