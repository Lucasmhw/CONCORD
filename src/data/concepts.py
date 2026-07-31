from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def _validate_history(x: torch.Tensor, tau: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"expected [batch, time, series], got shape={tuple(x.shape)}")
    if x.shape[1] < tau + 1:
        raise ValueError(f"history length {x.shape[1]} is smaller than tau + 1 = {tau + 1}")


def _amplitude_from_buffer(buffer: torch.Tensor) -> torch.Tensor:
    tau = buffer.shape[1]
    centered = buffer - buffer.mean(dim=1, keepdim=True)
    index = torch.arange(tau, device=buffer.device, dtype=buffer.dtype)
    angle = 2.0 * math.pi * index / tau
    cosine = torch.cos(angle).view(1, tau, 1)
    sine = torch.sin(angle).view(1, tau, 1)
    a1 = (centered * cosine).mean(dim=1)
    b1 = (centered * sine).mean(dim=1)
    return 2.0 * torch.sqrt((a1.square() + b1.square()).clamp_min(0.0))


def _concepts_for_scale(x: torch.Tensor, tau: int) -> torch.Tensor:
    _validate_history(x, tau)
    current = x[:, -tau:]
    previous = x[:, -tau - 1 : -1]
    current_level = current.mean(dim=1)
    previous_level = previous.mean(dim=1)
    level_velocity = current_level - previous_level
    interaction = previous_level * level_velocity
    variance = (current - current_level.unsqueeze(1)).square().mean(dim=1)
    volatility = torch.sqrt(variance.clamp_min(1e-12))
    amplitude = _amplitude_from_buffer(current)
    return torch.stack(
        [current_level, level_velocity, interaction, amplitude, volatility],
        dim=-1,
    )


def compute_concept_targets(x_hist: torch.Tensor, scales: Iterable[int]) -> torch.Tensor:
    """Compute Equations (8)-(12), returning [batch, series, 5 * scales]."""
    return torch.cat([_concepts_for_scale(x_hist, int(tau)) for tau in scales], dim=-1)


def _rolling_harmonic(
    sequence: torch.Tensor,
    first_end: int,
    horizon: int,
    tau: int,
) -> torch.Tensor:
    first_window = sequence[:, first_end - tau + 1 : first_end + 1]
    index = torch.arange(tau, device=sequence.device, dtype=sequence.dtype)
    angle = 2.0 * math.pi * index / tau
    real = (first_window * torch.cos(angle).view(1, tau, 1)).sum(dim=1)
    imag = (first_window * torch.sin(angle).view(1, tau, 1)).sum(dim=1)
    coefficient = torch.complex(real, imag)
    rotation = torch.polar(
        torch.ones((), device=sequence.device, dtype=sequence.dtype),
        torch.tensor(-2.0 * math.pi / tau, device=sequence.device, dtype=sequence.dtype),
    )
    coefficients = [coefficient]
    for step in range(1, horizon):
        new_end = first_end + step
        leaving = sequence[:, new_end - tau]
        entering = sequence[:, new_end]
        coefficient = rotation * (coefficient - leaving + entering)
        coefficients.append(coefficient)
    return 2.0 * torch.stack(coefficients, dim=1).abs() / tau


def compute_concept_trajectory(
    x_hist: torch.Tensor,
    pred: torch.Tensor,
    scales: Iterable[int],
) -> torch.Tensor:
    """Recompute mixed-trajectory concepts for h=0,...,H-1 without future labels."""
    if pred.ndim != 3:
        raise ValueError(f"expected pred [batch, horizon, series], got {tuple(pred.shape)}")
    horizon = pred.shape[1]
    if horizon < 1:
        raise ValueError("prediction horizon must be positive")
    sequence = torch.cat([x_hist, pred[:, :-1]], dim=1)
    history_length = x_hist.shape[1]
    end_indices = torch.arange(
        history_length - 1,
        history_length - 1 + horizon,
        device=x_hist.device,
    )
    cumulative = F.pad(sequence.cumsum(dim=1), (0, 0, 1, 0))
    cumulative_sq = F.pad(sequence.square().cumsum(dim=1), (0, 0, 1, 0))

    all_scales = []
    for tau_value in scales:
        tau = int(tau_value)
        _validate_history(x_hist, tau)
        right = end_indices + 1
        left = right - tau
        current_sum = cumulative[:, right] - cumulative[:, left]
        previous_sum = cumulative[:, right - 1] - cumulative[:, left - 1]
        level = current_sum / tau
        previous_level = previous_sum / tau
        velocity = level - previous_level
        interaction = previous_level * velocity

        current_sq_sum = cumulative_sq[:, right] - cumulative_sq[:, left]
        variance = (current_sq_sum / tau - level.square()).clamp_min(1e-12)
        volatility = torch.sqrt(variance)
        amplitude = _rolling_harmonic(
            sequence=sequence,
            first_end=history_length - 1,
            horizon=horizon,
            tau=tau,
        )
        all_scales.append(
            torch.stack([level, velocity, interaction, amplitude, volatility], dim=-1)
        )
    return torch.cat(all_scales, dim=-1)


def build_mixed_buffer(x_hist: torch.Tensor, pred: torch.Tensor, h: int, tau: int) -> torch.Tensor:
    sequence = x_hist if h == 0 else torch.cat([x_hist, pred[:, :h]], dim=1)
    return sequence[:, -tau:]


def compute_rollout_concepts(
    x_hist: torch.Tensor,
    pred: torch.Tensor,
    h: int,
    scales: Iterable[int],
) -> torch.Tensor:
    if not 0 <= h < pred.shape[1]:
        raise IndexError(h)
    return compute_concept_trajectory(x_hist, pred[:, : h + 1], scales)[:, h]
