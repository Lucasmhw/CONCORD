from __future__ import annotations

from typing import Any

import torch

from concord.data.concepts import compute_concept_trajectory
from concord.metrics import mse


def _active_scales(cfg: dict[str, Any]) -> list[int]:
    scales = list(cfg["model"]["scales"])
    if cfg["model"].get("use_multiscale", True):
        return scales
    return [scales[len(scales) // 2]]


def residual_warmup_weight(global_step: int, warmup_steps: int, target: float) -> float:
    if warmup_steps <= 0:
        return float(target)
    fraction = min(max(global_step / warmup_steps, 0.0), 1.0)
    return float(target) * fraction


def concept_residual_loss(output: Any, x_hist: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    expected = compute_concept_trajectory(x_hist, output.pred, _active_scales(cfg))
    actual = torch.stack(output.q_states[:-1], dim=1)
    weights = x_hist.new_tensor(cfg["loss"].get("lambda_k", [1.0] * 5))
    num_scales = len(_active_scales(cfg))
    squared = (actual - expected).square().view(*actual.shape[:-1], num_scales, 5)
    return (squared * weights).mean()


def observation_residual_loss(output: Any, x_hist: torch.Tensor) -> torch.Tensor:
    """Match the first rollout vector field to the last causal observed increment."""
    x_previous = x_hist[:, -2]
    x_current = x_hist[:, -1]
    observed_derivative = (x_current - x_previous) / output.delta
    laplacian = torch.einsum("bij,bj->bi", output.lap, x_current)
    predicted_derivative = (
        output.u_states[0]
        - output.gamma * (x_current - output.ell_states[0])
        - output.mu * laplacian
        + output.innov_states[0]
    )
    return (observed_derivative - predicted_derivative).square().mean()


def total_loss(
    output: Any,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    global_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_loss = mse(output.pred, batch["y"])
    concept_loss = mse(output.q_hat, output.q_target)
    relation_loss = mse(output.q0, output.q_hat)
    concept_residual = concept_residual_loss(output, batch["x_hist"], cfg)
    observation_residual = observation_residual_loss(output, batch["x_hist"])
    residual_loss = concept_residual + float(cfg["loss"].get("lambda_x", 1.0)) * observation_residual
    innovation_loss = torch.stack(output.innov_states, dim=1).square().mean()
    prediction_std = output.pred.std(dim=1, unbiased=False)
    flat_loss = torch.relu(float(cfg["loss"].get("flat_std_floor", 0.02)) - prediction_std).mean()

    residual_weight = residual_warmup_weight(
        global_step=global_step,
        warmup_steps=int(cfg["loss"].get("residual_warmup_steps", 500)),
        target=float(cfg["loss"].get("lambda_res", 0.3)),
    )
    total = (
        pred_loss
        + float(cfg["loss"].get("lambda_con", 0.1)) * concept_loss
        + float(cfg["loss"].get("lambda_rel", 0.01)) * relation_loss
        + residual_weight * residual_loss
        + float(cfg["loss"].get("lambda_innov", 1e-3)) * innovation_loss
        + float(cfg["loss"].get("lambda_flat", 1e-3)) * flat_loss
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
        "concept_loss": float(concept_loss.detach().cpu()),
        "relation_loss": float(relation_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "concept_residual": float(concept_residual.detach().cpu()),
        "observation_residual": float(observation_residual.detach().cpu()),
        "innovation_loss": float(innovation_loss.detach().cpu()),
        "flat_loss": float(flat_loss.detach().cpu()),
        "lambda_res_effective": residual_weight,
        "gamma": float(output.gamma.detach().cpu()),
        "mu": float(output.mu.detach().cpu()),
    }
    return total, stats
