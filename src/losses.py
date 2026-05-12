from __future__ import annotations

from typing import Any

import torch

from concord.data.concepts import compute_rollout_concepts
from concord.metrics import mse


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mse(pred, target)


def concept_alignment_loss(q_hat: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    return ((q_hat - q_target) ** 2).mean()


def _active_scales(cfg: dict[str, Any]) -> list[int]:
    scales = list(cfg["model"]["scales"])
    if cfg["model"].get("use_multiscale", True):
        return scales
    return [scales[len(scales) // 2]]


def residual_consistency_loss(output: Any, x_hist: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    scales = _active_scales(cfg)
    lambda_x = float(cfg["loss"].get("lambda_x", 1.0))
    lambda_k = list(cfg["loss"].get("lambda_k", [1.0] * 5))
    pred = output.pred
    q_states = output.q_states
    x_states = output.x_states
    total = x_hist.new_tensor(0.0)
    horizon = pred.shape[1]

    alpha = torch.softmax(output.alpha_logits[: len(scales)], dim=0)
    level_indices = [5 * i for i in range(len(scales))]

    for h in range(horizon):
        q_expected = compute_rollout_concepts(x_hist, pred, h, scales)
        q_curr = q_states[h]
        for k in range(5):
            total = total + float(lambda_k[k]) * ((q_curr[..., k::5] - q_expected[..., k::5]) ** 2).mean()

        x_curr = x_states[h]
        x_next = x_states[h + 1]
        levels = torch.stack([q_curr[..., idx] for idx in level_indices], dim=-1)
        ell = torch.einsum("bns,s->bn", levels, alpha)
        u = output.beta0 + torch.einsum("bnc,c->bn", q_curr, output.beta)
        x_rhs = x_curr + output.delta * (u - output.gamma * (x_curr - ell) - output.mu * torch.einsum("bij,bj->bi", output.lap, x_curr))
        total = total + lambda_x * ((x_next - x_rhs) ** 2).mean()
    return total / max(horizon, 1)


def total_loss(output: Any, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    pred = output.pred
    y = batch["y"]
    pred_loss = prediction_loss(pred, y)
    con_loss = concept_alignment_loss(output.q_hat, output.q_target)
    res_loss = residual_consistency_loss(output, batch["x_hist"], cfg)
    total = pred_loss + float(cfg["loss"]["lambda_con"]) * con_loss + float(cfg["loss"]["lambda_res"]) * res_loss
    stats = {
        "loss": float(total.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
        "con_loss": float(con_loss.detach().cpu()),
        "res_loss": float(res_loss.detach().cpu()),
    }
    return total, stats
