from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concord.losses import total_loss
from concord.metrics import METRICS


def _masked_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, names: list[str]) -> dict[str, float]:
    missing = (1.0 - mask).bool()
    out = {}
    if missing.sum() == 0:
        return {name: 0.0 for name in names}
    p = pred[missing]
    t = target[missing]
    for name in names:
        out[name] = float(METRICS[name](p, t).item())
    return out


def causal_impute(model: torch.nn.Module, batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    seq = batch["sequence"]
    observed = batch["observed"].clone()
    mask = batch["mask"]
    lookback = int(cfg["data"]["lookback"])
    b, t, n = seq.shape
    filled = observed.clone()
    outputs = []
    for step in range(lookback - 1, t - 1):
        x_hist = filled[:, step - lookback + 1: step + 1, :]
        pred_next = model(x_hist, horizon=1).pred[:, 0, :]
        next_obs = observed[:, step + 1, :]
        next_mask = mask[:, step + 1, :]
        next_filled = next_mask * next_obs + (1.0 - next_mask) * pred_next
        filled[:, step + 1, :] = next_filled
        outputs.append(pred_next)
    pred = torch.stack(outputs, dim=1)
    target = seq[:, lookback:, :]
    eval_mask = mask[:, lookback:, :]
    return pred, target, eval_mask


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def run_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, cfg: dict[str, Any], device: torch.device, training: bool) -> EpochResult:
    model.train(training)
    losses = []
    metric_sums = {name: 0.0 for name in cfg["metrics"]["report"]}
    num_batches = 0
    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(training):
            if "x_hist" in batch:
                output = model(batch["x_hist"], horizon=batch["y"].shape[1])
                loss, _ = total_loss(output, batch, cfg)
                pred_for_metrics = output.pred.detach()
                target_for_metrics = batch["y"]
                masked_metrics = None
            else:
                pred_for_metrics, target_for_metrics, eval_mask = causal_impute(model, batch, cfg)
                missing = 1.0 - eval_mask
                denom = missing.sum().clamp_min(1.0)
                loss = (((pred_for_metrics - target_for_metrics) ** 2) * missing).sum() / denom
                masked_metrics = _masked_metrics(pred_for_metrics.detach(), target_for_metrics, eval_mask, list(metric_sums.keys()))
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optim"].get("grad_clip", 1.0)))
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name in metric_sums:
            if masked_metrics is not None:
                metric_sums[name] += masked_metrics[name]
            else:
                metric_val = METRICS[name](pred_for_metrics, target_for_metrics).item()
                metric_sums[name] += float(metric_val)
        num_batches += 1
        iterator.set_description(f"{'train' if training else 'eval'} loss={losses[-1]:.4f}")
    metrics = {k: v / max(num_batches, 1) for k, v in metric_sums.items()}
    return EpochResult(loss=sum(losses) / max(len(losses), 1), metrics=metrics)
