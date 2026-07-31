from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from concord.losses import total_loss


class MetricAccumulator:
    def __init__(self, names: list[str], null_value: float | None = None) -> None:
        self.names = names
        self.null_value = null_value
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.absolute_percentage_error = 0.0
        self.count = 0
        self.percentage_count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        valid = torch.ones_like(target, dtype=torch.bool)
        if valid_mask is not None:
            valid &= valid_mask.bool()
        if self.null_value is not None:
            valid &= ~torch.isclose(target, target.new_tensor(self.null_value))
        valid &= torch.isfinite(pred) & torch.isfinite(target)
        if not valid.any():
            return
        error = pred[valid] - target[valid]
        self.squared_error += float(error.square().sum().detach().cpu())
        self.absolute_error += float(error.abs().sum().detach().cpu())
        self.count += int(valid.sum().item())

        percentage_valid = valid & target.abs().gt(1e-6)
        if percentage_valid.any():
            percentage_error = (
                (pred[percentage_valid] - target[percentage_valid]).abs()
                / target[percentage_valid].abs()
            )
            self.absolute_percentage_error += float(percentage_error.sum().detach().cpu())
            self.percentage_count += int(percentage_valid.sum().item())

    def compute(self) -> dict[str, float]:
        mse = self.squared_error / max(self.count, 1)
        values = {
            "mse": mse,
            "mae": self.absolute_error / max(self.count, 1),
            "rmse": mse**0.5,
            "mape": 100.0 * self.absolute_percentage_error / max(self.percentage_count, 1),
        }
        return {name: values[name] for name in self.names}


def _causal_forward_fill(observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = observed.clone()
    for step in range(1, observed.shape[1]):
        missing = mask[:, step].eq(0)
        filled[:, step] = torch.where(missing, filled[:, step - 1], filled[:, step])
    return filled


def causal_impute(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict masked positions from strictly preceding, causally filled windows."""
    sequence = batch["sequence"]
    observed = batch["observed"]
    mask = batch["mask"]
    lookback = int(cfg["data"]["lookback"])
    filled = _causal_forward_fill(observed, mask)
    num_origins = sequence.shape[1] - lookback
    windows = filled.unfold(dimension=1, size=lookback, step=1)[:, :num_origins]
    histories = windows.permute(0, 1, 3, 2).reshape(-1, lookback, sequence.shape[-1])
    targets = sequence[:, lookback:].reshape(-1, sequence.shape[-1])
    missing = mask[:, lookback:].eq(0).reshape(-1, sequence.shape[-1])

    valid_origins = torch.nonzero(missing.any(dim=-1), as_tuple=False).squeeze(-1)
    if model.training:
        maximum = int(cfg["data"].get("imputation_train_origins", 8))
        if maximum > 0 and valid_origins.numel() > maximum:
            order = torch.randperm(valid_origins.numel(), device=valid_origins.device)[:maximum]
            valid_origins = valid_origins[order]
        chunk_size = int(cfg["data"].get("imputation_train_origin_batch_size", 2))
    else:
        chunk_size = int(cfg["data"].get("imputation_eval_origin_batch_size", 16))
    if valid_origins.numel() == 0:
        raise ValueError("The imputation batch contains no masked evaluation positions")

    selected_histories = histories[valid_origins]
    predictions = [
        model(chunk, horizon=1).pred[:, 0]
        for chunk in selected_histories.split(max(chunk_size, 1), dim=0)
    ]
    return torch.cat(predictions, dim=0), targets[valid_origins], missing[valid_origins]


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]
    loss_terms: dict[str, float]
    global_step: int


def _accumulation_divisor(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> int:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _metric_tensors(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    scaler: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["y"]
    if str(cfg["metrics"].get("scale", "normalized")) != "original":
        return pred, target
    if scaler is None:
        raise ValueError("Original-scale metrics require scaler statistics")
    mean, std = scaler
    pred_original = pred * std + mean
    target_original = batch.get("y_raw", target * std + mean)
    return pred_original, target_original


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
    device: torch.device,
    training: bool,
    global_step: int = 0,
    scaler: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> EpochResult:
    model.train(training)
    names = list(cfg["metrics"]["report"])
    accumulator = MetricAccumulator(names, null_value=cfg["metrics"].get("null_value"))
    loss_sum = 0.0
    sample_count = 0
    term_sums: dict[str, float] = {}
    accumulation_steps = int(cfg["optim"].get("accumulation_steps", 1))
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, leave=False, disable=not bool(cfg["exp"].get("progress", True)))
    for batch_index, batch in enumerate(iterator):
        batch = _move_batch(batch, device)
        batch_size = next(value for value in batch.values() if torch.is_tensor(value)).shape[0]
        with torch.set_grad_enabled(training):
            if "x_hist" in batch:
                output = model(
                    batch["x_hist"],
                    horizon=batch["y"].shape[1],
                    x_hist_raw=batch.get("x_hist_raw"),
                )
                loss, terms = total_loss(output, batch, cfg, global_step=global_step)
                metric_pred, metric_target = _metric_tensors(
                    output.pred.detach(), batch, cfg, scaler
                )
                accumulator.update(metric_pred, metric_target)
            else:
                prediction, target, missing = causal_impute(model, batch, cfg)
                denominator = missing.sum().clamp_min(1)
                loss = (prediction.sub(target).square() * missing).sum() / denominator
                terms = {"loss": float(loss.detach().cpu())}
                accumulator.update(prediction.detach(), target, valid_mask=missing)

            if training:
                divisor = _accumulation_divisor(
                    batch_index,
                    total_batches=len(loader),
                    accumulation_steps=accumulation_steps,
                )
                (loss / divisor).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg["optim"].get("grad_clip", 1.0))
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
        for name, value in terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + float(value) * batch_size
        iterator.set_description(
            f"{'train' if training else 'eval'} loss={float(loss.detach()):.4f}"
        )

    return EpochResult(
        loss=loss_sum / max(sample_count, 1),
        metrics=accumulator.compute(),
        loss_terms={name: value / max(sample_count, 1) for name, value in term_sums.items()},
        global_step=global_step,
    )
