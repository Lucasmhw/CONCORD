from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from concord.data.datasets import ForecastingDataset, ImputationDataset, WindowSpec, load_processed_split
from concord.engine import run_epoch
from concord.models.concord import CONCORDModel
from concord.utils.checkpoint import save_checkpoint
from concord.utils.logging import dump_json, ensure_dir
from concord.utils.seed import set_seed


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_dir = cfg["data"]["processed_dir"]
    task = cfg["data"].get("task", "forecasting")
    if task == "imputation":
        train = ImputationDataset(load_processed_split(processed_dir, "train"), seq_len=int(cfg["data"]["sequence_length"]), stride=int(cfg["data"].get("stride", 32)), mask_ratios=cfg["data"]["mask_ratios"])
        val = ImputationDataset(load_processed_split(processed_dir, "val"), seq_len=int(cfg["data"]["sequence_length"]), stride=int(cfg["data"].get("stride", 32)), mask_ratios=cfg["data"]["mask_ratios"])
        test = ImputationDataset(load_processed_split(processed_dir, "test"), seq_len=int(cfg["data"]["sequence_length"]), stride=int(cfg["data"].get("stride", 32)), mask_ratios=cfg["data"]["mask_ratios"])
    else:
        spec = WindowSpec(
            lookback=int(cfg["data"]["lookback"]),
            horizon=int(cfg["data"]["horizon"]),
            stride=int(cfg["data"].get("stride", 1)),
        )
        train = ForecastingDataset(load_processed_split(processed_dir, "train"), spec)
        val = ForecastingDataset(load_processed_split(processed_dir, "val"), spec)
        test = ForecastingDataset(load_processed_split(processed_dir, "test"), spec)
    bs = int(cfg["optim"]["batch_size"])
    nw = int(cfg["exp"].get("num_workers", 0))
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        betas=tuple(cfg["optim"]["betas"]),
        eps=float(cfg["optim"]["eps"]),
        weight_decay=float(cfg["loss"]["weight_decay"]),
    )


def train_main(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["exp"]["seed"]))
    device_name = cfg["exp"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    run_dir = ensure_dir(Path(cfg["exp"]["output_dir"]) / cfg["exp"]["name"])
    ensure_dir(run_dir / "checkpoints")
    dump_json(cfg, run_dir / "config.resolved.json")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = CONCORDModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    best_val = float("inf")
    best_metrics: dict[str, Any] = {}
    for epoch in range(1, int(cfg["optim"]["epochs"]) + 1):
        train_res = run_epoch(model, train_loader, optimizer, cfg, device, training=True)
        val_res = run_epoch(model, val_loader, None, cfg, device, training=False)
        record = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "val_loss": val_res.loss,
            "train_metrics": train_res.metrics,
            "val_metrics": val_res.metrics,
        }
        dump_json(record, run_dir / f"epoch_{epoch:03d}.json")
        primary = cfg["metrics"]["primary"]
        if val_res.metrics[primary] < best_val:
            best_val = val_res.metrics[primary]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
            })
            test_res = run_epoch(model, test_loader, None, cfg, device, training=False)
            best_metrics = {
                "best_epoch": epoch,
                "val": val_res.metrics,
                "test": test_res.metrics,
            }
            dump_json(best_metrics, run_dir / "metrics.json")
    return best_metrics
