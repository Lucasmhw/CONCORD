from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from concord.data.datasets import (
    ForecastingDataset,
    ImputationDataset,
    WindowSpec,
    load_processed_metadata,
    load_processed_split,
)
from concord.engine import run_epoch
from concord.models.concord import CONCORDModel
from concord.utils.checkpoint import load_checkpoint, save_checkpoint
from concord.utils.environment import environment_metadata
from concord.utils.logging import dump_json, ensure_dir
from concord.utils.seed import set_seed


def _load_split_arrays(processed_dir: str, split: str) -> tuple[np.ndarray, np.ndarray | None]:
    normalized = load_processed_split(processed_dir, split)
    raw_path = Path(processed_dir) / f"{split}_raw.npy"
    raw = load_processed_split(processed_dir, split, raw=True) if raw_path.exists() else None
    return normalized, raw


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_dir = cfg["data"]["processed_dir"]
    task = cfg["data"].get("task", "forecasting")
    seed = int(cfg["exp"]["seed"])
    metadata = load_processed_metadata(processed_dir)
    datasets = []
    for split_index, split in enumerate(("train", "val", "test")):
        normalized, raw = _load_split_arrays(processed_dir, split)
        target_offset = int(metadata.get("splits", {}).get(split, {}).get("target_offset", 0))
        if task == "imputation":
            mask_seed = int(cfg["data"].get("mask_seed", seed)) + split_index * 10_000
            dataset = ImputationDataset(
                normalized,
                seq_len=int(cfg["data"]["sequence_length"]),
                stride=int(cfg["data"].get("stride", 32)),
                mask_ratios=cfg["data"]["mask_ratios"],
                seed=mask_seed,
                raw_array=raw,
                target_offset=target_offset,
            )
        else:
            split_stride = int(cfg["data"].get(f"{split}_stride", cfg["data"].get("stride", 1)))
            dataset = ForecastingDataset(
                normalized,
                WindowSpec(
                    lookback=int(cfg["data"]["lookback"]),
                    horizon=int(cfg["data"]["horizon"]),
                    stride=split_stride,
                ),
                raw_array=raw,
                target_offset=target_offset,
            )
        if len(dataset) == 0:
            raise ValueError(f"No valid {split} samples in {processed_dir}")
        datasets.append(dataset)

    batch_size = int(cfg["optim"]["batch_size"])
    num_workers = int(cfg["exp"].get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": bool(cfg["exp"].get("pin_memory", True)),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        datasets[0],
        shuffle=True,
        drop_last=bool(cfg["data"].get("drop_last", False)),
        generator=generator,
        **common,
    )
    val_loader = DataLoader(datasets[1], shuffle=False, **common)
    test_loader = DataLoader(datasets[2], shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        betas=tuple(cfg["optim"]["betas"]),
        eps=float(cfg["optim"]["eps"]),
        weight_decay=float(cfg["loss"]["weight_decay"]),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(cfg["optim"].get("scheduler", "none")).lower()
    epochs = int(cfg["optim"]["epochs"])
    warmup = int(cfg["optim"].get("warmup_epochs", 0))
    if name in {"none", "null", ""}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup],
    )


def _scaler_tensors(processed_dir: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats = np.load(Path(processed_dir) / "scaler_stats.npz")
    mean = torch.from_numpy(stats["mean"]).float().to(device)
    std = torch.from_numpy(stats["std"]).float().to(device)
    return mean, std


def train_main(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg["exp"].get("deterministic", True)),
    )
    requested_device = str(cfg["exp"].get("device", "cuda"))
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    run_dir = ensure_dir(Path(cfg["exp"]["output_dir"]) / cfg["exp"]["name"])
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    dump_json(cfg, run_dir / "config.resolved.json")
    dump_json(environment_metadata(), run_dir / "environment.json")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    scaler = _scaler_tensors(cfg["data"]["processed_dir"], device)
    model = CONCORDModel(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_value = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = int(cfg["optim"].get("early_stopping_patience", 0))
    global_step = 0
    primary = str(cfg["metrics"]["primary"])
    best_path = checkpoint_dir / "best.pt"

    for epoch in range(1, int(cfg["optim"]["epochs"]) + 1):
        train_result = run_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
            training=True,
            global_step=global_step,
            scaler=scaler,
        )
        global_step = train_result.global_step
        val_result = run_epoch(
            model,
            val_loader,
            None,
            cfg,
            device,
            training=False,
            global_step=global_step,
            scaler=scaler,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "train_metrics": train_result.metrics,
            "val_metrics": val_result.metrics,
            "train_terms": train_result.loss_terms,
            "val_terms": val_result.loss_terms,
            "lr": optimizer.param_groups[0]["lr"],
        }
        dump_json(record, run_dir / f"epoch_{epoch:03d}.json")

        current = val_result.metrics[primary]
        if current < best_value:
            best_value = current
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_metrics": val_result.metrics,
                },
            )
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if patience > 0 and epochs_without_improvement >= patience:
            break

    checkpoint = load_checkpoint(best_path, map_location=str(device))
    run_test = bool(cfg.get("eval", {}).get("run_test_after_training", True))
    test_result = None
    if run_test:
        model.load_state_dict(checkpoint["model_state"])
        test_result = run_epoch(
            model,
            test_loader,
            None,
            cfg,
            device,
            training=False,
            global_step=global_step,
            scaler=scaler,
        )
    summary = {
        "seed": int(cfg["exp"]["seed"]),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_val": checkpoint["val_metrics"],
        "test": test_result.metrics if test_result is not None else None,
        "test_loss": test_result.loss if test_result is not None else None,
    }
    dump_json(summary, run_dir / "metrics.json")
    return summary
