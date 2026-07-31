from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from concord.engine import run_epoch
from concord.models.concord import CONCORDModel
from concord.training.train import build_dataloaders
from concord.utils.checkpoint import load_checkpoint
from concord.utils.logging import dump_json
from concord.utils.seed import set_seed


def evaluate_main(cfg: dict) -> dict:
    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg["exp"].get("deterministic", True)),
    )
    requested_device = str(cfg["exp"].get("device", "cuda"))
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = load_checkpoint(cfg["eval"]["checkpoint"], map_location=str(device))
    model = CONCORDModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, _, test_loader = build_dataloaders(cfg)
    stats = np.load(Path(cfg["data"]["processed_dir"]) / "scaler_stats.npz")
    scaler = (
        torch.from_numpy(stats["mean"]).float().to(device),
        torch.from_numpy(stats["std"]).float().to(device),
    )
    result = run_epoch(
        model,
        test_loader,
        None,
        cfg,
        device,
        training=False,
        global_step=int(checkpoint.get("global_step", 0)),
        scaler=scaler,
    )
    metrics = {"loss": result.loss, **result.metrics}
    output_path = Path(cfg["exp"]["output_dir"]) / cfg["exp"]["name"] / "eval_metrics.json"
    dump_json(metrics, output_path)
    return metrics
