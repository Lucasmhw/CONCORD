from __future__ import annotations

from pathlib import Path

import torch

from concord.data.datasets import ForecastingDataset, ImputationDataset, WindowSpec, load_processed_split
from concord.engine import run_epoch
from concord.models.concord import CONCORDModel
from concord.utils.checkpoint import load_checkpoint
from concord.utils.logging import dump_json


def evaluate_main(cfg: dict) -> dict:
    device_name = cfg["exp"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    ckpt = load_checkpoint(cfg["eval"]["checkpoint"], map_location=device_name)
    model = CONCORDModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    if cfg["data"].get("task", "forecasting") == "imputation":
        test = ImputationDataset(load_processed_split(cfg["data"]["processed_dir"], "test"), seq_len=int(cfg["data"]["sequence_length"]), stride=int(cfg["data"].get("stride", 32)), mask_ratios=cfg["data"]["mask_ratios"])
    else:
        spec = WindowSpec(lookback=int(cfg["data"]["lookback"]), horizon=int(cfg["data"]["horizon"]), stride=int(cfg["data"].get("stride", 1)))
        test = ForecastingDataset(load_processed_split(cfg["data"]["processed_dir"], "test"), spec)
    loader = torch.utils.data.DataLoader(test, batch_size=int(cfg["optim"]["batch_size"]), shuffle=False)
    res = run_epoch(model, loader, None, cfg, device, training=False)
    metrics = {"loss": res.loss, **res.metrics}
    out_path = Path(cfg["exp"]["output_dir"]) / cfg["exp"]["name"] / "eval_metrics.json"
    dump_json(metrics, out_path)
    return metrics
