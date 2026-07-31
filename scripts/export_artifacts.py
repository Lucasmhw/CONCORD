from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from concord.data.concepts import compute_concept_targets, compute_concept_trajectory
from concord.models.concord import CONCORDModel
from concord.training.train import build_dataloaders
from concord.utils.checkpoint import load_checkpoint
from concord.utils.seed import set_seed


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic CONCORD figure artifacts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = checkpoint["cfg"]
    set_seed(
        int(cfg["exp"]["seed"]),
        deterministic=bool(cfg["exp"].get("deterministic", True)),
    )
    requested_device = args.device
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    cfg["exp"]["device"] = str(device)
    _, _, test_loader = build_dataloaders(cfg)
    model = CONCORDModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    stats = np.load(Path(cfg["data"]["processed_dir"]) / "scaler_stats.npz")
    mean = torch.from_numpy(stats["mean"]).float().to(device)
    std = torch.from_numpy(stats["std"]).float().to(device)
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "x_hist",
            "y_true",
            "y_pred",
            "x_hist_raw",
            "y_true_raw",
            "y_pred_raw",
            "q_hat",
            "q0",
            "q_target",
            "q_target_raw",
            "q_mixed",
            "q_mixed_raw",
            "adj",
            "lap",
            "u",
            "ell",
            "innovation",
        )
    }
    sample_count = 0
    with torch.inference_mode():
        for batch in test_loader:
            remaining = args.max_samples - sample_count
            if remaining <= 0:
                break
            batch = {
                key: value[:remaining].to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(
                batch["x_hist"],
                horizon=batch["y"].shape[1],
                x_hist_raw=batch.get("x_hist_raw"),
            )
            pred_raw = output.pred * std + mean
            x_hist_raw = batch.get("x_hist_raw", batch["x_hist"] * std + mean)
            y_raw = batch.get("y_raw", batch["y"] * std + mean)
            mixed = compute_concept_trajectory(batch["x_hist"], output.pred, model.scales)
            mixed_raw = compute_concept_trajectory(x_hist_raw, pred_raw, model.scales)
            q_target_raw = output.q_target_raw
            if q_target_raw is None:
                q_target_raw = compute_concept_targets(x_hist_raw, model.scales)
            values = {
                "x_hist": batch["x_hist"],
                "y_true": batch["y"],
                "y_pred": output.pred,
                "x_hist_raw": x_hist_raw,
                "y_true_raw": y_raw,
                "y_pred_raw": pred_raw,
                "q_hat": output.q_hat,
                "q0": output.q0,
                "q_target": output.q_target,
                "q_target_raw": q_target_raw,
                "q_mixed": mixed,
                "q_mixed_raw": mixed_raw,
                "adj": output.adj,
                "lap": output.lap,
                "u": torch.stack(output.u_states, dim=1),
                "ell": torch.stack(output.ell_states, dim=1),
                "innovation": torch.stack(output.innov_states, dim=1),
            }
            for key, tensor in values.items():
                collected[key].append(_to_numpy(tensor))
            sample_count += output.pred.shape[0]

    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset": cfg["data"]["dataset_name"],
        "seed": cfg["exp"]["seed"],
        "lookback": cfg["data"]["lookback"],
        "horizon": cfg["data"]["horizon"],
        "scales": model.scales,
        "samples": sample_count,
        "best_epoch": checkpoint["epoch"],
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Wrote {output_path} with {sample_count} held-out samples")


if __name__ == "__main__":
    main()
