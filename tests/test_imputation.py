import numpy as np
import torch

from concord.data.datasets import ImputationDataset
from concord.engine import causal_impute


def test_imputation_mask_is_seeded_by_sample() -> None:
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    ds1 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=123)
    ds2 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=123)
    ds3 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=456)

    assert torch.equal(ds1[2]["mask"], ds2[2]["mask"])
    assert not torch.equal(ds1[2]["mask"], ds3[2]["mask"])


class LastValueModel(torch.nn.Module):
    def forward(self, x_hist: torch.Tensor, horizon: int = 1):
        prediction = x_hist[:, -1:].expand(-1, horizon, -1)
        return type("Output", (), {"pred": prediction})()


def test_causal_imputation_uses_only_preceding_values() -> None:
    model = LastValueModel()
    model.eval()
    sequence = torch.arange(24, dtype=torch.float32).view(1, 12, 2)
    mask = torch.ones_like(sequence)
    mask[:, 6, 0] = 0
    observed = sequence * mask
    cfg = {
        "data": {
            "lookback": 4,
            "imputation_eval_origin_batch_size": 8,
        }
    }
    prediction_a, target_a, missing_a = causal_impute(
        model,
        {"sequence": sequence, "observed": observed, "mask": mask},
        cfg,
    )

    changed_future = sequence.clone()
    changed_future[:, 7:] += 1000
    changed_observed = changed_future * mask
    prediction_b, _, _ = causal_impute(
        model,
        {"sequence": changed_future, "observed": changed_observed, "mask": mask},
        cfg,
    )
    assert torch.equal(prediction_a, prediction_b)
    assert target_a.shape == missing_a.shape == prediction_a.shape
    assert missing_a.sum().item() == 1
