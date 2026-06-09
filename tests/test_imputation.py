import numpy as np
import torch

from concord.data.datasets import ImputationDataset


def test_imputation_mask_is_seeded_by_sample() -> None:
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    ds1 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=123)
    ds2 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=123)
    ds3 = ImputationDataset(x, seq_len=8, stride=2, mask_ratios=[0.25, 0.5], seed=456)

    assert torch.equal(ds1[2]["mask"], ds2[2]["mask"])
    assert not torch.equal(ds1[2]["mask"], ds3[2]["mask"])
