import torch

from concord.data.concepts import compute_concept_targets


def test_concept_shape() -> None:
    x = torch.randn(2, 128, 5)
    q = compute_concept_targets(x, [16, 32, 64])
    assert q.shape == (2, 5, 15)
