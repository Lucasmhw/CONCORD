import torch

from concord.models.graph import build_correlation_graph


def test_graph_shapes() -> None:
    x = torch.randn(2, 96, 4)
    a, l = build_correlation_graph(x, topk=2)
    assert a.shape == (2, 4, 4)
    assert l.shape == (2, 4, 4)
    assert torch.allclose(a, a.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(l, l.transpose(-1, -2), atol=1e-6)
    assert torch.linalg.eigvalsh(l).min() >= -1e-5


def test_graph_single_series_is_identity() -> None:
    x = torch.randn(3, 12, 1)
    a, l = build_correlation_graph(x, topk=4)
    assert torch.allclose(a, torch.ones_like(a))
    assert torch.allclose(l, torch.zeros_like(l))
