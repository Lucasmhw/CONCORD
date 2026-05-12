import torch

from concord.models.graph import build_correlation_graph


def test_graph_shapes() -> None:
    x = torch.randn(2, 96, 4)
    a, l = build_correlation_graph(x, topk=2)
    assert a.shape == (2, 4, 4)
    assert l.shape == (2, 4, 4)
