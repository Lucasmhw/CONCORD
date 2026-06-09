import torch

from concord.models.concord import CONCORDModel


def tiny_cfg() -> dict:
    return {
        "model": {
            "scales": [4, 8],
            "delta": 0.5,
            "gamma": 0.2,
            "mu": 0.1,
            "topk": 1,
            "corr_window": 8,
            "kappa": 2.0,
            "use_graph": True,
            "rollout_mode": "concept",
            "d_model": 8,
            "encoder_channels": [4, 8],
            "kernel_size": 3,
            "num_basis": 6,
            "spline_order": 2,
            "grid_min": -3.0,
            "grid_max": 3.0,
            "dropout": 0.0,
            "use_kan": True,
        }
    }


def test_forward_shape_and_linear_forcing_surface() -> None:
    model = CONCORDModel(tiny_cfg())
    out = model(torch.randn(2, 16, 3), horizon=5)
    assert out.pred.shape == (2, 5, 3)
    assert out.q_hat.shape == (2, 3, 10)
    assert len(out.u_states) == 5
    assert model.forcing_coefficients().shape == (2, 5)


def test_no_undocumented_forcing_heads() -> None:
    model = CONCORDModel(tiny_cfg())
    assert not hasattr(model, "step_emb")
    assert not hasattr(model, "u_head")
    assert not hasattr(model, "innov_head")
