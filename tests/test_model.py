import torch

from concord.losses import total_loss
from concord.models.concord import CONCORDModel


def tiny_cfg() -> dict:
    return {
        "model": {
            "scales": [4, 8],
            "delta": 0.5,
            "gamma_init": 0.05,
            "mu_init": 0.05,
            "step_emb_dim": 4,
            "step_emb_init_std": 1.0,
            "max_horizon": 8,
            "max_lookback": 32,
            "topk": 1,
            "corr_window": 8,
            "kappa": 2.0,
            "use_graph": True,
            "rollout_mode": "specialized",
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "d_ff": 16,
            "num_knots": 5,
            "grid_min": -1.0,
            "grid_max": 1.0,
            "dropout": 0.0,
            "use_kan": True,
            "norm_style": "post",
        },
        "loss": {
            "lambda_con": 0.1,
            "lambda_rel": 0.01,
            "lambda_res": 0.3,
            "lambda_x": 1.0,
            "lambda_k": [1.0] * 5,
            "residual_warmup_steps": 10,
            "lambda_innov": 1e-3,
            "lambda_flat": 1e-3,
            "flat_std_floor": 0.02,
        },
    }


def test_forward_shape_and_methods_readouts() -> None:
    model = CONCORDModel(tiny_cfg())
    x_hist = torch.randn(2, 16, 3)
    out = model(x_hist, horizon=5, x_hist_raw=2.0 * x_hist + 3.0)
    assert out.pred.shape == (2, 5, 3)
    assert out.q_hat.shape == (2, 3, 10)
    assert out.q_target_raw is not None
    assert len(out.u_states) == 5
    assert len(out.ell_states) == 5
    assert len(out.innov_states) == 5
    assert hasattr(model, "step_emb")
    assert hasattr(model, "u_head")
    assert hasattr(model, "ell_head")
    assert hasattr(model, "innov_head")
    assert model.gamma.item() > 0
    assert model.mu.item() > 0


def test_specialized_rollout_keeps_graph_refined_concept_state() -> None:
    model = CONCORDModel(tiny_cfg())
    out = model(torch.randn(1, 16, 2), horizon=4)
    for state in out.q_states[1:]:
        assert torch.equal(state, out.q0)


def test_step_embedding_uses_configured_initialization_scale() -> None:
    cfg = tiny_cfg()
    cfg["model"]["max_horizon"] = 512
    cfg["model"]["step_emb_init_std"] = 0.7
    torch.manual_seed(11)
    model = CONCORDModel(cfg)
    assert abs(float(model.step_emb.weight.detach().std()) - 0.7) < 0.05


def test_total_loss_backpropagates_through_methods_objective() -> None:
    cfg = tiny_cfg()
    model = CONCORDModel(cfg)
    batch = {"x_hist": torch.randn(2, 16, 3), "y": torch.randn(2, 4, 3)}
    output = model(batch["x_hist"], horizon=4)
    loss, terms = total_loss(output, batch, cfg, global_step=5)
    loss.backward()
    assert torch.isfinite(loss)
    assert terms["lambda_res_effective"] == 0.15
    assert terms["observation_residual"] > 0.0
    assert model.u_head.net[0].coeff.grad is not None
