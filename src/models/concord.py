from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from concord.data.concepts import compute_concept_targets
from concord.models.encoder import SeriesLocalEncoder
from concord.models.graph import build_correlation_graph
from concord.models.kan import KANLinear, KANMLP


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(value_tensor)))


@dataclass
class ForwardOutput:
    pred: torch.Tensor
    q_hat: torch.Tensor
    q0: torch.Tensor
    q_states: list[torch.Tensor]
    x_states: list[torch.Tensor]
    u_states: list[torch.Tensor]
    ell_states: list[torch.Tensor]
    innov_states: list[torch.Tensor]
    adj: torch.Tensor
    lap: torch.Tensor
    q_target: torch.Tensor
    q_target_raw: torch.Tensor | None
    delta: float
    gamma: torch.Tensor
    mu: torch.Tensor


class CONCORDModel(nn.Module):
    """Manuscript-aligned horizon-conditioned CONCORD implementation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        self.scales = list(mcfg["scales"])
        if not mcfg.get("use_multiscale", True):
            self.scales = [self.scales[len(self.scales) // 2]]
        self.num_concepts = 5 * len(self.scales)
        self.delta = float(mcfg.get("delta", 0.5))
        self.topk = int(mcfg["topk"])
        self.kappa = float(mcfg["kappa"])
        corr_window = mcfg.get("corr_window")
        self.corr_window = None if corr_window in {None, "null"} else int(corr_window)
        self.use_graph = bool(mcfg.get("use_graph", True))
        self.rollout_mode = str(mcfg.get("rollout_mode", "specialized"))
        self.max_horizon = int(mcfg.get("max_horizon", 720))

        d_model = int(mcfg["d_model"])
        num_knots = int(mcfg.get("num_knots", mcfg.get("num_basis", 9)))
        grid_min = float(mcfg.get("grid_min", -1.0))
        grid_max = float(mcfg.get("grid_max", 1.0))
        dropout = float(mcfg["dropout"])
        use_kan = bool(mcfg.get("use_kan", True))
        common = {
            "num_knots": num_knots,
            "grid_min": grid_min,
            "grid_max": grid_max,
            "dropout": dropout,
            "use_kan": use_kan,
        }

        self.encoder = SeriesLocalEncoder(
            d_model=d_model,
            num_heads=int(mcfg.get("num_heads", 4)),
            num_layers=int(mcfg.get("num_layers", 3)),
            d_ff=int(mcfg.get("d_ff", 128)),
            num_knots=num_knots,
            grid_min=grid_min,
            grid_max=grid_max,
            dropout=dropout,
            use_kan=use_kan,
            max_len=int(mcfg.get("max_lookback", 1024)),
            series_chunk_size=int(mcfg.get("series_chunk_size", 0)),
            norm_style=str(mcfg.get("norm_style", "pre")),
        )
        self.concept_head = KANMLP([d_model, d_model, self.num_concepts], **common)
        self.phi = KANMLP(
            [self.num_concepts, self.num_concepts, self.num_concepts],
            **common,
        )
        self.psi = KANMLP(
            [2 * self.num_concepts, self.num_concepts, self.num_concepts],
            **common,
        )

        step_dim = int(mcfg.get("step_emb_dim", 32))
        readout_dim = self.num_concepts + step_dim
        self.step_emb = nn.Embedding(self.max_horizon, step_dim)
        self.u_head = KANMLP([readout_dim, self.num_concepts, 1], **common)
        self.ell_head = KANMLP([readout_dim, self.num_concepts, 1], **common)
        self.innov_head = KANMLP([readout_dim, self.num_concepts, 1], **common)

        corrector_dim = 2 * self.num_concepts + 1
        self.omega = KANMLP([corrector_dim, self.num_concepts, self.num_concepts], **common)
        self.latent_update = KANMLP([d_model + 1, d_model, d_model], **common)
        self.latent_to_concept = KANMLP([d_model, d_model, self.num_concepts], **common)

        gamma_init = float(mcfg.get("gamma_init", 0.05))
        mu_init = float(mcfg.get("mu_init", 0.05))
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init), dtype=torch.float32))
        self.mu_raw = nn.Parameter(torch.tensor(_inverse_softplus(mu_init), dtype=torch.float32))
        nn.init.normal_(
            self.step_emb.weight,
            mean=0.0,
            std=float(mcfg.get("step_emb_init_std", 1.0)),
        )
        self._initialize_small_innovation()

    def _initialize_small_innovation(self) -> None:
        final_layer = self.innov_head.net[-1]
        if isinstance(final_layer, KANLinear):
            nn.init.normal_(final_layer.coeff, mean=0.0, std=1e-3)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
        elif isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw) + 1e-6

    @property
    def mu(self) -> torch.Tensor:
        return F.softplus(self.mu_raw)

    def _graph(self, x_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = x_hist.shape[1] if self.corr_window is None else min(self.corr_window, x_hist.shape[1])
        adj, lap = build_correlation_graph(x_hist[:, -window:, :], self.topk, self.kappa)
        if not self.use_graph:
            batch, _, num_series = x_hist.shape
            adj = torch.eye(num_series, device=x_hist.device, dtype=x_hist.dtype)
            adj = adj.unsqueeze(0).expand(batch, -1, -1)
            lap = torch.zeros_like(adj)
        return adj, lap

    def _message(self, adj: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bjd->bid", adj, self.phi(q))

    def _readouts(self, q: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_series, _ = q.shape
        step_ids = torch.full((batch, num_series), step, device=q.device, dtype=torch.long)
        inputs = torch.cat([q, self.step_emb(step_ids)], dim=-1)
        return (
            self.u_head(inputs).squeeze(-1),
            self.ell_head(inputs).squeeze(-1),
            self.innov_head(inputs).squeeze(-1),
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        horizon: int | None = None,
        x_hist_raw: torch.Tensor | None = None,
    ) -> ForwardOutput:
        horizon = int(horizon or 1)
        if horizon > self.max_horizon:
            raise ValueError(f"horizon={horizon} exceeds model.max_horizon={self.max_horizon}")
        if x_hist.shape[1] < max(self.scales) + 1:
            raise ValueError("lookback must be at least max(scales) + 1")

        q_target = compute_concept_targets(x_hist, self.scales)
        q_target_raw = (
            compute_concept_targets(x_hist_raw, self.scales) if x_hist_raw is not None else None
        )
        latent = self.encoder(x_hist)
        q_hat = self.concept_head(latent)
        adj, lap = self._graph(x_hist)
        q0 = q_hat + self.psi(torch.cat([q_hat, self._message(adj, q_hat)], dim=-1))

        x_curr = x_hist[:, -1]
        q_curr = q0
        q_states = [q_curr]
        x_states = [x_curr]
        u_states: list[torch.Tensor] = []
        ell_states: list[torch.Tensor] = []
        innov_states: list[torch.Tensor] = []

        for step in range(horizon):
            u_step, ell_step, innov_step = self._readouts(q_curr, step)
            laplacian_term = torch.einsum("bij,bj->bi", lap, x_curr)
            derivative = (
                u_step
                - self.gamma * (x_curr - ell_step)
                - self.mu * laplacian_term
                + innov_step
            )
            x_next = x_curr + self.delta * derivative

            if self.rollout_mode == "recursive":
                message = self._message(adj, q_curr)
                q_next = q_curr + self.delta * self.omega(
                    torch.cat([q_curr, x_curr.unsqueeze(-1), message], dim=-1)
                )
            elif self.rollout_mode == "latent":
                latent = latent + self.latent_update(torch.cat([latent, x_curr.unsqueeze(-1)], dim=-1))
                q_next = self.latent_to_concept(latent)
            elif self.rollout_mode == "specialized":
                q_next = q0
            else:
                raise ValueError(f"Unsupported rollout_mode: {self.rollout_mode}")

            u_states.append(u_step)
            ell_states.append(ell_step)
            innov_states.append(innov_step)
            x_curr = x_next
            q_curr = q_next
            x_states.append(x_curr)
            q_states.append(q_curr)

        return ForwardOutput(
            pred=torch.stack(x_states[1:], dim=1),
            q_hat=q_hat,
            q0=q0,
            q_states=q_states,
            x_states=x_states,
            u_states=u_states,
            ell_states=ell_states,
            innov_states=innov_states,
            adj=adj,
            lap=lap,
            q_target=q_target,
            q_target_raw=q_target_raw,
            delta=self.delta,
            gamma=self.gamma,
            mu=self.mu,
        )
