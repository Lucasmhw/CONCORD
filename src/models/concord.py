from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from concord.data.concepts import compute_concept_targets
from concord.models.encoder import SeriesLocalEncoder
from concord.models.graph import build_correlation_graph
from concord.models.kan import KANMLP


@dataclass
class ForwardOutput:
    pred: torch.Tensor
    q_hat: torch.Tensor
    q0: torch.Tensor
    q_states: list[torch.Tensor]
    x_states: list[torch.Tensor]
    adj: torch.Tensor
    lap: torch.Tensor
    q_target: torch.Tensor
    beta: torch.Tensor
    beta0: torch.Tensor
    alpha_logits: torch.Tensor
    delta: float
    gamma: float
    mu: float


class CONCORDModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        self.scales = list(mcfg["scales"])
        if not mcfg.get("use_multiscale", True):
            self.scales = [self.scales[len(self.scales) // 2]]
        self.num_concepts = 5 * len(self.scales)
        self.delta = float(mcfg["delta"])
        self.gamma = float(mcfg["gamma"])
        self.mu = float(mcfg["mu"])
        self.topk = int(mcfg["topk"])
        self.kappa = float(mcfg["kappa"])
        self.use_graph = bool(mcfg.get("use_graph", True))
        self.rollout_mode = str(mcfg.get("rollout_mode", "concept"))

        self.encoder = SeriesLocalEncoder(
            d_model=int(mcfg["d_model"]),
            channels=list(mcfg["encoder_channels"]),
            kernel_size=int(mcfg["kernel_size"]),
            num_basis=int(mcfg["num_basis"]),
            spline_order=int(mcfg["spline_order"]),
            grid_min=float(mcfg["grid_min"]),
            grid_max=float(mcfg["grid_max"]),
            dropout=float(mcfg["dropout"]),
            use_kan=bool(mcfg.get("use_kan", True)),
        )
        d_model = int(mcfg["d_model"])
        common = dict(
            num_basis=int(mcfg["num_basis"]),
            degree=int(mcfg["spline_order"]),
            grid_min=float(mcfg["grid_min"]),
            grid_max=float(mcfg["grid_max"]),
            dropout=float(mcfg["dropout"]),
            use_kan=bool(mcfg.get("use_kan", True)),
        )
        self.concept_head = KANMLP([d_model, d_model, self.num_concepts], **common)
        self.phi = KANMLP([self.num_concepts, d_model, self.num_concepts], **common)
        self.psi = KANMLP([2 * self.num_concepts, d_model, self.num_concepts], **common)
        self.omega = KANMLP([2 * self.num_concepts + 1, d_model, self.num_concepts], **common)
        self.latent_rollout = KANMLP([d_model + 1, d_model, d_model], **common)
        self.latent_to_concept = KANMLP([d_model, d_model, self.num_concepts], **common)
        self.register_parameter("beta", nn.Parameter(torch.zeros(self.num_concepts)))
        self.register_parameter("beta0", nn.Parameter(torch.zeros(1)))
        self.register_parameter("alpha_logits", nn.Parameter(torch.zeros(len(self.scales))))

    def _graph(self, x_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = x_hist.shape[1]
        adj, lap = build_correlation_graph(x_hist[:, -w:, :], self.topk, self.kappa)
        if not self.use_graph:
            b, _, n = x_hist.shape
            adj = torch.eye(n, device=x_hist.device, dtype=x_hist.dtype).unsqueeze(0).repeat(b, 1, 1)
            lap = torch.zeros_like(adj)
        return adj, lap

    def _message(self, adj: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        msg_in = self.phi(q)
        return torch.einsum("bij,bjd->bid", adj, msg_in)

    def _level_reference(self, q: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.alpha_logits, dim=0)
        level_idx = [5 * i for i in range(len(self.scales))]
        levels = torch.stack([q[..., idx] for idx in level_idx], dim=-1)
        return torch.einsum("bns,s->bn", levels, alpha)

    def _forcing(self, q: torch.Tensor) -> torch.Tensor:
        return self.beta0 + torch.einsum("bnc,c->bn", q, self.beta)

    def forward(self, x_hist: torch.Tensor, horizon: int | None = None) -> ForwardOutput:
        # x_hist: [B, L, N]
        horizon = int(horizon or 1)
        q_target = compute_concept_targets(x_hist, self.scales)
        h0 = self.encoder(x_hist)
        q_hat = self.concept_head(h0)
        adj, lap = self._graph(x_hist)
        m0 = self._message(adj, q_hat)
        q0 = q_hat + self.psi(torch.cat([q_hat, m0], dim=-1))

        x_curr = x_hist[:, -1, :]
        q_curr = q0
        q_states = [q_curr]
        x_states = [x_curr]
        latent = h0

        for _ in range(horizon):
            msg = self._message(adj, q_curr)
            if self.rollout_mode == "latent":
                latent = latent + self.latent_rollout(torch.cat([latent, x_curr.unsqueeze(-1)], dim=-1))
                q_next = self.latent_to_concept(latent)
            else:
                q_next = q_curr + self.delta * self.omega(torch.cat([q_curr, msg, x_curr.unsqueeze(-1)], dim=-1))
            ell = self._level_reference(q_curr)
            u = self._forcing(q_curr)
            x_next = x_curr + self.delta * (u - self.gamma * (x_curr - ell) - self.mu * torch.einsum("bij,bj->bi", lap, x_curr))
            q_curr = q_next
            x_curr = x_next
            q_states.append(q_curr)
            x_states.append(x_curr)
        pred = torch.stack(x_states[1:], dim=1)
        return ForwardOutput(
            pred=pred,
            q_hat=q_hat,
            q0=q0,
            q_states=q_states,
            x_states=x_states,
            adj=adj,
            lap=lap,
            q_target=q_target,
            beta=self.beta,
            beta0=self.beta0,
            alpha_logits=self.alpha_logits,
            delta=self.delta,
            gamma=self.gamma,
            mu=self.mu,
        )
