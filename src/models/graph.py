from __future__ import annotations

import torch


def build_correlation_graph(x_hist: torch.Tensor, topk: int, kappa: float = 2.0, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Build symmetric normalized adjacency and Laplacian from x_hist [B, W, N]."""
    b, w, n = x_hist.shape
    x = x_hist - x_hist.mean(dim=1, keepdim=True)
    x = x / (x_hist.std(dim=1, keepdim=True) + eps)
    corr = torch.einsum("bwn,bwm->bnm", x, x) / max(w - 1, 1)
    corr = corr.clamp(-1.0, 1.0)
    abs_corr = corr.abs()
    eye = torch.eye(n, device=x_hist.device, dtype=x_hist.dtype).unsqueeze(0)
    abs_corr = abs_corr * (1.0 - eye)

    k = min(topk, max(n - 1, 1))
    topk_vals, topk_idx = torch.topk(abs_corr, k=k, dim=-1)
    directed = torch.zeros_like(abs_corr)
    directed.scatter_(-1, topk_idx, torch.exp(kappa * topk_vals))

    sym = 0.5 * (directed + directed.transpose(-1, -2))
    degree = sym.sum(dim=-1)
    d_inv_sqrt = torch.pow(degree + eps, -0.5)
    d_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    adj = d_inv_sqrt @ sym @ d_inv_sqrt
    lap = eye - adj
    return adj, lap
