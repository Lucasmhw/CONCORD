
# -*- coding: utf-8 -*-
"""
CONCORD v2 (non-flat rollout): Concept- and CORrelation-coupled Dynamics forecaster
-------------------------------------------------------------------------------
Key fix vs v1:
- Step-conditioned drift and equilibrium: u_{t,h}, ell_{t,h} depend on horizon h
- Add small innovation forcing epsilon_{t,h} (learned) to avoid fixed-point collapse
- Small initial gamma/mu + physics warm-up to prevent early over-damping

Input:
  CSV file, each non-date column is a univariate series; all columns forecast jointly.

Protocol (NO leakage):
- Split at split = T - pred_len
- Fit standardization on [0, split) only
- Train windows sampled entirely within [0, split)
- Test: last lookback window before split -> predict last pred_len steps
- Metrics computed on standardized scale per series, then averaged across series.

Run (Windows example):
  python concord_run_v2.py --dataset electricity --data_dir 

This script also works in this sandbox if your CSVs are in /mnt/data.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
# Only clear argv in notebook kernels; keep CLI args in normal execution
if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
    sys.argv = ['']

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d-%H%M%S")


def find_csv_for_dataset(data_dir: str, dataset: str) -> str:
    dataset_l = dataset.lower()
    cand1 = os.path.join(data_dir, f"{dataset}.csv")
    if os.path.isfile(cand1):
        return cand1

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    for f in files:
        if dataset_l in f.lower():
            return os.path.join(data_dir, f)

    raise FileNotFoundError(
        f"Cannot find CSV for dataset='{dataset}' in '{data_dir}'. "
        f"Expected '{dataset}.csv' or a file name containing '{dataset}'."
    )


def load_csv_timeseries(csv_path: str, series_limit: int = 0) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    date_cols = []
    for c in df.columns:
        if re.fullmatch(r"(date|datetime|timestamp|time)", str(c).strip().lower()):
            date_cols.append(c)
    if len(date_cols) > 0:
        df = df.drop(columns=date_cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.ffill().bfill()
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {csv_path} after dropping date-like columns.")

    # optional cap number of series (for debugging large datasets)
    if series_limit and df.shape[1] > series_limit:
        df = df.iloc[:, :series_limit]

    col_names = list(df.columns)
    data = df.values.astype(np.float32)

    std = data.std(axis=0)
    keep = std > 1e-8
    if keep.sum() < data.shape[1]:
        dropped = [col_names[i] for i in range(len(col_names)) if not keep[i]]
        print(f"[WARN] Dropping constant columns: {dropped}")
        data = data[:, keep]
        col_names = [c for i, c in enumerate(col_names) if keep[i]]

    return data, col_names


def fit_standardizer(train_data: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean, std


def apply_standardizer(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data - mean) / std


def series_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mse = ((y_pred - y_true) ** 2).mean(dim=0)
    mae = (y_pred - y_true).abs().mean(dim=0)
    return mse, mae


# =========================================================
# KAN modules (hat-basis spline KAN)
# =========================================================
class KANLayer(nn.Module):
    """
    A lightweight KAN layer with triangular (hat) basis functions on fixed knots.
    """
    def __init__(self, in_features: int, out_features: int, n_knots: int = 9, x_min: float = -1.0, x_max: float = 1.0):
        super().__init__()
        assert n_knots >= 3
        self.in_features = in_features
        self.out_features = out_features
        self.n_knots = n_knots

        self.register_buffer("knots", torch.linspace(x_min, x_max, n_knots))
        self.h = (x_max - x_min) / (n_knots - 1)

        self.coeff = nn.Parameter(torch.randn(in_features, out_features, n_knots) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # per-input affine normalization into [x_min, x_max]
        self.scale = nn.Parameter(torch.ones(in_features))
        self.shift = nn.Parameter(torch.zeros(in_features))


    @torch.no_grad()
    def eval_phi(self, in_idx: int, out_idx: int, x_grid: torch.Tensor) -> torch.Tensor:
        assert 0 <= in_idx < self.in_features
        assert 0 <= out_idx < self.out_features
        x = x_grid * self.scale[in_idx] + self.shift[in_idx]
        x = torch.clamp(x, self.knots[0].item(), self.knots[-1].item())
        dist = torch.abs(x.unsqueeze(-1) - self.knots.view(1, -1))
        basis = torch.clamp(1.0 - dist / self.h, min=0.0)
        coeff = self.coeff[in_idx, out_idx, :]
        return (basis * coeff.unsqueeze(0)).sum(dim=-1)


class KANMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, n_knots: int = 9, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        self.layers = nn.ModuleList([KANLayer(dims[i], dims[i + 1], n_knots=n_knots) for i in range(len(dims) - 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        return x


# =========================================================
# Transformer encoder with KAN feed-forward
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T]


class KANFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_knots: int = 9, dropout: float = 0.1):
        super().__init__()
        self.fc1 = KANLayer(d_model, d_ff, n_knots=n_knots)
        self.fc2 = KANLayer(d_ff, d_model, n_knots=n_knots)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x.reshape(-1, x.size(-1)))
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y.view(*x.shape)
        return self.norm(x + y)


class KANTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, n_knots: int = 9):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = KANFeedForward(d_model, d_ff, n_knots=n_knots, dropout=dropout)



class KANTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 dropout: float = 0.1, n_knots: int = 9, max_len: int = 4096):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            KANTransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, n_knots=n_knots)
            for _ in range(n_layers)
        ])

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, return_attn: bool = False, attn_layers: int = 1):
        x = self.pos(x)
        attn_maps = []
        mask = self.causal_mask(x.size(1), x.device)
        for li, layer in enumerate(self.layers):
            ra = return_attn and (li >= len(self.layers) - attn_layers)
            x, attn = layer(x, attn_mask=mask, return_attn=ra)
            if ra and attn is not None:
                attn_maps.append(attn)
        return x, attn_maps


# =========================================================
# Graph utilities
# =========================================================
def corrcoef_batch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B, W, N]
    B, W, N = x.shape
    xm = x - x.mean(dim=1, keepdim=True)
    cov = torch.matmul(xm.transpose(1, 2), xm) / max(W - 1, 1)  # [B,N,N]
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp(min=eps)  # [B,N]
    std = torch.sqrt(var)
    corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + eps)
    return corr.clamp(-1.0, 1.0)


def build_topk_graph(corr: torch.Tensor, topk: int = 8, kappa: float = 5.0) -> torch.Tensor:
    B, N, _ = corr.shape
    abs_corr = corr.abs()
    eye = torch.eye(N, device=corr.device).unsqueeze(0)
    abs_corr = abs_corr.masked_fill(eye.bool(), float("-inf"))

    k = min(topk, max(N - 1, 1))
    vals, idx = torch.topk(abs_corr, k=k, dim=-1)  # [B,N,k]
    logits = kappa * vals
    w = torch.softmax(logits, dim=-1)

    W = torch.zeros((B, N, N), device=corr.device, dtype=corr.dtype)
    W.scatter_(-1, idx, w)
    return W


# =========================================================
# Multi-scale concept targets (NO leakage: from lookback only)
# =========================================================
def compute_multiscale_concepts_from_lookback(x: torch.Tensor, scales: List[int], eps: float = 1e-6):
    """
    x: [B,L,N] standardized, corresponds to y[s-L : s] (last index is s-1)
    concepts per scale:
      level, velocity, power, amplitude, volatility
    """
    B, L, N = x.shape
    max_tau = max(scales)
    if L < max_tau + 2:
        raise ValueError(f"lookback={L} must be >= max(scales)+2={max_tau+2}.")

    S = len(scales)
    c_cur_list = []
    level_prev_list, vel_prev_list, vol_prev_list = [], [], []

    for tau in scales:
        win = x[:, -tau:, :]               # end s-1
        win_prev = x[:, -tau-1:-1, :]      # end s-2
        win_prev2 = x[:, -tau-2:-2, :]     # end s-3


        vol = torch.sqrt(detr.pow(2).mean(dim=1).clamp(min=eps))
        c_cur = torch.stack([level, vel, power, amp, vol], dim=-1)  # [B,N,5]
        c_cur_list.append(c_cur)

        detr_prev = win_prev - level_prev.unsqueeze(1)
        vol_prev = torch.sqrt(detr_prev.pow(2).mean(dim=1).clamp(min=eps))

        level_prev_list.append(level_prev)
        vel_prev_list.append(vel_prev)
        vol_prev_list.append(vol_prev)


# =========================================================
# CONCORD v2 model
# =========================================================
class CONCORDModel(nn.Module):
    """
    Main difference vs v1: step-conditioned u/ell and a learnable innovation forcing.
    This prevents the rollout from collapsing to a fixed point (flatline).
    """
    def __init__(
        self,
        n_series: int,
        pred_len: int = 96,
        lookback: int = 336,
        scales: Optional[List[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        kan_knots: int = 9,
        graph_window: int = 96,
        topk: int = 8,
        kappa: float = 5.0,
        dt: float = 0.5,
        step_emb_dim: int = 16,
    ):
        super().__init__()
        self.n_series = n_series
        self.pred_len = pred_len
        self.lookback = lookback
        self.scales = scales if scales is not None else [24, 48, 96]
        self.S = len(self.scales)
        self.K = 5
        self.concept_dim = self.S * self.K

        self.graph_window = graph_window
        self.topk = topk
        self.kappa = kappa
        self.dt = dt

        # per-series temporal encoder (shared weights), implemented by flattening B*N
        self.in_proj = nn.Linear(1, d_model)
        self.encoder = KANTransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout, n_knots=kan_knots,
            max_len=lookback + 8
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # concept inference
        self.concept_head = KANMLP(d_model, hidden_dims=[d_model], out_dim=self.concept_dim, n_knots=kan_knots, dropout=dropout)

        # concept-space message passing
        self.phi = KANMLP(self.concept_dim, hidden_dims=[self.concept_dim], out_dim=self.concept_dim, n_knots=kan_knots, dropout=dropout)
        self.psi = KANMLP(self.concept_dim * 2, hidden_dims=[self.concept_dim], out_dim=self.concept_dim, n_knots=kan_knots, dropout=dropout)

        # --- Step-conditioned drift/equilibrium + innovation forcing ---
        self.step_emb = nn.Embedding(self.pred_len, step_emb_dim)
        self.u_head = KANMLP(self.concept_dim + step_emb_dim, hidden_dims=[self.concept_dim], out_dim=1, n_knots=kan_knots, dropout=dropout)
        self.ell_head = KANMLP(self.concept_dim + step_emb_dim, hidden_dims=[self.concept_dim], out_dim=1, n_knots=kan_knots, dropout=dropout)
        self.innov_head = KANMLP(self.concept_dim + step_emb_dim, hidden_dims=[self.concept_dim], out_dim=1, n_knots=kan_knots, dropout=dropout)

        # positive dynamics params via softplus; initialize small to avoid early over-damping
        self.gamma_raw = nn.Parameter(torch.tensor(-3.0))  # softplus ~ 0.05
        self.mu_raw = nn.Parameter(torch.tensor(-3.0))     # softplus ~ 0.05

    def gamma(self):
        return F.softplus(self.gamma_raw) + 1e-6

    def mu(self):
        return F.softplus(self.mu_raw)

    def forward(self, x: torch.Tensor, return_attn: bool = False, attn_layers: int = 1) -> Dict[str, torch.Tensor]:
        """
        x: [B,L,N] standardized
        """
        B, L, N = x.shape
        assert N == self.n_series
        assert L == self.lookback

        # encode each series independently via shared temporal encoder
        x_bn = x.permute(0, 2, 1).contiguous().view(B * N, L, 1)
        h = self.in_proj(x_bn)
        h, attn_maps = self.encoder(h, return_attn=return_attn, attn_layers=attn_layers)
        h = self.enc_norm(h)
        z = h[:, -1, :].view(B, N, -1)  # [B,N,d]

        c_hat = self.concept_head(z.reshape(B * N, -1)).view(B, N, -1)

        # concept-space coupling
        phi_c = self.phi(c_hat.reshape(B * N, -1)).view(B, N, -1)
        m = torch.matmul(W, phi_c)
        c_tilde = c_hat + self.psi(torch.cat([c_hat, m], dim=-1).reshape(B * N, -1)).view(B, N, -1)

        gamma = self.gamma()
        mu = self.mu()

        # rollout with step-conditioned drift/equilibrium and innovation forcing
        y = x[:, -1, :]  # [B,N]
        preds = []
        u0 = None
        ell0 = None
        innov_l2 = 0.0

        for hstep in range(self.pred_len):
            h_emb = self.step_emb(torch.tensor(hstep, device=x.device)).view(1, 1, -1).expand(B, N, -1)
            inp = torch.cat([c_tilde, h_emb], dim=-1).reshape(B * N, -1)

            u_h = self.u_head(inp).view(B, N)
            ell_h = self.ell_head(inp).view(B, N)
            eps_h = self.innov_head(inp).view(B, N)

            innov_l2 = innov_l2 + eps_h.pow(2).mean()
            if hstep == 0:
                u0 = u_h
                ell0 = ell_h

            Ly = torch.matmul(Lmat, y.unsqueeze(-1)).squeeze(-1)
            dy = u_h - gamma * (y - ell_h) - mu * Ly + eps_h
            y = y + self.dt * dy
            preds.append(y)

        y_hat = torch.stack(preds, dim=1)  # [B,H,N]

        return {
            "y_hat": y_hat,
            "c_hat": c_hat,
            "c_tilde": c_tilde,
            "c_star": c_star_flat,
            "aux": aux,
            "W": W,
            "L": Lmat,
            "corr": corr,
            "attn_maps": attn_maps,
            "gamma": gamma,
            "mu": mu,
            "u0": u0,
            "ell0": ell0,
            "innov_l2": innov_l2 / self.pred_len,
        }



def compute_total_loss(
    out: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y_true: torch.Tensor,
    cfg: "Config",
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    x: [B,L,N], y_true: [B,H,N] standardized
    """
    y_hat = out["y_hat"]

    l_data = F.mse_loss(y_hat, y_true)
    l_con = F.mse_loss(out["c_hat"], out["c_star"])
    l_rel = F.mse_loss(out["c_tilde"], out["c_hat"])

    # physics residuals R1..R4 (concept consistency on lookback-only stats)
    aux = out["aux"]
    level_prev = aux["level_prev"]  # [B,N,S]
    vel_prev = aux["vel_prev"]
    vol_prev = aux["vol_prev"]

    l_phys = (
        cfg.lambda_r1 * R1.pow(2).mean()
        + cfg.lambda_r2 * R2.pow(2).mean()
        + cfg.lambda_r3 * R3.pow(2).mean()
        + cfg.lambda_r4 * R4.pow(2).mean()
        + cfg.lambda_r5 * R5.pow(2).mean()
    )

    l_innov = out["innov_l2"]

    # anti-flatline: encourage non-degenerate horizon variation (standardized space)
    pred_std = y_hat.std(dim=1)  # [B,N]
    l_flat = F.relu(cfg.flat_std_floor - pred_std).mean()


# =========================================================
# Dataset windows (train windows only from pre-split region)
# =========================================================
class WindowDataset(Dataset):
    def __init__(self, data_std: np.ndarray, lookback: int, pred_len: int, train_end: int, stride: int = 1):
        self.data = data_std.astype(np.float32)
        self.L = lookback
        self.H = pred_len
        self.train_end = train_end
        self.stride = stride

        max_s = train_end - pred_len
        if max_s <= lookback:
            raise ValueError(
                f"Not enough training data. Need train_end - pred_len > lookback "
                f"but got train_end={train_end}, pred_len={pred_len}, lookback={lookback}."
            )
        self.starts = list(range(self.L, max_s + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        x = self.data[s - self.L : s]
        y = self.data[s : s + self.H]
        return torch.from_numpy(x), torch.from_numpy(y)


# =========================================================
# Plotting (same as v1: 20+ figs)
# =========================================================
def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_single_series_forecast(y_true: np.ndarray, y_pred: np.ndarray, series_name: str, out_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(f"Forecast vs True (standardized) - {series_name}")
    plt.xlabel("Horizon step")
    plt.ylabel("Standardized value")
    plt.legend()
    save_fig(out_path)


def plot_grid_forecasts(y_true_mat: np.ndarray, y_pred_mat: np.ndarray, names: List[str], out_path: str, n_show: int = 16):
    H, N = y_true_mat.shape
    idx = np.linspace(0, N - 1, min(n_show, N)).astype(int)
    n = len(idx)
    r = int(math.ceil(math.sqrt(n)))
    c = int(math.ceil(n / r))
    plt.figure(figsize=(4 * c, 3 * r))
    for k, j in enumerate(idx):
        plt.subplot(r, c, k + 1)
        plt.plot(y_true_mat[:, j], linewidth=1.5, label="T")
        plt.plot(y_pred_mat[:, j], linewidth=1.2, label="P")
        plt.title(names[j] if j < len(names) else f"series{j}")
        plt.xticks([])
    plt.suptitle("Grid of Forecasts (standardized)", y=1.02, fontsize=14)
    save_fig(out_path)


def plot_error_heatmap(err: np.ndarray, out_path: str, max_series: int = 200):
    # err: [H,N] -> show subset if N huge
    H, N = err.shape
    if N > max_series:
        idx = np.linspace(0, N - 1, max_series).astype(int)
        err = err[:, idx]
    plt.figure(figsize=(12, 6))
    plt.imshow(err.T, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title("Error heatmap (Pred - True), axes: horizon x series(subset)")
    plt.xlabel("Horizon step")
    plt.ylabel("Series index")
    save_fig(out_path)


def plot_horizon_mae_mse(y_true_mat: np.ndarray, y_pred_mat: np.ndarray, out_path: str):
    err = y_pred_mat - y_true_mat
    mae_h = np.mean(np.abs(err), axis=1)
    mse_h = np.mean(err ** 2, axis=1)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mae_h, linewidth=2)
    plt.title("Horizon-wise MAE (averaged over series)")
    plt.xlabel("Horizon step")
    plt.ylabel("MAE")

    plt.subplot(2, 1, 2)
    plt.plot(mse_h, linewidth=2)
    plt.title("Horizon-wise MSE (averaged over series)")
    plt.xlabel("Horizon step")
    plt.ylabel("MSE")
    save_fig(out_path)


def plot_series_metric_bars(mse: np.ndarray, mae: np.ndarray, names: List[str], out_path: str, top: int = 40):
    N = len(mse)
    idx = np.argsort(mae)[::-1][: min(top, N)]
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(idx)), mae[idx])
    plt.title("Top series by MAE (standardized)")
    plt.xticks(range(len(idx)), [names[i] if i < len(names) else str(i) for i in idx], rotation=90, fontsize=8)
    plt.ylabel("MAE")

    idx2 = np.argsort(mse)[::-1][: min(top, N)]
    plt.subplot(1, 2, 2)
    plt.bar(range(len(idx2)), mse[idx2])
    plt.title("Top series by MSE (standardized)")
    plt.xticks(range(len(idx2)), [names[i] if i < len(names) else str(i) for i in idx2], rotation=90, fontsize=8)
    plt.ylabel("MSE")
    save_fig(out_path)


def plot_scatter_true_pred(y_true_mat: np.ndarray, y_pred_mat: np.ndarray, out_path: str, sample: int = 5000):
    yt = y_true_mat.reshape(-1)
    yp = y_pred_mat.reshape(-1)
    if yt.shape[0] > sample:
        idx = np.random.choice(yt.shape[0], size=sample, replace=False)
        yt, yp = yt[idx], yp[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.4)
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    plt.plot([mn, mx], [mn, mx], linewidth=2)
    plt.title("True vs Pred (standardized)")
    plt.xlabel("True")
    plt.ylabel("Pred")
    save_fig(out_path)


def plot_error_hist(err: np.ndarray, out_path: str):
    e = err.reshape(-1)
    plt.figure(figsize=(8, 4))
    plt.hist(e, bins=80, alpha=0.9)
    plt.title("Error histogram (Pred-True)")
    plt.xlabel("Error")
    plt.ylabel("Count")
    save_fig(out_path)


def autocorr_1d(x: np.ndarray, max_lag: int = 40):
    x = x - x.mean()
    denom = np.sum(x * x) + 1e-8
    ac = []
    for lag in range(max_lag + 1):
        ac.append(np.sum(x[: len(x) - lag] * x[lag:]) / denom)
    return np.array(ac)


def plot_error_acf(err: np.ndarray, out_path: str, series_idx: int = 0, max_lag: int = 40):
    e = err[:, series_idx]
    ac = autocorr_1d(e, max_lag=max_lag)
    plt.figure(figsize=(8, 4))
    plt.stem(range(len(ac)), ac)
    plt.title(f"Error ACF over horizon (series {series_idx})")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    save_fig(out_path)


def plot_concept_heatmap(c_tilde: np.ndarray, out_path: str, series_idx: int = 0, scales: List[int] = None):
    N, D = c_tilde.shape
    S = len(scales)
    K = 5
    c_sk = c_tilde[series_idx].reshape(S, K)
    plt.figure(figsize=(10, 3))
    plt.imshow(c_sk, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Refined concept state (series {series_idx}) - rows=scales, cols=5 concepts")
    plt.xlabel("Concept k (1..5)")
    plt.ylabel("Scale index s")
    plt.yticks(range(S), [str(s) for s in scales])
    save_fig(out_path)


def plot_attention_avg(attn_maps: List[np.ndarray], out_path: str, max_T: int = 256):
    if len(attn_maps) == 0:
        return
    A = np.stack(attn_maps, axis=0)  # [L, B, T, T] because we average heads in torch
    A = A.mean(axis=(0, 1))          # [T, T]
    T = A.shape[0]
    if T > max_T:
        A = A[-max_T:, -max_T:]
    plt.figure(figsize=(7, 6))
    plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title("Average causal attention (last layers; averaged)")
    plt.xlabel("Key time")
    plt.ylabel("Query time")
    save_fig(out_path)


def plot_training_curves(history: List[Dict[str, float]], out_path: str):
    if len(history) == 0:
        return
    keys = ["loss", "l_data", "l_con", "l_rel", "l_phys", "l_innov", "l_flat", "lam_phys_eff", "grad_norm"]
    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot([h.get(k, np.nan) for h in history], linewidth=2, label=k)
    plt.title("Training curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    save_fig(out_path)


def plot_corr_heatmap(corr: np.ndarray, out_path: str, max_series: int = 120):
    # corr: [N,N]
    N = corr.shape[0]
    if N > max_series:
        idx = np.linspace(0, N - 1, max_series).astype(int)
        corr = corr[np.ix_(idx, idx)]
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Correlation heatmap (subset)")
    plt.xlabel("Series")
    plt.ylabel("Series")
    save_fig(out_path)


def plot_adj_heatmap(W: np.ndarray, out_path: str, max_series: int = 120):
    N = W.shape[0]
    if N > max_series:
        idx = np.linspace(0, N - 1, max_series).astype(int)
        W = W[np.ix_(idx, idx)]
    plt.figure(figsize=(7, 6))
    plt.imshow(W, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title("Graph adjacency W (subset)")
    plt.xlabel("Neighbor j")
    plt.ylabel("Node i")
    save_fig(out_path)




def plot_laplacian_spectrum(W: np.ndarray, out_path: str, max_eigs: int = 200):
    # L = D - W
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    # symmetric-ish: for numerical stability use (L+L^T)/2
    Ls = 0.5 * (L + L.T)
    # compute eigvals (may be heavy for huge N; subsample)
    N = Ls.shape[0]
    if N > 600:
        idx = np.linspace(0, N - 1, 600).astype(int)
        Ls = Ls[np.ix_(idx, idx)]
    eigs = np.linalg.eigvalsh(Ls)
    eigs = np.sort(eigs)
    eigs = eigs[: min(len(eigs), max_eigs)]
    plt.figure(figsize=(7, 4))
    plt.plot(eigs, linewidth=2)
    plt.title("Laplacian spectrum (smallest eigenvalues)")
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    save_fig(out_path)


def plot_concept_boxplots(c_tilde: np.ndarray, scales: List[int], out_path: str, max_series: int = 300):
    # c_tilde: [N,5S]
    N, D = c_tilde.shape
    if N > max_series:
        idx = np.random.choice(N, size=max_series, replace=False)
        c_tilde = c_tilde[idx]
    S = len(scales); K = 5
    c = c_tilde.reshape(-1, S, K)  # [N,S,K]
    labels = []
    data = []
    for si, tau in enumerate(scales):
        for k in range(K):
            data.append(c[:, si, k])
            labels.append(f"s{tau}:c{k+1}")
    plt.figure(figsize=(max(12, len(labels) * 0.35), 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title("Distribution of refined concepts across series (boxplots)")
    plt.xticks(rotation=60, ha="right")
    save_fig(out_path)


def plot_concept_correlation(c_tilde: np.ndarray, out_path: str):
    # corr across concept dimensions
    X = c_tilde - c_tilde.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    std = np.sqrt(np.diag(C) + 1e-8)
    Corr = C / (std[:, None] * std[None, :] + 1e-8)
    plt.figure(figsize=(8, 6))
    plt.imshow(Corr, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Concept-dimension correlation matrix")
    plt.xlabel("Concept dim")
    plt.ylabel("Concept dim")
    save_fig(out_path)


def plot_horizon_fan(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    # show quantiles over series across horizon
    err = y_pred - y_true
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    qv = np.quantile(err, qs, axis=1)  # [len(qs),H]
    H = err.shape[0]
    x = np.arange(H)
    plt.figure(figsize=(10, 4))
    plt.fill_between(x, qv[0], qv[-1], alpha=0.3, label="90% band")
    plt.fill_between(x, qv[1], qv[-2], alpha=0.5, label="50% band")
    plt.plot(x, qv[2], linewidth=2, label="median")
    plt.title("Forecast error fan chart over horizon (quantiles across series)")
    plt.xlabel("Horizon step")
    plt.ylabel("Error (Pred-True)")
    plt.legend()
    save_fig(out_path)


def plot_pred_std_bars(y_pred: np.ndarray, out_path: str, top: int = 60):
    # y_pred: [H,N]
    std = y_pred.std(axis=0)
    idx = np.argsort(std)[::-1][: min(top, len(std))]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(idx)), std[idx])
    plt.title("Top series by prediction variability (std over horizon)")
    plt.xlabel("rank")
    plt.ylabel("std")
    save_fig(out_path)


def plot_error_vs_true_variance(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, sample: int = 2000):
    # per-series variance vs per-series MAE
    err = y_pred - y_true
    var = y_true.var(axis=0)
    mae = np.mean(np.abs(err), axis=0)
    if len(var) > sample:
        idx = np.random.choice(len(var), size=sample, replace=False)
        var = var[idx]; mae = mae[idx]
    plt.figure(figsize=(6, 5))
    plt.scatter(var, mae, s=10, alpha=0.5)
    plt.title("Per-series variance vs per-series MAE (standardized)")
    plt.xlabel("Var(true)")
    plt.ylabel("MAE")
    save_fig(out_path)


def plot_gamma_mu(history: List[Dict[str, float]], out_path: str):
    if len(history) == 0:
        return
    g = [h.get("gamma", np.nan) for h in history]
    m = [h.get("mu", np.nan) for h in history]
    plt.figure(figsize=(10, 4))
    plt.plot(g, linewidth=2, label="gamma")
    plt.plot(m, linewidth=2, label="mu")
    plt.title("Dynamics parameters over training")
    plt.xlabel("Iteration")
    plt.ylabel("value")
    plt.legend()
    save_fig(out_path)






# =========================================================
# Extra rich / complex plotting (v3+): diverse, colorful, paper-style diagnostics
# =========================================================
def _subsample_indices(n: int, k: int, seed: int = 0):
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)

def _safe_call(plot_fn, *args, **kwargs):
    try:
        plot_fn(*args, **kwargs)
    except Exception as e:
        # Never crash training/eval because of plotting
        print(f"[WARN] plot '{plot_fn.__name__}' failed: {e}")

def apply_rich_plot_style():
    # Rich, conference-like look (no seaborn)
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "savefig.bbox": "tight",
    })

def plot_spaghetti_forecasts(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, n_series: int = 60, seed: int = 0):
    # Many series overlayed to show diversity (spaghetti plot)
    H, N = y_true.shape
    idx = _subsample_indices(N, min(n_series, N), seed=seed)
    x = np.arange(H)
    plt.figure(figsize=(11, 5))
    cmap = plt.cm.turbo
    for k, j in enumerate(idx):
        col = cmap(k / max(len(idx)-1, 1))
        plt.plot(x, y_true[:, j], color=col, alpha=0.20)
        plt.plot(x, y_pred[:, j], color=col, alpha=0.55)
    # mean trajectory
    plt.plot(x, y_true[:, idx].mean(axis=1), color="black", alpha=0.85, label="True mean")
    plt.plot(x, y_pred[:, idx].mean(axis=1), color="magenta", alpha=0.85, label="Pred mean")
    plt.title("Spaghetti forecasts (subset; standardized)")
    plt.xlabel("Horizon step")
    plt.ylabel("value")
    plt.legend(loc="best")
    save_fig(out_path)

def plot_lookback_plus_forecast_panel(x_lookback: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                      series_names: List[str], out_path: str, n_show: int = 9, seed: int = 0):
    # Multi-panel: last lookback followed by forecast; shows continuity.
    # x_lookback: [L,N], y_true/y_pred: [H,N]
    L, N = x_lookback.shape
    H = y_true.shape[0]
    idx = _subsample_indices(N, min(n_show, N), seed=seed)
    r = int(math.ceil(math.sqrt(len(idx))))
    c = int(math.ceil(len(idx) / r))
    plt.figure(figsize=(4.2 * c, 3.2 * r))
    for k, j in enumerate(idx):
        plt.subplot(r, c, k + 1)
        t0 = np.arange(L)
        t1 = np.arange(L, L + H)
        plt.plot(t0, x_lookback[:, j], color="gray", alpha=0.85, label="History" if k == 0 else None)
        plt.plot(t1, y_true[:, j], color="black", alpha=0.9, label="True" if k == 0 else None)
        plt.plot(t1, y_pred[:, j], color="crimson", alpha=0.9, label="Pred" if k == 0 else None)
        plt.axvline(L-1, color="blue", alpha=0.35, linestyle="--")
        nm = series_names[j] if j < len(series_names) else f"series{j}"
        plt.title(nm, fontsize=10)
        plt.xticks([])
    plt.suptitle("History + Forecast continuity panels (standardized)", y=1.02)
    plt.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02))
    save_fig(out_path)

def plot_fft_spectrum_compare(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, n_series: int = 120, seed: int = 0):
    # Compare frequency spectrum of true vs pred (across series subset)
    H, N = y_true.shape
    idx = _subsample_indices(N, min(n_series, N), seed=seed)
    yt = y_true[:, idx] - y_true[:, idx].mean(axis=0, keepdims=True)
    yp = y_pred[:, idx] - y_pred[:, idx].mean(axis=0, keepdims=True)
    ft = np.abs(np.fft.rfft(yt, axis=0))
    fp = np.abs(np.fft.rfft(yp, axis=0))
    mean_t = ft.mean(axis=1)
    mean_p = fp.mean(axis=1)
    q10_t, q90_t = np.quantile(ft, [0.1, 0.9], axis=1)
    q10_p, q90_p = np.quantile(fp, [0.1, 0.9], axis=1)
    f = np.arange(len(mean_t))
    plt.figure(figsize=(10.5, 4.8))
    plt.fill_between(f, q10_t, q90_t, alpha=0.25, label="True 10–90% band")
    plt.plot(f, mean_t, color="black", label="True mean")
    plt.fill_between(f, q10_p, q90_p, alpha=0.25, label="Pred 10–90% band")
    plt.plot(f, mean_p, color="crimson", label="Pred mean")
    plt.title("Frequency-domain amplitude comparison (subset; standardized)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Amplitude")
    plt.legend(loc="best", ncol=2)
    save_fig(out_path)

def plot_residual_corr_heatmap(err: np.ndarray, out_path: str, max_series: int = 120, seed: int = 0):
    # Residual correlation across series (complex dependency diagnostic)
    H, N = err.shape
    idx = _subsample_indices(N, min(max_series, N), seed=seed)
    E = err[:, idx]
    E = E - E.mean(axis=0, keepdims=True)
    C = (E.T @ E) / max(H - 1, 1)
    s = np.sqrt(np.diag(C) + 1e-8)
    Corr = C / (s[:, None] * s[None, :] + 1e-8)
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(Corr, vmin=-1, vmax=1, interpolation="nearest", aspect="auto", cmap="coolwarm")
    plt.colorbar()
    plt.title("Residual correlation matrix (subset)")
    plt.xlabel("Series")
    plt.ylabel("Series")
    save_fig(out_path)

def plot_violin_metrics(mse: np.ndarray, mae: np.ndarray, out_path: str):
    # Distribution view (non-Gaussian tails are common)
    plt.figure(figsize=(9, 4))
    plt.violinplot([mae, mse], showmeans=True, showextrema=True, widths=0.8)
    plt.xticks([1, 2], ["MAE", "MSE"])
    plt.title("Metric distribution across series (standardized)")
    plt.ylabel("value")
    save_fig(out_path)

def plot_ecdf_abs_error(err: np.ndarray, out_path: str):
    e = np.abs(err.reshape(-1))
    e = np.sort(e)
    y = np.linspace(0, 1, len(e), endpoint=True)
    plt.figure(figsize=(7, 5))
    plt.plot(e, y, color="purple")
    plt.title("ECDF of absolute error |Pred-True|")
    plt.xlabel("Absolute error")
    plt.ylabel("ECDF")
    save_fig(out_path)

def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, sample: int = 6000, seed: int = 0):
    # Complex scatter diagnostic: bias & heteroscedasticity
    rng = np.random.default_rng(seed)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    n = len(yt)
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        yt = yt[idx]; yp = yp[idx]
    m = 0.5 * (yt + yp)
    d = yp - yt
    mu = d.mean()
    sd = d.std() + 1e-8
    plt.figure(figsize=(7, 5))
    plt.scatter(m, d, s=8, alpha=0.35, c=m, cmap="viridis")
    plt.axhline(mu, color="black", linestyle="--", label="mean diff")
    plt.axhline(mu + 1.96 * sd, color="crimson", linestyle="--", label="±1.96σ")
    plt.axhline(mu - 1.96 * sd, color="crimson", linestyle="--")
    plt.title("Bland–Altman: difference vs mean (standardized)")
    plt.xlabel("Mean of (True, Pred)")
    plt.ylabel("Pred - True")
    plt.legend(loc="best")
    save_fig(out_path)

def plot_ranked_abs_error_heatmap(err: np.ndarray, mae: np.ndarray, out_path: str, max_series: int = 240):
    # Sort series by MAE; show abs error heatmap with structure
    H, N = err.shape
    order = np.argsort(mae)[::-1]
    if N > max_series:
        order = order[:max_series]
    A = np.abs(err[:, order]).T  # [Nsub,H]
    plt.figure(figsize=(12, 6))
    plt.imshow(A, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar()
    plt.title("Ranked absolute error heatmap (series sorted by MAE; subset)")
    plt.xlabel("Horizon step")
    plt.ylabel("Series (ranked)")
    save_fig(out_path)

def _pca2(X: np.ndarray):
    # Lightweight PCA (no sklearn), returns 2D embedding
    X = X - X.mean(axis=0, keepdims=True)
    # covariance in feature space
    C = (X.T @ X) / max(X.shape[0]-1, 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    V = V[:, idx[:2]]
    Z = X @ V
    return Z

def plot_pca_concept_embedding(c_tilde: np.ndarray, mae: np.ndarray, y_pred: np.ndarray, out_path: str, max_series: int = 1500, seed: int = 0):
    # Complex embedding: concepts -> 2D; color by MAE; size by pred variability
    N, D = c_tilde.shape
    idx = _subsample_indices(N, min(max_series, N), seed=seed)
    Z = _pca2(c_tilde[idx])
    stdp = y_pred.std(axis=0)[idx]
    s = 10 + 80 * (stdp - stdp.min()) / (stdp.ptp() + 1e-8)
    plt.figure(figsize=(7.2, 6.2))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=mae[idx], s=s, cmap="turbo", alpha=0.75, edgecolors="none")
    plt.colorbar(sc, label="MAE (std)")
    plt.title("PCA embedding of refined concepts (color=MAE, size=pred std)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(out_path)

def plot_spectral_graph_embedding(W: np.ndarray, mae: np.ndarray, out_path: str, max_series: int = 500, seed: int = 0):
    # 2D spectral embedding from Laplacian eigenvectors (subset)
    N = W.shape[0]
    idx = _subsample_indices(N, min(max_series, N), seed=seed)
    Ws = W[np.ix_(idx, idx)]
    deg = Ws.sum(axis=1) + 1e-8
    Dm12 = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(Ws.shape[0]) - Dm12 @ Ws @ Dm12  # normalized Laplacian
    Ls = 0.5 * (L + L.T)
    # smallest non-trivial eigenvectors (skip the first)
    w, V = np.linalg.eigh(Ls)
    order = np.argsort(w)
    V = V[:, order]
    if V.shape[1] < 3:
        return
    Z = V[:, 1:3]
    plt.figure(figsize=(7.2, 6.2))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=mae[idx], s=18, cmap="coolwarm", alpha=0.8)
    plt.colorbar(sc, label="MAE (std)")
    plt.title("Spectral embedding of graph (subset; color=MAE)")
    plt.xlabel("eigvec 2")
    plt.ylabel("eigvec 3")
    save_fig(out_path)

def plot_laplacian_eigenvectors_heatmap(W: np.ndarray, out_path: str, k: int = 16, max_series: int = 350, seed: int = 0):
    # Heatmap of first eigenvectors; shows clustered communities & smoothness
    N = W.shape[0]
    idx = _subsample_indices(N, min(max_series, N), seed=seed)
    Ws = W[np.ix_(idx, idx)]
    deg = Ws.sum(axis=1) + 1e-8
    L = np.diag(deg) - Ws
    Ls = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(Ls)
    order = np.argsort(w)
    V = V[:, order[:min(k, V.shape[1])]]  # [n,k]
    plt.figure(figsize=(10.5, 5.2))
    plt.imshow(V.T, aspect="auto", interpolation="nearest", cmap="Spectral")
    plt.colorbar()
    plt.title("Laplacian eigenvectors (subset; rows=eigenvector index)")
    plt.xlabel("Node (subset index)")
    plt.ylabel("Eigenvector")
    save_fig(out_path)

def plot_attention_entropy(attn_maps: List[np.ndarray], out_path: str, max_T: int = 256):
    if len(attn_maps) == 0:
        return
    A = np.stack(attn_maps, axis=0)  # [L,B,T,T]
    A = A.mean(axis=(0, 1))          # [T,T]
    T = A.shape[0]
    if T > max_T:
        A = A[-max_T:, -max_T:]
    P = np.maximum(A, 1e-12)
    P = P / P.sum(axis=-1, keepdims=True)
    H = -(P * np.log(P)).sum(axis=-1)  # [T]
    plt.figure(figsize=(9, 4))
    plt.plot(H, color="teal")
    plt.title("Causal attention entropy over query time (avg)")
    plt.xlabel("Query time (truncated)")
    plt.ylabel("Entropy")
    save_fig(out_path)

def plot_loss_stack(history: List[Dict[str, float]], out_path: str):
    if len(history) == 0:
        return
    keys = ["l_data", "l_con", "l_rel", "l_phys", "l_innov", "l_flat"]
    X = np.arange(len(history))
    Ys = [np.array([h.get(k, 0.0) for h in history], dtype=float) for k in keys]
    plt.figure(figsize=(11, 5))
    plt.stackplot(X, Ys, labels=keys, alpha=0.85)
    plt.title("Stacked loss components over training")
    plt.xlabel("Iteration")
    plt.ylabel("loss component")
    plt.legend(loc="upper right", ncol=3)
    save_fig(out_path)

def plot_gamma_mu_phase(history: List[Dict[str, float]], out_path: str):
    if len(history) == 0:
        return
    g = np.array([h.get("gamma", np.nan) for h in history], dtype=float)
    m = np.array([h.get("mu", np.nan) for h in history], dtype=float)
    t = np.arange(len(g))
    plt.figure(figsize=(6.2, 5.6))
    sc = plt.scatter(g, m, c=t, cmap="plasma", s=14, alpha=0.75)
    plt.colorbar(sc, label="iteration")
    plt.title("Dynamics phase portrait: (gamma, mu)")
    plt.xlabel("gamma")
    plt.ylabel("mu")
    save_fig(out_path)

def plot_pred_true_density(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, bins: int = 80):
    # 2D density via histogram (true vs pred)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    H2, xedges, yedges = np.histogram2d(yt, yp, bins=bins)
    plt.figure(figsize=(6.5, 6.2))
    plt.imshow(H2.T, origin="lower", aspect="auto",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap="inferno")
    plt.colorbar(label="count")
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    plt.plot([mn, mx], [mn, mx], color="cyan", linewidth=2, alpha=0.9)
    plt.title("True vs Pred density (2D histogram; standardized)")
    plt.xlabel("True")
    plt.ylabel("Pred")
    save_fig(out_path)

def plot_calibration_binned(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, n_bins: int = 12):
    # Calibration-like diagnostic: bin by pred; compare mean true vs mean pred
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    order = np.argsort(yp)
    yt = yt[order]; yp = yp[order]
    bins = np.array_split(np.arange(len(yp)), n_bins)
    mp = np.array([yp[b].mean() for b in bins])
    mt = np.array([yt[b].mean() for b in bins])
    plt.figure(figsize=(6.8, 5.8))
    plt.plot(mp, mt, marker="o", color="navy", label="binned means")
    mn = min(mp.min(), mt.min())
    mx = max(mp.max(), mt.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray", label="ideal")
    plt.title("Binned calibration (standardized)")
    plt.xlabel("Mean predicted (bin)")
    plt.ylabel("Mean true (bin)")
    plt.legend(loc="best")
    save_fig(out_path)

def plot_concept_radar_grid(c_tilde: np.ndarray, scales: List[int], series_names: List[str], out_path: str, n_show: int = 6, seed: int = 0):
    # Radar charts for multi-scale concept profile (complex, high-style)
    N, D = c_tilde.shape
    S = len(scales); K = 5
    idx = _subsample_indices(N, min(n_show, N), seed=seed)
    labels = [f"L@{s}" for s in scales] + [f"V@{s}" for s in scales] + [f"P@{s}" for s in scales] + [f"A@{s}" for s in scales] + [f"Vol@{s}" for s in scales]
    # normalize dims for radar comparability
    X = c_tilde - c_tilde.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-8)

    M = X.shape[1]
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False).tolist()
    angles += angles[:1]

    r = int(math.ceil(math.sqrt(len(idx))))
    c = int(math.ceil(len(idx) / r))
    plt.figure(figsize=(4.4 * c, 4.0 * r))
    for k, j in enumerate(idx):
        ax = plt.subplot(r, c, k + 1, polar=True)
        vals = X[j].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=plt.cm.turbo(k / max(len(idx)-1, 1)), linewidth=2)
        ax.fill(angles, vals, alpha=0.18)
        nm = series_names[j] if j < len(series_names) else f"series{j}"
        ax.set_title(nm, fontsize=10, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Radar: multi-scale concept fingerprints (standardized)", y=1.02)
    save_fig(out_path)

def plot_parallel_coordinates_concepts(c_tilde: np.ndarray, mae: np.ndarray, out_path: str, max_series: int = 200, seed: int = 0):
    # Parallel coordinates without pandas.plotting (matplotlib only)
    N, D = c_tilde.shape
    idx = _subsample_indices(N, min(max_series, N), seed=seed)
    X = c_tilde[idx]
    # normalize per-dim
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    x = np.arange(D)
    plt.figure(figsize=(12, 5))
    # color by MAE
    cm = plt.cm.viridis
    m = mae[idx]
    m = (m - m.min()) / (m.ptp() + 1e-8)
    for i in range(len(idx)):
        plt.plot(x, X[i], color=cm(m[i]), alpha=0.35)
    plt.title("Parallel coordinates: refined concepts (subset; color=MAE)")
    plt.xlabel("Concept dimension")
    plt.ylabel("Standardized value")
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("relative MAE")
    save_fig(out_path)

def plot_dashboard_3x3(y_true: np.ndarray, y_pred: np.ndarray, mse: np.ndarray, mae: np.ndarray,
                      W: Optional[np.ndarray], corr: Optional[np.ndarray], history: List[Dict[str, float]],
                      out_path: str, seed: int = 0):
    # A single, very complex "paper dashboard" figure using GridSpec
    from matplotlib.gridspec import GridSpec

    H, N = y_true.shape
    err = y_pred - y_true
    mae_h = np.mean(np.abs(err), axis=1)
    mse_h = np.mean(err**2, axis=1)

    plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3)

    # (0,0): horizon MAE
    ax = plt.subplot(gs[0, 0])
    ax.plot(mae_h, color="darkorange")
    ax.set_title("Horizon MAE")
    ax.set_xlabel("h")
    ax.set_ylabel("MAE")

    # (0,1): horizon MSE
    ax = plt.subplot(gs[0, 1])
    ax.plot(mse_h, color="slateblue")
    ax.set_title("Horizon MSE")
    ax.set_xlabel("h")
    ax.set_ylabel("MSE")

    # (0,2): ECDF abs error
    ax = plt.subplot(gs[0, 2])
    e = np.sort(np.abs(err.reshape(-1)))
    y = np.linspace(0, 1, len(e))
    ax.plot(e, y, color="purple")
    ax.set_title("ECDF |error|")
    ax.set_xlabel("|error|")
    ax.set_ylabel("ECDF")

    # (1,0): metric violin (approx)
    ax = plt.subplot(gs[1, 0])
    parts = ax.violinplot([mae, mse], showmeans=True, showextrema=True, widths=0.9)
    for pc in parts.get('bodies', []):
        pc.set_facecolor('cyan')
        pc.set_alpha(0.35)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MAE", "MSE"])
    ax.set_title("Metrics distribution")

    # (1,1): correlation heatmap
    ax = plt.subplot(gs[1, 1])
    if corr is not None:
        Ns = min(corr.shape[0], 80)
        idx = _subsample_indices(corr.shape[0], Ns, seed=seed)
        C = corr[np.ix_(idx, idx)]
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest", aspect="auto")
        ax.set_title("Corr (subset)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "corr=None", ha="center", va="center")
        ax.set_axis_off()

    # (1,2): graph adjacency heatmap
    ax = plt.subplot(gs[1, 2])
    if W is not None:
        Ns = min(W.shape[0], 80)
        idx = _subsample_indices(W.shape[0], Ns, seed=seed+1)
        A = W[np.ix_(idx, idx)]
        im = ax.imshow(A, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title("Adjacency W (subset)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "W=None", ha="center", va="center")
        ax.set_axis_off()

    # (2,0): error histogram
    ax = plt.subplot(gs[2, 0])
    ax.hist(err.reshape(-1), bins=80, alpha=0.9, color="gray")
    ax.set_title("Error histogram")
    ax.set_xlabel("Pred-True")

    # (2,1): gamma/mu curves
    ax = plt.subplot(gs[2, 1])
    if len(history) > 0:
        g = [h.get("gamma", np.nan) for h in history]
        m = [h.get("mu", np.nan) for h in history]
        ax.plot(g, label="gamma", color="teal")
        ax.plot(m, label="mu", color="crimson")
        ax.set_title("Dynamics params")
        ax.set_xlabel("iter")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "history empty", ha="center", va="center")
        ax.set_axis_off()

    # (2,2): true vs pred density
    ax = plt.subplot(gs[2, 2])
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    H2, xedges, yedges = np.histogram2d(yt, yp, bins=60)
    im = ax.imshow(H2.T, origin="lower", aspect="auto",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap="inferno")
    mn = min(yt.min(), yp.min()); mx = max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], color="cyan", linewidth=2, alpha=0.9)
    ax.set_title("True vs Pred density")
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("CONCORD diagnostic dashboard (standardized)", y=1.02)
    save_fig(out_path)



# =========================================================
# KNN-only Rich Visualizations (Concept-space kNN diagnostics)
# =========================================================
def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances for X [n,d] -> [n,n] (float32-friendly)."""
    X = X.astype(np.float32, copy=False)
    G = X @ X.T
    s = np.sum(X * X, axis=1, keepdims=True)
    D2 = s + s.T - 2.0 * G
    return np.maximum(D2, 0.0)


def _knn_indices_from_d2(D2: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return knn indices and distances from a full distance matrix D2 [n,n]."""
    n = D2.shape[0]
    # exclude self
    D2 = D2.copy()
    np.fill_diagonal(D2, np.inf)
    k = int(max(1, min(k, n - 1)))
    idx = np.argpartition(D2, kth=k, axis=1)[:, :k]  # [n,k], unordered
    d2 = np.take_along_axis(D2, idx, axis=1)
    # sort each row for prettier plots
    order = np.argsort(d2, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    d2 = np.take_along_axis(d2, order, axis=1)
    return idx, np.sqrt(d2 + 1e-12)


def _subsample_Xy(X: np.ndarray, y: np.ndarray, max_points: int, seed: int = 0):
    n = X.shape[0]
    if max_points and n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=max_points, replace=False)
        sel = np.sort(sel)
        return X[sel], y[sel], sel
    return X, y, np.arange(X.shape[0])


def _knn_regress_1d(x: np.ndarray, y: np.ndarray, k: int = 25, grid_n: int = 120,
                    weight: str = "inv", trim_q: float = 0.02):
    """
    1D kNN smoothing curve: estimate E[y | x≈t] via kNN in x-space.
    Returns grid, mean, std.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    lo = np.quantile(x, trim_q)
    hi = np.quantile(x, 1.0 - trim_q)
    grid = np.linspace(lo, hi, int(grid_n))
    mu = np.zeros_like(grid)
    sd = np.zeros_like(grid)
    for i, t in enumerate(grid):
        d = np.abs(x - t)
        kk = min(int(k), len(x))
        nn = np.argpartition(d, kth=kk - 1)[:kk]
        dn = d[nn]
        yn = y[nn]
        if weight == "gauss":
            # bandwidth tied to local scale
            bw = np.maximum(np.median(dn) * 1.2, 1e-6)
            w = np.exp(-(dn ** 2) / (2.0 * bw ** 2))
        elif weight == "inv":
            w = 1.0 / (dn + 1e-6)
        else:
            w = np.ones_like(dn)
        w = w / (w.sum() + 1e-12)
        m = (w * yn).sum()
        v = (w * (yn - m) ** 2).sum()
        mu[i] = m
        sd[i] = np.sqrt(np.maximum(v, 0.0))
    return grid, mu, sd


def plot_knn_concept_effect_curves(c_tilde: np.ndarray, mae: np.ndarray, scales: List[int],
                                  out_path: str, k: int = 35, grid_n: int = 140,
                                  weight: str = "inv", seed: int = 0):
    """
    Multi-panel kNN function curves: for each (scale s, concept k), plot kNN-smoothed MAE as a function of concept value.
    c_tilde: [N, 5S], mae: [N]
    """
    N, D = c_tilde.shape
    S = len(scales); K = 5
    assert D == S * K, f"Expected c_tilde dim {S*K}, got {D}"
    X = c_tilde.reshape(N, S, K)

    plt.figure(figsize=(4.2 * K, 2.9 * S))
    cm = plt.cm.turbo
    for si in range(S):
        for ki in range(K):
            ax = plt.subplot(S, K, si * K + ki + 1)
            x = X[:, si, ki]
            g, m, s = _knn_regress_1d(x, mae, k=k, grid_n=grid_n, weight=weight)
            # color gradient along curve
            for ii in range(len(g) - 1):
                ax.plot(g[ii:ii+2], m[ii:ii+2], color=cm(ii / max(len(g) - 2, 1)), linewidth=2.0, alpha=0.95)
            ax.fill_between(g, m - 1.0 * s, m + 1.0 * s, alpha=0.18, color="gray")
            ax.set_title(f"Scale {scales[si]} | Concept {ki+1}", fontsize=10)
            if si == S - 1:
                ax.set_xlabel("Concept value (std)")
            if ki == 0:
                ax.set_ylabel("kNN E[MAE]")
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    plt.suptitle("kNN concept-effect curves on per-series MAE (standardized)", y=1.02, fontsize=14)
    save_fig(out_path)


def plot_knn_graph_on_pca(c_tilde: np.ndarray, mae: np.ndarray, out_path: str,
                          k: int = 12, max_points: int = 700, seed: int = 0):
    """
    kNN graph visualized on 2D PCA embedding (nodes colored by MAE).
    """
    X, y, sel = _subsample_Xy(c_tilde, mae, max_points=max_points, seed=seed)
    Z = _pca2(X)
    D2 = _pairwise_sq_dists((X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6))
    nn, dist = _knn_indices_from_d2(D2, k=k)

    plt.figure(figsize=(10.5, 8.0))
    # edges
    for i in range(Z.shape[0]):
        zi = Z[i]
        for j in nn[i]:
            zj = Z[j]
            plt.plot([zi[0], zj[0]], [zi[1], zj[1]], color="lightgray", alpha=0.25, linewidth=0.7)

    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=36, alpha=0.95, edgecolors="black", linewidths=0.25)
    plt.colorbar(sc, label="Per-series MAE (std)")
    plt.title(f"kNN concept-space graph on PCA embedding (k={k}, n={Z.shape[0]})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2, linestyle="--")
    save_fig(out_path)


def plot_knn_adjacency_heatmap(c_tilde: np.ndarray, mae: np.ndarray, out_path: str,
                               k: int = 12, max_points: int = 520, seed: int = 0):
    """
    Dense-looking but sparse kNN adjacency (after reordering by MAE) for visual complexity.
    """
    X, y, sel = _subsample_Xy(c_tilde, mae, max_points=max_points, seed=seed)
    n = X.shape[0]
    Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    D2 = _pairwise_sq_dists(Xn)
    nn, dist = _knn_indices_from_d2(D2, k=k)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, nn[i]] = 1.0 / (dist[i] + 1e-6)

    # reorder by MAE to create structure
    order = np.argsort(y)
    A = A[order][:, order]

    plt.figure(figsize=(8.5, 7.0))
    plt.imshow(A, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(label="Edge weight (1/d)")
    plt.title(f"kNN adjacency heatmap (reordered by MAE; k={k}, n={n})")
    plt.xlabel("Series (sorted by MAE)")
    plt.ylabel("Series (sorted by MAE)")
    save_fig(out_path)


def plot_knn_neighbor_distance_distribution(c_tilde: np.ndarray, out_path: str,
                                            k: int = 12, max_points: int = 1800, seed: int = 0):
    """
    Distribution of kNN distances in concept space (subsampled for large N).
    """
    X, _, _ = _subsample_Xy(c_tilde, np.zeros((c_tilde.shape[0],), dtype=np.float32), max_points=max_points, seed=seed)
    Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    D2 = _pairwise_sq_dists(Xn)
    nn, dist = _knn_indices_from_d2(D2, k=k)
    d = dist.reshape(-1)

    plt.figure(figsize=(10.5, 4.6))
    plt.subplot(1, 2, 1)
    plt.hist(d, bins=90, alpha=0.9, color="steelblue")
    plt.title("kNN distance histogram")
    plt.xlabel("Distance (concept space, z-scored)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    d_sorted = np.sort(d)
    ecdf = np.linspace(0, 1, len(d_sorted))
    plt.plot(d_sorted, ecdf, color="darkmagenta", linewidth=2.2)
    plt.title("kNN distance ECDF")
    plt.xlabel("Distance")
    plt.ylabel("ECDF")
    plt.grid(alpha=0.25, linestyle="--")
    save_fig(out_path)


def _knn_loo_predict(X: np.ndarray, y: np.ndarray, k: int = 25) -> np.ndarray:
    """
    Leave-one-out kNN regression predictions for y given X.
    O(n^2) but okay for typical benchmark N (<=~1000), with subsampling upstream when needed.
    """
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    n = X.shape[0]
    Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    D2 = _pairwise_sq_dists(Xn)
    np.fill_diagonal(D2, np.inf)
    kk = min(int(k), n - 1)
    idx = np.argpartition(D2, kth=kk, axis=1)[:, :kk]
    d2 = np.take_along_axis(D2, idx, axis=1)
    w = 1.0 / (np.sqrt(d2 + 1e-12) + 1e-6)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    yhat = (w * y[idx]).sum(axis=1)
    return yhat


def plot_knn_per_scale_permutation_importance(c_tilde: np.ndarray, mae: np.ndarray, scales: List[int],
                                              out_path: str, k: int = 30, max_points: int = 900,
                                              seed: int = 0):
    """
    KNN-only 'importance' via permutation: within each scale (5 concepts),
    permute one concept dimension and measure degradation of kNN LOO fit to MAE.
    Output: heatmap [S,5] of ΔMSE.
    """
    X, y, _ = _subsample_Xy(c_tilde, mae, max_points=max_points, seed=seed)
    n = X.shape[0]
    S = len(scales); K = 5
    X = X.reshape(n, S, K)

    base_pred = _knn_loo_predict(X.reshape(n, S*K), y, k=k)
    base_mse = float(np.mean((base_pred - y) ** 2))

    rng = np.random.default_rng(seed)
    imp = np.zeros((S, K), dtype=np.float32)

    for si in range(S):
        for ki in range(K):
            Xp = X.copy()
            perm = rng.permutation(n)
            Xp[:, si, ki] = Xp[perm, si, ki]
            pred = _knn_loo_predict(Xp.reshape(n, S*K), y, k=k)
            mse = float(np.mean((pred - y) ** 2))
            imp[si, ki] = max(mse - base_mse, 0.0)

    plt.figure(figsize=(9.6, 3.6))
    plt.imshow(imp, aspect="auto", interpolation="nearest", cmap="plasma")
    plt.colorbar(label="ΔMSE after permuting concept")
    plt.xticks(range(K), [f"C{j+1}" for j in range(K)])
    plt.yticks(range(S), [str(s) for s in scales])
    plt.xlabel("Concept (within scale)")
    plt.ylabel("Scale")
    plt.title(f"kNN permutation importance wrt MAE (k={k}, n={n})")
    save_fig(out_path)


def plot_knn_error_surface_top2(c_tilde: np.ndarray, mae: np.ndarray, scales: List[int],
                                out_path: str, k: int = 35, max_points: int = 900,
                                grid_n: int = 70, seed: int = 0):
    """
    Pick two most 'important' concept dims (global, via simple corr with MAE) and plot
    a 2D kNN-regressed MAE surface on their plane (others marginalized implicitly by kNN).
    """
    X, y, _ = _subsample_Xy(c_tilde, mae, max_points=max_points, seed=seed)
    n, D = X.shape
    # choose two dims by absolute correlation with MAE
    Xc = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    yc = (y - y.mean()) / (y.std() + 1e-6)
    corr = np.abs((Xc.T @ yc) / max(n - 1, 1))
    d1, d2 = int(np.argmax(corr)), int(np.argsort(corr)[-2])

    x1 = Xc[:, d1]; x2 = Xc[:, d2]
    g1 = np.linspace(np.quantile(x1, 0.03), np.quantile(x1, 0.97), int(grid_n))
    g2 = np.linspace(np.quantile(x2, 0.03), np.quantile(x2, 0.97), int(grid_n))
    G1, G2 = np.meshgrid(g1, g2)
    Z = np.zeros_like(G1, dtype=np.float64)

    # kNN in (d1,d2) plane
    P = np.stack([x1, x2], axis=1)
    D2 = _pairwise_sq_dists(P)
    # for each grid point, compute distances to points and knn average
    for i in range(G1.shape[0]):
        for j in range(G1.shape[1]):
            t = np.array([G1[i, j], G2[i, j]], dtype=np.float64)
            d = np.sqrt(np.sum((P - t[None, :]) ** 2, axis=1) + 1e-12)
            kk = min(int(k), len(d))
            nn = np.argpartition(d, kth=kk - 1)[:kk]
            dn = d[nn]
            w = 1.0 / (dn + 1e-6)
            w = w / (w.sum() + 1e-12)
            Z[i, j] = (w * y[nn]).sum()



# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    pred_len: int = 96
    lookback: int = 336
    scales: List[int] = None
    graph_window: int = 96

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1
    kan_knots: int = 12
    topk: int = 5
    kappa: float = 5.0
    dt: float = 0.5
    step_emb_dim: int = 16

    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    stride: int = 1

    lambda_con: float = 0.1
    lambda_rel: float = 0.01
    lambda_phys: float = 0.1
    phys_warmup_steps: int = 500  # warm-up over first N iterations
    lambda_innov: float = 1e-3

    # anti-flatline regularizer (optional)
    lambda_flat: float = 0.001
    flat_std_floor: float = 0.02

    lambda_r1: float = 1.0
    lambda_r2: float = 1.0
    lambda_r3: float = 1.0
    lambda_r4: float = 1.0
    lambda_r5: float = 1.0

    seed: int = 42
    device: str = "cuda"
    make_plots: int = 1
    save_attn: int = 0
    attn_layers: int = 1
    series_limit: int = 0

    @property
    def num_scales(self):
        return len(self.scales)


# =========================================================
# Training / Evaluation
# =========================================================
def train_and_test_one_dataset(dataset: str, csv_path: str, out_root: str, cfg: Config):
    print(f"\n========== Dataset: {dataset} ==========")
    print(f"CSV: {csv_path}")

    data, col_names = load_csv_timeseries(csv_path, series_limit=cfg.series_limit)
    T, N = data.shape
    print(f"Loaded data shape: T={T}, N={N}")

    H = cfg.pred_len
    if T <= cfg.lookback + H + 5:
        raise ValueError(
            f"Dataset too short for lookback={cfg.lookback} and pred_len={H}. "
            f"Need T > lookback + pred_len + 5, but got T={T}."
        )
    split = T - H
    train_data = data[:split]

    mean, std = fit_standardizer(train_data)
    data_std = apply_standardizer(data, mean, std).astype(np.float32)

    ds = WindowDataset(data_std, lookback=cfg.lookback, pred_len=cfg.pred_len, train_end=split, stride=cfg.stride)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    model = CONCORDModel(
        n_series=N,
        pred_len=cfg.pred_len,
        lookback=cfg.lookback,
        scales=cfg.scales,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        kan_knots=cfg.kan_knots,
        graph_window=cfg.graph_window,
        topk=min(cfg.topk, max(N - 1, 1)),
        kappa=cfg.kappa,
        dt=cfg.dt,
        step_emb_dim=cfg.step_emb_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    global_step = 0
    model.train()
    for ep in range(cfg.epochs):
        for it, (x_b, y_b) in enumerate(dl):
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            out = model(x_b, return_attn=False)
            loss, logs = compute_total_loss(out, x_b, y_b, cfg, global_step)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if cfg.grad_clip and cfg.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                try:
                    grad_norm = float(gn.detach().cpu())
                except Exception:
                    grad_norm = float(gn)
            logs['grad_norm'] = grad_norm if grad_norm is not None else float('nan')
            opt.step()

            history.append(logs)
            global_step += 1

            if (it + 1) % 50 == 0:
                print(f"Epoch {ep+1}/{cfg.epochs} iter {it+1}/{len(dl)} | "
                      f"loss={logs['loss']:.4f} data={logs['l_data']:.4f} phys={logs['l_phys']:.4f} "
                      f"innov={logs['l_innov']:.6f} lam_phys={logs['lam_phys_eff']:.4f} "
                      f"gamma={logs['gamma']:.4f} mu={logs['mu']:.4f}")

        print(f"[Epoch {ep+1}] last loss: {history[-1]['loss']:.4f}")

    # test
    model.eval()
    x_test = torch.from_numpy(data_std[split - cfg.lookback: split]).unsqueeze(0).to(device)
    y_test = torch.from_numpy(data_std[split: split + cfg.pred_len]).unsqueeze(0).to(device)

    with torch.no_grad():
        out_t = model(x_test, return_attn=bool(cfg.save_attn), attn_layers=cfg.attn_layers)
        y_hat = out_t["y_hat"].squeeze(0).detach().cpu()
        y_true = y_test.squeeze(0).detach().cpu()

    mse_s, mae_s = series_metrics(y_hat, y_true)
    mse_avg = float(mse_s.mean())
    mae_avg = float(mae_s.mean())

    # flatline diagnostics
    pred_std = y_hat.std(dim=0)
    pred_std_avg = float(pred_std.mean())
    pred_std_min = float(pred_std.min())

    print(f"[TEST] standardized MSE(avg)={mse_avg:.6f}, MAE(avg)={mae_avg:.6f}")
    print(f"[DIAG] pred std over horizon: avg={pred_std_avg:.6f}, min={pred_std_min:.6f}  (should NOT be ~0)")

    if pred_std_avg < 1e-3:
        print("[WARN] Predictions look almost flat. Try: smaller gamma/mu init, larger dt, higher lr, lower lambda_phys, or increase step_emb_dim.")
    elif pred_std_min < 1e-6:
        print("[WARN] At least one series is nearly flat; consider increasing model capacity or decreasing graph topk/kappa for stability.")

    # save outputs
    run_dir = os.path.join(out_root, dataset, now_str())
    ensure_dir(run_dir)

    np.save(os.path.join(run_dir, "y_true_std.npy"), y_true.numpy())
    np.save(os.path.join(run_dir, "y_pred_std.npy"), y_hat.numpy())
    np.save(os.path.join(run_dir, "mse_per_series.npy"), mse_s.numpy())
    np.save(os.path.join(run_dir, "mae_per_series.npy"), mae_s.numpy())

    out_csv = os.path.join(run_dir, "pred_vs_true_std.csv")
    df_out = pd.DataFrame()
    for j in range(y_true.shape[1]):
        nm = col_names[j] if j < len(col_names) else f"series{j}"
        df_out[f"{nm}_true"] = y_true[:, j].numpy()
        df_out[f"{nm}_pred"] = y_hat[:, j].numpy()
        if (j + 1) >= 25 and y_true.shape[1] > 25:
            break
    df_out.to_csv(out_csv, index=False)

    metrics = {
        "dataset": dataset,
        "csv_path": csv_path,
        "T": int(T),
        "N": int(N),
        "lookback": int(cfg.lookback),
        "pred_len": int(cfg.pred_len),
        "mse_avg_std": mse_avg,
        "mae_avg_std": mae_avg,
        "pred_std_avg": pred_std_avg,
        "pred_std_min": pred_std_min,
        "config": cfg.__dict__,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plots (subset of 20+ in v1, still multi-panel and diagnostics)
    if cfg.make_plots:
        plot_dir = os.path.join(run_dir, "plots")
        ensure_dir(plot_dir)

        c_tilde = out_t["c_tilde"].squeeze(0).detach().cpu().numpy()
        attn_maps = [a.detach().cpu().numpy() for a in out_t["attn_maps"]] if cfg.save_attn else []

        make_all_plots(
            out_dir=plot_dir,
            series_names=col_names,
            x_lookback=x_test.squeeze(0).detach().cpu().numpy(),
            y_true=y_true.numpy(),
            y_pred=y_hat.numpy(),
            mse=mse_s.numpy(),
            mae=mae_s.numpy(),
            c_tilde=c_tilde,
            scales=cfg.scales,
            corr=out_t['corr'].squeeze(0).detach().cpu().numpy(),
            W=out_t['W'].squeeze(0).detach().cpu().numpy(),
            attn_maps=attn_maps,
            history=history,
        )

    print(f"Saved results to: {run_dir}")
    return metrics


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=r"/root/autodl-tmp/S")
    p.add_argument("--dataset", type=str, default="electricity",
                   choices=["electricity", "ETTh1", "exchange_rate", "national_illness", "traffic", "weather", "all"])
    p.add_argument("--csv", type=str, default="", help="Optional explicit CSV path (overrides auto-detect).")
    p.add_argument("--series_limit", type=int, default=0, help="Optional cap number of series columns for debugging.")

    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--lookback", type=int, default=200)
    p.add_argument("--scales", type=str, default="48,96,192")
    p.add_argument("--graph_window", type=int, default=96)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--kan_knots", type=int, default=60)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--kappa", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.8)
    p.add_argument("--step_emb_dim", type=int, default=128)

    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0000001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=30.0)
    p.add_argument("--stride", type=int, default=30)

    p.add_argument("--lambda_con", type=float, default=0.001)
    p.add_argument("--lambda_rel", type=float, default=0.001)
    p.add_argument("--lambda_phys", type=float, default=0.001)
    p.add_argument("--phys_warmup_steps", type=int, default=500)
    p.add_argument("--lambda_innov", type=float, default=1e-3)

    p.add_argument("--lambda_flat", type=float, default=0.001)
    p.add_argument("--flat_std_floor", type=float, default=0.02)

    p.add_argument("--lambda_r1", type=float, default=0.001)
    p.add_argument("--lambda_r2", type=float, default=0.001)
    p.add_argument("--lambda_r3", type=float, default=0.001)
    p.add_argument("--lambda_r4", type=float, default=0.001)
    p.add_argument("--lambda_r5", type=float, default=0.001)

    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--make_plots", type=int, default=1)
    p.add_argument("--save_attn", type=int, default=0)
    p.add_argument("--attn_layers", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="concord_outputs_v2")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]

    cfg = Config(
        pred_len=args.pred_len,
        lookback=args.lookback,
        scales=scales,
        graph_window=args.graph_window,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        kan_knots=args.kan_knots,
        topk=args.topk,
        kappa=args.kappa,
        dt=args.dt,
        step_emb_dim=args.step_emb_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        stride=args.stride,
        lambda_con=args.lambda_con,
        lambda_rel=args.lambda_rel,
        lambda_phys=args.lambda_phys,
        phys_warmup_steps=args.phys_warmup_steps,
        lambda_innov=args.lambda_innov,
        lambda_flat=args.lambda_flat,
        flat_std_floor=args.flat_std_floor,
        lambda_r1=args.lambda_r1,
        lambda_r2=args.lambda_r2,
        lambda_r3=args.lambda_r3,
        lambda_r4=args.lambda_r4,
        lambda_r5=args.lambda_r5,
        seed=args.seed,
        device=args.device,
        make_plots=args.make_plots,
        save_attn=args.save_attn,
        attn_layers=args.attn_layers,
        series_limit=args.series_limit,
    )

    if cfg.lookback < max(cfg.scales) + 2:
        raise ValueError(f"lookback must be >= max(scales)+2. Got lookback={cfg.lookback}, scales={cfg.scales}")

    ensure_dir(args.out_dir)

    # auto fallback: if Windows data_dir doesn't exist, use current folder or /mnt/data
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        if os.path.isdir("/mnt/data"):
            data_dir = "/mnt/data"
            print(f"[INFO] data_dir not found, fallback to {data_dir}")

    datasets = ["electricity", "ETTh1", "exchange_rate", "national_illness", "traffic", "weather"]
    run_list = datasets if args.dataset == "all" else [args.dataset]

    all_metrics = []
    for d in run_list:
        csv_path = args.csv if args.csv else find_csv_for_dataset(data_dir, d)
        m = train_and_test_one_dataset(d, csv_path, args.out_dir, cfg)
        all_metrics.append(m)

    with open(os.path.join(args.out_dir, f"summary_{now_str()}.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
