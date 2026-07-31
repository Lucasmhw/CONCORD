from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from concord.models.kan import KANFeedForward


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    """Four-head causal attention using PyTorch's memory-efficient SDPA kernels."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(attended)


class KANTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_knots: int,
        grid_min: float,
        grid_max: float,
        dropout: float,
        use_kan: bool,
        norm_style: str,
    ) -> None:
        super().__init__()
        if norm_style not in {"post", "pre"}:
            raise ValueError("norm_style must be 'post' or 'pre'")
        self.norm_style = norm_style
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = KANFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            num_knots=num_knots,
            grid_min=grid_min,
            grid_max=grid_max,
            dropout=dropout,
            use_kan=use_kan,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.dropout(self.attn(self.attn_norm(x)))
            return x + self.ffn(self.ffn_norm(x))
        x = self.attn_norm(x + self.dropout(self.attn(x)))
        return self.ffn_norm(x + self.ffn(x))


class SeriesLocalEncoder(nn.Module):
    """Shared causal Transformer applied independently to every observed series."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_knots: int,
        grid_min: float,
        grid_max: float,
        dropout: float,
        use_kan: bool,
        max_len: int,
        series_chunk_size: int = 0,
        norm_style: str = "pre",
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.position = SinusoidalPositionEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                KANTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_knots=num_knots,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    dropout=dropout,
                    use_kan=use_kan,
                    norm_style=norm_style,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.series_chunk_size = int(series_chunk_size)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.position(self.input_projection(x))
        for layer in self.layers:
            h = layer(h)
        return self.output_norm(h[:, -1])

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        batch, length, num_series = x_hist.shape
        flattened = x_hist.permute(0, 2, 1).reshape(batch * num_series, length, 1)
        chunk_size = self.series_chunk_size or flattened.shape[0]
        encoded = [self._encode(chunk) for chunk in flattened.split(chunk_size, dim=0)]
        return torch.cat(encoded, dim=0).view(batch, num_series, -1)
