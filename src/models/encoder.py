from __future__ import annotations

import torch
import torch.nn as nn

from concord.models.kan import KANMLP


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class SeriesLocalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        channels: list[int],
        kernel_size: int,
        num_basis: int,
        spline_order: int,
        grid_min: float,
        grid_max: float,
        dropout: float,
        use_kan: bool,
    ) -> None:
        super().__init__()
        convs = []
        in_ch = 1
        for i, ch in enumerate(channels):
            convs.append(CausalConv1d(in_ch, ch, kernel_size=kernel_size, dilation=2 ** i))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(dropout))
            in_ch = ch
        self.conv = nn.Sequential(*convs)
        self.proj = KANMLP(
            [channels[-1], d_model, d_model],
            num_basis=num_basis,
            degree=spline_order,
            grid_min=grid_min,
            grid_max=grid_max,
            dropout=dropout,
            use_kan=use_kan,
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        # x_hist: [B, L, N]
        b, l, n = x_hist.shape
        x = x_hist.permute(0, 2, 1).reshape(b * n, 1, l)
        h = self.conv(x)
        h = h[..., -1]
        h = self.proj(h)
        return h.view(b, n, -1)
