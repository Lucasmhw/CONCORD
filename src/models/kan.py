from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bspline_values(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Cox--de Boor evaluation; zero outside the full extended knot support.

    x ends in the input-feature dimension; knots has shape [inputs, knots].
    No input clamping or continuation beyond the outermost knots is applied.
    """
    z = x.unsqueeze(-1)
    values = ((z >= knots[..., :-1]) & (z < knots[..., 1:])).to(x.dtype)
    for order in range(1, degree + 1):
        left = (z - knots[..., : -(order + 1)]) / (
            knots[..., order:-1] - knots[..., : -(order + 1)]
        )
        right = (knots[..., order + 1 :] - z) / (
            knots[..., order + 1 :] - knots[..., 1:-order]
        )
        values = left * values[..., :-1] + right * values[..., 1:]
    return values


class BSplineBasis(nn.Module):
    """Per-input cubic splines: 15 core intervals, 18 bases, 22 total knots."""

    def __init__(self, in_features: int, grid_size: int = 15, degree: int = 3,
                 grid_min: float = -1.0, grid_max: float = 1.0) -> None:
        super().__init__()
        if grid_size < 1 or degree < 1 or grid_max <= grid_min:
            raise ValueError("Require grid_size >= 1, degree >= 1 and grid_max > grid_min")
        self.grid_size = int(grid_size)
        self.degree = int(degree)
        self.num_basis = self.grid_size + self.degree
        step = (grid_max - grid_min) / self.grid_size
        grid = grid_min + torch.arange(-degree, grid_size + degree + 1) * step
        self.register_buffer("knots", grid.expand(in_features, -1).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bspline_values(x, self.knots.to(dtype=x.dtype), self.degree)


class KANLinear(nn.Module):
    """Inspectable edges: w_base SiLU(a*x+c) + w_spline sum_k coeff_k B_k."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 15,
                 spline_degree: int = 3, grid_min: float = -1.0,
                 grid_max: float = 1.0, bias: bool = True, grid_eps: float = 0.02,
                 grid_margin: float = 0.01) -> None:
        super().__init__()
        if not 0.0 < grid_eps <= 1.0 or grid_margin <= 0.0:
            raise ValueError("grid_eps must be in (0,1] and grid_margin must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_eps = float(grid_eps)
        self.grid_margin = float(grid_margin)
        self.basis = BSplineBasis(in_features, grid_size, spline_degree, grid_min, grid_max)
        self.coeff = nn.Parameter(torch.empty(in_features, out_features, self.basis.num_basis))
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.spline_scale = nn.Parameter(torch.ones(in_features, out_features))
        self.input_scale = nn.Parameter(torch.ones(in_features))
        self.input_shift = nn.Parameter(torch.zeros(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.base_weight.T, a=math.sqrt(5))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.in_features) * self.input_scale + self.input_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_input(x)
        weights = self.coeff * self.spline_scale.unsqueeze(-1)
        out = F.silu(normalized) @ self.base_weight
        out = out + torch.einsum("bik,iok->bo", self.basis(normalized), weights)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], self.out_features)

    @torch.no_grad()
    def evaluate_edge(self, input_index: int, output_index: int, x: torch.Tensor) -> torch.Tensor:
        """An edge excludes the output bias shared across its inputs."""
        if not 0 <= input_index < self.in_features or not 0 <= output_index < self.out_features:
            raise IndexError((input_index, output_index))
        z = x * self.input_scale[input_index] + self.input_shift[input_index]
        basis = bspline_values(z.unsqueeze(-1), self.basis.knots[input_index:input_index+1],
                               self.basis.degree).squeeze(-2)
        return (F.silu(z) * self.base_weight[input_index, output_index]
                + (basis @ self.coeff[input_index, output_index])
                * self.spline_scale[input_index, output_index])


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANMLP(nn.Module):
    def __init__(self, dims: list[int], grid_size: int = 15, spline_degree: int = 3,
                 grid_min: float = -1.0, grid_max: float = 1.0, dropout: float = 0.0,
                 use_kan: bool = True, grid_eps: float = 0.02, grid_margin: float = 0.01) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must include input and output widths")
        if not use_kan:
            self.net = MLP(dims, dropout=dropout)
            return
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i + 1], grid_size, spline_degree,
                                    grid_min, grid_max, grid_eps=grid_eps, grid_margin=grid_margin))
            if i < len(dims) - 2:
                layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, grid_size: int, spline_degree: int,
                 grid_min: float, grid_max: float, dropout: float, use_kan: bool,
                 grid_eps: float = 0.02, grid_margin: float = 0.01) -> None:
        super().__init__()
        if use_kan:
            self.fc1: nn.Module = KANLinear(d_model, d_ff, grid_size, spline_degree,
                                            grid_min, grid_max, grid_eps=grid_eps, grid_margin=grid_margin)
            self.fc2: nn.Module = KANLinear(d_ff, d_model, grid_size, spline_degree,
                                            grid_min, grid_max, grid_eps=grid_eps, grid_margin=grid_margin)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))
