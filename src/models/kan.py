from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class BSplineBasis(nn.Module):
    def __init__(self, num_basis: int, degree: int = 3, grid_min: float = -3.0, grid_max: float = 3.0) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.degree = degree
        knots = torch.linspace(grid_min, grid_max, num_basis - degree + 1)
        left = knots[0].repeat(degree)
        right = knots[-1].repeat(degree)
        full = torch.cat([left, knots, right])
        self.register_buffer("knots", full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...]
        x = x.clamp(float(self.knots[0]), float(self.knots[-1]))
        basis = []
        for i in range(self.num_basis):
            b = self._basis_function(x, i, self.degree)
            basis.append(b)
        return torch.stack(basis, dim=-1)

    def _basis_function(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        knots = self.knots
        if k == 0:
            left = knots[i]
            right = knots[i + 1]
            return ((x >= left) & (x < right)).to(x.dtype)
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]
        term1 = 0.0
        term2 = 0.0
        if float(denom1) > 0:
            term1 = (x - knots[i]) / denom1 * self._basis_function(x, i, k - 1)
        if float(denom2) > 0:
            term2 = (knots[i + k + 1] - x) / denom2 * self._basis_function(x, i + 1, k - 1)
        return term1 + term2


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 16,
        degree: int = 3,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis = BSplineBasis(num_basis=num_basis, degree=degree, grid_min=grid_min, grid_max=grid_max)
        self.coeff = nn.Parameter(torch.zeros(out_features, in_features, num_basis))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        basis_vals = self.basis(x_flat.unsqueeze(-1)).squeeze(-2)  # [B, in_features, num_basis]
        out = torch.einsum("bin,oin->bo", basis_vals, self.coeff)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape, self.out_features)


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANMLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        num_basis: int = 16,
        degree: int = 3,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        dropout: float = 0.0,
        use_kan: bool = True,
    ) -> None:
        super().__init__()
        if not use_kan:
            self.net = MLP(dims, dropout=dropout)
            return
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i + 1], num_basis, degree, grid_min, grid_max))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
