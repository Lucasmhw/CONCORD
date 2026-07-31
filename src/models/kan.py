from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HatBasis(nn.Module):
    """Fixed triangular basis on a one-dimensional knot grid."""

    def __init__(self, num_knots: int = 9, grid_min: float = -1.0, grid_max: float = 1.0) -> None:
        super().__init__()
        if num_knots < 3:
            raise ValueError("num_knots must be at least 3")
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        knots = torch.linspace(grid_min, grid_max, num_knots)
        self.register_buffer("knots", knots)
        self.spacing = float((grid_max - grid_min) / (num_knots - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(float(self.knots[0]), float(self.knots[-1]))
        distance = (x.unsqueeze(-1) - self.knots).abs()
        return (1.0 - distance / self.spacing).clamp_min(0.0)


class KANLinear(nn.Module):
    """KAN edge layer with inspectable univariate hat-basis responses."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 9,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.basis = HatBasis(num_knots=num_knots, grid_min=grid_min, grid_max=grid_max)
        self.coeff = nn.Parameter(torch.empty(self.in_features, self.out_features, num_knots))
        self.input_scale = nn.Parameter(torch.ones(self.in_features))
        self.input_shift = nn.Parameter(torch.zeros(self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        nn.init.normal_(self.coeff, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        normalized = x_flat * self.input_scale + self.input_shift
        basis = self.basis(normalized)
        out = torch.einsum("bik,iok->bo", basis, self.coeff)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape, self.out_features)

    @torch.no_grad()
    def evaluate_edge(self, input_index: int, output_index: int, x: torch.Tensor) -> torch.Tensor:
        """Evaluate one learned univariate edge function for interpretation."""
        if not 0 <= input_index < self.in_features:
            raise IndexError(input_index)
        if not 0 <= output_index < self.out_features:
            raise IndexError(output_index)
        normalized = x * self.input_scale[input_index] + self.input_shift[input_index]
        basis = self.basis(normalized)
        return basis @ self.coeff[input_index, output_index]


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
    def __init__(
        self,
        dims: list[int],
        num_knots: int = 9,
        grid_min: float = -1.0,
        grid_max: float = 1.0,
        dropout: float = 0.0,
        use_kan: bool = True,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must include input and output widths")
        if not use_kan:
            self.net = MLP(dims, dropout=dropout)
            return

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(
                KANLinear(
                    dims[i],
                    dims[i + 1],
                    num_knots=num_knots,
                    grid_min=grid_min,
                    grid_max=grid_max,
                )
            )
            if i < len(dims) - 2:
                layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_knots: int,
        grid_min: float,
        grid_max: float,
        dropout: float,
        use_kan: bool,
    ) -> None:
        super().__init__()
        if use_kan:
            self.fc1: nn.Module = KANLinear(d_model, d_ff, num_knots, grid_min, grid_max)
            self.fc2: nn.Module = KANLinear(d_ff, d_model, num_knots, grid_min, grid_max)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))
