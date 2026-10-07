"""Exact, full-training-set grid calibration with disk-backed layer inputs.

Calibration is a separate deterministic pass at fixed weights after epochs
5, 10, ... . No validation/test windows, reservoir, or batch cap are used.
Per-feature quantiles and the coefficient least-squares fit use every captured
activation, including every recursive invocation of a shared KAN layer.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from concord.engine import causal_impute
from concord.models.kan import KANLinear, bspline_values


def grid_update_due(epoch: int, cfg: dict[str, Any]) -> bool:
    every = int(cfg["model"].get("grid_update_every", 5))
    if every < 0:
        raise ValueError("grid_update_every must be nonnegative (0 disables calibration)")
    return every > 0 and epoch > 0 and epoch % every == 0


@torch.no_grad()
def fit_grid_from_samples(
    layer: KANLinear, samples: np.ndarray, scratch: Path, chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return new knots and unscaled coefficients without mutating the layer.

    Samples are already transformed by the learned input affine map. Exact
    linearly interpolated quantiles use disk-backed in-place order statistics.
    Streaming QR avoids forming the squared-condition-number normal equations.
    Regridding generally approximates the old spline; it is not exact for an
    arbitrary change of knots. Both residual error and sample counts are logged.
    """
    if samples.ndim != 2 or samples.shape[1] != layer.in_features or not len(samples):
        raise ValueError("Expected nonempty [samples, input_features] calibration data")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    scratch.mkdir(parents=True, exist_ok=True)
    count, width = samples.shape
    size, degree = layer.basis.grid_size, layer.basis.degree
    basis_count = size + degree
    proposed_grid = layer.basis.knots.detach().cpu().clone()
    proposed_coeff = layer.coeff.detach().cpu().clone()
    total_error = total_energy = 0.0
    position = np.linspace(0.0, count - 1, size + 1)
    lo, hi = np.floor(position).astype(np.int64), np.ceil(position).astype(np.int64)
    for feature in range(width):
        column_path = scratch / 'column.bin'
        column = np.memmap(column_path, mode='w+', dtype=samples.dtype, shape=(count,))
        try:
            for offset in range(0, count, chunk_size):
                values = samples[offset:offset + chunk_size, feature]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite layer input during grid calibration")
                column[offset:offset + len(values)] = values
            column.partition(np.unique(np.concatenate([lo, hi])))
            adaptive = np.array(column[lo], dtype=np.float64) * (hi - position)
            adaptive += np.array(column[hi], dtype=np.float64) * (position - lo)
            exact = lo == hi
            adaptive[exact] = column[lo[exact]]
            lower, upper = float(column[0]), float(column[-1])
            step = (upper - lower + 2.0 * layer.grid_margin) / size
            uniform = lower - layer.grid_margin + np.arange(size + 1) * step
            core = (1.0 - layer.grid_eps) * adaptive + layer.grid_eps * uniform
            extended = np.concatenate([
                core[:1] - step * np.arange(degree, 0, -1),
                core,
                core[-1:] + step * np.arange(1, degree + 1),
            ])
            new_grid = torch.as_tensor(extended, dtype=proposed_grid.dtype)
            if not torch.all(new_grid[1:] > new_grid[:-1]):
                raise ValueError("Grid knots are not strictly increasing at model precision")
            old_grid = layer.basis.knots[feature:feature+1].detach().cpu().double()
            new_grid64 = new_grid.unsqueeze(0).double()
            reduced = torch.empty((0, 2 * basis_count), dtype=torch.float64)
            # Order statistics permute a feature's samples; univariate refitting
            # depends only on their values, so the same complete sample set is used.
            for offset in range(0, count, chunk_size):
                z = torch.from_numpy(np.array(column[offset:offset + chunk_size], dtype=np.float64)).unsqueeze(-1)
                old = bspline_values(z, old_grid, degree).squeeze(1)
                new = bspline_values(z, new_grid64, degree).squeeze(1)
                reduced = torch.linalg.qr(
                    torch.cat([reduced, torch.cat([new, old], dim=1)], dim=0), mode='r'
                ).R
            design, reference = reduced[:, :basis_count], reduced[:, basis_count:]
            transfer = torch.linalg.lstsq(design, reference, driver='gelsd').solution
            original = layer.coeff[feature].detach().cpu().double().T
            fitted = transfer @ original
            if not torch.isfinite(fitted).all():
                raise ValueError("Nonfinite coefficients after grid refit")
            # Fit the unscaled spline, leaving the separate learned scale intact.
            scale = layer.spline_scale[feature].detach().cpu().double()
            target = (reference @ original) * scale
            error = (design @ fitted) * scale - target
            total_error += float(error.square().sum())
            total_energy += float(target.square().sum())
            proposed_grid[feature] = new_grid
            proposed_coeff[feature] = fitted.T.to(proposed_coeff.dtype)
        finally:
            column.flush()
            del column
            column_path.unlink(missing_ok=True)
    return proposed_grid, proposed_coeff, {
        'sample_rows': int(count), 'input_features': width,
        'refit_rmse': (total_error / (count * width * layer.out_features)) ** 0.5,
        'relative_refit_l2': (total_error / max(total_energy, 1e-30)) ** 0.5,
    }


@torch.no_grad()
def update_grids_from_training_set(
    model: torch.nn.Module, train_loader: DataLoader, cfg: dict[str, Any],
    device: torch.device, scratch_root: Path, optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    layers = {name: m for name, m in model.named_modules() if isinstance(m, KANLinear)}
    if not layers:
        return {'training_examples': 0, 'layers': {}, 'mode': 'no_kan_layers'}
    scratch_root.mkdir(parents=True, exist_ok=True)
    modes = {m: m.training for m in model.modules()}
    model.eval()
    handles = []
    sinks: dict[str, dict[str, Any]] = {}
    seen = 0
    try:
        with TemporaryDirectory(prefix='grid-calibration-', dir=scratch_root) as temporary:
            folder = Path(temporary)
            for index, (name, layer) in enumerate(layers.items()):
                path = folder / f'layer-{index}.bin'
                sink = {'path': path, 'stream': path.open('wb'), 'rows': 0, 'dtype': None}
                sinks[name] = sink

                def capture(module, args, sink=sink):
                    normalized = module.normalize_input(args[0]).detach().cpu()
                    if normalized.dtype not in {torch.float32, torch.float64}:
                        normalized = normalized.float()
                    array = normalized.contiguous().numpy()
                    if sink['dtype'] is not None and sink['dtype'] != array.dtype:
                        raise ValueError('Layer input dtype changed during calibration')
                    sink['dtype'] = array.dtype
                    array.tofile(sink['stream'])
                    sink['rows'] += len(array)

                handles.append(layer.register_forward_pre_hook(capture))
            # Own sampler: all training windows, including any final partial batch.
            calibration = DataLoader(
                train_loader.dataset, batch_size=train_loader.batch_size or 1,
                shuffle=False, drop_last=False, num_workers=0,
                generator=torch.Generator().manual_seed(int(cfg['exp']['seed'])),
            )
            try:
                for batch in calibration:
                    batch = {key: value.to(device) if torch.is_tensor(value) else value
                             for key, value in batch.items()}
                    seen += next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
                    if 'x_hist' in batch:
                        model(batch['x_hist'], horizon=batch['y'].shape[1],
                              x_hist_raw=batch.get('x_hist_raw'))
                    else:
                        causal_impute(model, batch, cfg)
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()
                for sink in sinks.values():
                    sink['stream'].close()
            proposals = {}
            for name, layer in layers.items():
                sink = sinks[name]
                if not sink['rows']:
                    continue  # An explicitly inactive rollout module.
                inputs = np.memmap(sink['path'], mode='r', dtype=sink['dtype'],
                                   shape=(sink['rows'], layer.in_features))
                try:
                    proposals[name] = fit_grid_from_samples(
                        layer, inputs, folder / 'refit',
                        int(cfg['model'].get('grid_refit_chunk_size', 8192)),
                    )
                finally:
                    del inputs
                sink['path'].unlink()
            # Commit only after every layer has a finite, valid proposal.
            for name, (knots, coeff, _) in proposals.items():
                layer = layers[name]
                layer.basis.knots.copy_(knots.to(layer.basis.knots))
                layer.coeff.copy_(coeff.to(layer.coeff))
                if optimizer is not None:
                    optimizer.state.pop(layer.coeff, None)
            return {
                'training_examples': int(seen), 'dataset_examples': len(train_loader.dataset),
                'mode': 'full_training_set', 'dropout': False,
                'layers': {name: proposal[2] for name, proposal in proposals.items()},
                'optimizer_coeff_moments_reset': optimizer is not None,
            }
    finally:
        for handle in handles:
            handle.remove()
        for sink in sinks.values():
            sink['stream'].close()
        for module, training in modes.items():
            module.training = training
