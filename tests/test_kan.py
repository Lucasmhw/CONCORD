import copy
import numpy as np
import torch
from scipy.interpolate import BSpline
from torch.utils.data import DataLoader, Dataset

from concord.models.kan import BSplineBasis, KANLinear
from concord.models.concord import CONCORDModel
from concord.training.grid import fit_grid_from_samples, grid_update_due, update_grids_from_training_set
from test_model import tiny_cfg


def test_cubic_basis_matches_independent_scipy_reference_and_extended_support():
    basis = BSplineBasis(1).double()
    assert basis.knots.shape == (1, 22)
    x = torch.linspace(-1.6, 1.6, 801, dtype=torch.float64)
    values = basis(x[:, None])[:, 0]
    knots = basis.knots[0].numpy()
    reference = np.column_stack([
        np.nan_to_num(BSpline.basis_element(knots[i:i+5], extrapolate=False)(x.numpy()))
        for i in range(18)
    ])
    np.testing.assert_allclose(values.numpy(), reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(values[x.abs() <= 1].sum(-1), torch.ones_like(x[x.abs() <= 1]))
    assert basis(torch.tensor([[1.05]], dtype=torch.float64)).sum() > 0
    assert basis(torch.tensor([[1.5]], dtype=torch.float64)).sum() == 0


def test_edge_evaluation_includes_both_branches_and_learned_affine_transform():
    torch.manual_seed(8)
    layer = KANLinear(2, 3).double()
    with torch.no_grad():
        layer.input_scale.copy_(torch.tensor([0.7, 1.5]))
        layer.input_shift.copy_(torch.tensor([0.1, -0.2]))
        layer.spline_scale.fill_(2.5)
    x = torch.randn(11, 2, dtype=torch.float64)
    expected = torch.stack([
        sum(layer.evaluate_edge(i, j, x[:, i]) for i in range(2)) + layer.bias[j]
        for j in range(3)
    ], -1)
    torch.testing.assert_close(layer(x), expected)
    x.requires_grad_()
    assert torch.autograd.gradcheck(layer, (x,))


def test_full_sample_quantiles_refit_preserves_linear_spline_and_scaler(tmp_path):
    layer = KANLinear(1, 2).double()
    # Greville coefficients reproduce affine functions over the original core.
    t = layer.basis.knots[0]
    greville = torch.stack([t[i+1:i+4].mean() for i in range(18)])
    with torch.no_grad():
        layer.coeff[0, 0].copy_(greville)
        layer.coeff[0, 1].copy_(2 * greville + 0.4)
        layer.spline_scale.copy_(torch.tensor([[2.7, -0.4]]))
    values = np.concatenate([np.linspace(-0.7, -0.1, 601), np.linspace(0.1, 0.8, 99)])[:, None]
    before = layer(torch.from_numpy(values)).detach()
    grid, coeff, record = fit_grid_from_samples(layer, values, tmp_path, chunk_size=53)
    expected_core = 0.98 * np.quantile(values[:, 0], np.linspace(0, 1, 16))
    expected_core += 0.02 * np.linspace(values.min() - 0.01, values.max() + 0.01, 16)
    np.testing.assert_allclose(grid[0, 3:-3].numpy(), expected_core, rtol=1e-12, atol=1e-12)
    assert record['sample_rows'] == len(values)
    with torch.no_grad():
        layer.basis.knots.copy_(grid)
        layer.coeff.copy_(coeff)
    torch.testing.assert_close(layer(torch.from_numpy(values)), before, atol=1e-10, rtol=1e-10)
    assert record['relative_refit_l2'] < 1e-10


class Histories(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(9)
        self.x = torch.randn(5, 16, 2, generator=generator)
        self.x[-1] += 3.0  # Tail example excluded by the ordinary drop_last loader.

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {'x_hist': self.x[i], 'y': torch.zeros(3, 2)}


def test_grid_calibration_uses_whole_training_set_and_checkpoints_new_knots(tmp_path):
    torch.manual_seed(7)
    cfg = tiny_cfg()
    cfg['exp'] = {'seed': 42}
    cfg['model']['grid_refit_chunk_size'] = 97
    cfg['model']['grid_update_every'] = 5
    model = CONCORDModel(cfg)
    dataset = Histories()
    loader = DataLoader(dataset, batch_size=2, drop_last=True)
    before = {name: layer.basis.knots.clone() for name, layer in model.named_modules()
              if isinstance(layer, KANLinear)}
    report = update_grids_from_training_set(model, loader, cfg, torch.device('cpu'), tmp_path)
    assert report['training_examples'] == report['dataset_examples'] == 5
    assert report['layers']['concept_head.net.0']['sample_rows'] == 5 * 2
    assert report['layers']['omega.net.0']['sample_rows'] == 5 * 2 * 3
    assert 'latent_update.net.0' not in report['layers']
    assert model.training
    assert not torch.equal(model.concept_head.net[0].basis.knots, before['concept_head.net.0'])
    saved = copy.deepcopy(model.state_dict())
    restored = CONCORDModel(cfg)
    restored.load_state_dict(saved)
    model.eval()
    restored.eval()
    with torch.no_grad():
        first = model(dataset.x[:1], horizon=3).pred
        second = restored(dataset.x[:1], horizon=3).pred
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, layer in restored.named_modules():
        if isinstance(layer, KANLinear):
            torch.testing.assert_close(layer.basis.knots, saved[name+'.basis.knots'])
    assert list(tmp_path.iterdir()) == []
    assert [epoch for epoch in range(1, 16) if grid_update_due(epoch, cfg)] == [5, 10, 15]


def test_repeated_inputs_produce_strict_knots_and_finite_refit(tmp_path):
    layer = KANLinear(2, 1)
    samples = np.full((50, 2), 0.3, dtype=np.float32)
    knots, coeff, report = fit_grid_from_samples(layer, samples, tmp_path)
    assert torch.isfinite(coeff).all()
    assert torch.all(knots[:, 1:] > knots[:, :-1])
    assert report['relative_refit_l2'] < 1e-5
