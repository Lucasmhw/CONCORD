import torch

from concord.data.concepts import compute_concept_targets, compute_concept_trajectory


def test_concept_shape() -> None:
    x = torch.randn(2, 128, 5)
    q = compute_concept_targets(x, [16, 32, 64])
    assert q.shape == (2, 5, 15)


def test_level_velocity_and_interaction_follow_methods() -> None:
    x = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1)
    concepts = compute_concept_targets(x, [3])[0, 0]
    current_level = torch.tensor(6.0)
    previous_level = torch.tensor(5.0)
    assert torch.allclose(concepts[0], current_level)
    assert torch.allclose(concepts[1], current_level - previous_level)
    assert torch.allclose(concepts[2], previous_level * (current_level - previous_level))


def test_mixed_trajectory_h0_matches_initial_target() -> None:
    x = torch.randn(2, 32, 4)
    pred = torch.randn(2, 6, 4)
    initial = compute_concept_targets(x, [8, 16])
    trajectory = compute_concept_trajectory(x, pred, [8, 16])
    assert trajectory.shape == (2, 6, 4, 10)
    assert torch.allclose(trajectory[:, 0], initial, atol=1e-5, rtol=1e-5)


def test_concept_trajectory_does_not_use_future_predictions() -> None:
    x = torch.randn(1, 32, 2)
    pred_a = torch.randn(1, 5, 2)
    pred_b = pred_a.clone()
    pred_b[:, 3:] += 1000.0
    trajectory_a = compute_concept_trajectory(x, pred_a, [8])
    trajectory_b = compute_concept_trajectory(x, pred_b, [8])
    assert torch.allclose(trajectory_a[:, :4], trajectory_b[:, :4], atol=1e-5, rtol=1e-5)
