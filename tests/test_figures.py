import json
from pathlib import Path

import numpy as np

from scripts.reproduce_figures import (
    _diagnostic_neighbor_count,
    figure2,
    figure3,
    figure4,
)


def _write_artifact(path: Path, dataset: str) -> None:
    rng = np.random.default_rng(7)
    samples, horizon, series, concepts = 2, 24, 30, 15
    q0 = rng.normal(size=(samples, series, concepts)).astype(np.float32)
    adjacency = np.eye(series, dtype=np.float32)
    adjacency += 0.1 * np.roll(np.eye(series, dtype=np.float32), 1, axis=1)
    adjacency = 0.5 * (adjacency + adjacency.T)
    adjacency /= adjacency.sum(axis=1, keepdims=True)
    adjacency = np.broadcast_to(adjacency, (samples, series, series)).copy()
    laplacian = np.eye(series, dtype=np.float32)[None] - adjacency
    truth = rng.normal(size=(samples, horizon, series)).astype(np.float32)
    prediction = truth + 0.1 * rng.normal(size=truth.shape).astype(np.float32)
    np.savez_compressed(
        path,
        q0=q0,
        adj=adjacency,
        lap=laplacian,
        y_true=truth,
        y_pred=prediction,
        y_true_raw=truth,
        y_pred_raw=prediction,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "scales": [48, 96, 192],
                "seed": 42,
                "horizon": horizon,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def test_manuscript_figure_pipeline_writes_png_and_pdf(tmp_path: Path) -> None:
    artifacts = {}
    for name in ("Electricity", "Illness", "Traffic", "Exchange", "Weather", "ETT"):
        path = tmp_path / f"{name.lower()}.npz"
        _write_artifact(path, name)
        artifacts[name] = path

    outputs = [tmp_path / "figure2", tmp_path / "figure3", tmp_path / "figure4"]
    figure2(str(artifacts["Electricity"]), outputs[0])
    figure3(str(artifacts["Electricity"]), outputs[1])
    figure4([f"{name}={path}" for name, path in artifacts.items()], outputs[2])

    for output in outputs:
        assert output.with_suffix(".png").stat().st_size > 0
        assert output.with_suffix(".pdf").stat().st_size > 0


def test_diagnostic_neighbor_count_preserves_small_dataset_sensitivity() -> None:
    assert _diagnostic_neighbor_count(7) == 2
    assert _diagnostic_neighbor_count(321) == 25
