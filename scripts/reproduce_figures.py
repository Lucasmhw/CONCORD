from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


CONCEPT_NAMES = ["Level", "Velocity", "Interaction", "Amplitude", "Volatility"]
CONCEPT_LABELS = ["C1", "C2", "C3", "C4", "C5"]
FIGURE4_ORDER = ["Electricity", "Illness", "Traffic", "Exchange", "Weather", "ETT"]


def _load(path: str) -> tuple[dict[str, np.ndarray], dict]:
    artifact_path = Path(path)
    arrays = dict(np.load(artifact_path))
    with artifact_path.with_suffix(".json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return arrays, metadata


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _feature_label(index: int, scales: list[int]) -> str:
    scale_index, concept_index = divmod(int(index), 5)
    return f"{CONCEPT_LABELS[concept_index]} (tau={scales[scale_index]})"


def _representative_nodes(features: np.ndarray, count: int) -> np.ndarray:
    standardized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    centered = standardized - standardized.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    score = centered @ right[0]
    ordered = np.argsort(score)
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return ordered[positions]


def figure2(artifact: str, output: Path) -> None:
    arrays, metadata = _load(artifact)
    q = arrays["q0"].mean(axis=0)
    adj = arrays["adj"].mean(axis=0)
    lap = arrays["lap"].mean(axis=0)
    scales = metadata["scales"]
    num_series = q.shape[0]
    series_index = int(np.argsort(np.linalg.norm(q, axis=1))[num_series // 2])
    concept_matrix = q[series_index].reshape(len(scales), 5)

    figure = plt.figure(figsize=(15, 7.2), constrained_layout=True)
    figure.suptitle(
        f"CONCORD concept-state and graph diagnostics ({metadata['dataset']})",
        fontsize=14,
        fontweight="bold",
    )
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 1.18),
        height_ratios=(1.0, 1.0, 0.1),
    )
    ax_a = figure.add_subplot(grid[0, :2])
    image = ax_a.imshow(concept_matrix, aspect="auto", cmap="coolwarm")
    ax_a.set_xticks(range(5), CONCEPT_LABELS)
    ax_a.set_yticks(range(len(scales)), [str(scale) for scale in scales])
    ax_a.set_xlabel("Concept coordinate")
    ax_a.set_ylabel("Temporal scale")
    ax_a.set_title("a   Refined concept state by scale and concept", loc="left")
    figure.colorbar(image, ax=ax_a, shrink=0.78, pad=0.02)

    subset_size = min(8, num_series)
    degree_order = np.argsort(adj.sum(axis=1))[::-1][:subset_size]
    ax_b = figure.add_subplot(grid[1, 0])
    adjacency_image = ax_b.imshow(adj[np.ix_(degree_order, degree_order)], cmap="viridis")
    ax_b.set_title("b   Top-K correlation graph adjacency", loc="left")
    ax_b.set_xlabel("Series subset")
    ax_b.set_ylabel("Series subset")
    figure.colorbar(adjacency_image, ax=ax_b, shrink=0.75, pad=0.02)

    lap_subset = lap[np.ix_(degree_order, degree_order)]
    _, eigenvectors = np.linalg.eigh(0.5 * (lap_subset + lap_subset.T))
    ax_c = figure.add_subplot(grid[1, 1])
    eigen_image = ax_c.imshow(
        eigenvectors[:, : min(8, subset_size)].T,
        aspect="auto",
        cmap="coolwarm",
    )
    ax_c.set_title("c   Laplacian eigenvectors", loc="left")
    ax_c.set_xlabel("Series subset")
    ax_c.set_ylabel("Eigenvector")
    figure.colorbar(eigen_image, ax=ax_c, shrink=0.75, pad=0.02)

    concept_by_type = q.reshape(num_series, len(scales), 5).mean(axis=1)
    standardized = (concept_by_type - concept_by_type.mean(axis=0)) / (
        concept_by_type.std(axis=0) + 1e-8
    )
    node_indices = _representative_nodes(concept_by_type, count=6)
    radar_grid = grid[:2, 2].subgridspec(4, 2, height_ratios=(0.13, 1.0, 1.0, 1.0))
    radar_title = figure.add_subplot(radar_grid[0, :])
    radar_title.axis("off")
    radar_title.text(
        0.0,
        0.5,
        "d   Node-level multi-scale fingerprints",
        fontsize=10,
        ha="left",
        va="center",
    )
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(node_indices)))
    for radar_index, (node, color) in enumerate(zip(node_indices, colors)):
        axis = figure.add_subplot(
            radar_grid[1 + radar_index // 2, radar_index % 2],
            projection="polar",
        )
        values = np.r_[standardized[node], standardized[node, 0]]
        axis.plot(closed_angles, values, color=color, linewidth=1.4)
        axis.fill(closed_angles, values, color=color, alpha=0.14)
        axis.set_xticks(angles, CONCEPT_LABELS, fontsize=7)
        axis.set_yticklabels([])
        axis.set_title(f"Series {node}", fontsize=8, pad=2)
    note_axis = figure.add_subplot(grid[2, :])
    note_axis.axis("off")
    note_axis.text(
        0.0,
        0.5,
        "Concept order: C1 level, C2 level velocity, C3 signal-drift interaction, "
        "C4 first-harmonic amplitude, C5 volatility.",
        fontsize=8,
        ha="left",
        va="center",
    )
    _save(figure, output)


def _loo_knn_mse(features: np.ndarray, target: np.ndarray, k: int) -> float:
    if len(features) < 2:
        raise ValueError("Leave-one-out kNN diagnostics require at least two series")
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(features))).fit(features)
    indices = neighbors.kneighbors(features, return_distance=False)
    prediction = target[indices[:, 1:]].mean(axis=1)
    return float(np.mean((prediction - target) ** 2))


def _diagnostic_neighbor_count(num_series: int) -> int:
    if num_series < 2:
        raise ValueError("Concept-space diagnostics require at least two series")
    return min(25, max(2, num_series // 4), num_series - 1)


def figure3(artifact: str, output: Path) -> None:
    arrays, metadata = _load(artifact)
    q = arrays["q0"].mean(axis=0)
    mae = np.abs(arrays["y_pred"] - arrays["y_true"]).mean(axis=(0, 1))
    scales = metadata["scales"]
    neighbor_count = _diagnostic_neighbor_count(len(q))
    base = _loo_knn_mse(q, mae, k=neighbor_count)
    rng = np.random.default_rng(0)
    importance = np.zeros(q.shape[1], dtype=np.float64)
    for feature_index in range(q.shape[1]):
        permuted = q.copy()
        permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
        importance[feature_index] = (
            _loo_knn_mse(permuted, mae, k=neighbor_count) - base
        )

    figure = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    figure.suptitle("Quantitative concept-space diagnostics", fontsize=14, fontweight="bold")
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.35),
        height_ratios=(1.0, 0.08),
    )
    importance_axis = figure.add_subplot(outer[0, 0])
    image = importance_axis.imshow(
        importance.reshape(len(scales), 5),
        aspect="auto",
        cmap="plasma",
    )
    importance_axis.set_xticks(range(5), CONCEPT_LABELS)
    importance_axis.set_yticks(range(len(scales)), [str(scale) for scale in scales])
    importance_axis.set_xlabel("Concept coordinate")
    importance_axis.set_ylabel("Temporal scale")
    importance_axis.set_title("a   Permutation sensitivity in concept space", loc="left")
    figure.colorbar(image, ax=importance_axis, label="Change in leave-one-out kNN MSE")

    top_two = np.argsort(importance)[-2:]
    x_pair = q[:, top_two]
    regressor = KNeighborsRegressor(n_neighbors=neighbor_count, weights="distance")
    regressor.fit(x_pair, mae)
    x_grid = np.linspace(np.percentile(x_pair[:, 0], 2), np.percentile(x_pair[:, 0], 98), 80)
    y_grid = np.linspace(np.percentile(x_pair[:, 1], 2), np.percentile(x_pair[:, 1], 98), 80)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = regressor.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    landscape_grid = outer[0, 1].subgridspec(2, 2, height_ratios=(0.12, 1.0))
    landscape_title = figure.add_subplot(landscape_grid[0, :])
    landscape_title.axis("off")
    landscape_title.text(
        0.0,
        0.5,
        "b   kNN error landscape and ridge slices",
        fontsize=10,
        ha="left",
        va="center",
    )
    landscape_axis = figure.add_subplot(landscape_grid[1, 0])
    ridge_axis = figure.add_subplot(landscape_grid[1, 1])
    contour = landscape_axis.contourf(xx, yy, zz, levels=30, cmap="viridis")
    landscape_axis.scatter(
        x_pair[:, 0],
        x_pair[:, 1],
        c=mae,
        s=12,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.25,
    )
    landscape_axis.set_xlabel(_feature_label(top_two[0], scales))
    landscape_axis.set_ylabel(_feature_label(top_two[1], scales))
    landscape_axis.set_title("2D kNN MAE surface + samples", fontsize=9)
    figure.colorbar(contour, ax=landscape_axis, label="Per-series MAE")

    ridge_indices = np.linspace(0, len(y_grid) - 1, 8, dtype=int)
    ridge_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ridge_indices)))
    for row, color in zip(ridge_indices, ridge_colors):
        ridge_axis.plot(x_grid, zz[row], color=color, linewidth=1.4)
    ridge_axis.set_xlabel(_feature_label(top_two[0], scales))
    ridge_axis.set_ylabel("kNN expected MAE")
    ridge_axis.set_title("Surface slices (ridge plot)", fontsize=9)
    note_axis = figure.add_subplot(outer[1, :])
    note_axis.axis("off")
    note_axis.text(
        0.0,
        0.5,
        "Permutation score is the change in leave-one-out kNN MSE after permuting "
        f"one concept coordinate (k={neighbor_count}); positive values indicate degradation.",
        fontsize=8,
        ha="left",
        va="center",
    )
    _save(figure, output)


def figure4(artifacts: list[str], output: Path) -> None:
    named: dict[str, dict[str, np.ndarray]] = {}
    for item in artifacts:
        name, path = item.split("=", 1)
        named[name] = _load(path)[0]
    missing = [name for name in FIGURE4_ORDER if name not in named]
    if missing:
        raise ValueError(f"Figure 4 requires artifacts for: {', '.join(missing)}")

    figure = plt.figure(figsize=(15, 7.0), constrained_layout=True)
    figure.suptitle(
        "Time-domain forecasts and frequency-domain consistency",
        fontsize=14,
        fontweight="bold",
    )
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.8, 1.45),
        height_ratios=(1.0, 0.08),
    )
    time_grid = outer[0, 0].subgridspec(2, 1)
    for panel_index, name in enumerate(("Electricity", "Weather")):
        axis = figure.add_subplot(time_grid[panel_index, 0])
        arrays = named[name]
        error = np.abs(arrays["y_pred"][0] - arrays["y_true"][0]).mean(axis=0)
        channel = int(np.argsort(error)[len(error) // 2])
        steps = np.arange(arrays["y_true_raw"].shape[1])
        axis.plot(
            steps,
            arrays["y_true_raw"][0, :, channel],
            label="Reference",
            linewidth=1.6,
            color="#2878B5",
        )
        axis.plot(
            steps,
            arrays["y_pred_raw"][0, :, channel],
            label="CONCORD",
            linewidth=1.5,
            color="#E07B39",
        )
        label = "a" if panel_index == 0 else "b"
        axis.set_title(
            f"{label}   {name}, median-error held-out channel, last "
            f"{len(steps)} steps",
            loc="left",
            fontsize=9,
        )
        axis.set_xlabel("Forecast step")
        axis.set_ylabel("Original units")
        axis.legend(frameon=False, fontsize=8)

    spectrum_grid = outer[0, 1].subgridspec(4, 2, height_ratios=(0.1, 1.0, 1.0, 1.0))
    spectrum_title = figure.add_subplot(spectrum_grid[0, :])
    spectrum_title.axis("off")
    spectrum_title.text(
        0.0,
        0.5,
        "c   Amplitude spectra across datasets",
        fontsize=10,
        ha="left",
        va="center",
    )
    for dataset_index, name in enumerate(FIGURE4_ORDER):
        axis = figure.add_subplot(
            spectrum_grid[1 + dataset_index // 2, dataset_index % 2]
        )
        arrays = named[name]
        reference = arrays["y_true"]
        prediction = arrays["y_pred"]
        reference_spectrum = np.abs(np.fft.rfft(reference, axis=1)).mean(axis=(0, 2))
        prediction_spectrum = np.abs(np.fft.rfft(prediction, axis=1)).mean(axis=(0, 2))
        denominator = max(reference_spectrum.max(), 1e-8)
        frequency = np.fft.rfftfreq(reference.shape[1])
        axis.plot(
            frequency,
            reference_spectrum / denominator,
            label="Reference",
            color="#2878B5",
            linewidth=1.2,
        )
        axis.plot(
            frequency,
            prediction_spectrum / denominator,
            label="CONCORD",
            color="#E07B39",
            linewidth=1.2,
        )
        axis.set_title(name, loc="left", fontsize=9)
        axis.set_xlabel("Frequency bin")
        axis.set_ylabel("Normalized amplitude")
        axis.legend(frameon=False, fontsize=7)
    note_axis = figure.add_subplot(outer[1, :])
    note_axis.axis("off")
    note_axis.text(
        0.0,
        0.5,
        "Reference trajectories and CONCORD forecasts use held-out windows only. "
        "Illness is included as an additional frequency-domain diagnostic.",
        fontsize=8,
        ha="left",
        va="center",
    )
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce CONCORD manuscript figures.")
    subparsers = parser.add_subparsers(dest="figure", required=True)
    for number in ("2", "3"):
        subparser = subparsers.add_parser(f"fig{number}")
        subparser.add_argument("--artifact", required=True)
        subparser.add_argument("--output", required=True)
    subparser = subparsers.add_parser("fig4")
    subparser.add_argument("--artifacts", nargs="+", required=True, help="name=artifact.npz")
    subparser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.figure == "fig2":
        figure2(args.artifact, Path(args.output))
    elif args.figure == "fig3":
        figure3(args.artifact, Path(args.output))
    else:
        figure4(args.artifacts, Path(args.output))


if __name__ == "__main__":
    main()
