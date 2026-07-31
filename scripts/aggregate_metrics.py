from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def collect_records(
    runs: Path,
    run_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(runs.glob("*/metrics.json")):
        if run_names is not None and metrics_path.parent.name not in run_names:
            continue
        metrics = _load_json(metrics_path)
        test_metrics = metrics.get("test")
        if not isinstance(test_metrics, dict):
            evaluation_path = metrics_path.parent / "eval_metrics.json"
            if evaluation_path.exists():
                evaluation = _load_json(evaluation_path)
                test_metrics = {
                    name: value for name, value in evaluation.items() if name != "loss"
                }
        config_path = metrics_path.parent / "config.resolved.json"
        if not isinstance(test_metrics, dict) or not config_path.exists():
            continue
        cfg = _load_json(config_path)
        mask_ratios = cfg["data"].get("mask_ratios", [])
        mask_ratio = mask_ratios[0] if len(mask_ratios) == 1 else None
        for metric, value in test_metrics.items():
            records.append(
                {
                    "run": metrics_path.parent.name,
                    "task": cfg["data"].get("task", "forecasting"),
                    "dataset": cfg["data"]["dataset_name"],
                    "horizon": cfg["data"].get("horizon"),
                    "mask_ratio": mask_ratio,
                    "seed": metrics["seed"],
                    "best_epoch": metrics["best_epoch"],
                    "metric": metric,
                    "value": value,
                }
            )
    return records


def _summary_row(
    scope: str,
    key: tuple[Any, ...],
    values: Iterable[float],
) -> dict[str, Any]:
    values = list(values)
    task, dataset, horizon, mask_ratio, metric = key
    return {
        "scope": scope,
        "task": task,
        "dataset": dataset,
        "horizon": horizon,
        "mask_ratio": mask_ratio,
        "metric": metric,
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_setting: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        key = (
            record["task"],
            record["dataset"],
            record["horizon"],
            record["mask_ratio"],
            record["metric"],
        )
        by_setting[key].append(float(record["value"]))
    rows.extend(_summary_row("setting", key, values) for key, values in sorted(by_setting.items()))

    per_seed_setting: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        per_seed_setting[
            (
                str(record["task"]),
                str(record["dataset"]),
                int(record["seed"]),
                str(record["metric"]),
                record["horizon"],
                record["mask_ratio"],
            )
        ].append(float(record["value"]))
    per_seed: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for (task, dataset, seed, metric, _, _), values in per_seed_setting.items():
        per_seed[(task, dataset, seed, metric)].append(statistics.mean(values))
    dataset_seed_means = {
        key: statistics.mean(values) for key, values in per_seed.items()
    }
    by_dataset: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (task, dataset, _, metric), value in dataset_seed_means.items():
        by_dataset[(task, dataset, None, None, metric)].append(value)
    rows.extend(
        _summary_row("dataset_average", key, values)
        for key, values in sorted(by_dataset.items())
    )

    ett_names = {"ett_h1", "ett_h2", "ett_m1", "ett_m2"}
    pems_names = {"pems03", "pems04", "pems07", "pems08"}
    benchmark_requirements = {
        "ETT average": ett_names,
        "PEMS average": pems_names,
        "ETT imputation average": ett_names,
    }
    benchmark_seed: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for (task, dataset, seed, metric), value in dataset_seed_means.items():
        dataset_lower = dataset.lower()
        if task == "forecasting" and dataset_lower in ett_names:
            benchmark_seed[("ETT average", seed, metric)][dataset_lower] = value
        if task == "forecasting" and dataset_lower in pems_names:
            benchmark_seed[("PEMS average", seed, metric)][dataset_lower] = value
        if task == "imputation" and dataset_lower in ett_names:
            benchmark_seed[("ETT imputation average", seed, metric)][dataset_lower] = value
    benchmark_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (name, _, metric), dataset_values in benchmark_seed.items():
        if set(dataset_values) != benchmark_requirements[name]:
            continue
        benchmark_values[("benchmark", name, None, None, metric)].append(
            statistics.mean(dataset_values.values())
        )
    rows.extend(
        _summary_row("benchmark_average", key, values)
        for key, values in sorted(benchmark_values.items())
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seeded CONCORD test metrics.")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="runs/metrics_summary.csv")
    parser.add_argument("--aggregate-output", default="runs/metrics_aggregated.csv")
    parser.add_argument(
        "--run-names",
        nargs="+",
        help="Restrict aggregation to these explicit run-directory names.",
    )
    args = parser.parse_args()

    selected_runs = set(args.run_names) if args.run_names else None
    records = collect_records(Path(args.runs), run_names=selected_runs)
    summaries = aggregate_records(records)
    record_fields = [
        "run",
        "task",
        "dataset",
        "horizon",
        "mask_ratio",
        "seed",
        "best_epoch",
        "metric",
        "value",
    ]
    summary_fields = [
        "scope",
        "task",
        "dataset",
        "horizon",
        "mask_ratio",
        "metric",
        "n",
        "mean",
        "std",
    ]
    _write_csv(Path(args.output), records, record_fields)
    _write_csv(Path(args.aggregate_output), summaries, summary_fields)
    print(f"Wrote {len(records)} records and {len(summaries)} aggregate rows")


if __name__ == "__main__":
    main()
