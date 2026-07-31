import json

from scripts.aggregate_metrics import aggregate_records, collect_records


def test_collect_records_accepts_explicit_frozen_evaluation(tmp_path) -> None:
    run = tmp_path / "selected"
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps({"seed": 42, "best_epoch": 3, "test": None}),
        encoding="utf-8",
    )
    (run / "eval_metrics.json").write_text(
        json.dumps({"loss": 0.5, "mse": 0.4, "mae": 0.3}),
        encoding="utf-8",
    )
    (run / "config.resolved.json").write_text(
        json.dumps(
            {
                "data": {
                    "task": "forecasting",
                    "dataset_name": "ett_h1",
                    "horizon": 96,
                    "mask_ratios": [],
                }
            }
        ),
        encoding="utf-8",
    )

    records = collect_records(tmp_path)

    assert {record["metric"] for record in records} == {"mse", "mae"}
    assert all(record["run"] == "selected" for record in records)


def test_collect_records_can_filter_out_old_candidates(tmp_path) -> None:
    for name, value in (("selected", 0.4), ("old_candidate", 9.9)):
        run = tmp_path / name
        run.mkdir()
        (run / "metrics.json").write_text(
            json.dumps(
                {"seed": 42, "best_epoch": 1, "test": {"mse": value}}
            ),
            encoding="utf-8",
        )
        (run / "config.resolved.json").write_text(
            json.dumps(
                {
                    "data": {
                        "task": "forecasting",
                        "dataset_name": "ett_h1",
                        "horizon": 96,
                        "mask_ratios": [],
                    }
                }
            ),
            encoding="utf-8",
        )

    records = collect_records(tmp_path, run_names={"selected"})
    assert [(record["run"], record["value"]) for record in records] == [
        ("selected", 0.4)
    ]


def test_imputation_ett_average_is_computed_within_seed_first() -> None:
    records = []
    datasets = ("ett_h1", "ett_h2", "ett_m1", "ett_m2")
    for seed, values in (
        (41, [1.0, 3.0, 1.0, 3.0]),
        (42, [5.0, 7.0, 5.0, 7.0]),
    ):
        for dataset, value in zip(datasets, values):
            records.append(
                {
                    "task": "imputation",
                    "dataset": dataset,
                    "horizon": 1,
                    "mask_ratio": 0.25,
                    "seed": seed,
                    "metric": "mse",
                    "value": value,
                }
            )

    rows = aggregate_records(records)
    average = next(
        row
        for row in rows
        if row["scope"] == "benchmark_average"
        and row["dataset"] == "ETT imputation average"
    )
    assert average["n"] == 2
    assert average["mean"] == 4.0


def test_incomplete_benchmark_family_is_not_labeled_as_an_average() -> None:
    rows = aggregate_records(
        [
            {
                "task": "forecasting",
                "dataset": "ett_h1",
                "horizon": 96,
                "mask_ratio": None,
                "seed": 41,
                "metric": "mse",
                "value": 0.4,
            }
        ]
    )
    assert not any(row["scope"] == "benchmark_average" for row in rows)
