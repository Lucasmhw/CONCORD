from concord.config import apply_overrides


def test_nested_cli_overrides() -> None:
    cfg = {"data": {"split": {"train": 0.7}}, "model": {"scales": [1]}}
    updated = apply_overrides(
        cfg,
        [
            "data.split.train_end=8640",
            "model.scales=[48,96,192]",
            "data.expected_channels=7",
        ],
    )
    assert updated["data"]["split"]["train_end"] == 8640
    assert updated["model"]["scales"] == [48, 96, 192]
    assert updated["data"]["expected_channels"] == 7


def test_null_override() -> None:
    updated = apply_overrides({"data": {"date_column": "date"}}, ["data.date_column=null"])
    assert updated["data"]["date_column"] is None
