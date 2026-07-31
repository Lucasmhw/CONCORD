import json

import numpy as np

from concord.data.datasets import ForecastingDataset, WindowSpec
from concord.data.io import load_time_series
from concord.data.preprocess import preprocess_dataset


def test_preprocess_fits_scaler_on_train_and_preserves_context(tmp_path) -> None:
    raw = np.arange(60, dtype=np.float32).reshape(30, 2)
    raw_path = tmp_path / "toy.npy"
    np.save(raw_path, raw)
    processed = tmp_path / "processed"
    cfg = {
        "data": {
            "dataset_name": "toy",
            "raw_path": str(raw_path),
            "processed_dir": str(processed),
            "date_column": None,
            "expected_channels": 2,
            "lookback": 4,
            "scaler": "standard",
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        }
    }
    preprocess_dataset(cfg)
    stats = np.load(processed / "scaler_stats.npz")
    assert np.allclose(stats["mean"], raw[:18].mean(axis=0, keepdims=True))

    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["splits"]["val"]["source_start"] == 14
    assert metadata["splits"]["val"]["target_offset"] == 4
    val = np.load(processed / "val.npy")
    val_raw = np.load(processed / "val_raw.npy")
    dataset = ForecastingDataset(
        val,
        WindowSpec(lookback=4, horizon=2),
        raw_array=val_raw,
        target_offset=4,
    )
    first = dataset[0]
    assert np.array_equal(first["x_hist_raw"].numpy(), raw[14:18])
    assert np.array_equal(first["y_raw"].numpy(), raw[18:20])


def test_text_loader_detects_optional_header(tmp_path) -> None:
    with_header = tmp_path / "with_header.txt"
    with_header.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    without_header = tmp_path / "without_header.txt"
    without_header.write_text("1,2\n3,4\n", encoding="utf-8")
    expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert np.array_equal(load_time_series(with_header, date_column=None), expected)
    assert np.array_equal(load_time_series(without_header, date_column=None), expected)
