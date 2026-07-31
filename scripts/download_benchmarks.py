from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TSLIB_BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"


@dataclass(frozen=True)
class DatasetFile:
    url: str
    target: str
    rows: int
    channels: int
    sha256: str | tuple[str, ...] | None = None
    gzip_source: bool = False
    target_sha256: str | tuple[str, ...] | None = None


DATASETS = {
    "electricity": DatasetFile(
        f"{TSLIB_BASE}/electricity/electricity.csv?download=true",
        "data/raw/electricity/electricity.csv",
        26_304,
        321,
        "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad",
    ),
    "traffic": DatasetFile(
        f"{TSLIB_BASE}/traffic/traffic.csv?download=true",
        "data/raw/traffic/traffic.csv",
        17_544,
        862,
        "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16",
    ),
    "weather": DatasetFile(
        f"{TSLIB_BASE}/weather/weather.csv?download=true",
        "data/raw/weather/weather.csv",
        52_696,
        21,
        "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
    ),
    "exchange": DatasetFile(
        f"{TSLIB_BASE}/exchange_rate/exchange_rate.csv?download=true",
        "data/raw/exchange/exchange_rate.csv",
        7_588,
        8,
        "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
    ),
    "illness": DatasetFile(
        f"{TSLIB_BASE}/illness/national_illness.csv?download=true",
        "data/raw/illness/national_illness.csv",
        966,
        7,
        "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
    ),
    "ett_h1": DatasetFile(
        f"{TSLIB_BASE}/ETT-small/ETTh1.csv?download=true",
        "data/raw/ett/ETTh1.csv",
        17_420,
        7,
        (
            "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
            "aaf53becad03ad8a7e93cf96452244f423480df960d769e12d66d4a5c570ddb0",
        ),
    ),
    "ett_h2": DatasetFile(
        f"{TSLIB_BASE}/ETT-small/ETTh2.csv?download=true",
        "data/raw/ett/ETTh2.csv",
        17_420,
        7,
        (
            "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
            "ae5c99b925fb8b4e77d1bd6a897b311d1167766076f3d00e14a8654de38e7468",
        ),
    ),
    "ett_m1": DatasetFile(
        f"{TSLIB_BASE}/ETT-small/ETTm1.csv?download=true",
        "data/raw/ett/ETTm1.csv",
        69_680,
        7,
        (
            "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
            "f8a9f83e816439447a4b0c523039ac618bd9b97b9e97d48d0c3f71feb1899ddc",
        ),
    ),
    "ett_m2": DatasetFile(
        f"{TSLIB_BASE}/ETT-small/ETTm2.csv?download=true",
        "data/raw/ett/ETTm2.csv",
        69_680,
        7,
        (
            "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
            "fa470b4ffeac8f96f325abf027fe1c3cad1149560eabffc6d254abb72de21a62",
        ),
    ),
    "solar": DatasetFile(
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
        "master/solar-energy/solar_AL.txt.gz",
        "data/raw/solar/solar_AL.txt",
        52_560,
        137,
        "1ae9329c3e1032acbb8dd080bfc397c44507006812a715d2788f172820489a1c",
        gzip_source=True,
        target_sha256="230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2",
    ),
    "pems03": DatasetFile(
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/"
        "resolve/main/PEMS03.npz?download=true",
        "data/raw/pems/PEMS03.npz",
        26_208,
        358,
        "34859517425eded66a0014db63de54b95d0428c38890741a1710ed4d3b043de8",
    ),
    "pems04": DatasetFile(
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/"
        "resolve/main/PEMS04.npz?download=true",
        "data/raw/pems/PEMS04.npz",
        16_992,
        307,
        "95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a",
    ),
    "pems07": DatasetFile(
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/"
        "resolve/main/PEMS07.npz?download=true",
        "data/raw/pems/PEMS07.npz",
        28_224,
        883,
        "a07887e610aa986276209bc9384d14ed7858fdcad1ded67f53dca19eb828600b",
    ),
    "pems08": DatasetFile(
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/"
        "resolve/main/PEMS08.npz?download=true",
        "data/raw/pems/PEMS08.npz",
        17_856,
        170,
        "e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(
    path: Path,
    expected: str | tuple[str, ...] | None,
    label: str,
) -> None:
    if expected is None:
        return
    allowed = {expected} if isinstance(expected, str) else set(expected)
    actual = _sha256(path)
    if actual not in allowed:
        raise ValueError(
            f"Checksum mismatch for {label}: got {actual}; "
            f"expected one of {sorted(allowed)}"
        )


def _download(spec: DatasetFile, force: bool) -> Path:
    target = ROOT / spec.target
    if target.exists() and not force:
        existing_hash = spec.target_sha256
        if existing_hash is None and not spec.gzip_source:
            existing_hash = spec.sha256
        _verify_sha256(target, existing_hash, str(target))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    urllib.request.urlretrieve(spec.url, temporary)
    try:
        _verify_sha256(temporary, spec.sha256, spec.url)
    except ValueError:
        temporary.unlink(missing_ok=True)
        raise
    if spec.gzip_source:
        with gzip.open(temporary, "rb") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.unlink()
    else:
        temporary.replace(target)
    target_hash = spec.target_sha256 if spec.gzip_source else spec.sha256
    _verify_sha256(target, target_hash, str(target))
    return target


def _validate(path: Path, spec: DatasetFile) -> None:
    if path.suffix == ".txt":
        array = np.loadtxt(path, delimiter=",", dtype=np.float32)
        shape = array.shape
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            array = archive["data"] if "data" in archive else archive[list(archive.keys())[0]]
        shape = (array.shape[0], array.shape[1])
    else:
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        shape = (len(frame), numeric.shape[1])
    expected = (spec.rows, spec.channels)
    if shape != expected:
        raise ValueError(f"{path} has shape {shape}; expected {expected}")
    print(f"validated {path.relative_to(ROOT)} shape={shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate CONCORD benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=sorted(DATASETS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        spec = DATASETS[name]
        _validate(_download(spec, force=args.force), spec)


if __name__ == "__main__":
    main()
