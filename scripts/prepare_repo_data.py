from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CSV_MAP = {
    "ETTh1.csv": "data/raw/ett/ETTh1.csv",
    "ETTh2.csv": "data/raw/ett/ETTh2.csv",
    "ETTm1.csv": "data/raw/ett/ETTm1.csv",
    "ETTm2.csv": "data/raw/ett/ETTm2.csv",
    "exchange_rate.csv": "data/raw/exchange/exchange_rate.csv",
    "national_illness.csv": "data/raw/illness/ili.csv",
    "solar_energy_137_10min.csv": "data/raw/solar/solar_energy_137_10min.csv",
    "weather.csv": "data/raw/weather/weather.csv",
}

ZIP_MAP = {
    "electricity.zip": "data/raw/electricity",
    "traffic.zip": "data/raw/traffic",
}


def copy_if_present(src_name: str, dst_name: str) -> None:
    src = ROOT / src_name
    if not src.exists():
        return
    dst = ROOT / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src_name} -> {dst_name}")


def extract_if_present(src_name: str, dst_name: str) -> None:
    src = ROOT / src_name
    if not src.exists():
        return
    dst = ROOT / dst_name
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)
    print(f"extracted {src_name} -> {dst_name}")


def main() -> None:
    for src, dst in CSV_MAP.items():
        copy_if_present(src, dst)
    for src, dst in ZIP_MAP.items():
        extract_if_present(src, dst)


if __name__ == "__main__":
    main()
