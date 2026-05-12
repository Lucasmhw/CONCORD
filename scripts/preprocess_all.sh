#!/usr/bin/env bash
set -euo pipefail

python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=electricity data.raw_path=data/raw/electricity/electricity.csv data.processed_dir=data/processed/electricity
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=traffic data.raw_path=data/raw/traffic/traffic.csv data.processed_dir=data/processed/traffic
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=weather data.raw_path=data/raw/weather/weather.csv data.processed_dir=data/processed/weather
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=illness data.raw_path=data/raw/illness/ili.csv data.processed_dir=data/processed/illness
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=exchange data.raw_path=data/raw/exchange/exchange_rate.csv data.processed_dir=data/processed/exchange
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=ett data.raw_path=data/raw/ett/ETTh1.csv data.processed_dir=data/processed/ett_h1
python -m concord.cli preprocess --config configs/pems.yaml data.dataset_name=pems data.raw_path=data/raw/pems/PEMS04.npz data.processed_dir=data/processed/pems04
python -m concord.cli preprocess --config configs/imputation.yaml data.dataset_name=electricity data.raw_path=data/raw/electricity/electricity.csv data.processed_dir=data/processed/electricity_imp
