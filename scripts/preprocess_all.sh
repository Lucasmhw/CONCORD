#!/usr/bin/env bash
set -euo pipefail

python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=electricity data.raw_path=data/raw/electricity/electricity.csv data.processed_dir=data/processed/electricity data.expected_channels=321
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=traffic data.raw_path=data/raw/traffic/traffic.csv data.processed_dir=data/processed/traffic data.expected_channels=862
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=weather data.raw_path=data/raw/weather/weather.csv data.processed_dir=data/processed/weather data.expected_channels=21
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=exchange data.raw_path=data/raw/exchange/exchange_rate.csv data.processed_dir=data/processed/exchange data.expected_channels=8
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=solar data.raw_path=data/raw/solar/solar_AL.txt data.processed_dir=data/processed/solar data.date_column=null data.expected_channels=137

python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=ett_h1 data.raw_path=data/raw/ett/ETTh1.csv data.processed_dir=data/processed/ett_h1 data.expected_channels=7 data.split.train_end=8640 data.split.val_end=11520 data.split.test_end=14400
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=ett_h2 data.raw_path=data/raw/ett/ETTh2.csv data.processed_dir=data/processed/ett_h2 data.expected_channels=7 data.split.train_end=8640 data.split.val_end=11520 data.split.test_end=14400
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=ett_m1 data.raw_path=data/raw/ett/ETTm1.csv data.processed_dir=data/processed/ett_m1 data.expected_channels=7 data.split.train_end=34560 data.split.val_end=46080 data.split.test_end=57600
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=ett_m2 data.raw_path=data/raw/ett/ETTm2.csv data.processed_dir=data/processed/ett_m2 data.expected_channels=7 data.split.train_end=34560 data.split.val_end=46080 data.split.test_end=57600

# Illness is used only for the qualitative spectrum diagnostic in Figure 4.
python -m concord.cli preprocess --config configs/long_term.yaml data.dataset_name=illness data.raw_path=data/raw/illness/national_illness.csv data.processed_dir=data/processed/illness data.expected_channels=7
