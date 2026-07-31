#!/usr/bin/env bash
set -euo pipefail

python scripts/download_benchmarks.py
bash scripts/preprocess_all.sh
bash scripts/train_long_term.sh
bash scripts/train_pems.sh
bash scripts/train_imputation.sh
bash scripts/evaluate.sh

echo "Core reproduction finished. See runs/metrics_summary.csv."
