#!/usr/bin/env bash
set -euo pipefail

find runs -name best.pt | while read -r ckpt; do
  run_dir=$(dirname "$(dirname "$ckpt")")
  cfg=${run_dir}/config.resolved.json
  echo "Checkpoint: $ckpt"
  python -m concord.cli evaluate --config "$cfg" eval.checkpoint="$ckpt"
done

python scripts/aggregate_metrics.py --runs runs --output runs/metrics_summary.csv
