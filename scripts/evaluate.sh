#!/usr/bin/env bash
set -euo pipefail

RUN_NAMES=${RUN_NAMES:-""}
aggregate_args=()

if [[ -n "${RUN_NAMES}" ]]; then
  read -r -a selected_runs <<< "${RUN_NAMES}"
  for run_name in ${RUN_NAMES}; do
    run_dir="runs/${run_name}"
    ckpt="${run_dir}/checkpoints/best.pt"
    cfg="${run_dir}/config.resolved.json"
    if [[ ! -f "${ckpt}" || ! -f "${cfg}" ]]; then
      echo "Missing frozen run artifacts for ${run_name}" >&2
      exit 1
    fi
    echo "Evaluating frozen run: ${run_name}"
    python -m concord.cli evaluate --config "${cfg}" eval.checkpoint="${ckpt}"
  done
  aggregate_args=(--run-names "${selected_runs[@]}")
else
  echo "RUN_NAMES is empty; no new test evaluation will be performed."
fi

python scripts/aggregate_metrics.py \
  --runs runs \
  --output runs/metrics_summary.csv \
  "${aggregate_args[@]}"
