#!/usr/bin/env bash
set -euo pipefail

find runs -name best.pt | while read -r ckpt; do
  run_dir=$(dirname "$(dirname "$ckpt")")
  cfg=${run_dir}/config.resolved.json
  echo "Checkpoint: $ckpt"
done
