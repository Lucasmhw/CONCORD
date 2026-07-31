#!/usr/bin/env bash
set -euo pipefail

SEEDS=${SEEDS:-"41 42 43"}

for ds in PEMS03 PEMS04 PEMS07 PEMS08; do
  case "${ds}" in
    PEMS03) channels=358 ;;
    PEMS04) channels=307 ;;
    PEMS07) channels=883 ;;
    PEMS08) channels=170 ;;
  esac
  python -m concord.cli preprocess --config configs/pems.yaml \
    data.dataset_name="${ds}" \
    data.raw_path="data/raw/pems/${ds}.npz" \
    data.processed_dir="data/processed/${ds}" \
    data.expected_channels="${channels}"
  for seed in ${SEEDS}; do
    python -m concord.cli train --config configs/pems.yaml \
      data.dataset_name="${ds}" \
      data.processed_dir="data/processed/${ds}" \
      data.expected_channels="${channels}" \
      exp.seed="${seed}" \
      exp.name="concord_${ds}_s${seed}"
  done
done
