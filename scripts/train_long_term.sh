#!/usr/bin/env bash
set -euo pipefail

SEEDS=${SEEDS:-"41 42 43"}
HORIZONS=${HORIZONS:-"96 192 336 720"}
DATASETS=${DATASETS:-"electricity traffic weather exchange solar ett_h1 ett_h2 ett_m1 ett_m2"}

for dataset in ${DATASETS}; do
  case "${dataset}" in
    traffic)
      batch_size=1
      accumulation_steps=16
      ;;
    electricity)
      batch_size=2
      accumulation_steps=8
      ;;
    solar)
      batch_size=4
      accumulation_steps=4
      ;;
    weather)
      batch_size=16
      accumulation_steps=1
      ;;
    *)
      batch_size=16
      accumulation_steps=1
      ;;
  esac

  for horizon in ${HORIZONS}; do
    for seed in ${SEEDS}; do
      python -m concord.cli train --config configs/long_term.yaml \
        data.dataset_name="${dataset}" \
        data.processed_dir="data/processed/${dataset}" \
        data.horizon="${horizon}" \
        exp.seed="${seed}" \
        exp.name="concord_${dataset}_h${horizon}_s${seed}" \
        optim.batch_size="${batch_size}" \
        optim.accumulation_steps="${accumulation_steps}"
    done
  done
done
