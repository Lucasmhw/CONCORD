#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-42}
HORIZON=${HORIZON:-96}
TRAIN_MISSING=${TRAIN_MISSING:-1}
ARTIFACT_DIR=${ARTIFACT_DIR:-runs/figure_artifacts}
FIGURE_DIR=${FIGURE_DIR:-runs/figures}

mkdir -p "${ARTIFACT_DIR}" "${FIGURE_DIR}"

datasets=(electricity weather illness traffic exchange ett_h1)
for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    traffic) batch_size=1; accumulation_steps=16 ;;
    electricity) batch_size=2; accumulation_steps=8 ;;
    *) batch_size=16; accumulation_steps=1 ;;
  esac
  run_name="figure_${dataset}_h${HORIZON}_s${SEED}"
  checkpoint="runs/${run_name}/checkpoints/best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    if [[ "${TRAIN_MISSING}" != "1" ]]; then
      echo "Missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    python -m concord.cli train --config configs/long_term.yaml \
      data.dataset_name="${dataset}" \
      data.processed_dir="data/processed/${dataset}" \
      data.horizon="${HORIZON}" \
      exp.seed="${SEED}" \
      exp.name="${run_name}" \
      optim.batch_size="${batch_size}" \
      optim.accumulation_steps="${accumulation_steps}"
  fi
  python scripts/export_artifacts.py \
    --checkpoint "${checkpoint}" \
    --output "${ARTIFACT_DIR}/${dataset}.npz" \
    --max-samples 8
done

python scripts/reproduce_figures.py fig2 \
  --artifact "${ARTIFACT_DIR}/electricity.npz" \
  --output "${FIGURE_DIR}/figure2"
python scripts/reproduce_figures.py fig3 \
  --artifact "${ARTIFACT_DIR}/electricity.npz" \
  --output "${FIGURE_DIR}/figure3"
python scripts/reproduce_figures.py fig4 \
  --artifacts \
    "Electricity=${ARTIFACT_DIR}/electricity.npz" \
    "Illness=${ARTIFACT_DIR}/illness.npz" \
    "Traffic=${ARTIFACT_DIR}/traffic.npz" \
    "Exchange=${ARTIFACT_DIR}/exchange.npz" \
    "Weather=${ARTIFACT_DIR}/weather.npz" \
    "ETT=${ARTIFACT_DIR}/ett_h1.npz" \
  --output "${FIGURE_DIR}/figure4"

echo "Figures written to ${FIGURE_DIR}"
