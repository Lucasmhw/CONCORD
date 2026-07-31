#!/usr/bin/env bash
set -euo pipefail

SEEDS=${SEEDS:-"41 42 43"}
MASK_RATIOS=${MASK_RATIOS:-"0.125 0.25 0.375 0.5"}
DATASETS=${DATASETS:-"ett_h1 ett_h2 ett_m1 ett_m2 electricity weather"}

for ds in ${DATASETS}; do
  for ratio in ${MASK_RATIOS}; do
    ratio_tag=${ratio/./p}
    for seed in ${SEEDS}; do
      python -m concord.cli train --config configs/imputation.yaml \
        data.dataset_name="${ds}" \
        data.processed_dir="data/processed/${ds}" \
        data.mask_ratios="[${ratio}]" \
        exp.seed="${seed}" \
        exp.name="concord_imputation_${ds}_m${ratio_tag}_s${seed}"
    done
  done
done
