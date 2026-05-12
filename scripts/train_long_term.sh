#!/usr/bin/env bash
set -euo pipefail

for ds in electricity traffic weather illness exchange; do
  for horizon in 96 192 336 720; do
    python -m concord.cli train --config configs/long_term.yaml \
      data.dataset_name=${ds} \
      data.processed_dir=data/processed/${ds} \
      data.horizon=${horizon} \
      exp.name=concord_${ds}_h${horizon}
  done
done
