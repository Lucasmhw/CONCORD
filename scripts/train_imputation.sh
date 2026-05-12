#!/usr/bin/env bash
set -euo pipefail

for ds in electricity traffic weather; do
  python -m concord.cli train --config configs/imputation.yaml \
    data.processed_dir=data/processed/${ds}_imp \
    exp.name=concord_imputation_${ds}
done
