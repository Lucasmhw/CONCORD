#!/usr/bin/env bash
set -euo pipefail

for ds in PEMS03 PEMS04 PEMS07 PEMS08; do
  python -m concord.cli preprocess --config configs/pems.yaml data.raw_path=data/raw/pems/${ds}.npz data.processed_dir=data/processed/${ds}
  python -m concord.cli train --config configs/pems.yaml data.processed_dir=data/processed/${ds} exp.name=concord_${ds}
done
