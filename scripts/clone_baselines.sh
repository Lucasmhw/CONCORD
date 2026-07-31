#!/usr/bin/env bash
set -euo pipefail

mkdir -p external

if [[ ! -d external/iTransformer/.git ]]; then
  git clone https://github.com/thuml/iTransformer.git external/iTransformer
fi
git -C external/iTransformer fetch origin
git -C external/iTransformer checkout c2426e68ca13f74aaec08045c5c724d8ad328124

if [[ ! -d external/TimeMixer/.git ]]; then
  git clone https://github.com/kwuking/TimeMixer.git external/TimeMixer
fi
git -C external/TimeMixer fetch origin
git -C external/TimeMixer checkout e24610583b36fdd8c76cc17a8df4e65759a5f460
