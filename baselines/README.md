# Baseline Reproduction Protocol

This directory records how the paper baselines are launched from their official implementations. Baseline source code is intentionally not vendored into CONCORD; clone each official repository under `external/` and run the recorded configs through `scripts/run_baseline.py`.

```bash
mkdir -p external
git clone https://github.com/thuml/iTransformer.git external/iTransformer
git clone https://github.com/kwuking/TimeMixer.git external/TimeMixer

python scripts/run_baseline.py --config configs/baselines/itransformer_long_term.yaml --dry-run
python scripts/run_baseline.py --config configs/baselines/timemixerpp_long_term.yaml --dry-run
```

Remove `--dry-run` to execute. The runner writes the exact commands to `runs/baselines/<baseline>/commands.json`.

For fairness, use the same raw data files, forecast horizons, lookback length, train/validation/test ordering, random seed, and metric definitions reported in the CONCORD configs. If a baseline repository exposes a dataset-specific flag with a different spelling, modify only the YAML argument name while keeping the value unchanged.
