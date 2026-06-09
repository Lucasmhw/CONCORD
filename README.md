# CONCORD

Reproducible PyTorch reference implementation of CONCORD, a concept-oriented graph-coupled dynamical forecaster for multivariate time series.

This repository is the manuscript-aligned implementation. In particular, the observation forcing term follows Eq. 15 exactly:

```text
u_{t,i}^{(h)} = beta_0 + sum_{m,k} beta_{k,m} q_{k,m,t,i}^{(h)}
```

The released model therefore does not use a step embedding, a nonlinear `u_head`, an `innov_head`, an innovation penalty, or a physics warmup schedule. The learnable forcing coefficients are exposed as `CONCORDModel.forcing_coefficients()` and are also available through `ForwardOutput.beta`.

## Repository Layout

```text
configs/
  base.yaml
  long_term.yaml
  pems.yaml
  imputation.yaml
  baselines/
baselines/
  README.md
scripts/
  aggregate_metrics.py
  evaluate.sh
  preprocess_all.sh
  run_baseline.py
  train_imputation.sh
  train_long_term.sh
  train_pems.sh
src/
  cli.py
  config.py
  engine.py
  losses.py
  metrics.py
  data/
  models/
  training/
  utils/
tests/
```

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Data Layout

Place raw data under `data/raw/`:

```text
data/raw/
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  illness/ili.csv
  exchange/exchange_rate.csv
  solar/solar_energy_137_10min.csv
  ett/ETTh1.csv
  ett/ETTh2.csv
  ett/ETTm1.csv
  ett/ETTm2.csv
  pems/PEMS03.npz
  pems/PEMS04.npz
  pems/PEMS07.npz
  pems/PEMS08.npz
```

CSV loaders use every numeric column except an optional `date` column as a time-series channel. NPZ loaders expect `data` or `x`.

If you use the mirrored root-level files in this repository, arrange them into the expected layout first:

```bash
python scripts/prepare_repo_data.py
```

Preprocessing always splits chronologically first and fits the scaler only on the training split:

```bash
bash scripts/preprocess_all.sh
```

Single dataset example:

```bash
python -m concord.cli preprocess --config configs/long_term.yaml \
  data.dataset_name=electricity \
  data.raw_path=data/raw/electricity/electricity.csv \
  data.processed_dir=data/processed/electricity
```

## Main CONCORD Runs

Long-term forecasting:

```bash
bash scripts/train_long_term.sh
```

PEMS forecasting:

```bash
bash scripts/train_pems.sh
```

Imputation:

```bash
bash scripts/train_imputation.sh
```

Evaluate saved checkpoints and aggregate metrics:

```bash
bash scripts/evaluate.sh
```

Each run writes `config.resolved.json`, per-epoch logs, checkpoints, and `metrics.json` under `runs/<exp.name>/`.

## Key Implementation Details

- Concepts: level, velocity, instantaneous power, first-harmonic amplitude, and local volatility at each configured scale.
- Default scales: `[48, 96, 192]`.
- Encoder: shared series-local causal convolutional encoder followed by a KAN projection.
- KAN: cubic B-spline basis with `num_basis=16`, `grid_min=-3.0`, and `grid_max=3.0`.
- Graph: causal correlation window `model.corr_window=96`, top-K sparsity `model.topk=6`, symmetric normalized adjacency, and Laplacian `L = I - A`.
- Dynamics: graph-coupled concept rollout and linear beta forcing for observation rollout.
- Loss: prediction loss + initial concept alignment + residual consistency. No innovation loss or warmup coefficient is used.
- Optimizer: AdamW with cosine schedule and `warmup_epochs=2`.
- Imputation masks: deterministic per sample using `data.mask_seed`.

## Baseline Protocol

Baseline configs for iTransformer and TimeMixer++ style runs are recorded under `configs/baselines/`. Clone official baseline repositories under `external/`, inspect commands with `--dry-run`, and then execute:

```bash
python scripts/run_baseline.py --config configs/baselines/itransformer_long_term.yaml --dry-run
python scripts/run_baseline.py --config configs/baselines/timemixerpp_long_term.yaml --dry-run
```

The runner stores the exact command lines in `runs/baselines/<baseline>/commands.json`.

## Reproducibility Checklist

See `REPRODUCTION.md` for the full checklist covering environment, data preparation, splits, hyperparameters, imputation masks, baseline commands, and reporting.
