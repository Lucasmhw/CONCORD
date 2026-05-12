# CONCORD: Reproducible Reference Codebase

This repository contains a complete, modular PyTorch reference implementation of **CONCORD** (Concept-Oriented Graph-Coupled Dynamics for Forecasting), together with configuration files, preprocessing utilities, training/evaluation scripts, ablation toggles, and reproduction instructions.

The implementation follows the manuscript's core design:

- explicit **multi-scale causal concept states**;
- **correlation-induced graph coupling** in concept space;
- **graph-coupled concept dynamics** and **graph-coupled observation dynamics**;
- **residual-consistent learning** to keep rolled-out concepts equal to the causal statistics they claim to represent;
- **KAN-parameterized** nonlinear maps.

## Repository layout

```text
concord_repro/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── long_term.yaml
│   ├── pems.yaml
│   └── imputation.yaml
├── scripts/
│   ├── preprocess_all.sh
│   ├── train_long_term.sh
│   ├── train_pems.sh
│   ├── train_imputation.sh
│   └── evaluate.sh
├── src/concord/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── losses.py
│   ├── metrics.py
│   ├── engine.py
│   ├── data/
│   │   ├── concepts.py
│   │   ├── datasets.py
│   │   ├── io.py
│   │   ├── preprocess.py
│   │   └── scalers.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── graph.py
│   │   ├── kan.py
│   │   └── concord.py
│   ├── training/
│   │   ├── evaluate.py
│   │   └── train.py
│   └── utils/
│       ├── checkpoint.py
│       ├── logging.py
│       └── seed.py
└── tests/
    ├── test_concepts.py
    └── test_graph.py
```

## 1. Environment

Create a fresh environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Data layout

Place raw data under:

```text
data/
├── raw/
│   ├── electricity/electricity.csv
│   ├── traffic/traffic.csv
│   ├── weather/weather.csv
│   ├── illness/ili.csv
│   ├── exchange/exchange_rate.csv
│   ├── ett/ETTh1.csv
│   ├── ett/ETTh2.csv
│   ├── ett/ETTm1.csv
│   ├── ett/ETTm2.csv
│   ├── pems/PEMS03.npz
│   ├── pems/PEMS04.npz
│   ├── pems/PEMS07.npz
│   └── pems/PEMS08.npz
└── processed/
```

For CSV files, all columns except an optional `date` column are treated as time-series channels.
For NPZ files, the loader expects either `data` or `x` to be present.

## 3. Preprocessing

Preprocessing fits the scaler **only on the training split**, then serializes processed arrays and metadata for exact reuse.

```bash
bash scripts/preprocess_all.sh
```

Or run a single dataset:

```bash
python -m concord.cli preprocess --config configs/long_term.yaml \
  data.dataset_name=electricity \
  data.raw_path=data/raw/electricity/electricity.csv \
  data.processed_dir=data/processed/electricity
```

## 4. Reproducing the main long-term forecasting results

The manuscript uses the following defaults unless otherwise stated:

- scales: `{48, 96, 192}`
- graph top-K sparsity: `6`
- correlation window: equal to the input look-back
- residual-consistency weight: `0.3`
- optimizer: `AdamW`

Run a single long-term forecasting experiment:

```bash
python -m concord.cli train --config configs/long_term.yaml \
  data.dataset_name=electricity \
  data.processed_dir=data/processed/electricity \
  exp.name=concord_electricity
```

To reproduce the averaged long-horizon setting across prediction lengths `{96, 192, 336, 720}`:

```bash
bash scripts/train_long_term.sh
```

This script launches four runs per dataset and writes metrics to `runs/<exp_name>/metrics.json`.

## 5. Reproducing PEMS short-term traffic forecasting

```bash
bash scripts/train_pems.sh
```

## 6. Reproducing the imputation setting

The imputation pipeline uses sequence length `1024` and averages over masking ratios `{0.125, 0.25, 0.375, 0.5}`.

```bash
bash scripts/train_imputation.sh
```

## 7. Evaluation

Evaluate a saved checkpoint:

```bash
python -m concord.cli evaluate --config configs/long_term.yaml \
  exp.name=concord_electricity \
  eval.checkpoint=runs/concord_electricity/checkpoints/best.pt
```

Or aggregate all experiment folders:

```bash
bash scripts/evaluate.sh
```

## 8. Ablations

The codebase exposes the main ablations via config switches:

- `model.use_multiscale=false`
- `model.use_graph=false`
- `loss.lambda_res=0.0`
- `model.use_kan=false`
- `loss.lambda_con=0.0`
- `model.rollout_mode=latent`

Example:

```bash
python -m concord.cli train --config configs/long_term.yaml \
  model.use_graph=false \
  exp.name=ablation_no_graph
```

## 9. Notes on exact reproducibility

1. Set a fixed seed (`exp.seed`) for each run.
2. Keep the processed artifacts and scaler files under versioned paths.
3. Use the same split boundaries for all baselines and CONCORD.
4. For ETT average reporting, average metrics across the ETT subsets you evaluate.
5. For imputation, average final results over the four masking ratios.
6. Record the exact config YAML copied into each run folder.

## 10. Practical interpretation of the code

The implementation intentionally mirrors the paper’s semantic structure:

- `concord.data.concepts` implements the five causal descriptors;
- `concord.models.graph` implements correlation-induced graph construction;
- `concord.models.concord.CONCORDModel` implements concept inference, graph refinement, concept rollout, observation rollout, and residual consistency hooks;
- `concord.losses` contains prediction, concept-alignment, and residual-consistency losses;
- `configs/*.yaml` expose all hyperparameters needed to reproduce the reported settings.

## 11. Recommended workflow

1. Preprocess each dataset.
2. Reproduce the main model with default settings.
3. Reproduce the ablations by changing one switch at a time.
4. Reproduce sensitivity runs for `num_scales`, `topk`, and `lambda_res`.
5. Use the saved JSON metrics to build the paper tables.

## 12. Minimal command set

```bash
bash scripts/preprocess_all.sh
bash scripts/train_long_term.sh
bash scripts/train_pems.sh
bash scripts/train_imputation.sh
bash scripts/evaluate.sh
```
