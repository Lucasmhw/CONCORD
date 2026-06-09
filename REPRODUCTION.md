# CONCORD Reproduction Checklist

Use this checklist for every reported table, figure, ablation, and baseline comparison.

## 1. Environment

- Python >= 3.10.
- Install exactly from `requirements.txt`.
- Install the package with `pip install -e .`.
- Record CUDA, PyTorch, NumPy, and pandas versions.
- Run `python -m pytest -q` before launching experiments.

## 2. Data Preparation

- Keep raw datasets under the `data/raw/` layout in `README.md`.
- Use chronological splits from `configs/base.yaml`: train 0.7, validation 0.1, test 0.2 unless a benchmark-specific protocol is explicitly required.
- Fit normalization on the training split only.
- Store processed arrays under `data/processed/<dataset>/`.
- Do not fit scalers, choose masks, construct graphs, or tune hyperparameters using validation/test targets.

## 3. CONCORD Architecture

- Concept targets: level, velocity, instantaneous power, first-harmonic amplitude, local volatility.
- Concept scales: `[48, 96, 192]` by default.
- Encoder: shared series-local causal convolutional stack with channels `[32, 64, 64]`, kernel size 3, dropout 0.1, and KAN projection to `d_model=64`.
- KAN maps: cubic B-spline basis, `num_basis=16`, `spline_order=3`, grid `[-3, 3]`.
- Graph: causal correlation window 96, top-K 6, `kappa=2.0`.
- Observation dynamics: fixed `delta=1.0`, `gamma=0.4`, `mu=0.15`.
- Forcing: explicit linear beta coefficients only; no step embedding, nonlinear forcing head, innovation head, innovation loss, flatline penalty, or physics warmup.

## 4. Training

- Seed: `exp.seed=42`.
- Workers: `exp.num_workers=0` by default for strict reproducibility.
- Optimizer: AdamW, learning rate `1e-3`, betas `[0.9, 0.999]`, eps `1e-8`, weight decay `1e-4`.
- Scheduler: cosine, two warmup epochs.
- Gradient clipping: `1.0`.
- Record each resolved config in the run directory.
- Report mean and standard deviation across repeated seeds if multiple-run results are claimed.

## 5. Forecasting Tables

- Long-term forecasting: run horizons `{96, 192, 336, 720}` and aggregate per dataset.
- PEMS: run PEMS03, PEMS04, PEMS07, and PEMS08 with `configs/pems.yaml`.
- Metrics: use the metric list in each resolved config.
- Aggregation: run `bash scripts/evaluate.sh` and inspect `runs/metrics_summary.csv`.

## 6. Imputation

- Sequence length: 1024.
- Mask ratios: `{0.125, 0.25, 0.375, 0.5}`.
- Mask generation: deterministic per sample from `data.mask_seed`.
- Report masked-position metrics only.

## 7. Baselines

- Clone official baseline repositories under `external/`.
- Run `scripts/run_baseline.py` with configs under `configs/baselines/`.
- Use the same raw data, chronological ordering, input length, prediction horizons, seed, and metric definitions.
- Store exact baseline commands in `runs/baselines/<baseline>/commands.json`.

## 8. Ablations

Change one switch at a time:

- `model.use_multiscale=false`
- `model.use_graph=false`
- `loss.lambda_res=0.0`
- `model.use_kan=false`
- `loss.lambda_con=0.0`
- `model.rollout_mode=latent`

## 9. Audit Points

- `src/models/concord.py` contains the graph-coupled state dynamics.
- `src/data/concepts.py` contains the causal concept definitions.
- `src/losses.py` contains prediction, concept alignment, and residual consistency losses.
- `tests/test_model.py` checks that the released implementation has no `step_emb`, `u_head`, or `innov_head`.
