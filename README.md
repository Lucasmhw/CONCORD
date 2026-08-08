# CONCORD

Modular PyTorch reference implementation of CONCORD, a concept-oriented,
graph-coupled dynamical model for multivariate time-series forecasting and causal
imputation. 

## What Is Implemented

For each observed series, CONCORD uses a shared causal KAN-Transformer encoder to
infer five operational concepts at several temporal scales:

1. level;
2. level velocity;
3. signal-drift interaction;
4. first-harmonic amplitude;
5. local volatility.

A causal top-K correlation graph refines the inferred concept state. The default
reported rollout retains this graph-refined state and uses a learned lead-time
embedding to obtain horizon-specific KAN readouts:

```text
r_i(h)       = concat(q_i(0), step_embedding(h))
u_i(h)       = KAN_u(r_i(h))
ell_i(h)     = KAN_ell(r_i(h))
epsilon_i(h) = KAN_epsilon(r_i(h))
x_i(h+1)     = x_i(h) + delta * [
                 u_i(h)
                 - gamma * (x_i(h) - ell_i(h))
                 - mu * (L x(h))_i
                 + epsilon_i(h)
               ]
```

`gamma` and `mu` are positive learnable scalars. The innovation is explicitly
penalized. The residual weight is warmed up over optimizer steps. The causal
observation residual compares the first rollout vector field with the final
observed increment, so it is nonzero and does not use future labels.

## Repository Layout

```text
configs/                 model and task configurations
baselines/               published-result provenance
scripts/                 download, training, evaluation, and figure entry points
src/data/                loading, chronological preprocessing, concepts, datasets
src/models/              KAN, causal encoder, graph, and CONCORD dynamics
src/training/            train/validation/test orchestration
tests/                   leakage, shape, graph, preprocessing, and model tests
REPRODUCTION.md          detailed audit checklist
```

## Environment

Python 3.10 or newer is supported. A portable environment can be installed with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The audited GPU runs used Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
cuDNN 9.1, and two NVIDIA RTX 5090 with 32 GB memory. Recreate the direct package
versions with:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-server.txt
pip install -e .
```

Every training run writes its detected package, CUDA, cuDNN, platform, and GPU
versions, deterministic-algorithm state, and source-tree SHA-256 to
`runs/<run>/environment.json`.

## Data

Download and validate all long-term and PEMS benchmarks:

```bash
python scripts/download_benchmarks.py
```

The downloader validates known SHA-256 digests and expected shapes. Long-term data
come from the THUML Time-Series-Library mirror, Solar-Energy comes from the
multivariate-time-series-data release, and PEMS NPZ files come from a public
Hugging Face mirror of the standard PEMS archives.

The repository also contains legacy root-level data archives. To arrange those
files into the expected directory layout:

```bash
python scripts/prepare_repo_data.py
```

Preprocess the long-term and qualitative datasets:

```bash
bash scripts/preprocess_all.sh
```

Preprocessing is chronological. The scaler is fitted only on the training segment,
and validation/test arrays contain only the preceding lookback context needed to
form their first causal window. Each processed dataset records the raw-file hash,
split boundaries, context offsets, and scaler in `data/processed/<dataset>/`.

The split protocols are:

| Family | Train / validation / test |
|---|---|
| ETTh1, ETTh2 | fixed 12/4/4 month boundaries |
| ETTm1, ETTm2 | fixed 12/4/4 month boundaries at 15-minute resolution |
| Electricity, Traffic, Weather, Exchange, Solar | chronological 70/10/20 |
| PEMS03, PEMS04, PEMS07, PEMS08 | chronological 60/20/20 |

## Smoke Test

After preprocessing ETTh1, run a one-epoch end-to-end check:

```bash
python -m concord.cli train --config configs/long_term.yaml \
  data.dataset_name=ett_h1 \
  data.processed_dir=data/processed/ett_h1 \
  data.horizon=96 \
  exp.name=smoke_ett_h1_h96 \
  optim.epochs=1 \
  optim.batch_size=16 \
  optim.accumulation_steps=1
```

Training selects the checkpoint using validation MSE. The held-out test split is
evaluated only after training has finished. For validation-only hyperparameter
screening, add:

```text
eval.run_test_after_training=false
```

## Main Experiments

Long-term forecasting uses horizons 96, 192, 336, and 720 and seeds 41, 42, and
43:

```bash
bash scripts/train_long_term.sh
```

Restrict a run without editing the script:

```bash
DATASETS="ett_h1 weather" HORIZONS="96 192" SEEDS="41 42 43" \
  bash scripts/train_long_term.sh
```

PEMS uses input length 96, prediction length 12, a 60/20/20 chronological split,
and non-overlapping 12-step test windows:

```bash
bash scripts/train_pems.sh
```

Causal imputation is run separately for each mask ratio and seed on ETTh1, ETTh2,
ETTm1, ETTm2, Electricity (ECL), and Weather. The reported ETT value averages the
four ETT datasets within each seed. A masked value is predicted only from preceding
observations and causal forward fills:

```bash
bash scripts/train_imputation.sh
```

Run every core task:

```bash
bash scripts/run_reproduction.sh
```

This full command launches many GPU runs. Use the environment variables above for
incremental checks.

## Metrics And Reporting

After validation-only tuning, freeze the selected run name and evaluate only that
checkpoint:

```bash
RUN_NAMES="selected_run_name" bash scripts/evaluate.sh
```

When `RUN_NAMES` is provided, both evaluation and aggregation are restricted to
that explicit whitelist, so old candidate runs cannot enter the reported mean.

Calling the script without `RUN_NAMES` performs no new test evaluation and only
aggregates test records that already exist:

```bash
bash scripts/evaluate.sh
```

Outputs:

```text
runs/metrics_summary.csv
runs/metrics_aggregated.csv
```

The aggregated file reports per-setting mean and sample standard deviation, each
dataset's horizon/mask average, the four-dataset ETT average, and the four-dataset
PEMS average.

Long-term forecasting metrics are reported in standardized space because the
published TimeMixer++ comparison table uses the standard normalized LTSF metrics.
PEMS metrics are inverse-transformed to original units. Imputation metrics are
computed only at masked positions in standardized space.

## Figures 2-4

Generate the manuscript-layout diagnostics from held-out predictions:

```bash
bash scripts/reproduce_manuscript_figures.sh
```

The script trains missing seed-42, horizon-96 checkpoints for Electricity, Weather,
Illness, Traffic, Exchange, and ETTh1; exports deterministic held-out artifacts; and
writes PNG and PDF versions to:

```text
runs/figures/figure2.{png,pdf}
runs/figures/figure3.{png,pdf}
runs/figures/figure4.{png,pdf}
```

To render from existing checkpoints only:

```bash
TRAIN_MISSING=0 bash scripts/reproduce_manuscript_figures.sh
```

Figure 2 contains the concept heatmap, adjacency, Laplacian eigenvectors, and six
node radar fingerprints. Figure 3 contains permutation degradation, a two-concept
kNN error surface, and ridge slices. Figure 4 contains Electricity/Weather
time-domain examples and six frequency-domain panels.

## Published Baselines

The baseline numbers cited in the unified comparison are published point estimates
transcribed from Tables 1, 3, and 4 of the
[TimeMixer++ ICLR 2025 paper](https://openreview.net/pdf?id=1CLzLXSFNn).
They were not produced by this repository. The complete machine-readable
transcription and source fields are in:

```text
baselines/timemixerpp_published_results.csv
baselines/timemixerpp_published_pems_results.csv
baselines/timemixerpp_published_imputation_results.csv
```

The source table does not report a standard deviation for every baseline entry.
Accordingly, these published values must not be presented as local repeated-run
means or assigned an unsupported standard deviation.

The published imputation table uses standard random-mask reconstruction, while this
repository's CONCORD imputation workflow is strictly causal. Its Table 4 values are
retained for provenance but are not labeled as matched-protocol causal reruns.

TimeMixer++ has no public implementation linked by that paper. Optional,
version-pinned matched-protocol checks are provided for the distinct public
TimeMixer and iTransformer implementations:

```bash
bash scripts/clone_baselines.sh
python scripts/run_baseline.py \
  --config configs/baselines/timemixer_long_term.yaml --dry-run
python scripts/run_baseline.py \
  --config configs/baselines/itransformer_long_term.yaml --dry-run
```

See `baselines/README.md` for the provenance boundary between published values and
optional local reruns.


## Leakage Controls

- Graphs and concept targets use only the lookback window.
- Scalers are fitted on the training split only.
- Validation selects checkpoints; the test split does not select hyperparameters.
- Mixed-trajectory concept residuals use generated predictions, not future labels.
- Step embeddings encode only the known lead index.
- Imputation contexts contain only preceding observations and causal fills.
- Imputation masks are deterministic functions of split, seed, and sample index.
- Deterministic PyTorch/cuDNN algorithms are enabled by default, and the source-tree
  hash is retained with every run.

See `REPRODUCTION.md` for the complete reviewer-facing checklist and audit paths.
