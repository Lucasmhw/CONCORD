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

A causal top-K correlation graph refines the inferred concept state. The reported
configuration uses `rollout_mode: recursive`: all concept coordinates are updated
at each forecast step, and the current state plus a learned lead-time embedding
provides the KAN readouts:

```text
r_i(h)       = concat(q_i(h), step_embedding(h))
u_i(h)       = KAN_u(r_i(h))
ell_i(h)     = KAN_ell(r_i(h))
epsilon_i(h) = KAN_epsilon(r_i(h))
x_i(h+1)     = x_i(h) + delta * [
                 u_i(h)
                 - gamma * (x_i(h) - ell_i(h))
                 - mu * (L x(h))_i
                 + epsilon_i(h)
               ]
q_i(h+1)     = q_i(h) + delta * KAN_omega(concat(
                 q_i(h), x_i(h), sum_j A_ij * KAN_phi(q_j(h))
               ))
```

Both updates use the current `(q(h), x(h))`; the updated concepts enter the next
step's readout. The graph stays fixed within a forecast, while its messages change
with the concept states. `specialized` is an explicitly selected constant-concept
alternative, and `latent` is a separate latent-update alternative.

KAN edges use a learned affine input transform followed by
`w_base * SiLU(z) + w_spline * sum_k coeff_k * B_k(z)`. The default cubic B-splines
have 15 core intervals and 18 basis functions per input, initialized on `[-1,1]`
with three extra knot intervals on each side (22 knots in total). Inputs are not
clamped. Splines remain nonzero near the core boundary within their extended
support and vanish beyond the outermost knots; the SiLU branch remains active.

After every fifth epoch, a separate pass at fixed weights uses the **entire
training dataset** to collect each active KAN layer's affine-transformed inputs,
including all recursive calls. Dropout is disabled during calibration. For each
input coordinate, the new core grid is 0.98 times its linearly interpolated
empirical quantiles plus 0.02 times a uniform grid over the observed range with a
0.01 margin. Three intervals are extended on each side. Streaming least squares
refits the unscaled spline coefficients to their previous responses over every
captured sample; learned spline scales are preserved. Regridding approximates the
previous spline and does not guarantee exact preservation for arbitrary grids.

Calibration writes disk-backed activation files under the run directory and
removes them after use. It does not substitute a reservoir, one mini-batch, or
validation/test inputs. Allow storage for all layer activations in this full-data
pass. Refitted coefficient optimizer moments are reset; other optimizer states
are preserved. `grid_update_005.json`, etc. record per-layer sample counts and
refit errors. Validation and checkpoint selection follow calibration, and
checkpoints save the learned knots as well as all parameters. Inference never
updates grids. `model.grid_update_every=0` explicitly disables calibration for
controlled ablations, rather than silently altering the reported default.

This release aligns the repository with the authors' confirmed recursive,
adaptive B-spline experiment configuration. The earlier fixed-hat,
constant-concept public snapshot is not the implementation to attribute to those
experiments. Existing checkpoints from that snapshot are incompatible with the
new basis and branch parameters and must not be silently loaded as this model.

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

## Baseline results and provenance

The manuscript's baseline results were rerun and checked by the authors using
seeds **7, 50, 81**. The [baseline audit](baselines/provenance/README.md) covers all
20 named methods, with original-paper links, frozen official source commits,
seed-setting evidence, and standard-deviation comparability. The corresponding
[JSON manifest](baselines/provenance/baseline_protocols.json) is machine-readable.
This documentation update does not change benchmark values or independently
rerun the full benchmark.

Separate TimeMixer++ transcriptions provide public reference means:

```text
baselines/timemixerpp_published_results.csv
baselines/timemixerpp_published_pems_results.csv
baselines/timemixerpp_published_imputation_results.csv
```

These public references are not the run archive for the author experiments.
Their seed identifiers, uncertainty coverage and information sets must not be
inferred from our reruns, or vice versa. In particular, source imputation results
may use non-causal reconstruction, while the released CONCORD workflow is causal.

The paper-linked TimeMixer snapshot contains no separately identifiable
TimeMixer++ implementation. Its source-table results can be independently
cross-checked, but a TimeMixer run must not be relabelled as TimeMixer++.

Optional version-pinned TimeMixer and iTransformer wrappers remain available:

```bash
bash scripts/clone_baselines.sh
python scripts/run_baseline.py \
  --config configs/baselines/timemixer_long_term.yaml --dry-run
python scripts/run_baseline.py \
  --config configs/baselines/itransformer_long_term.yaml --dry-run
```

They request `itr=3` from upstream code seeded once to 2021/2023, respectively;
they do not implement the manuscript's 7/50/81 author-run protocol. See
[baselines/README.md](baselines/README.md) for these separate provenance classes.


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
