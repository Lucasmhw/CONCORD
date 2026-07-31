# CONCORD Reproduction Checklist

Use this checklist for every reported table, figure, ablation, and baseline
comparison.

## 1. Environment

- Record Python, PyTorch, CUDA, cuDNN, GPU, and direct dependency versions.
- Keep deterministic algorithms enabled and retain the source-tree SHA-256.
- Use `requirements-server.txt` for the audited CUDA 12.8 environment.
- Run `python -m pytest -q` before experiments.
- Retain each run's `environment.json` and `config.resolved.json`.

## 2. Data Integrity

- Run `python scripts/download_benchmarks.py`.
- Verify the printed shape for every selected dataset.
- Retain `raw_sha256` from each processed `metadata.json`.
- Fit normalization statistics on the training segment only.
- Never interpolate a missing value from a later observation.
- Retain the exact fixed ETT boundaries and PEMS 60/20/20 boundaries.

## 3. Architecture

- Shared series-local causal KAN-Transformer encoder.
- Three pre-norm layers, four heads, width 64, feed-forward width 128.
- Five concepts at scales `[48, 96, 192]` for LTSF.
- Nine fixed triangular spline knots on `[-1, 1]`.
- Symmetric normalized top-K absolute-correlation graph.
- Six neighbors and a correlation window equal to the lookback.
- Thirty-two-dimensional lead-time embedding.
- KAN forcing, equilibrium, and innovation heads.
- Positive learnable damping and graph-coupling scalars.
- Specialized graph-refined concept state for reported runs.

## 4. Objective

- Prediction MSE.
- Initial concept alignment, weight 0.1.
- Graph-refinement relation penalty, weight 0.01.
- Mixed observed-predicted concept consistency.
- Causal origin-step observation-dynamics residual.
- Residual weight 0.3 after a 500-step linear warm-up.
- Innovation L2 and variance-floor penalties, each weight `1e-3`.
- Fixed learning rate `1e-3`, 20-epoch budget, and validation patience 5.
- Long-term effective batch size 16, using gradient accumulation for high-channel datasets.
- AdamW weight decay `1e-4`.

## 5. Forecasting

- Run horizons `{96, 192, 336, 720}`.
- Run seeds `{41, 42, 43}`.
- Select the checkpoint only from validation MSE.
- Evaluate the test split after the configuration is frozen.
- Report normalized MSE/MAE for the LTSF comparison.
- Average horizon metrics within each seed before computing mean and s.d. across
  seeds.
- Average the four ETT datasets within each seed for the ETT aggregate.

## 6. PEMS

- Use PEMS03, PEMS04, PEMS07, and PEMS08.
- Use input 96, prediction 12, and chronological 60/20/20 splits.
- Use stride 1 for training/validation and stride 12 for test windows.
- Fit one channel-wise scaler on the training split.
- Report inverse-transformed MAE, MAPE, and RMSE.
- Average datasets within seed before computing the PEMS mean and s.d.

## 7. Imputation

- Use ETTh1, ETTh2, ETTm1, ETTm2, Electricity, and Weather processed splits.
- Use sequence length 1024.
- Train separate runs for mask ratios `{0.125, 0.25, 0.375, 0.5}`.
- Generate masks deterministically from split, seed, and sample index.
- Predict a masked position only from its preceding causal window.
- Report MSE/MAE only at masked positions.
- Average mask-ratio results within seed before computing mean and s.d.
- Average the four ETT datasets within seed for the reported ETT imputation result.

## 8. Baselines

- Treat `baselines/timemixerpp_published_results.csv` as a transcription of
  published point estimates, not local output.
- Cite TimeMixer++ ICLR 2025 Table 1 for every long-term baseline row.
- Do not attach invented standard deviations to published point estimates.
- Keep optional TimeMixer and iTransformer reruns separate from TimeMixer++.
- Do not label the published non-causal random-mask imputation values as
  matched-protocol comparisons to CONCORD's causal imputation workflow.
- Verify external Git commits before optional reruns.
- Retain expanded baseline commands under `runs/baselines/`.

## 9. Figures

- Export artifacts only from the held-out test loader.
- Retain artifact JSON sidecars with dataset, seed, horizon, and checkpoint epoch.
- Generate Figures 2-4 with `scripts/reproduce_manuscript_figures.sh`.
- Do not manually alter plotted values.

## 10. Audit Paths

- Concepts: `src/data/concepts.py`
- Graph: `src/models/graph.py`
- KAN: `src/models/kan.py`
- Encoder: `src/models/encoder.py`
- Dynamics: `src/models/concord.py`
- Objective: `src/losses.py`
- Splits/scaling: `src/data/preprocess.py`
- Training selection: `src/training/train.py`
- Metrics/imputation: `src/engine.py`
- Figure definitions: `scripts/reproduce_figures.py`
- Baseline provenance: `baselines/README.md`
