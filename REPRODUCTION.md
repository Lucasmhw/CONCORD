# Reproduction Checklist for CONCORD

1. Prepare a clean Python environment.
2. Install dependencies from `requirements.txt`.
3. Place each benchmark under `data/raw/` using the directory layout described in `README.md`.
4. Run preprocessing once per dataset so that train-only normalization is serialized.
5. Reproduce the default model first:
   - scales = `[48, 96, 192]`
   - top-K = `6`
   - residual weight = `0.3`
   - optimizer = `AdamW`
6. Reproduce the long-term forecasting table by running the four horizons `{96, 192, 336, 720}` and averaging metrics per dataset.
7. Reproduce the PEMS table with the PEMS config.
8. Reproduce the imputation table by averaging over mask ratios `{0.125, 0.25, 0.375, 0.5}`.
9. Reproduce ablations by toggling one switch at a time in the config.
10. Reproduce sensitivity by scanning:
    - `model.scales`
    - `model.topk`
    - `loss.lambda_res`

All resolved configs and metrics are copied into the run directory for exact auditability.
