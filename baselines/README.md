# Baseline provenance and optional reruns
 
## Author reruns and published cross-check references

The authors confirm that the baseline summaries in manuscript Tables 1 and 2
come from reruns with seeds **7, 50, 81**, checked against the relevant published
benchmarks. The [20-method provenance audit](provenance/README.md) identifies
each original paper, official repository, frozen source version and upstream
seed behavior. Its [JSON manifest](provenance/baseline_protocols.json) separates
the author-confirmed seeds from upstream defaults.

The existing CSV files contain independent reference point estimates transcribed
from Tables 1, 3, and 4 of the
[TimeMixer++ ICLR 2025 paper](https://openreview.net/pdf?id=1CLzLXSFNn).
They include the published TimeMixer++, TimeMixer, iTransformer, PatchTST,
Crossformer, TiDE, TimesNet, DLinear, SCINet, MICN, FEDformer, Stationary, and
Autoformer rows where those models appear in the source tables. Stationary is
retained in this historical source transcription but is not a displayed baseline
in the current manuscript. The CSVs are not author-run output or the source of
the manuscript's standard deviations. The complete
transcriptions, including the source table and URL on every row, are stored in:

- `baselines/timemixerpp_published_results.csv` (long-term forecasting);
- `baselines/timemixerpp_published_pems_results.csv` (PEMS average);
- `baselines/timemixerpp_published_imputation_results.csv` (imputation).

The paper points readers to `https://github.com/kwuking/TimeMixer`. At the audited
commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`, that repository contains
`models/TimeMixer.py` but no separately identifiable implementation of the
TimeMixer++ MRTI/TID/MRM architecture. CONCORD therefore does not relabel the
public TimeMixer implementation as TimeMixer++. Published point estimates must not
be presented as CONCORD-side repeated-run means or assigned standard deviations
unless the corresponding repeated-run records are available.

The source PDF used for the transcription has SHA-256
`fe0314e9017764d4404b291cbee5662908c17753a67896fe2e0f0712d81091d6`.
Recording the digest avoids silently mixing values from different paper revisions.

These are source-protocol references. They do not establish that every baseline was
rerun with CONCORD's 336-step lookback or other CONCORD-specific settings; exact
matched-protocol reruns must be labeled separately.

Two protocol differences are particularly important. TimeMixer++ Table 1 uses an
input length of 96, whereas the canonical CONCORD long-term configuration uses 336.
TimeMixer++ Table 4 reports standard random-mask reconstruction over length-1024
sequences, whereas the released CONCORD imputation task is deliberately causal: each
masked value is predicted only from preceding context. The Table 4 transcription is
therefore a source benchmark reference, not evidence of a strictly matched causal
imputation comparison.

## Versioned runnable baselines

For an independent matched-protocol check, this repository provides runnable
manifests for the official TimeMixer and iTransformer repositories. These are
separate models from TimeMixer++ and are named accordingly.

```bash
bash scripts/clone_baselines.sh

python scripts/run_baseline.py \
  --config configs/baselines/timemixer_long_term.yaml \
  --dry-run
python scripts/run_baseline.py \
  --config configs/baselines/itransformer_long_term.yaml \
  --dry-run
```

Remove `--dry-run` to execute. The manifests pin an exact external commit, use the
same 336-step input length and four forecast horizons as CONCORD, and request three
iterations (`itr=3`). The upstream entry points seed once, to 2021 and 2023
respectively, before that loop. These optional wrappers therefore do not
implement the author-confirmed baseline seed list 7/50/81; their results remain
a separate provenance class. The runner verifies the checked-out commit and writes every expanded command,
the expected revision, and the actual revision to
`runs/baselines/<model>/commands.json`.
