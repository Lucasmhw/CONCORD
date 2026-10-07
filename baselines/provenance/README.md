# Baseline source and random-seed audit

Audit date: 2026-10-07. This record covers all 20 named baselines in manuscript
Tables 1 and 2. The authors confirmed that the reported baseline results were
rerun and checked. This documentation audit does not change any manuscript
table value or the horizon tables in Appendix A, and does not independently
rerun the full benchmark.

Three sources have different roles:

1. **Reported experiments:** author-run baseline and CONCORD results in Tables 1
   and 2. Their displayed dispersion belongs to those runs, where reported.
2. **Model provenance:** the original method papers and official implementations
   below. Frozen commits identify what was inspected for this audit; they are
   not asserted to be the historical commits used for every author-run result.
3. **Public cross-check references:** the three existing TimeMixer++ CSVs and
   Appendix A's published horizon entries. They are transcriptions, not an
   archive of the author reruns. A matching mean does not prove a matching seed,
   run protocol or standard deviation.

The authors explicitly confirmed a common baseline rerun seed list **7, 50, 81**
(three training runs). This author-confirmed protocol is recorded separately
from **upstream code defaults** below. It is not inferred from public benchmark
tables and is not an independent execution audit of the author runs. Hard-coded
upstream seeds must be overridden in an adapted rerun implementation; merely
setting an environment variable or requesting `itr=3` is not equivalent.
Where upstream run records are absent, only public benchmark tables are
available for independent numerical cross-checking. No per-run metric is
reconstructed from a mean or SD.

The machine-readable companion is [baseline_protocols.json](baseline_protocols.json).
It records full commits, file SHA-256 digests, line-specific evidence links,
the inspected file inventory, task scope and the distinction between upstream
defaults and reported rerun identifiers.

## Method-by-method findings

### TimeMixer++

- Method: [original paper](https://arxiv.org/abs/2410.16032v5); [official repository](https://github.com/kwuking/TimeMixer).
- Audited version: [`e24610583b36fdd8c76cc17a8df4e65759a5f460`](https://github.com/kwuking/TimeMixer/tree/e24610583b36fdd8c76cc17a8df4e65759a5f460).
- Upstream seed behavior: No independently identifiable TimeMixer++ entry point at the paper-linked repository snapshot; no numeric seed list recovered for TimeMixer++.
- Repetition scope: Paper Appendices A/G state three repetitions. This does not identify individual seeds or establish the protocol of every inherited baseline row.
- Code evidence: [README.md:1-5](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/README.md#L1-L5), [run.py:1-16](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/run.py#L1-L16).
- Source boundary: The paper-linked repository implements TimeMixer. Its fixed seed 2021 must not be assigned to TimeMixer++ or treated as the code that generated the reported TimeMixer++ reruns. Only the public benchmark tables were available for independent numerical cross-checking of TimeMixer++; the author-side adapted implementation/run ledger was not supplied to this audit.

### TimeMixer

- Method: [original paper](https://openreview.net/forum?id=7oLshfEIC2); [official repository](https://github.com/kwuking/TimeMixer).
- Audited version: [`e24610583b36fdd8c76cc17a8df4e65759a5f460`](https://github.com/kwuking/TimeMixer/tree/e24610583b36fdd8c76cc17a8df4e65759a5f460).
- Upstream seed behavior: Python random, NumPy and PyTorch initialized once to 2021 before the repetition loop.
- Repetition scope: run.py default itr=1; the optional CONCORD wrapper requests itr=3, without assigning three distinct explicit seed integers.
- Code evidence: [run.py:10-16](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/run.py#L10-L16), [run.py:83-88](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/run.py#L83-L88), [run.py:135-145](https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/run.py#L135-L145).

### iTransformer

- Method: [original paper](https://arxiv.org/abs/2310.06625); [official repository](https://github.com/thuml/iTransformer).
- Audited version: [`c2426e68ca13f74aaec08045c5c724d8ad328124`](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124).
- Upstream seed behavior: Python random, NumPy and PyTorch initialized once to 2023 before the repetition loop.
- Repetition scope: run.py default itr=1; the optional CONCORD wrapper requests itr=3, without assigning three distinct explicit seed integers.
- Code evidence: [run.py:8-12](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/run.py#L8-L12), [run.py:59-63](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/run.py#L59-L63), [run.py:108-119](https://github.com/thuml/iTransformer/blob/c2426e68ca13f74aaec08045c5c724d8ad328124/run.py#L108-L119).

### PatchTST

- Method: [original paper](https://arxiv.org/abs/2211.14730); [official repository](https://github.com/yuqinie98/PatchTST).
- Audited version: [`204c21efe0b39603ad6e2ca640ef5896646ab1a9`](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9).
- Upstream seed behavior: Supervised entry point: random_seed defaults to 2021 and initializes Python random, NumPy and PyTorch once.
- Repetition scope: Supervised parser default itr=2; dataset launch scripts can override it. No per-iteration seed list is set by this loop.
- Code evidence: [PatchTST_supervised/run_longExp.py:11-13](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/run_longExp.py#L11-L13), [PatchTST_supervised/run_longExp.py:76-78](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/run_longExp.py#L76-L78), [PatchTST_supervised/run_longExp.py:97-101](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/run_longExp.py#L97-L101), [PatchTST_supervised/run_longExp.py:117-124](https://github.com/yuqinie98/PatchTST/blob/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/run_longExp.py#L117-L124).

### Crossformer

- Method: [original paper](https://openreview.net/forum?id=vSVLM2j9eie); [official repository](https://github.com/Thinklab-SJTU/Crossformer).
- Audited version: [`c10c8eadb153d1dd9798250967747ca3ebb81383`](https://github.com/Thinklab-SJTU/Crossformer/tree/c10c8eadb153d1dd9798250967747ca3ebb81383).
- Upstream seed behavior: No explicit RNG seed initialization found in the inspected entry point and released Python source.
- Repetition scope: main_crossformer.py exposes itr (default 1); an iteration number is not a seed identifier.
- Code evidence: [main_crossformer.py:1-40](https://github.com/Thinklab-SJTU/Crossformer/blob/c10c8eadb153d1dd9798250967747ca3ebb81383/main_crossformer.py#L1-L40), [main_crossformer.py:74-89](https://github.com/Thinklab-SJTU/Crossformer/blob/c10c8eadb153d1dd9798250967747ca3ebb81383/main_crossformer.py#L74-L89).
- Source boundary: Original README distinguishes its 12-variable WTH dataset from the 21-variable Weather benchmark. Original-repository settings are not automatically the settings of our reruns.

### TiDE

- Method: [original paper](https://arxiv.org/abs/2304.08424); [official repository](https://github.com/google-research/google-research).
- Audited version: [`e49bbfe381c9c0e564b937f1c4e163a2273c65cc`](https://github.com/google-research/google-research/tree/e49bbfe381c9c0e564b937f1c4e163a2273c65cc).
- Upstream seed behavior: train.py initially seeds NumPy/TensorFlow to 1024, but training() resets both from random_seed, whose default is None. Thus 1024 is not an effective fixed training default.
- Repetition scope: Official scripts omit random_seed. Original paper Table 2 uses five independent runs; Table 5 reports standard errors for those runs, not standard deviations.
- Code evidence: [tide/train.py:63-65](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/tide/train.py#L63-L65), [tide/train.py:116-117](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/tide/train.py#L116-L117), [tide/train.py:130-133](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/tide/train.py#L130-L133), [tide/scripts/electricity.sh:17-52](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/tide/scripts/electricity.sh#L17-L52).
- Source boundary: The original release is TensorFlow. A Time-Series Library adaptation and author-run results need their own implementation/configuration records.

### TimesNet

- Method: [original paper](https://arxiv.org/abs/2210.02186); [official repository](https://github.com/thuml/Time-Series-Library).
- Audited version: [`4e938a1767106324dd753b2a44832bf870a0252e`](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e).
- Upstream seed behavior: Time-Series Library run.py initializes Python random, NumPy and PyTorch to 2021. Separate augmentation seed defaults to 2; it is not the global training seed.
- Repetition scope: Current parser default itr=1. Original TimesNet paper Appendix A states three repetitions, without a list of seed identifiers.
- Code evidence: [run.py:9-13](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/run.py#L9-L13), [run.py:87-90](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/run.py#L87-L90), [run.py:116-119](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/run.py#L116-L119), [utils/augmentation.py:332-340](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/utils/augmentation.py#L332-L340), [run.py:202-208](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/run.py#L202-L208).

### DLinear

- Method: [original paper](https://arxiv.org/abs/2205.13504); [official repository](https://github.com/cure-lab/LTSF-Linear).
- Audited version: [`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6).
- Upstream seed behavior: run_longExp.py initializes Python random, NumPy and PyTorch once to 2021.
- Repetition scope: Parser default itr=2; repetition loop does not assign separate explicit seeds.
- Code evidence: [run_longExp.py:8-11](https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/run_longExp.py#L8-L11), [run_longExp.py:64-66](https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/run_longExp.py#L64-L66), [run_longExp.py:96-105](https://github.com/cure-lab/LTSF-Linear/blob/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6/run_longExp.py#L96-L105).

### SCINet

- Method: [original paper](https://arxiv.org/abs/2106.09305); [official repository](https://github.com/cure-lab/SCINet).
- Audited version: [`02e6b0af2d58243de09aaa1eac3840237b659847`](https://github.com/cure-lab/SCINet/tree/02e6b0af2d58243de09aaa1eac3840237b659847).
- Upstream seed behavior: ETT and PEMS entry points set PyTorch CPU/CUDA seeds to 4321 and deterministic cuDNN; no Python/NumPy seed initialization found in these entry points.
- Repetition scope: No multi-seed list recovered from the inspected ETT/PEMS entry points.
- Code evidence: [run_ETTh.py:102-106](https://github.com/cure-lab/SCINet/blob/02e6b0af2d58243de09aaa1eac3840237b659847/run_ETTh.py#L102-L106), [run_pems.py:69-77](https://github.com/cure-lab/SCINet/blob/02e6b0af2d58243de09aaa1eac3840237b659847/run_pems.py#L69-L77).

### MICN

- Method: [original paper](https://openreview.net/forum?id=zt53IDUR1U); [official repository](https://github.com/wanghq21/MICN).
- Audited version: [`370c69b841d72246556ca05dd23163c560c22b5a`](https://github.com/wanghq21/MICN/tree/370c69b841d72246556ca05dd23163c560c22b5a).
- Upstream seed behavior: setup_seed(2021) initializes Python random, NumPy, PyTorch CPU and CUDA once.
- Repetition scope: Parser default itr=1; repetition loop does not assign separate explicit seeds.
- Code evidence: [run.py:7-14](https://github.com/wanghq21/MICN/blob/370c69b841d72246556ca05dd23163c560c22b5a/run.py#L7-L14), [run.py:51-54](https://github.com/wanghq21/MICN/blob/370c69b841d72246556ca05dd23163c560c22b5a/run.py#L51-L54), [run.py:111-119](https://github.com/wanghq21/MICN/blob/370c69b841d72246556ca05dd23163c560c22b5a/run.py#L111-L119).

### FEDformer

- Method: [original paper](https://proceedings.mlr.press/v162/zhou22g.html); [official repository](https://github.com/MAZiqing/FEDformer).
- Audited version: [`c0f6b972def125691434d62be1ecadf710ae921a`](https://github.com/MAZiqing/FEDformer/tree/c0f6b972def125691434d62be1ecadf710ae921a).
- Upstream seed behavior: run.py initializes Python random, NumPy and PyTorch once to 2021.
- Repetition scope: Parser default itr=3; repetition loop does not assign separate explicit seeds.
- Code evidence: [run.py:10-14](https://github.com/MAZiqing/FEDformer/blob/c0f6b972def125691434d62be1ecadf710ae921a/run.py#L10-L14), [run.py:76-79](https://github.com/MAZiqing/FEDformer/blob/c0f6b972def125691434d62be1ecadf710ae921a/run.py#L76-L79), [run.py:109-117](https://github.com/MAZiqing/FEDformer/blob/c0f6b972def125691434d62be1ecadf710ae921a/run.py#L109-L117).

### Autoformer

- Method: [original paper](https://arxiv.org/abs/2106.13008); [official repository](https://github.com/thuml/Autoformer).
- Audited version: [`51c7d416ae120b805fd5beef2f4ccf7de496a6ff`](https://github.com/thuml/Autoformer/tree/51c7d416ae120b805fd5beef2f4ccf7de496a6ff).
- Upstream seed behavior: run.py initializes Python random, NumPy and PyTorch once to 2021.
- Repetition scope: Parser default itr=2; repetition loop does not assign separate explicit seeds.
- Code evidence: [run.py:9-13](https://github.com/thuml/Autoformer/blob/51c7d416ae120b805fd5beef2f4ccf7de496a6ff/run.py#L9-L13), [run.py:62-66](https://github.com/thuml/Autoformer/blob/51c7d416ae120b805fd5beef2f4ccf7de496a6ff/run.py#L62-L66), [run.py:95-103](https://github.com/thuml/Autoformer/blob/51c7d416ae120b805fd5beef2f4ccf7de496a6ff/run.py#L95-L103).

### DCRNN

- Method: [original paper](https://arxiv.org/abs/1707.01926); [official repository](https://github.com/liyaguang/DCRNN).
- Audited version: [`602afd9d767d3aa1c9b3eac51710d6aeee12c227`](https://github.com/liyaguang/DCRNN/tree/602afd9d767d3aa1c9b3eac51710d6aeee12c227).
- Upstream seed behavior: No explicit training RNG seed initialization found in the TensorFlow entry point, supervisor, utilities or distributed YAML configurations.
- Repetition scope: No identifiable multi-seed list for the reported four-PEMS aggregate in the inspected release.
- Code evidence: [dcrnn_train.py:1-36](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/dcrnn_train.py#L1-L36), [lib/utils.py:30-34](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/lib/utils.py#L30-L34).
- Source boundary: Original official benchmarks are METR-LA and PEMS-BAY; they do not by themselves establish a PEMS03/04/07/08 rerun protocol.

### STGCN

- Method: [original paper](https://www.ijcai.org/proceedings/2018/0505.pdf); [official repository](https://github.com/VeritasYin/STGCN_IJCAI-18).
- Audited version: [`4ab75ab69af354d6a0ab86aefa082d389b187127`](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/4ab75ab69af354d6a0ab86aefa082d389b187127).
- Upstream seed behavior: No explicit training RNG seed initialization found in main.py, trainer and data utilities of the original TensorFlow release.
- Repetition scope: No multi-seed list recovered in the inspected release.
- Code evidence: [main.py:25-43](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/4ab75ab69af354d6a0ab86aefa082d389b187127/main.py#L25-L43), [data_loader/data_utils.py:99-107](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/4ab75ab69af354d6a0ab86aefa082d389b187127/data_loader/data_utils.py#L99-L107).
- Source boundary: The original PeMSD7(M/L) evaluation does not identify the configuration of a PEMS03/04/07/08 aggregate.

### STFGNN

- Method: [original paper](https://arxiv.org/abs/2012.09641); [official repository](https://github.com/MengzhangLI/STFGNN).
- Audited version: [`a02feee56fbb04d6cabb8f04f625948072fa9583`](https://github.com/MengzhangLI/STFGNN/tree/a02feee56fbb04d6cabb8f04f625948072fa9583).
- Upstream seed behavior: No explicit Python/NumPy/MXNet seed initialization found in the entry point, training utilities and PEMS JSON configurations.
- Repetition scope: Original paper says its model was evaluated more than ten times per dataset; explicit seed identifiers and their mapping to our reruns are unavailable.
- Code evidence: [main_4n0_3layer_12T_res.py:1-45](https://github.com/MengzhangLI/STFGNN/blob/a02feee56fbb04d6cabb8f04f625948072fa9583/main_4n0_3layer_12T_res.py#L1-L45), [config/PEMS04/individual_3layer_12T.json:1-27](https://github.com/MengzhangLI/STFGNN/blob/a02feee56fbb04d6cabb8f04f625948072fa9583/config/PEMS04/individual_3layer_12T.json#L1-L27).
- Source boundary: The original paper Table 2 supplies a separate four-PEMS benchmark; it is not substituted for the author-rerun aggregate in our Table 2.

### Graph WaveNet

- Method: [original paper](https://arxiv.org/abs/1906.00121); [official repository](https://github.com/nnzhan/Graph-WaveNet).
- Audited version: [`6b162e80c59a1d494809252eca055cff93dc66b1`](https://github.com/nnzhan/Graph-WaveNet/tree/6b162e80c59a1d494809252eca055cff93dc66b1).
- Upstream seed behavior: The seed=99 argument and PyTorch/NumPy seed calls in train.py are commented out. They are inactive and do not establish a fixed default seed.
- Repetition scope: No active multi-seed loop or seed list recovered from train.py.
- Code evidence: [train.py:28-40](https://github.com/nnzhan/Graph-WaveNet/blob/6b162e80c59a1d494809252eca055cff93dc66b1/train.py#L28-L40).
- Source boundary: Original repository targets METR-LA and PEMS-BAY; the four-PEMS reruns are a distinct evaluation.

### AGCRN

- Method: [original paper](https://arxiv.org/abs/2007.02842); [official repository](https://github.com/LeiBAI/AGCRN).
- Audited version: [`7fbbf2aeb099242098a3cf482b55cd45d7295c28`](https://github.com/LeiBAI/AGCRN/tree/7fbbf2aeb099242098a3cf482b55cd45d7295c28).
- Upstream seed behavior: PEMSD4 config seed=10; PEMSD8 config seed=12. Run.py calls init_seed, which sets Python random, NumPy and PyTorch CPU/CUDA RNGs.
- Repetition scope: These are dataset-specific defaults, not a repeated-run seed list; no corresponding PEMS03/07 seed configurations found.
- Code evidence: [model/PEMSD4_AGCRN.conf:20-24](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/PEMSD4_AGCRN.conf#L20-L24), [model/PEMSD8_AGCRN.conf:20-24](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/PEMSD8_AGCRN.conf#L20-L24), [model/Run.py:69-73](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/Run.py#L69-L73), [model/Run.py:91-95](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/model/Run.py#L91-L95), [lib/TrainInits.py:5-14](https://github.com/LeiBAI/AGCRN/blob/7fbbf2aeb099242098a3cf482b55cd45d7295c28/lib/TrainInits.py#L5-L14).

### BRITS

- Method: [original paper](https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html); [official repository](https://github.com/caow13/BRITS).
- Audited version: [`fc0a3a472a6d99a6934e471d40c25ed0b029b501`](https://github.com/caow13/BRITS/tree/fc0a3a472a6d99a6934e471d40c25ed0b029b501).
- Upstream seed behavior: No explicit RNG seed initialization found in original main.py, data_loader.py or input_process.py; validation sampling and artificial masking use NumPy randomness.
- Repetition scope: No explicit repeated-training seed list recovered from the original release.
- Code evidence: [main.py:22-32](https://github.com/caow13/BRITS/blob/fc0a3a472a6d99a6934e471d40c25ed0b029b501/main.py#L22-L32), [data_loader.py:10-23](https://github.com/caow13/BRITS/blob/fc0a3a472a6d99a6934e471d40c25ed0b029b501/data_loader.py#L10-L23), [input_process.py:113-121](https://github.com/caow13/BRITS/blob/fc0a3a472a6d99a6934e471d40c25ed0b029b501/input_process.py#L113-L121).
- Source boundary: The SAITS repository also contains a BRITS implementation; its seed 26 must not be attributed to original BRITS without identifying which implementation was run.

### SAITS

- Method: [original paper](https://arxiv.org/abs/2202.08516); [official repository](https://github.com/WenjieDu/SAITS).
- Audited version: [`660b87f19c1277065f314f24134f646229e89ca9`](https://github.com/WenjieDu/SAITS/tree/660b87f19c1277065f314f24134f646229e89ca9).
- Upstream seed behavior: Global_Config.py defines RANDOM_SEED=26; run_models.py seeds NumPy and PyTorch with it.
- Repetition scope: This default is not a list of repeated-training seeds. The original paper Table 5 reports five independent runs for downstream classification, which must not be transferred to imputation runs.
- Code evidence: [Global_Config.py:28-28](https://github.com/WenjieDu/SAITS/blob/660b87f19c1277065f314f24134f646229e89ca9/Global_Config.py#L28-L28), [run_models.py:43-43](https://github.com/WenjieDu/SAITS/blob/660b87f19c1277065f314f24134f646229e89ca9/run_models.py#L43-L43), [run_models.py:60-61](https://github.com/WenjieDu/SAITS/blob/660b87f19c1277065f314f24134f646229e89ca9/run_models.py#L60-L61).

### CSDI

- Method: [original paper](https://arxiv.org/abs/2107.03502); [official repository](https://github.com/ermongroup/CSDI).
- Audited version: [`7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`](https://github.com/ermongroup/CSDI/tree/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27).
- Upstream seed behavior: Physio entry point seed defaults to 1 and is passed to dataset construction: NumPy uses it for missing targets and fold/split permutations. No torch.manual_seed call found in the inspected release. The forecasting parser declares seed=1 but does not use it in the entry point.
- Repetition scope: Paper imputation experiments run five times; forecasting experiments run three times with different seeds. The code exposes five-fold evaluation and nsample (diffusion draws); neither is automatically a count of independent training seeds.
- Code evidence: [exe_physio.py:12-22](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/exe_physio.py#L12-L22), [exe_physio.py:43-48](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/exe_physio.py#L43-L48), [dataset_physio.py:74-82](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/dataset_physio.py#L74-L82), [dataset_physio.py:144-163](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/dataset_physio.py#L144-L163), [exe_forecasting.py:12-19](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/exe_forecasting.py#L12-L19), [exe_forecasting.py:41-49](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/exe_forecasting.py#L41-L49).

## Public benchmark scope

The [TimeMixer++ versioned paper](https://arxiv.org/html/2410.16032v5)
provides summary means in Tables 1/3/4 and LTSF horizon entries in Table 16.
Its Appendix A distinguishes inherited TimesNet results from Time-Series Library
reproductions. Appendices A/G state three repetitions, without the seed integers.
Table 13 provides uncertainty only for TimeMixer++ and iTransformer on selected
LTSF datasets; Table 14 covers TimeMixer++ and TimeMixer separately for each PEMS
dataset. These do not supply every baseline SD, an ETT-family SD or a four-PEMS
aggregate SD. No cell-level imputation SD is supplied in Table 4. The paper's
LTSF lookback is 96; its imputation reference uses length-1024 random-mask
reconstruction. Those public settings are distinct from an author rerun's
configuration and from CONCORD's causal imputation information set.

For TimeMixer++, a separate runnable MRTI/TID/MRM implementation was not found
at the linked TimeMixer snapshot. Only published benchmark tables were available
for an independent numerical cross-check. This does not contradict the authors'
confirmation of their own reruns; their adapted implementation is a separate
provenance item not recovered from that upstream snapshot.

The task-specialized models are not included in those TimeMixer++ tables.
Their original papers/repositories establish method provenance. The
[STFGNN paper, Table 2](https://arxiv.org/abs/2012.09641) provides a separate
PEMS03/04/07/08 benchmark for several graph models. Its per-dataset entries are
not the author-run aggregate in CONCORD Table 2. DCRNN/Graph WaveNet originally
evaluate METR-LA and PEMS-BAY, STGCN PeMSD7, and AGCRN PEMS04/08; these scopes must
not be conflated merely because they are all traffic datasets. Original BRITS,
SAITS and CSDI papers similarly do not identify the ETT/ECL/Weather rerun ledger
in CONCORD Table 2. Numerical checking beyond the available public tables needs
the corresponding author-run records.

## What standard deviations can be compared

- A run SD describes variability under that model's particular training,
  checkpoint selection and evaluation protocol. An SD, a standard error of a
  mean and a confidence interval are different quantities. TiDE's original
  Table 5 reports standard errors over five independent runs; they must not be
  copied as SDs. Conversion by `SD = SE * sqrt(n)` is valid only when the reported
  SE is defined that way for the same independent runs and metric aggregation.
- Comparable dispersions require the same evaluated dataset/split, prediction
  horizons or mask ratios, scaling/metric definition, information available to
  the model, and compatible run counts and aggregation. Stochastic imputation
  also requires distinguishing variability across trained models from variability
  across masks, folds or conditional draws from one trained model.
- Identical numeric seeds are neither necessary nor sufficient for comparable
  marginal SDs. Different implementations consume RNG states differently. Using
  the same seed integer does not by itself create statistically paired runs.
- For metrics `m[s,j]` across run `s` and setting `j`, first form
  `a[s] = mean_j m[s,j]`, then compute the mean and SD of `a[s]` across runs.
  The variance of this average is `sum_{j,k} Cov_s(m[s,j],m[s,k]) / J^2`.
  Averaging per-horizon, per-dataset or per-mask-ratio SDs is not generally the
  SD of the aggregate. Public marginal SDs cannot recover the covariances.
- Table highlighting orders displayed means. SD overlap or separation alone
  is not a test of the difference between models. No paired significance or
  identical stochastic protocol is inferred from the supplied summaries.
- The original entry points of TimeMixer, iTransformer, PatchTST, DLinear,
  MICN, FEDformer and Autoformer seed once before their `itr` loop. The loop
  consumes successive RNG states; `itr=3` is not the explicit seed list
  `{seed, seed+1, seed+2}`. Restarting the program resets its initial seed.
  CONCORD's two optional wrappers retain this upstream behavior.

## Mapping a rerun to its result

An independently traceable rerun record identifies the exact implementation
commit/adaptation, command and resolved configuration; all training, mask,
data-loader and split seeds (or the actual RNG-state/repetition policy); dataset
version/splits, normalization, lookback/horizon or imputation masking/information
set; checkpoint selection; individual run metrics and the aggregation formula
(including SD denominator). No such record is reconstructed from a table mean.
The JSON records author-confirmed seeds `[7, 50, 81]` for every displayed baseline
family, separately from the versioned upstream seed findings. The audit does not
assert that the public upstream snapshots, without the author-side adaptations,
automatically reproduce the manuscript tables.
