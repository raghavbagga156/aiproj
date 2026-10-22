# Prediction collapse: Part A/B analysis and the proposed controlled experiment

**Status (2026-10-19, revision 2 after review):** analysis only. Nothing in this document has been run on a GPU. The proposed experiment, notebook 07, waits for approval. It is off by default.

**Inputs read**

| File | Source | Notes |
|---|---|---|
| `06_step_diagnostic.json` (Part B, 207 kB) | user's Downloads folder | Read as data; not modified |
| `05_smoke.json` | user's Downloads folder | Read as data; not modified |
| `STAGE_REPORT.md` | user's Downloads folder | Read as data; not modified |
| `06_collapse_inspection__*.json` (Part A) | **not available to me** | The Part A statements below are as you reported them. I could not check them independently. Their consequences are corroborated by `05_smoke.json` (§1.1). |

## 1. Verification of the reported findings

### 1.1 Part A (reported) and `05_smoke.json` (verified)

| Smoke epoch | Train loss | Grad norm mean / max | Non-finite norms | Grad scale | Val loss | Val avg Spearman |
|---|---|---|---|---|---|---|
| 1 | 0.0193185 | 0.0044485 / **0.9923** | 0 | 65536 | 0.0112228689005456 | 0.0 |
| 2 | 0.0192293 | 0.0 / 0.0 | 0 | 65536 | 0.0112228689005456 | 0.0 |
| 3 (resumed) | 0.0192293 | 0.0 / 0.0 | 0 | 65536 | 0.0112228689005456 | 0.0 |

**Readings:**
- **The validation loss is identical to 16 digits after epochs 1, 2 and 3.** The validation predictions were therefore already constant (collapsed) at the **end of epoch 1**. This agrees with Part A as you reported it: all raw head outputs ≤ 0, all predictions 0, and training loss equal to the zero-predictor loss.
- **No GradScaler event.** The scale stayed at its initial 65536, and there were no non-finite norms or skipped steps.
- **The collapse was probably early in epoch 1. This is an inference, not a measurement.**
  - The epoch-1 norms sum to 0.0044485 × 251 = 1.117. One step alone was 0.992, so the other 250 steps sum to at most 0.125.
  - In Part B, the cumulative norm passes 0.125 after 2 steps in both precisions; the mean per-step norm there is 0.045.
  - If the smoke run's pre-collapse gradients were similar, at most a handful of its 250 other steps had a non-zero gradient.
  - The step log of the proposed experiment measures this directly.

### 1.2 Part B: your summary checked against the JSON

| Claim | JSON | ✓ |
|---|---|---|
| 60 FP32 + 60 FP16 optimizer steps | `optimizer_steps_ran` 60 / 60 | ✓ |
| No exactly-zero-gradient steps | `steps_with_exactly_zero_grad` 0 / 0 | ✓ |
| 6 training graphs, 2 validation graphs | train = first sample of each of the 6 training designs; val = `250-` and `255-mgc_matrix_mult_a` | ✓ |
| Validation from one design only | both validation samples are mgc_matrix_mult_a | ✓ |
| FP32 avg Spearman 0.16433 at step 60; peak 0.21915 at step 40 | 0.164334 / 0.219153 | ✓ |
| FP16 avg Spearman 0.20831 at step 60; peak 0.21187 at step 40 | 0.208305 / 0.211869 | ✓ |
| Cell raw outputs ≤ 0 at step 60: ≈ 98.6–98.8 % | 0.98631 (FP32) / 0.98824 (FP16) | ✓ |
| Grid raw outputs ≤ 0 at step 60: ≈ 85.7–85.9 % | 0.85655 / 0.85945 | ✓ |

**Additional facts from the JSON:**
- **Labels match the smoke run:** c_max = 3.465736, identical to the official-split c_max in `STAGE_REPORT.md` §6.
- **Scaler:** the GradScaler scale was 65536 at every FP16 step, with 0 skipped and 0 non-finite steps.
- **Gradient norms:** the per-step maximum was 0.137 (FP32) and 0.187 (FP16), against **0.992** in smoke epoch 1. No step was clipped.
- **Part B nearly collapsed too.** The grid head reached 99.67 % ≤ 0 at FP32 step 8, and 99.60 % at FP16 step 51. The cell head was at ≥ 99.99 % ≤ 0 (printed 1.0000) at steps 10 and 15 in both precisions. The Part B runs were not robustly healthy; they stayed just short of the absorbing state.
- **FP32 and FP16 are almost the same trajectory early on:**

| Step | FP32 train loss | FP16 train loss | FP32 grad norm | FP16 grad norm |
|---|---|---|---|---|
| 1 | 0.01998 | 0.01998 | 1.39e-2 | 1.46e-2 |
| 2 | 0.02441 | 0.02449 | 1.23e-1 | 1.27e-1 |

  They diverge gradually afterwards, as rounding differences compound.
- **At step 0 the validation predictions were constant for the grid head** (100 % zeros) and nearly so for the cell head. The official metric reports 0.0 for the grid head, and the independent implementation reports NaN.

## 2. Part B loop compared with the normal training engine

Part B is `tools/diagnose_training.py`. The normal engine is `Trainer.run`, as used by the notebook-05 smoke test.

| Aspect | Part B | Smoke (normal engine) | Meaningful difference? |
|---|---|---|---|
| Step code | `Trainer._train_epoch` called with a 1-graph batch list | `Trainer._train_epoch` over the training split | **No.** Same function. The Probe instrumentation was proven bit-identical (new test `test_part_b_instrumentation_does_not_change_the_step`). |
| Training graphs | **6**: first sample (sorted by name) of each training design | **251**: all of them | **Yes, main difference.** Size *and* composition. |
| Sample order | `randperm(6)` with Generator(42), reshuffled with seed 42 + step after each pass | `randperm(251)` from `shuffle_gen` seeded 42 | **Yes.** A different first-step sequence. |
| Optimizer steps | 60 | 251 per epoch | Partly. The smoke collapse was inside epoch 1, probably early (§1.1), so both cover the critical window. |
| Labels / c_max / features | per_dataset, c_max 3.465736 (full training split), paper_eq3 | same | No |
| Model initialization | `_init_state`, seed 42 | `_init_state`, seed 42 | No: identical initial weights (same code and seed) |
| Dropout RNG | seed 42, consumed over the Part B graph sequence | seed 42, consumed over the smoke sequence | Yes, but only as a consequence of the order. Different masks after step 1. |
| Loss | official `verihgn_loss`, α 10, λ 0.5 | same | No |
| Optimizer / clipping | AdamW lr 5e-4, wd 1e-4; clip 1.0 | same | No |
| Scheduler | StepLR never stepped | stepped once per epoch; lr stays 5e-4 until epoch 50 | No (lr 5e-4 throughout both) |
| Precision | FP32 and FP16 (separate runs) | FP16 | Not isolated. No FP16 event in either run (§1). |
| GradScaler | init 65536; stayed 65536 | init 65536; 65536 at each epoch end | No skips in either |
| Gradient checkpointing, TF32 | on, off | on, off | No |
| Data placement | re-collated per step, `.to(cuda)` | lazy collation per step (`dataset_resident_on_gpu: false`) | No: same tensors |
| Evaluation during training | FP32 eval-mode forward on the training graph before and after **every** step; 2-graph validation every 20 steps | FP32 validation on 45 graphs after each epoch | No effect on training (no_grad, eval mode, no RNG use). Yes for the *reported* validation numbers: 2 samples of 1 design. |
| Train metrics | computed (1 graph per call) | computed (pooled per epoch) | No (observation only) |
| Checkpoint / resume | none | last/best each epoch; resumed at epoch 3 | No: the collapse happened before any checkpoint was written |
| Early stopping | n/a | patience 20, not reached | No |
| Software / hardware | A100-SXM4-40GB, torch 2.11.0+cu130 | same (`05_smoke.json` env); torch_scatter 2.1.2 (atomic, not bit-deterministic) | No. GPU non-determinism affects both. |

**Conclusion of the comparison:** the step mathematics, labels, initialization and optimizer are the same. The **training-set size, its composition and the resulting sample/dropout sequence** are the only substantive differences. Precision differs between the smoke run (FP16) and Part B's FP32 run, but not between the smoke run and Part B's FP16 run.

## 3. Root-cause analysis

**Confirmed (from earlier work, `SMOKE_TRAINING_DIAGNOSIS.md`):**
- The absorbing state is "every raw output of both official heads ≤ 0". The official `clamp(min=0)` then makes the gradient exactly 0.
- Our training step is bit-identical to official `train.py` in FP32.
- The official metrics report 0.0 for constant predictions.

**Established by Part A/B and `05_smoke.json`:**
1. The smoke run reached that state **during epoch 1**, probably within its first few steps (§1.1).
2. **Part B started from the identical initialization and nearly collapsed** (99.67 % / 99.60 % grid ≤ 0), but recovered, in both precisions.
3. **The initialization sits at the edge of the absorbing state.** At step 1, 97.65 % of cell and 99.21 % of grid raw outputs are already ≤ 0 on the first Part B graph.
4. **The smoke run had one step with gradient norm 0.992,** 5–7 × larger than any Part B step. Its position in the epoch, and whether it came before or after the collapse, is **unknown**.

**Not established: why one run tipped over and the other did not.** The data are consistent with collapse being **a trajectory-dependent tipping event from an edge-of-collapse initialization**. Which way a run goes would then depend on the early sample sequence, the composition of the training set, and GPU non-determinism. They are equally consistent with a **systematic effect of training-set size or diversity**. One run of each kind cannot separate the two. FP16 is **not implicated** by any observed event, but it is not excluded either: the smoke run was FP16, and Part B's FP16 run used a different data sequence.

The reconstructed smoke order is: fft_a, edit_dist_a, fft_a, des_perf_a, matrix_mult_b, … This assumes `torch.randperm` behaves the same in torch 2.14 locally and 2.11 on Colab, which is **unverified**. Part B began with des_perf_a, matrix_mult_b, fft_a, pci_bridge32_a, …. The experiment's step log records the real order.

## 4. Hypotheses and how the experiment decides them

With 3 seeds per arm, only qualitative differences can be decided, such as 3 of 3 collapsed against 0 of 3. A split such as 1 of 3 against 2 of 3 is **inconclusive**, and the report must say so rather than pick a side.

| # | Hypothesis | Supported if | Rejected if |
|---|---|---|---|
| H1 | **Stochastic tipping:** collapse is a sequence- and seed-dependent event from the edge-of-collapse initialization | outcomes differ across seeds within the same arm, or within the replicas | every arm is uniform across its seeds (all collapse or none) |
| H2 | **Training-set size** (number of distinct graphs) at a fixed step budget | collapse count rises with size: FULL ≥ DIVERSE36 ≥ DIVERSE6, with a clear gap | DIVERSE6 collapses as often as FULL |
| H3 | **Design diversity** at a fixed size (6 graphs) | DIVERSE6 and SINGLE6 differ clearly (e.g. 3/3 vs 0/3) | similar counts |
| H4 | **Precision (FP16 vs FP32)** on identical data and order | REPLICA FP16 collapses in clearly more seeds than REPLICA FP32 | similar counts |
| H5 | **Large-gradient trigger or specific samples** | in collapsed runs, the step before the collapse step has a gradient norm far above the run's median (e.g. > 0.5), or the same samples recur just before collapse across seeds | collapses follow ordinary-norm steps on varied samples |
| H6 | **The smoke run is reproducible** | REPLICA FP16 seed 42 collapses in epoch 1 | it does not, which shows GPU non-determinism matters (supports H1) |
| H7 | **Part B's harness changed the training** | — | **already rejected locally:** the instrumented step is bit-identical. The experiment uses the normal engine anyway. |
| H8 | **Part B's learning is specific to matrix_mult_a** (scope of the validation) | non-collapsed runs show near-zero Spearman on most of the 7 validation designs | correlations are positive across designs |

## 5. Proposed controlled experiment (notebook 07, normal engine)

> **Revision 2 (after review).** All counts below are generated from `configs/collapse_experiment_manifest.json`. Notebook 07 recomputes the plan from the processed data and **refuses to run** if any selection, budget or run differs from that file.

### 5.0 Design roles: why "per design" means 6 designs

The official9 scope has **9 designs** and 383 samples. The approved official seed-42 split (D5; `STAGE_REPORT.md` §5) assigns them to three roles:

| Design | Samples | Role | Used by this experiment |
|---|---|---|---|
| mgc_des_perf_a | 45 | train | training pool; 1 sample held out for validation |
| mgc_edit_dist_a | 45 | train | training pool; 1 sample held out for validation |
| mgc_fft_a | 45 | train | training pool; 1 sample held out for validation |
| mgc_fft_b | 43 | test | **never loaded** |
| mgc_matrix_mult_a | 45 | val | validation only (6 samples) |
| mgc_matrix_mult_b | 44 | train | training pool; 1 sample held out for validation |
| mgc_matrix_mult_c | 44 | test | **never loaded** |
| mgc_pci_bridge32_a | 27 | train | training pool; 1 sample held out for validation |
| mgc_pci_bridge32_b | 45 | train | training pool; 1 sample held out for validation |

Every training arm draws only from the **6 training designs**. Hence:
- "one sample per design" = **6 graphs**;
- "6 samples per design" = **36 graphs** (not 54);
- the validation design and the two test designs are never trained on.

Including matrix_mult_a, fft_b or matrix_mult_c in training would break the approved split and leak validation or test data.

### 5.1 Fixed for every run (baseline preserved)

- **Model, loss, optimizer, labels, features:** official model and `verihgn_loss` (α 10, λ 0.5); AdamW lr 5e-4, wd 1e-4; clip 1.0; StepLR (no decay is reached); dropout 0.15; gradient checkpointing on; batch size 1 graph.
- **Variant:** labels per_dataset with **c_max = 3.465736 from all 251 official training samples for every run**, so all arms see identical targets. Notebook 07 asserts that this equals the c_max that passed notebook 04. paper_eq3 grid features.
- **Diagnostic-only settings, identical in all runs and recorded in each run's `config.yaml`:**
  - patience disabled
  - `keep_every: 0`; `last.pt` and `best.pt` are still written and runs resume
  - `step_log: true`
  - `keep_dataset_on_gpu: "false"`, matching the smoke run, which recorded `dataset_resident_on_gpu: false`

### 5.2 Sample-selection rules (deterministic, on names only, no labels or results)

Samples are sorted per design by name, as `list_processed` sorts them; the plan checks this order on Drive. Write `n` for the number of samples in a design, `linspace(0, n-1, k)` for k evenly spaced positions in that sorted list, and the **pool** for a training design's samples minus its held-out one.

1. **Hold-out**, per training design: the middle sample, position `round(linspace(0, n-1, 3))[1]`. It goes to validation and is removed from every arm. That leaves a pool of n − 1.
2. **Validation, main part:** 6 samples of mgc_matrix_mult_a at positions `round(linspace(0, 44, 6))`.
3. **FULL:** every pool sample of every training design, giving 251 − 6 = 245.
4. **DIVERSE6:** the first pool sample of each training design. This is identical to Part B's 6 training graphs.
5. **DIVERSE36:** 6 pool samples per training design, at positions `round(linspace(0, n_pool-1, 6))`. It always contains the DIVERSE6 sample.
6. **SINGLE6:** 6 pool samples of **mgc_fft_a**, at positions `round(linspace(0, 43, 6))`.
   - fft_a is now **pre-registered** rather than chosen from the data at plan time, so the selection is fixed before anything runs.
   - Reason: it is the lower-median grid non-zero fraction among the 6 per-design samples measured in Part B (pci_a 3.0 %, pci_b 6.1 %, **fft_a 10.5 %**, mm_b 12.5 %, des 13.7 %, edit 38.1 %).
   - The plan also reports the lower-median design by pool mean; that is informational only.
7. **REPLICA:** all 251 official training samples, in the same order as the smoke run's graph list (§5.6). These 251 include the 6 samples that rule 1 holds out from the *other* arms, which are validation rows 7–12. The replica validates on rows 1–6 only (§5.7a).

### 5.3 Corrected experiment table (exact)

| Run group | Training graphs | Training designs (graphs each) | Epochs | Optimizer steps | Validation graphs / designs | Precision | Seeds |
|---|---|---|---|---|---|---|---|
| REPLICA | 251 | 6 (des 45, edit 45, fft_a 45, mm_b 44, pci_a 27, pci_b 45) | 1 | **251** | 6 / 1 (matrix_mult_a) | FP16 | 42, 43, 44 |
| REPLICA | 251 | 6 (des 45, edit 45, fft_a 45, mm_b 44, pci_a 27, pci_b 45) | 1 | **251** | 6 / 1 (matrix_mult_a) | FP32 | 42, 43, 44 |
| FULL | 245 | 6 (des 44, edit 44, fft_a 44, mm_b 43, pci_a 26, pci_b 44) | 1 | **245** | 12 / 7 | FP32 | 42, 43, 44 |
| DIVERSE6 | 6 | 6 (1 each) | 41 | **246** | 12 / 7 | FP32 | 42, 43, 44 |
| DIVERSE36 | 36 | 6 (6 each) | 7 | **252** | 12 / 7 | FP32 | 42, 43, 44 |
| SINGLE6 | 6 | 1 (fft_a 6) | 41 | **246** | 12 / 7 | FP32 | 42, 43, 44 |

**Total: 18 runs.** Run order: the replicas first, then FULL, DIVERSE6, DIVERSE36, SINGLE6.

### 5.4 Step budgets and whether they bias the comparisons

| Run | Steps | Difference from FULL (245) |
|---|---|---|
| REPLICA | 251 | +6 (+2.4 %) |
| FULL | 245 | 0 |
| DIVERSE6 / SINGLE6 | 246 | +1 (+0.4 %) |
| DIVERSE36 | 252 | +7 (+2.9 %) |

- **Where the differences come from.** The engine trains in whole epochs. Each arm runs `ceil(245 / n)` epochs: the smallest whole number of passes that reaches FULL's 245 steps. For n = 6 that is 41 epochs = 246 steps; for n = 36 it is 7 epochs = 252 steps. The replicas use the complete 251-graph smoke epoch.
- **How the bias is removed.**
  - The collapse outcome is computed twice: on each run's full length, and on its **first 245 steps only** (`at_common_steps`). The comparisons between arms use the 245-step outcome.
  - Collapse is identified from the per-step log, so it doesn't depend on epoch boundaries.
- **What remains.** The final validation metrics are taken after each run's last epoch, so at 245, 246 or 252 steps; DIVERSE36 has no epoch boundary at 245. These metrics are secondary, and the report shows the step count beside each one.
- **Learning rate.** It is 5e-4 throughout every run. StepLR would first decay at epoch 50, which no run reaches.

### 5.5 What each comparison isolates, and what it does not

| Comparison | Held equal | Varied | Isolates | Remaining confounds (stated in the report) |
|---|---|---|---|---|
| **Size:** FULL (245) vs DIVERSE36 (36) vs DIVERSE6 (6) | same 6 training designs; step budget (245 common); init seed; labels; validation set; FP32 | number of distinct graphs (and so repetitions per graph: 1 vs ≈ 7 vs 41) | effect of training-set size at a fixed number of optimizer steps | **1.** Distinct graphs and repetitions per graph cannot be separated at a fixed step budget. **2.** FULL is unbalanced (pci_bridge32_a 26 of 245 = 10.6 %), while the DIVERSE arms are balanced (16.7 % each). **3.** Different samples give different early order. Seeds partly address this. |
| **Diversity:** DIVERSE6 vs SINGLE6 | 6 graphs; 41 epochs, 246 steps; seed; labels; validation; FP32 | 6 designs vs 1 design (fft_a) | effect of design diversity at a fixed size | **Only one single-design arm**, so "1 design" is confounded with fft_a-specific properties: size and label sparsity. A difference supports "diversity *or* fft_a"; adding single-design arms for other designs would separate them. They aren't proposed now, to keep the experiment small. |
| **Precision:** REPLICA FP16 vs FP32, seeds 42/43/44 | identical 251 graphs, same graph-list order, same seed, so the same shuffle order and initialization | precision only | FP16 autocast + GradScaler vs FP32 on the smoke data | GPU scatter non-determinism. FP16 and FP32 dropout masks match only while the forward passes stay identical. |
| **Reproducibility:** REPLICA FP16 seed 42 vs the original smoke run | everything in §5.6 | nothing intended | whether the smoke collapse reproduces | GPU non-determinism; the validation set (no effect on training) |
| **Seeds within an arm** | everything except the seed (initialization + order + dropout) | seed | the stochastic-tipping hypothesis (H1) | 3 seeds detect only qualitative effects (§4) |

### 5.6 Fidelity of the smoke replica (REPLICA FP16, seed 42)

| Item | Original smoke run | Replica | Same? |
|---|---|---|---|
| Training graphs and their list order | `build_experiment_graphs` → `list_processed` over the 6 training designs, 251 graphs | the same 251 names in the same order (test `test_replica_order_equals_the_smoke_runs_graph_order`; on Drive, the plan checks `list_processed` order = sorted) | ✓ |
| Shuffle order of epoch 1 | `randperm(251)` from `shuffle_gen` seeded 42 | same code, same seed, same list, so the same permutation | ✓ |
| Initialization and RNG | `_init_state`: `seed_everything(42)`, then the model | same | ✓ (GPU non-determinism excepted) |
| Labels and features | per_dataset, c_max 3.465736; paper_eq3 | same (asserted against the 04 report) | ✓ |
| Optimizer, loss, clipping, AMP | AdamW 5e-4 / 1e-4, α 10, λ 0.5, clip 1.0, FP16 + GradScaler(65536) | same (test compares every training field with base + smoke.yaml) | ✓ |
| Dataset placement | lazy per-step collation (`dataset_resident_on_gpu: false`) | forced `keep_dataset_on_gpu: "false"` | ✓ |
| Epoch-1 length | 251 steps | 251 steps | ✓ |
| Total epochs / session limit | 3 epochs, the first session stopping at 2 | 1 epoch | differs; only epoch 1 is compared |
| Patience / periodic checkpoints | 20 / every 10 epochs (neither applies within epoch 1) | off / off | no effect within epoch 1 |
| Step log | not present | on: observation only, bit-identical training (test) | no effect |
| Validation set | 45 matrix_mult_a graphs, after the epoch | 6 matrix_mult_a graphs, after the epoch | no effect on training; the replica's validation numbers are not comparable to the smoke run's |
| Experiment name / run directory | `verihgn_ispd15_smoke…` | `collapsediag_REPLICA…` under `runs/collapse_diag/` | naming only; the smoke run is untouched |
| Code version | epochs 1–2 ran with the bundle before the resume fix | current bundle | training step unchanged (only checkpoint loading, logging and the step log changed) |

### 5.7 Complete validation manifest (12 graphs, 7 designs)

| # | Sample | Design | Role | Why |
|---|---|---|---|---|
| 1 | `250-mgc_matrix_mult_a-util_0.85` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 2 | `279-mgc_matrix_mult_a-util_0.9` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 3 | `301-mgc_matrix_mult_a-util_0.9` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 4 | `329-mgc_matrix_mult_a-util_0.75` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 5 | `362-mgc_matrix_mult_a-util_0.7` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 6 | `383-mgc_matrix_mult_a-util_0.75` | mgc_matrix_mult_a | official validation design | evenly spaced over 45 |
| 7 | `215-mgc_des_perf_a-util_0.75` | mgc_des_perf_a | training design | held-out middle sample (not in any arm) |
| 8 | `204-mgc_edit_dist_a-util_0.7` | mgc_edit_dist_a | training design | held-out middle sample (not in any arm) |
| 9 | `166-mgc_fft_a-util_0.75` | mgc_fft_a | training design | held-out middle sample (not in any arm) |
| 10 | `315-mgc_matrix_mult_b-util_0.8` | mgc_matrix_mult_b | training design | held-out middle sample (not in any arm) |
| 11 | `3-mgc_pci_bridge32_a-util_0.85` | mgc_pci_bridge32_a | training design | held-out middle sample (not in any arm) |
| 12 | `50-mgc_pci_bridge32_b-util_0.9` | mgc_pci_bridge32_b | training design | held-out middle sample (not in any arm) |

### 5.7a Validation sets: replica vs arms (intersections from the JSON manifest)

**"The six held-out samples"** are rows 7–12 above: the middle sample of each of the 6 training designs (des_perf_a, edit_dist_a, fft_a, matrix_mult_b, pci_bridge32_a, pci_bridge32_b). They are held out only from FULL, DIVERSE6, DIVERSE36 and SINGLE6, so that those arms have within-design validation samples.

The replica must reproduce the smoke run, which trained on all 251 official training samples, so it keeps these 6 in training. It therefore cannot validate on them, and validates on rows 1–6 only (6 × matrix_mult_a, the official validation design).

| Training set | ∩ rows 1–6 (matrix_mult_a) | ∩ rows 7–12 (held-out) |
|---|---|---|
| REPLICA (251 = all official training samples) | ∅ | **all 6:** 215-des_perf_a, 204-edit_dist_a, 166-fft_a, 315-matrix_mult_b, 3-pci_bridge32_a, 50-pci_bridge32_b. Hence rows 7–12 are **not** used to validate the replica. |
| FULL (245) | ∅ | ∅ |
| DIVERSE6 (6) | ∅ | ∅ |
| DIVERSE36 (36) | ∅ | ∅ |
| SINGLE6 (6) | ∅ | ∅ |

These were printed from `configs/collapse_experiment_manifest.json` and are asserted by `test_replica_and_arm_validation_intersections`.

**How results are compared:**
- **Primary outcome: collapse, from the training diagnostics in `steps.csv`.** That means exactly-zero-gradient steps, the collapse step and the fraction of raw head outputs ≤ 0, evaluated at the common 245-step budget. It does not depend on any validation set, so it is comparable across all 18 runs, replicas included.
- **Secondary: validation metrics.** These are compared only between runs with the same validation set:
  - the replicas with each other (rows 1–6; FP16 vs FP32);
  - the four arms with each other (rows 1–12).
- Replica validation numbers are never compared with arm validation numbers, nor with the smoke run's 45-sample validation.

The report carries `n_val` for every run and states this rule in `by_arm._note`.

### 5.8 Complete training selections

| Arm | Samples |
|---|---|
| DIVERSE6 (6) | `108-mgc_des_perf_a-util_0.9`, `100-mgc_edit_dist_a-util_0.7`, `101-mgc_fft_a-util_0.85`, `240-mgc_matrix_mult_b-util_0.85`, `10-mgc_pci_bridge32_a-util_0.7`, `1-mgc_pci_bridge32_b-util_0.9` |
| DIVERSE36 (36) | `108-mgc_des_perf_a-util_0.9`, `181-mgc_des_perf_a-util_0.85`, `199-mgc_des_perf_a-util_0.7`, `229-mgc_des_perf_a-util_0.8`, `239-mgc_des_perf_a-util_0.75`, `257-mgc_des_perf_a-util_0.7`, `100-mgc_edit_dist_a-util_0.7`, `165-mgc_edit_dist_a-util_0.7`, `197-mgc_edit_dist_a-util_0.9`, `211-mgc_edit_dist_a-util_0.9`, `225-mgc_edit_dist_a-util_0.85`, `99-mgc_edit_dist_a-util_0.7`, `101-mgc_fft_a-util_0.85`, `134-mgc_fft_a-util_0.7`, `153-mgc_fft_a-util_0.75`, `178-mgc_fft_a-util_0.75`, `60-mgc_fft_a-util_0.75`, `94-mgc_fft_a-util_0.85`, `240-mgc_matrix_mult_b-util_0.85`, `274-mgc_matrix_mult_b-util_0.7`, `304-mgc_matrix_mult_b-util_0.7`, `325-mgc_matrix_mult_b-util_0.9`, `355-mgc_matrix_mult_b-util_0.75`, `377-mgc_matrix_mult_b-util_0.75`, `10-mgc_pci_bridge32_a-util_0.7`, `15-mgc_pci_bridge32_a-util_0.7`, `27-mgc_pci_bridge32_a-util_0.75`, `311-mgc_pci_bridge32_a-util_0.7`, `4-mgc_pci_bridge32_a-util_0.7`, `9-mgc_pci_bridge32_a-util_0.7`, `1-mgc_pci_bridge32_b-util_0.9`, `25-mgc_pci_bridge32_b-util_0.7`, `42-mgc_pci_bridge32_b-util_0.75`, `58-mgc_pci_bridge32_b-util_0.75`, `78-mgc_pci_bridge32_b-util_0.75`, `97-mgc_pci_bridge32_b-util_0.8` |
| SINGLE6 (6) | `101-mgc_fft_a-util_0.85`, `134-mgc_fft_a-util_0.7`, `153-mgc_fft_a-util_0.75`, `178-mgc_fft_a-util_0.75`, `60-mgc_fft_a-util_0.75`, `94-mgc_fft_a-util_0.85` |
| FULL (245) | all 251 official training samples except the 6 held out above (full list in `configs/collapse_experiment_manifest.json`) |

### 5.9 Leakage checks (all verified by tests on the real names)

- **No validation sample is in any training set:** none of the 12 is in FULL, DIVERSE6, DIVERSE36 or SINGLE6. The held-out samples are excluded from every arm; the replicas validate on matrix_mult_a only.
- **No training sample comes from the validation design or a test design.**
- **The test designs are never used:** no sample of mgc_fft_b or mgc_matrix_mult_c appears anywhere, in training or validation.
- **No duplicates:** no duplicate sample in any training set, and every name is unique.
- `plan_experiment` re-checks all of this on the real data before anything runs.

### 5.10 What each run records

- **`steps.csv`** (observation only), one row per step:
  - sample and loss
  - gradient norm before clipping; whether it was exactly zero; whether it was clipped
  - GradScaler scale before and after, and whether the step was skipped
  - fraction of raw cell and grid outputs ≤ 0, and their maximum
- **`history.csv`**, per epoch: the validation metrics, the zero-gradient count, and the validation prediction std and zero fraction.
- **Run summary in the report:**
  - collapsed (yes/no) and the collapse step, on the full run and **at 245 steps**
  - largest gradient: its step and its sample
  - first 10 samples
  - near-collapse count
  - scaler skips
  - final and best validation average Spearman
  - per-design final validation metrics

**Collapse rule:** the run (or its first 245 steps) ends with ≥ 20 consecutive exactly-zero-gradient steps, or consists entirely of them.

### 5.11 Cost and outputs (estimates, not measurements)

**Measured inputs to the estimate:**
- smoke run, FP16: 53–60 s per 251-step epoch, and 8.5 s to validate 45 graphs
- Part B: 120 steps plus evaluations in 60.7 s

**FP32 assumption:** 1–2 × the FP16 step time.

| Item | Estimate |
|---|---|
| Graph building (257 graphs: 251 training + 6 matrix_mult_a) | 5–10 min |
| 6 replicas | 6–10 min |
| 12 arm runs | 25–45 min |
| **Total A100 time** | **≈ 35–65 min** |
| GPU memory | as in the smoke run |
| Drive | ≈ 0.5 GB |

**Outputs (never overwritten; a timestamp is added if a name is taken):**
- `reports/07_collapse_plan.json`
- `reports/07_collapse_experiment.json`, updated after each run
- `runs/collapse_diag/<run>/`, containing config, env, status, `history.csv`, `steps.csv`, and the last and best checkpoints

## 6. FP32 or FP16 for the next diagnostic

**Recommendation: FP32 for the four arms, and both precisions only in the replica pairs.** This is a choice of experimental control, not a verdict on precision.

**What the current data do and do not show:**
- **No precision winner.** Part B's step-60 difference (FP16 0.208 vs FP32 0.164) is one seed, two validation samples of one design, and a single checkpoint in time. FP32's own peak (0.219 at step 40) is above FP16's peak (0.212). The two trajectories are nearly identical for the first steps, and they separate only through compounding rounding differences.
- **No FP16-specific event** was observed in Part B or the smoke run: the GradScaler scale stayed at 65536, with zero skips and zero non-finite steps.

**Why FP32 for the arms:**
- The question is about data size and diversity, and FP32 removes autocast and GradScaler as variables.
- Evaluation is FP32 in every case.
- The replica pairs still test precision directly, on identical data and order across 3 seeds, and include the exact smoke configuration (FP16, seed 42).
- This choice is a diagnostic control. It is not the recommended training precision; that remains the I-3 benchmark's question.

## 7. What changed in the code (no methodology change)

| File | Change |
|---|---|
| `training/trainer.py` | `_train_epoch` rewritten without changing the step: the same statements in the same order. New opt-in `training.step_log` writes `steps.csv` through read-only head hooks. A resumed epoch replaces its own rows. |
| `config.py` | `TrainCfg.step_log: bool = False` (default off; `run_id` unchanged). |
| `diagnostics/collapse_experiment.py` (new) | `select_samples` (rules on names only), `build_runs` (budgets), `plan_experiment` (checks the real data against the pre-registered manifest), runner, collapse summary (full run and common 245-step budget), aggregation. |
| `configs/official9_sample_names.txt`, `configs/collapse_experiment_manifest.json` (new) | The 383 sample names, from the public CircuitNet archive listing, and the pre-registered selection generated from them by `tools/make_collapse_manifest.py`. |
| `notebooks/07_collapse_experiment.ipynb` (new, generated) | Cell 1 plans and enforces the manifest and the c_max. The run cell is gated by `RUN_COLLAPSE_EXPERIMENT = False`. |

**Fixed during this review** (found by inspecting the code, before any GPU run):

1. **The duplicate-sample check could never fire.** It compared a set with itself. It now checks the name list itself, plus per-run training duplicates.
2. **The manifest comparison would have blocked the real run.** It compared informational keys (`comment`, `split`, `samples_per_design`) as if they were selections, so notebook 07 would have refused to run on correct data. It now compares only the selection and budget keys, and a test uses the real 383 names.
3. **Dataset placement.** It is now forced to lazy per-step collation, as in the smoke run. With `auto`, the smaller graph sets could have become GPU-resident. That is numerically identical, but it is no longer a difference from the smoke run.
4. **SINGLE6's design is pre-registered** (fft_a) instead of computed from labels at plan time.
5. **A common 245-step collapse outcome** removes the 245–252-step budget differences from the collapse comparisons.

The model, clamp, loss, labels, features, optimizer settings and normalization are unchanged.

**Tests** (`tests/test_collapse_experiment.py`, 14):
- **Step log and Part B harness:**
  - the step log leaves training bit-identical
  - the step log survives a resume without duplicated rows
  - the Part B instrumentation leaves the step bit-identical
- **Plan:**
  - the plan follows its fixed rules and is deterministic
  - it rejects any overlap between training and test designs
  - it refuses to differ from the manifest, including a missing key, and ignores informational keys
- **Shipped manifest (real names):**
  - it is exactly what the generator produces from the 383 names
  - exact counts, designs, steps, validation composition, and the leakage checks of §5.9
  - the plan on a stub tree with the real 383 names reproduces it exactly
- **Replica:** it presents the smoke run's graph list in the same order, with the same c_max and the same training configuration as base + smoke.yaml
- **Runner:**
  - it works end to end and detects a forced collapse
  - the collapse rule behaves correctly at its edge cases
  - the common-step-budget rule works
- **Notebook 07:** its own cells run on a synthetic Drive layout: plan only by default, then the gated run, with no overwrite on a second plan

## 8. Test results

- **Full suite (laptop, CPU, with the real-data fixtures): 79 passed, 1 skipped.** The skipped test needs CUDA and runs in notebook 05 cell 4b. The total includes the 15 collapse-experiment tests.
- **Notebook dry run (03–06, 3 real fixture samples, CPU): passed.** The smoke resume check passed, and the step log is off by default.
- **Notebook 07** is covered by `test_notebook_07_cells_run_on_synthetic_layout`, plus the real-name manifest tests. The 3-sample fixture tree cannot support per-design hold-outs.
- **Bundle** rebuilt and verified (see the final summary). No GPU work was done.

## 9. Pre-execution checklist (all must be ✓ before `RUN_COLLAPSE_EXPERIMENT = True`)

**On the laptop, done in this revision:**
- [x] Model, clamp, loss, labels, features, optimizer and normalization are unchanged (diff limited to §7).
- [x] Selections, budgets and runs are pre-registered in `configs/collapse_experiment_manifest.json`, generated from names only.
- [x] No validation or test sample in any training set; no test design anywhere (tests on the real names).
- [x] The replica matches the smoke run's graph order, seed, precision, c_max, training configuration and dataset placement.
- [x] Step budgets are documented, and the collapse comparisons use a common 245-step budget.
- [x] Intersections verified from the manifest (§5.7a): REPLICA ∩ rows 1–6 = ∅; REPLICA ∩ rows 7–12 = the 6 held-out samples (so the replica validates on rows 1–6 only); every arm ∩ all 12 = ∅.
- [x] The primary comparison uses training diagnostics (`steps.csv`). Validation metrics are compared only within the same validation set.
- [x] The step log is proven not to change training.
- [x] Full test suite and notebook dry run pass (§8).
- [x] The bundle is rebuilt and verified; all GPU flags are off; `third_party` is unchanged.

**On Colab, cell 1 only (no training; needs your go-ahead to upload):**
- [ ] Cell 1 completes without a manifest mismatch. A mismatch stops here; send me the message.
- [ ] Printed c_max = 3.465736, equal to the notebook-04 report (asserted).
- [ ] Printed validation list equals §5.7, and the runs table equals §5.3 (18 runs; steps 251 / 245 / 246 / 252 / 246).
- [ ] `07_collapse_plan.json` written.

**Approval:**
- [ ] Your explicit approval to set `RUN_COLLAPSE_EXPERIMENT = True` (≈ 35–65 min A100, estimate).
