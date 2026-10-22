# Smoke-test training diagnosis (notebook 05, A100, 2026-10-19)

**Status:** local investigation complete. The A100-specific parts are **not yet confirmed**: they need notebook 06 Part A, which is read-only. All local numbers below come from CPU/FP32 runs on the 3 real fixture samples and the synthetic test data. None of them is an experimental result.

## 1. What was observed (reported from `05_smoke.json`)

| | Epoch 1 | Epoch 2 | Epoch 3 (resumed) |
|---|---:|---:|---:|
| Training loss | 0.0193185 | 0.0192293 | 0.0192293 |
| Mean gradient norm (before clipping) | 0.0044485 | 0.0000000 | 0.0000000 |
| Validation Spearman / Pearson / Kendall | 0 | 0 | 0 |

No non-finite steps were reported. Configuration: official_seed42 / paper_eq3 / per_dataset / fp16 / seed 42.

## 2. Conclusion

| | Status |
|---|---|
| **The signature is prediction collapse.** Every raw output of both official prediction heads is ≤ 0, so `PredictionHead.forward` (`third_party/VeriHGN/models/verihgn.py:204`, `.clamp(min=0)`) returns exactly 0 for every cell and every tile. From then on the gradient of the whole network is exactly 0 and finite. The loss equals the all-zero predictor's loss and no longer depends on the parameters, so it is identical in every epoch. The official metrics return exactly 0.0 for a constant prediction. | **Mechanism confirmed** (tests and experiments below). It reproduces every logged quantity of the A100 signature. |
| The A100 run is in that state | **Strongly indicated, not yet confirmed.** Notebook 06 Part A checks the raw head outputs of the saved checkpoint, and checks that the epoch-2/3 training loss equals the mean all-zero-predictor loss over the 251 training graphs. |
| Our training loop, loss call, clipping, AdamW, checkpoint/resume or metrics cause it | **Ruled out.** Our FP32 step is bit-identical to official `train.py::train_one_epoch`. The metrics agree with an independent implementation. Resume restored the state exactly (§4). |
| Single-head collapse is specific to FP16 | **Ruled out.** In FP32 on CPU the **cell head** collapses within 2–3 steps for seeds 42 and 43 and for both normalizations. In the 80-step seed-42 run, 100 % of raw cell outputs were ≤ 0 at steps 3–6, 8–11 and 13, and every eval-mode cell prediction was exactly 0 at steps 1–15. |
| The **full** (two-head, absorbing) collapse also happens in FP32 | **Not reproduced locally.** In the 80-step FP32 run (2 training graphs), the grid head stayed partly active (59–99 % of raw outputs ≤ 0). The gradient was never exactly zero (0 of 80 steps), and the cell head partly recovered after step 15. Whether the full collapse needs FP16, the 251-graph training distribution, or both is **open**: notebook 06 Part B measures FP32 against FP16 on the A100. |
| Contributing factor: sparse, small targets | **Supported (measured), but causality is a hypothesis.** The D1 targets (GR overflow) are 78–99 % exactly zero, with means 0.002–0.009 (§5.1). With a ReLU-clamped head, predicting 0 is almost optimal for most nodes, and "all raw outputs ≤ 0" is absorbing. |

### How the official clamped heads collapse

1. **Each head clamps its output.** `PredictionHead` is `Linear(128→64) → ReLU → Linear(64→1) → clamp(min=0)`. Where the raw output is ≤ 0, the prediction is exactly 0 and the local gradient is exactly 0. This uses `torch.clamp`'s backward pass, which zeroes the gradient below the threshold even when the incoming gradient is NaN or infinite, as it is for the std term at std = 0.
2. **Most raw outputs start below 0 (seed 42).** At the official initialization, 96.9 % of the raw cell outputs and 98.7 % of the raw grid outputs are ≤ 0. The eval-mode validation grid prediction is already 100 % zero at step 0.
3. **Training pushes the rest down.** With zero-inflated targets, most still-positive outputs have target 0, so their gradient lowers them. Only the few non-zero targets pull outputs up, and those carry the official weight α = 10. AdamW's step size is about lr per parameter however small the gradient is, so the push is fast: the cell head reaches 100 % ≤ 0 at step 3.
4. **One dead head can recover; two dead heads cannot.** Both heads read from the same message-passing layers. While the grid head still has positive outputs, its gradient keeps changing the shared cell embeddings, and some cell outputs can become positive again (seen locally from step 16). Once **both** heads are entirely ≤ 0, the gradient of every parameter is exactly 0. After that only AdamW weight decay and the decaying Adam momentum move the parameters, and both keep pushing in the direction that killed the heads. Nothing can bring the outputs back above 0: the state is **absorbing**.
5. **Metrics.** A constant prediction makes the official metrics return exactly 0.0, and the training loss equals the all-zero predictor's loss, identical in every epoch, including across the resume.

The A100 smoke run shows step 5 and is consistent with step 4: `grad_norm_mean` is exactly 0.0, which needs both heads dead. Step 4 was reached during epoch 1; its mean gradient norm of 0.0044 is below every single-step norm measured locally (0.013–0.18), which suggests most epoch-1 steps already had zero gradient. Part A checks this.

No defect was found in code that this project wrote. The collapse arises from three things together:

- the **unmodified official model** (clamped heads)
- the **official optimizer settings** (AdamW, lr 5e-4)
- the **approved D1 labels** (sparse GR-overflow maps)

So there was nothing to "fix" without changing methodology. I made no change to the model, the loss, the labels, the optimizer settings or the normalization. **Section 8** lists the options that need your decision.

## 3. Affected code locations

| Location | Role |
|---|---|
| `third_party/VeriHGN/models/verihgn.py:194-204` (`PredictionHead`: Linear → ReLU → Linear → `squeeze` → `clamp(min=0)`) | Outputs ≤ 0 become exactly 0 and have zero gradient. Unmodified official code. |
| `third_party/VeriHGN/utils/loss.py` (`weighted_mse`, `variance_reg`) | At an all-zero prediction the loss is constant. The variance term `relu(std_y − std_ŷ)²` would push the spread up, but its gradient reaches the parameters only through unclamped outputs, of which there are none. |
| `third_party/VeriHGN/utils/evaluation.py:26-55` (`compute_pearson/spearman/kendall`) | `pred.std() < 1e-8 → return 0.0`. A constant prediction is reported as exactly 0 (Pearson, Spearman, Kendall). The correlation is mathematically undefined, not zero. |
| `src/verihgn_repro/training/trainer.py` | **Changed, logging only.** History now records `grad_zero_steps` / `train_steps` and, on validation, `val_{cell,grid}_pred_std` and `val_{cell,grid}_pred_zero_frac`. A warning is printed when every step of an epoch had an exactly-zero gradient. Training mathematics, RNG consumption and resume behaviour are unchanged; the resume tests still pass. |
| `src/verihgn_repro/evaluation/evaluate.py` | New `prediction_health()` for those columns. |

## 4. Answers to the specific questions

1. **Are gradients genuinely zero before `optimizer.step()`?**
   - **Locally, yes, in the collapsed state.** In the synthetic collapse test the gradient of every parameter is exactly 0 (not NaN, not skipped), and `clip_grad_norm_` returns 0.0.
   - **On the A100, consistent with the logs, not yet measured directly.** The history computes `grad_norm_mean` only over finite norms. A run where the GradScaler skipped every step would log NaN, not 0.0. So the steps behind the logged 0.0 had finite, exactly-zero gradients.
2. **Is the GradScaler skipping optimizer steps?**
   - **Not as the cause.** Skipped steps produce infinite or NaN norms, not 0.0.
   - Whether *some* epoch-1 steps were skipped can be read from the `grad_scale` and `grad_norm_nonfinite` columns of `05_smoke.json`, which weren't in the summary I received. Part B records the scale and whether each step ran.
3. **Are model parameters actually changing?**
   - **Yes, slightly.** AdamW applies decoupled weight decay (lr · wd · p) and the Adam momentum from epoch 1 still decays out, both even with zero gradients (`test_parameters_still_move_while_outputs_stay_collapsed`).
   - **The outputs don't change, though:** they stay ≤ 0 → 0, which is why the loss is identical to 7 digits.
   - Before collapse, every step moves each parameter by about lr = 5.0e-4: exactly lr on the first Adam step, and up to 1.8 × lr later as the moment estimates change (measured max |Δp| per step = 5.0e-4 in the first steps, rising to 8.8e-4 by step 80; L2 ‖Δp‖ ≈ 0.09–0.47 over 1,012,753 parameters). Adam's step size doesn't depend on how small the gradient is, so even the tiny gradients from a few non-zero targets produce full-size steps.
4. **Are predictions constant or incorrectly normalized?**
   - **Constant: exactly 0** (pred_std = 0, 100 % zeros for the cell head locally after 2 steps).
   - The targets are normalized as approved (D4). per_dataset divides log1p((H+V)/2) by the training-split c_max, and is identical to the Phase 1 code that passed all gates. Nothing in the pipeline rescales predictions.
5. **Are the correlation metrics zero because of a bug, or because predictions have no variance?**
   - **No variance.** Official `compute_*` returns 0.0 when `pred.std() < 1e-8`.
   - The official metrics agree with an independent implementation (numpy float64, own tie-averaged ranks) to < 1e-5 on non-constant data. That implementation agrees with scipy in float64 to < 1e-9.
   - For a constant prediction, the independent implementation returns NaN (undefined), where the official one returns 0.0. Reporting should say "undefined (constant prediction)" rather than "0 correlation".
6. **FP16-specific, or also FP32?**
   - **The cell-head collapse also happens in FP32** (steps 2–15 locally, both seeds, both normalizations).
   - **The full two-head collapse seen on the A100 did not happen** in 80 FP32 steps on 2 training graphs. There, every step had a non-zero gradient, the training loss was below the all-zero predictor's on 79 of 80 steps, and the validation average Spearman reached 0.20 at step 20 (§5.3).
   - **Open:** whether the full collapse needs FP16, the full 251-graph training split, or both. Part B answers this.
7. **Checkpoint save/load and restored optimizer state?** Epoch 3 resumed from the epoch-2 checkpoint and reproduced the epoch-2 loss to 7 digits. That is exactly what a correctly restored collapsed state gives. The resume regression tests (`tests/test_checkpoint_resume.py`, 17 tests) still pass, so resume didn't cause the collapse.
8. **Does the official implementation behave the same?** Our FP32 step is **bit-identical** to `third_party/VeriHGN/train.py::train_one_epoch`: same init, same graph order, identical losses and parameters after 3 epochs (`test_trainer_step_equals_official_train_one_epoch`). Under these labels the official code collapses in the same way. Whether it collapses on the authors' (unpublished) labels can't be determined.

## 5. Diagnostic experiments (local, CPU, FP32)

Tool: `tools/diagnose_training.py`. It runs the **unchanged** `Trainer._train_epoch` one graph at a time and observes it through hooks: it wraps `clip_grad_norm_` and the official loss, and adds optimizer step hooks and head forward hooks. The data are the 3 real CircuitNet-ISPD15 fixture samples:

- train: fft_a and pci_bridge32_a (1 sample each)
- val: matrix_mult_a
- c_max = 2.7881, taken over these samples only

### 5.1 Targets (real samples)

| Sample | Labels | Cell mean / std / max / non-zero | Grid mean / std / max / non-zero |
|---|---|---|---|
| 68-mgc_fft_a | per_dataset | 0.00196 / 0.0244 / 0.503 / 1.2 % | 0.00593 / 0.0547 / 1.000 / 1.8 % |
| 3-mgc_pci_bridge32_a | per_dataset | 0.00261 / 0.0207 / 0.544 / 8.3 % | 0.00434 / 0.0293 / 0.544 / 7.9 % |
| 380-mgc_matrix_mult_a | per_dataset | 0.00679 / 0.0282 / 0.712 / 22.1 % | 0.00605 / 0.0281 / 0.712 / 12.5 % |
| 380-mgc_matrix_mult_a | per_sample | 0.00953 / 0.0396 / 1.000 / 22.1 % | 0.00850 / 0.0395 / 1.000 / 12.5 % |

The all-zero predictor's loss on the two training graphs is 0.03805 (fft_a) and 0.01373 (pci_bridge32_a).

### 5.2 Mechanism (unit level)

- **Raw outputs all ≤ 0** (FP32 and FP16 head outputs):
  - the gradient is exactly 0 and finite, including through the variance term at std = 0
  - the loss equals the all-zero predictor's loss
  - `test_all_clamped_outputs_give_exactly_zero_finite_gradient`
- **Collapse forced at initialization** (head biases −10, synthetic data, the real Trainer, 3 epochs):
  - `grad_norm_mean` = 0 and `grad_zero_steps` = `train_steps` every epoch
  - identical `train_loss` every epoch, equal to the mean all-zero-predictor loss (difference < 1e-7)
  - all six validation correlations and `val_avg_spearman` exactly 0
  - `skipped_nonfinite` = 0
  - This is the A100 signature exactly (`test_collapse_reproduces_smoke_signature`).

### 5.3 Step-level runs (real samples, FP32, official hyperparameters)

`raw≤0` is the fraction of raw head outputs that the clamp sets to 0, measured in training mode during the step.

| Run | Step 1 (initialization) | Step 2 | Afterwards |
|---|---|---|---|
| per_dataset, seed 42 (**complete, 80 steps**, 1002 s) | cell 0.969, grid 0.987 | cell 0.99997, grid 0.655 | See the table below |
| per_dataset, seed 43 (stopped by me after 14 steps to free the CPU) | cell 0.613, grid 0.627 | cell 1.000*, grid 0.981 | cell 1.000* through step 14; grid 0.999 at step 4, then 0.78–0.93 |
| per_sample, seed 42 (stopped by me after 15 steps to free the CPU) | cell 0.969, grid 0.987 | cell 1.000*, grid 0.655 | cell 1.000* through step 15; grid 0.63–0.88 |

\* The two stopped runs only printed 3 decimals, so 1.000 means ≥ 0.9995. Their JSON was not written.

**Seed-42 run, step blocks** (every optimizer step ran; no step was clipped; no gradient was exactly 0):

| Steps | Cell raw ≤ 0 | Grid raw ≤ 0 | Gradient norm (before = after clipping) | Train loss − all-zero loss | Eval loss fell after the step | Max \|Δp\| | Eval-mode cell predictions = 0 | Grid prediction std |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.969 | 0.987 | 2.0e-2 | −0.0003 | 0 / 1 | 5.0e-4 | 100 % | 0.072 |
| 2–10 | 0.99993–1.000 | 0.655–0.986 | 1.6e-2 – 1.8e-1 | −0.0022 … +0.0018 | 7 / 9 | 5.1e-4 | 100 % | 0.005–0.070 |
| 11–20 | 0.992–1.000 | 0.666–0.840 | 2.1e-2 – 6.7e-2 | −0.0049 … −0.0014 | 10 / 10 | 5.6e-4 | ≥ 98.8 % | 0.033–0.048 |
| 21–40 | 0.964–0.998 | 0.595–0.885 | 2.2e-2 – 1.0e-1 | −0.0207 … −0.0019 | 18 / 20 | 6.9e-4 | ≥ 94.1 % | 0.028–0.065 |
| 41–60 | 0.975–0.993 | 0.724–0.947 | 1.3e-2 – 1.8e-1 | −0.0287 … −0.0033 | 16 / 20 | 7.6e-4 | ≥ 96.8 % | 0.019–0.081 |
| 61–80 | 0.986–0.994 | 0.881–0.960 | 1.5e-2 – 8.9e-2 | −0.0308 … −0.0053 | 14 / 20 | 8.8e-4 | ≥ 98.8 % | 0.026–0.077 |

**Validation (matrix_mult_a, FP32 eval).** "Independent" is the numpy float64 implementation:

| Step | Official avg Spearman | Cell Spearman (official / independent) | Grid Spearman (official / independent) | Grid Pearson (official / independent) | Cell predictions = 0 | Grid predictions = 0 | Val loss |
|---|---|---|---|---|---|---|---|
| 0 | −0.0007 | −0.0014 / −0.0014 | 0.0000 / **NaN** (constant) | 0.0000 / NaN | 99.9993 % (1 cell > 0) | **100 %** | 0.01746 |
| 20 | 0.2024 | 0.0843 / 0.0843 | 0.3204 / 0.3204 | 0.1806 / 0.1806 | 98.8 % | 76.2 % | 0.02654 |
| 40 | 0.2006 | 0.1088 / 0.1088 | 0.2923 / 0.2923 | 0.1494 / 0.1494 | 98.0 % | 81.8 % | 0.02054 |
| 60 | 0.1215 | 0.0799 / 0.0799 | 0.1631 / 0.1631 | 0.1198 / 0.1198 | 99.4 % | 89.3 % | 0.02019 |
| 80 | 0.0900 | 0.0524 / 0.0524 | 0.1276 / 0.1276 | 0.1111 / 0.1111 | 99.7 % | 87.8 % | 0.02386 |

**Readings:**
- The official and independent metrics agree to 4 decimals whenever the prediction varies. When it is constant, the official code reports 0.0 and the independent one reports NaN.
- **Step 0, cell head.** Exactly 1 of the 149,655 validation cells has a positive prediction (0.0061); the zero fraction is 0.9999933. That gives a std of 1.6e-5, which passes the official `1e-8` check, so a near-zero correlation (−0.0014) is computed. Both implementations give it. A head with almost every output clamped therefore reports either exactly 0 or a correlation near 0.
- With only 2 training graphs the model overfits: the validation correlation peaks at step 20, then declines. These are diagnostic numbers, not results.

- **At initialization (seed 42),** the eval-mode grid predictions on the validation sample are already 100 % zero (official Spearman 0.0 at step 0).
- **Per-step quantities, seed-42 run, first 4 steps:**

| Step | Eval loss before → after | Gradient norm before = after clipping | Scaler |
|---|---|---|---|
| 1 | 0.03805 → 0.04066 | 0.0201 | n/a (FP32) |
| 2 | 0.01726 → 0.01309 | 0.183 | n/a (FP32) |
| 3 | 0.03701 → 0.03722 | 0.0160 | n/a (FP32) |
| 4 | 0.01343 → 0.01322 | 0.0255 | n/a (FP32) |

- No step was clipped, and every optimizer step ran.

## 6. Regression tests

`tests/test_training_diagnosis.py` (9 tests):

- independent metrics vs scipy and vs official
- the official constant-prediction convention
- zero gradient at all-clamped outputs
- the collapse signature reproduced end-to-end
- parameters still moving under weight decay
- no false positives from the collapse monitor
- `prediction_health`
- the diagnostic tool's instrumentation (including that it removes its hooks)
- bit-identity with the official `train_one_epoch`

Full suite (laptop CPU, with the real-data fixtures): **64 passed, 1 skipped**; the skipped test is CUDA-only and runs in notebook 05 cell 4b. Notebook dry run of 03–06 on the 3 real fixture samples: **passed**, including notebook 06 Part A and a 2-step CPU version of Part B. The bundle was rebuilt and verified (84 entries; `third_party` at 28a6f1e with no changes). No test was removed or weakened.

## 7. Safe to benchmark?

**No, not yet.** A precision benchmark on a model whose predictions collapse to 0 would compare dead models: every precision would score correlation 0 after the collapse. Any timing or VRAM differences would be measured on degenerate training. The engineering (checkpointing, resume, data pipeline) is fine. What's missing is a methodology decision (§8), which is yours.

## 8. Options that need your decision (none implemented)

| Option | Fidelity | Comment |
|---|---|---|
| A. Keep everything official; report the collapse as a finding under D1 labels | highest | An honest negative result. Training to 200 epochs would very likely stay collapsed, which wastes GPU time. |
| B. Training-time head change as a documented deviation, e.g. clamp only at inference, or softplus. Implemented in our wrapper; `third_party` untouched. | deviation (model) | The smallest change that removes the absorbing state. Run it alongside A as an ablation, never in place of A. |
| C. Revisit D1 (e.g. GR utilization-based maps, which are dense) | deviation (label) | Changes the task. The paper's sparse-congestion description and the WMSE `y > 0` weighting point to sparse labels, which is why D1 was chosen. |
| D. Lower lr or other optimizer changes | deviation (optimizer) | Not recommended. It deviates from the official settings and only delays an absorbing state. |
| per_sample normalization (D4 alternative) | approved variant | **Does not prevent it:** the cell head collapsed at step 2 under per_sample too. |

My recommendation:

1. Confirm the A100 state with notebook 06 Part A.
2. Measure FP32 vs FP16 with Part B.
3. If the collapse is confirmed, approve B as a clearly labelled deviation, run in parallel with the official configuration A.

## 9. Next Colab steps (exact)

1. Upload the rebuilt `dist/verihgn_code.zip` to `MyDrive/VeriHGN/code/`, replacing only the zip. Also upload `notebooks/06_training_diagnosis.ipynb`.
2. Open notebook 06 and run the bootstrap cell, then the setup cell. A CPU runtime is enough for Part A; the A100 is faster. Notebook 02 is not needed.
3. Run **Part A** (read-only, about 5–10 minutes). For each smoke run that has a checkpoint, it writes `reports/06_collapse_inspection__<run>.json` and never overwrites anything. Read two things:
   - `forward[*].cell/grid.raw_frac_le0`: 1.0 means collapsed
   - `history_minus_zero_predictor`: about 0 for epochs 2–3 means the model predicted 0 everywhere
4. **Only if you authorize it:** set `RUN_GPU_DIAGNOSTIC = True` and run **Part B** (60 FP32 and 60 FP16 steps on 6 training graphs, about 5–15 minutes on the A100). It writes `reports/06_step_diagnostic*.json`.
5. Send back:
   - `reports/06_*.json`
   - the full `reports/05_smoke.json`, which includes the `grad_scale` and `grad_norm_nonfinite` columns not in the summary I received

Then decide on §8. No benchmark or training is started by any of these steps.
