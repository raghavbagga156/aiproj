# Post-collapse forensic report (notebook 07, 18 runs)

**Date:** 2026-10-20. **Type:** read-only investigation. No source, config, label, result or checkpoint was changed, and no training or GPU work was started. The only project change is this file. The local checks in Appendix B ran on the laptop CPU in a scratch directory, outside the project.

> **Revision note (2026-10-20, after review; the original text below is kept unchanged).**
> - **Audit closed.** The Appendix A audits were run on Drive by the user, including the original first-session `07_collapse_experiment.json`. All 18 runs pass the JSON/log consistency and Adam-step checks (user's report).
> - **§11.3 is superseded by `docs/HEADPATH_PLAN.md`.** Three corrections:
>   1. **Statistics.** The Fisher p-values quoted for 5 seeds treated the arms as independent, but the design is paired by seed. HEADPATH uses **6 paired seeds** and a **paired exact two-sided sign test**, where only 6/6 discordant pairs reach p ≤ 0.05.
>   2. **"The grid head serves as a within-run official control" (§11.2) is inexact.** Its code is official, but 67 % of the parameters are shared, so its trajectory differs between arms.
>   3. **P1 is tightened (R2).** The cell head must be non-degenerate (validation std ≥ 1e-4, zero fraction < 0.999). Activity (DEAD / DEGENERATE / HEALTHY) is now reported separately from correlation, which removes the "ρ ≤ 0" clause from §3.3's DEGENERATE.
> - **Research decisions R1–R9** are recorded in `docs/HEADPATH_PLAN.md` §9.

**Evidence labels used throughout**

| Label | Meaning |
|---|---|
| **[C] confirmed** | Follows from the source code, or was reproduced locally, or is read directly from a run artifact |
| **[S] strongly supported** | Several independent artifacts agree, but the decisive per-step log or checkpoint was not available to me |
| **[H] hypothesis** | Plausible, consistent with the data, not tested |

Each claim also says where it comes from: *code* (file:line), *log* (a JSON field and its run), or *local* (Appendix B).

## 0. Inputs: what was and was not available

| Artifact | Location | Status |
|---|---|---|
| `07_collapse_experiment__20261019_191331.json` (149 kB) | `~/Downloads` (not in the project) | **Read.** Summary of all 18 runs |
| `07_collapse_plan.json` | `~/Downloads` | **Read.** Equal to `configs/collapse_experiment_manifest.json` on every compared key and all 18 run rows [C] |
| `06_step_diagnostic.json` (Part B, 60 FP32 + 60 FP16 steps) | `~/Downloads` | **Read.** The only artifact with per-step, per-head loss components |
| `05_smoke.json` | `~/Downloads` | **Read** |
| `06_collapse_inspection__*.json` | not found on this machine (Desktop, Downloads, Documents, OneDrive searched) | **Not available** |
| `runs/collapse_diag/<run>/` (`steps.csv`, `history.csv`, `status.json`, `checkpoints/last.pt`, `best.pt`) | Google Drive only | **Not available.** No local Drive sync exists |
| The original `07_collapse_experiment.json` (no timestamp) | Google Drive only | **Not available.** Its existence follows from the file name we have (§8, D6) |

**Consequence.** Task C (checkpoint inspection) and Task D (reconstruction from per-step logs) could be done only as far as the summary JSON allows. Every item that needs `steps.csv` or a checkpoint is marked as such. Appendix A gives a tested, read-only, CPU-only audit script that closes those items on Colab in a few minutes.

---

## 1. Executive summary

1. **The dominant failure is the cell head, not "collapse".** The cell head's validation predictions are exactly 0 on every validation graph at the end of **17 of 18 runs** [C, log]. All 8 runs labelled "not collapsed" include 7 with a completely dead cell head. Only SINGLE6 seed 42 ends with a live cell head, and even there its correlation is weak and mixed in sign.
2. **The collapse detector only detects the death of both heads.** A step counts as "zero gradient" only if the gradient of the whole model is exactly 0. That requires every raw output of *both* heads to be ≤ 0 on that step's graph (`collapse_experiment.py:222-230`, `trainer.py:204,216`) [C, code]. A dead cell head with a live grid head is invisible to it.
3. **A dead cell head freezes 18.8 % of the model while training continues.** The parameters that receive gradient only through the cell head are 190,598 of 1,012,753. They are the cell head itself plus the whole cell/net path of the last layer. They get exactly zero gradient, while the grid head and all shared layers keep learning [C, local]. The reported average Spearman then equals the grid Spearman divided by 2. The official metric reports the undefined cell correlation as 0.0 (`utils/evaluation.py:34-39`).
4. **Even before it dies, the cell head barely beats predicting zero.** Part B was the healthiest run that has per-head losses. There the cell WMSE was only 0.5–1.1 % below the all-zero predictor over 60 steps, against 7.8–9.4 % for the grid head [C, log].
5. **When both heads collapse, it is early and depends on the run's seed.** 9 of the 10 both-head collapses occurred by step 32, and 4 of them by step 4 [C, log]. FULL seed 44 and REPLICA-FP32 seed 44 took a bit-identical first step (same initialization, same first graph). They then diverged on their second graph: one collapsed at step 19, the other never collapsed [S, log].
6. **The infinite gradient norm is one GradScaler-skipped first step, not a hidden error.** Contrary to the premise in the task, the JSON records `scaler_skipped_steps: 1` for REPLICA FP16 seed 44 [C, log]. The inf is the pre-clip norm of step 1. The FP32 run with the same seed took that step with norm 12.35, the largest in the experiment. Under the official GradScaler order this step updates nothing, halves the scale and leaves the parameters finite [C, code + local]. The recorded `max_grad_norm = inf` and `clipped = True` describe a step that was never applied, so they are misleading.
7. **Precision is not implicated.** FP16 and FP32 give the same outcome for seeds 42 (both collapse at step 15) and 43 (neither collapses). The one disagreement, seed 44, coincides with the skipped FP16 first step: the FP16 run never took the 12.35-norm update that the FP32 run took. FP32 collapsed in 2 of 3 seeds [C, log].
8. **The experiment cannot test training-set size or diversity.** Up to step k a run has seen only its first k graphs. A collapse at step ≤ 6 therefore cannot depend on whether the set holds 6, 36 or 245 graphs [C, logic]. Two of the three DIVERSE36 collapses were at steps 3 and 4.
9. **The summary JSON is internally consistent, with misleading names in places.** Collapse arithmetic, the tail rule, and zero fractions against correlations all agree across 18 runs [C]. The reloaded `last.pt` reproduces the logged final validation metric to ≤ 7.7e-8 (6 REPLICA runs) [C]. One field is misnamed: `steps_both_heads_ge_0999` counts steps where *either* head is ≥ 99.9 % dead. One metric is misleading: `max_grad_norm` includes a non-applied step (§8).
10. **Recommendation.** Do **not** start full training now: it is safe to run but scientifically uninformative. In the official configuration it would very likely end with a dead cell head, and quite possibly a fully collapsed model. Model selection on average Spearman cannot detect a dead cell head. First, run the read-only Drive audit (Appendix A). Then run the smallest experiment that separates the two leading explanations: does the clamp kill a learnable cell head, or does the cell branch lack learnable signal? That is a 2-arm × 5-seed, 490-step FP32 design (§11), about 25–50 A100 minutes (estimate). It needs one clearly labelled, training-only deviation, explained in §11.

---

## 2. Confirmed facts

| # | Fact | Evidence | Status |
|---|---|---|---|
| F1 | Both official heads are `Linear(128→64) → ReLU → Linear(64→1) → squeeze → clamp(min=0)` | `third_party/VeriHGN/models/verihgn.py:194-204` | [C, code] |
| F2 | The clamp's backward pass is `grad · 1[raw > 0]`. It is 0 at raw = 0 exactly, and 0 even when the incoming gradient is NaN | local: raw [-1, 0, 1], upstream [NaN, 5, 7] → grad [0, 0, 7] (torch 2.14.1) | [C, local] |
| F3 | The cell target is the grid target of the cell's L0 tile: `cell.y = grid_y[cell_tile]`, with `grid_y = log1p((H+V)/2)/c_max` | `src/verihgn_repro/data/graph.py:30-32,75,79` | [C, code] |
| F4 | The cell-to-tile index and the label flattening use the same row-major `[x, y]` convention. The same tile extents reproduce CircuitNet `cell_density` bit for bit | `features.py:104-107`, `graph.py:32`, `preprocess.py:90-95`, `PHASE_1_REPORT.md:37` | [C, code] / [S] that the check passed for every sample (the 03 report is not available here) |
| F5 | The official model never uses the `grid_l0 → cell` (`contains`) edges, although the builder creates them. Cells receive tile information only through nets (grid→net, Op2, then net→cell, Op5) | `graph_builder.py:282`; the only edge reads in `verihgn.py` are lines 105, 113, 125, 135, 141, 170, 175, 181 | [C, code] |
| F6 | Exactly-zero total gradient ⇔ every raw output of both heads is ≤ 0 on that step's graph | F2; partition below; `test_all_clamped_outputs_give_exactly_zero_finite_gradient` | [C] |
| F7 | 190,598 parameters (18.8 %) get gradient only from the cell head, 140,417 (13.9 %) only from the grid head, and 681,738 (67.3 %) from both | local, Appendix B.1 | [C, local] |
| F8 | With the cell head dead and the grid head alive, every cell-only parameter has exactly zero gradient; all 18 grid-only and 80 shared tensors get non-zero, finite gradients; the cell WMSE equals the zero-predictor WMSE | local, Appendix B.1 | [C, local] |
| F9 | The first AdamW step moves every parameter by exactly lr (5.0e-4), whatever the gradient's magnitude | 06: `param_delta_maxabs` = 0.0005 at step 1 in FP32 and FP16; Adam's bias-corrected first step is m̂/√v̂ = sign(g) | [C, log + math] |
| F10 | Within a seed, all arms start from identical weights (`seed_everything(seed)` precedes `build_model`, `trainer.py:109-110`) | REPLICA-FP32 s44 and FULL s44 share the same first graph and log a bit-identical `max_grad_norm` (12.350273132324219) and `min_cell_raw_le0` (0.21756736934185028) | [C, code + log] |
| F11 | REPLICA seed 42 reproduces the smoke run's order and collapse | first samples `68-fft_a, 205-edit_dist_a, 46-fft_a, 199-des_perf_a, 376-mm_b` = the order predicted in `COLLAPSE_EXPERIMENT_PLAN.md` §3; max norm 0.99213 at step 2 (FP16) vs the smoke epoch-1 max 0.99229; collapse at step 15, inside epoch 1 | [S, log] (GPU non-determinism explains the 1.6e-4 relative difference) |
| F12 | DIVERSE6 seed 42 reproduces Part B FP32 step 1 exactly | `min_cell_raw_le0` 0.9764710068702698 = Part B FP32 step-1 `cell_raw_head.frac_le0`, bit for bit; same 6 graphs, same first-pass order | [C, log] |
| F13 | The 07 report JSON was produced in a session in which every run was already complete | `seconds_this_session` is 1.3–3.0 s for every run (a 251-step epoch takes 53–60 s); `Trainer.run` returns early for completed runs (`trainer.py:282-285`); the timestamped name is created only when `07_collapse_experiment.json` already exists (notebook 07 cell 2, `new_report`) | [C, code + log] |
| F14 | The plan used for the runs is the pre-registered one | `07_collapse_plan.json` = the manifest on all `MANIFEST_KEYS` and all 18 run rows; the same plan is embedded in the experiment JSON | [C, log] |

---

## 3. Collapse detector (Task A)

### 3.1 What the code does

| Question | Answer | Evidence |
|---|---|---|
| What makes a run "collapsed"? | The trailing run of steps with an exactly-zero gradient norm (`grad_norm == 0.0` after `clip_grad_norm_`) is ≥ 20 steps long, or covers the whole run. `collapse_step` is the global step at which that trailing run starts | `collapse_experiment.py:222-230`, `trainer.py:216` [C] |
| Does it consider both heads jointly? | **Only jointly, and only indirectly.** An exactly-zero gradient needs both heads fully clamped (F6). Neither head's state enters the rule on its own | [C] |
| Can one head die while the other stays active? | **Yes.** Shown by the partition (F7, F8), observed in Part B (cell raw ≤ 0 ≥ 0.999 on 23 of 60 FP32 steps, grid on 0), and at the end of 7 of 18 runs (§7) | [C, local + log] |
| Can a run be "not collapsed" with a dead head? | **Yes: 7 of the 8 "not collapsed" runs.** Their cell validation predictions are 100 % zero | [C, log] |
| Are the collapse steps computed correctly? | **Yes, for the definition used.** In all 18 runs, `collapse_step + zero_tail_len − 1 = steps_logged`; `at_common_steps` follows the same arithmetic with 245; `zero_grad_steps ≥ zero_tail_len` | re-checked from the JSON [C] |
| Do they mean what the name suggests? | Partly. (a) They mark the start of the *final* zero run. Earlier zero steps that recovered are ignored: FULL s42 has 226 zero steps but a 214-step tail, and DIVERSE36 s44 has 164 and 158. (b) Zero gradient is measured per training graph. A step is "zero" if both heads are dead **on that graph, in training mode with that dropout mask**. It says nothing about the other graphs | [C] |
| Are the per-step logs enough to reconstruct the classification? | **For the training-side head state, yes; for anything about loss or gradient per head, no.** `steps.csv` has `cell_raw_le0`, `grid_raw_le0` and `*_raw_max` per step (`trainer.py:220-222`), so per-head death on each training graph can be reconstructed. It does **not** log per-head loss terms, per-head or per-partition gradient norms, or any eval-mode quantity. Eval-mode per-head health (`val_{cell,grid}_pred_std`, `_zero_frac`) is in `history.csv` once per epoch only, so the 1-epoch arms (REPLICA, FULL) have a single snapshot at the end. 07 has no step-0 baseline | `trainer.py:211-223, 313` [C] |

### 3.2 Defects in the summary fields (reporting only; training is unaffected)

| # | Field | Problem | Evidence |
|---|---|---|---|
| A1 | `collapsed` | Detects only both heads dead. The JSON has no per-head classification. `final_val_pred_zero_frac` carries the per-head end state, but the aggregate `by_arm` ignores it | `collapse_experiment.py:251, 307-318` [C] |
| A2 | `steps_both_heads_ge_0999` | Computes `max(cell, grid) ≥ 0.999`, i.e. **either** head ≥ 99.9 % clamped, not both. On the Part B data, "either" = 23 of 60 FP32 steps while "both" = **0** | `collapse_experiment.py:248, 258`; recomputed from 06 [C] |
| A3 | `min_cell_raw_le0`, `max_grid_raw_le0` | They report the cell head's best step and the grid head's worst step, with no matching values for the other head and no per-head tail. They cannot classify per-head state | `collapse_experiment.py:256-257` [C] |
| A4 | `max_grad_norm` | Taken with `np.nanargmax` over all rows, so it includes non-finite norms of scaler-skipped steps (inf). `history.csv` `grad_norm_max` uses finite values only, so the same name has two definitions | `collapse_experiment.py:246-247, 253`; `trainer.py:198-201` [C] |
| A5 | `final_val_avg_spearman` | The official metric returns 0.0 for a constant prediction, so a dead head contributes "0.0" as if it were a measured correlation. The average hides a dead head | `utils/evaluation.py:34-39`; `evaluate.py:46` [C] |
| A6 | Per-head prediction spread | `history.csv` has `val_{cell,grid}_pred_std`, but the summary exports only `_zero_frac` | `collapse_experiment.py:263` [C] |

### 3.3 Proposed diagnostic classification (not implemented)

**Per training step** (from `steps.csv`; head *h* is "dead on this graph" when `h_raw_le0 == 1.0`):

| Step state | Rule |
|---|---|
| `INVALID` | non-finite loss (empty `grad_norm`), non-finite `grad_norm`, or `step_skipped_by_scaler` |
| `BOTH_DEAD` | cell and grid dead. The gradient must be exactly 0; anything else is flagged as an inconsistency |
| `CELL_DEAD` | cell dead, grid not |
| `GRID_DEAD` | grid dead, cell not |
| `BOTH_ALIVE` | neither dead |

**Per head, at the end of a run** (eval mode, validation set, from `history.csv`; the training-side `dead_tail` from `steps.csv` corroborates):

| Head state | Rule |
|---|---|
| `DEAD` | `val_h_pred_zero_frac == 1.0` (constant prediction; correlation **undefined**, reported as such, not as 0) |
| `DEGENERATE` | not dead, but `val_h_pred_std` < 1e-4, or zero fraction ≥ 0.999, or pooled validation Spearman ≤ 0 |
| `HEALTHY` | otherwise |

**Per run:** the pair (cell state, grid state) gives the five requested classes: healthy/healthy, dead cell/healthy grid, healthy cell/dead grid, both dead, and **INVALID**. A run is INVALID if any parameter or Adam moment in `last.pt` is non-finite, if any loss is non-finite, if the Adam `step` count ≠ the number of `optimizer.step()` calls in the log, or if there are GradScaler skips beyond the initial calibration. A per-head death step is also reported: the first step of each head's trailing dead run, plus the fraction of the last 50 steps spent in each state. One FP16 skip at step 1 is a normal calibration event and is flagged, not counted as invalid.

Appendix A implements the training-side half of this, read-only, so the 18 existing runs can be reclassified from their Drive logs without re-running anything.

**Provisional end-state classification from the JSON alone** (eval-mode validation zero fraction plus per-design Spearman):

| Class | Runs | Count |
|---|---|---|
| Both dead | REPLICA16 s42; REPLICA32 s42, s44; FULL s42, s43; D6 s43; D36 s42, s43, s44; S6 s43 | 10 |
| Dead cell / healthy grid | REPLICA16 s43, s44; REPLICA32 s43; FULL s44; D6 s42; S6 s44 | 6 |
| Dead cell / degenerate grid | D6 s44 (grid 95.5 % zeros, negative Spearman on all 7 designs) | 1 |
| Healthy cell (weak) / healthy grid | S6 s42 | 1 |
| Healthy cell / dead grid | — | 0 |
| Invalid | — (one FP16 calibration skip in REPLICA16 s44, flagged) | 0 |

---

## 4. Cell-head forensic analysis (Task B)

### 4.1 Complete computation, from source (facts, [C, code])

| Stage | Computation | Source |
|---|---|---|
| Input | `h_cell` from layer 3: `dropout(LayerNorm(ReLU(W_cell_fuse[h_c ‖ m_net ‖ m_geom]) + h_c))`, shape (N_cell, 128). LayerNorm (γ=1, β=0 at init) makes each node's 128 features zero-mean and unit-variance. Dropout 0.15 is active in training mode only | `verihgn.py:164-167` |
| Where its grid information comes from | `m_net` (Op5) aggregates **updated** net states. Those include `h_net_grid`, the grid→net messages over `spans` edges (Op2; nets spanning > 100 tiles are dropped). There is no grid→cell edge (F5) | `verihgn.py:111-121, 134-138` |
| Linear 1 | `Linear(128, 64)` | `verihgn.py:198` |
| Activation | `ReLU` | `verihgn.py:199` |
| Output layer | `Linear(64, 1)`, then `squeeze(-1)` | `verihgn.py:200, 204` |
| Final clamp | `clamp(min=0)` | `verihgn.py:204` |
| Cast | `out["cell_pred"].float()` | `trainer.py:162` |
| Target | `data["cell"].y = grid_y[cell_tile]` (per_dataset: c_max = 3.465736, from all 251 official training samples) | `graph.py:75,79`; plan `c_max` |
| Loss | `cell_wmse = mean(w_i (ŷ_i − y_i)²)`, w = 10 if y > 0 else 1; `var_cell = relu(σ_y − σ_ŷ)²`; total = cell_wmse + grid_wmse + 0.5 (var_cell + var_grid) | `loss.py:8-41`; `trainer.py:164-165` |
| Backprop to the predictions | ∂L/∂ŷ_i = (2/N) w_i (ŷ_i − y_i) − 2λ relu(σ_y − σ_ŷ)(ŷ_i − mean ŷ)/((N−1) σ_ŷ) | derived from `loss.py` |
| Through the clamp | multiplied by 1[raw_i > 0] (F2) | `verihgn.py:204` |
| Update | global clip at 1.0 (`clip_grad_norm_`), then AdamW (lr 5e-4, wd 1e-4) | `trainer.py:181-184` (FP32), `173-179` (FP16) |

### 4.2 Where the learning signal disappears

- **At the clamp, element by element [C].** For every cell with raw ≤ 0, the gradient from both loss terms is exactly 0, including the variance term's NaN at σ_ŷ = 0 (F2).
- **When every raw cell output is ≤ 0, it is gone for 18.8 % of the model [C, local].**
  - The cell loss then sends no gradient anywhere.
  - The cell head (8,321 params) and the last layer's cell/net path receive **exactly zero** gradient. That path is `mlp_pin`, `W_grid_to_net`, `W_net_update`, `net_norm`, `W_net_to_cell`, `W_gate`, `W_geom_msg`, `W_cell_fuse` and `cell_norm`: 182,277 params.
  - These parameters move only through AdamW's decaying momentum and decoupled weight decay (lr·wd = 5e-8 per step). Neither of these is a learning signal.
- **The rest of the model keeps learning [C, local].** The shared layers 1–2 and the input projections still receive the grid head's gradient. This changes the embeddings the dead cell head reads, so the cell head **can** revive, but only passively: nothing in the loss pushes raw cell outputs above 0 once they are all below it.
- **The loss value hides the death.** With a dead cell head, `cell_wmse` is the constant zero-predictor WMSE α·E[y²]. The analytic value reproduces the logged per-graph zero-predictor loss to 6 digits; for des_perf_a it is 0.020215 in both. `var_cell` is the constant λσ_y². The total training loss still falls because the grid terms fall.

### 4.3 Observations from the logs (separated from the code facts)

**Part B (06), seed 42, the 6 DIVERSE6 graphs, 60 steps, FP32 and FP16. This is the only artifact with per-head losses** [C, log]:

| Quantity | FP32 | FP16 |
|---|---|---|
| Cell raw ≤ 0, step 1 (train mode) | 0.97647 | 0.97646 |
| Cell raw ≤ 0, step 2 (after one sign-like Adam step) | 0.99983 | 0.99983 |
| Steps with cell raw ≤ 0 ≥ 0.999 / exactly 1.0 | 23 / 6 of 60 | 30 / 9 of 60 |
| Steps with grid raw ≤ 0 ≥ 0.999 | 0 of 60 | 0 of 60 |
| Cell WMSE vs zero predictor, all 60 steps (Σ over steps) | −1.08 % | −0.48 % |
| Cell WMSE vs zero predictor, last 20 steps | −3.08 % | −1.78 % |
| Grid WMSE vs zero predictor, all 60 steps | −7.79 % | −9.41 % |
| Grid WMSE vs zero predictor, last 20 steps | −13.33 % | −14.42 % |
| Eval-mode cell predictions all 0 on the training graph after the step | 13 of 60 steps | 14 of 60 steps |
| Validation (2 × mm_a) cell / grid Spearman at step 60 | 0.051 / 0.278 | 0.071 / 0.345 |
| Validation cell zero fraction at step 0 → 60 | 0.99998 → 0.903 | 0.99998 → 0.966 |

**Readings.**
- The cell head was alive in the narrow sense for most of Part B, but it **never learned materially better than predicting 0** on its training graphs [C, log].
- The grid head learned several times more relative to the same baseline [C, log].
- After a single Adam step the cell head went from 97.6 % to 99.98 % clamped. In the same step the grid head went from 99.2 % to 67.5 % clamped. The two heads' trajectories diverge from step 1 [C, log].

**Targets (the six Part B training graphs).** The cell targets are **not** sparser than the grid targets:
- Cell non-zero fractions: 0.309, 0.619, 0.377, 0.397, 0.014, 0.082 (des, edit, fft_a, mm_b, pci_a, pci_b).
- Grid non-zero fractions: 0.137, 0.381, 0.105, 0.125, 0.030, 0.061.
- The cell fraction is higher on 5 of the 6 graphs; cells sit preferentially in congested tiles. Target means are 0.0006–0.021 (cell) and 0.0013–0.009 (grid) [C, log].
- **So label sparsity does not explain why the cell head, rather than the grid head, dies** [S]. The enrichment of non-zero targets at cell locations is also what correct cell–tile alignment predicts. It argues against a scrambled mapping (F4) [S].

**07 (18 runs)** [C, log]:
- **End state.** The final eval-mode cell zero fraction is 1.0 in 17 of 18 runs; the exception is SINGLE6 s42 at 0.810. Per-design cell Spearman is exactly 0.0 (undefined) in every design of those 17 runs.
- **Initialization by seed.** `min_cell_raw_le0` clusters by seed: 0.970–0.977 (seed 42), 0.571–0.633 (43), 0.201–0.229 (44). The initial fraction of clamped cell outputs is therefore mainly a property of the initialization, not of the graph [S]: the value is the minimum over steps, and is most likely the step-1 value.
- **Initialization does not decide the outcome.** All four non-collapsed seed-44 runs still end with a dead cell head, although they started with only ≈ 22 % of cell outputs clamped [S].
- **Not logged.** Per-head losses, per-head gradients and per-head prediction std are not in the 07 JSON. The std is in `history.csv` on Drive (Appendix A).

**Requested items, as far as the evidence allows:**

| Item | Value | Status |
|---|---|---|
| Fraction of raw cell outputs ≤ 0 | Part B: 0.877–1.000 per step; 07: start-of-run minimum by seed as above; end of run: eval-mode cell all-zero on validation in 17 of 18 | [C, log] |
| Fraction of cell predictions exactly 0 | = raw ≤ 0 fraction (F2). Validation, end of run: 1.0 in 17 of 18 runs, 0.81 in 1 | [C] |
| Cell prediction std | Part B step 60 (training graph): 0.0155 (FP32), 0.0065 (FP16); cell target std 0.006–0.051. 07: in `history.csv` only | [C, log] / not available |
| Target distribution and zero fraction | Table above (Part B graphs); all graphs per design: `grid_nonzero_fraction_by_design` in the plan (grid only) | [C, log] |
| Per-head loss contribution | Part B only (table above). At a dead cell head, the cell terms are the constants α·E[y²] and λσ_y² | [C] |
| Per-head gradients | Not logged anywhere. The local partition shows which parameters each head can reach | [C, local] |
| Do the shared layers get useful gradients? | They receive the grid head's gradient (F7, F8). Whether that helps the cell head is not measured | [C] / [H] |
| Can the grid head learn while the cell head is dead? | Yes, by construction (F8), and observed: grid Spearman 0.15–0.45 on mm_a in 6 runs with a dead cell head | [C] |

---

## 5. Grid-head forensic analysis

**Computation [C, code].** The head is the same as the cell head (`verihgn.py:194-204, 251`), applied to `h_grid_l0_out = γ ⊙ h̃⁽⁰⁾ + (1 − γ) ⊙ h′⁽⁰⁾` (`verihgn.py:186-189`):
- `h′⁽⁰⁾` is `dropout(LayerNorm(...))` from Op4 (`verihgn.py:129-132`).
- `h̃⁽⁰⁾` is the L1→L0 message `scatter_mean(phi_cf(h⁽¹⁾))`, where `phi_cf = Linear + ReLU`. It is therefore elementwise ≥ 0 (`verihgn.py:181-184`).

So the grid head's input distribution differs structurally from the cell head's pure LayerNorm output. Whether this explains the grid head's greater resilience is **[H]**. The grid target `grid_y` and its loss terms mirror the cell ones.

**Gradient reach [C, local].** The grid-only parameters are the grid head plus the last layer's `W_cell_to_grid`, `W_grid_update`, `grid_norm`, `phi_fc`, `W_l1_spatial`, `phi_cf` and `W_gate_hier`: 140,417 params.

**Observations [C, log].**
- **At initialization.** Part B, seed 42: the eval-mode grid predictions on validation are **100 % zero** at step 0. In training mode at step 1, 99.2 % of grid raw outputs are ≤ 0. The grid head starts on the edge of death and **revives** in the first step: 67.5 % clamped at step 2.
- **Never dead in Part B.** The grid head was never ≥ 99.9 % clamped on any of the 120 Part B steps.
- **End state in 07.**
  - The grid head is alive in 8 of 18 runs.
  - Even when alive, it predicts exactly 0 for **54–96 % of validation tiles**. "Alive" is a relative term: the grid head is mostly clamped too.
  - Grid Spearman on the 6 matrix_mult_a graphs: 0.146–0.445 in 7 runs, −0.024 in DIVERSE6 s44.
  - Per-design grid Spearman is positive on all 7 designs in FULL s44, DIVERSE6 s42, SINGLE6 s42 and SINGLE6 s44, and negative on all 7 designs in DIVERSE6 s44. Six of these designs have a single validation sample each.
- **Collapse is a grid event.** In all 10 both-dead runs the grid head is dead too. Given that the cell head is dead or near-dead almost everywhere, a "collapse" in 07 is in effect **the grid head dying**. This is [S]: Part B shows the cell head dying first, but the per-step order in the 07 runs needs `steps.csv`.

---

## 6. Infinite gradient norm (Task C)

**Run.** `REPLICA__fp16__seed44`.

**Record.** `max_grad_norm = Infinity` at `max_grad_step = 1` (sample `278-mgc_matrix_mult_b-util_0.9`), with **`scaler_skipped_steps = 1`**, `zero_grad_steps = 0`, `collapsed = false`. This is the only `Infinity` in the file, and the only run with a scaler skip [C, log]. The premise that this run had zero scaler-skipped steps is not what the JSON says.

### 6.1 Exact order of operations (FP16 branch)

| # | Operation | Source | Behaviour relevant here |
|---|---|---|---|
| 1 | `optimizer.zero_grad()` | `trainer.py:159` | `set_to_none=True` by default, so any NaN gradient left from the previous step is discarded |
| 2 | Forward pass under `autocast(fp16)`; heads → `.float()` | `trainer.py:160-163` | |
| 3 | Loss `verihgn_loss(...)["total"]` in FP32; `scale_before = scaler.get_scale()` | `trainer.py:164-166` | |
| 4 | `if not isfinite(loss)`: skip (never taken here; the norm was logged) | `trainer.py:167-172` | |
| 5 | `scaler.scale(loss).backward()` | `trainer.py:174` | gradients scaled by 2¹⁶ = 65536; FP16 intermediates saturate at 65504 |
| 6 | `scaler.unscale_(optimizer)` | `trainer.py:175`; `torch/amp/grad_scaler.py:302-361` | divides by the scale and records `found_inf` if any gradient is non-finite |
| 7 | `gn = clip_grad_norm_(..., 1.0)` | `trainer.py:176`; `torch/nn/utils/clip_grad.py:164-181` | with an inf gradient, `total_norm = inf`, `clip_coef = 1/(inf + 1e-6) = 0`, every gradient × 0: finite → 0, inf → NaN |
| 8 | `scaler.step(optimizer)` | `trainer.py:178`; `grad_scaler.py:371-372` | **`optimizer.step()` is called only if `found_inf` is 0**; here it is skipped |
| 9 | `scaler.update()` | `trainer.py:179`; `grad_scaler.py:484-560` | scale × 0.5 (backoff) → 32768; growth needs 2000 clean steps, so the scale stays 32768 for the rest of the run |
| 10 | Log: `grad_norm = inf`, `clipped = (inf > 1.0) = True`, `step_skipped_by_scaler = (32768 < 65536) = True` | `trainer.py:211-219` | |

The order is identical to official `train.py:64-69` [C, code].

### 6.2 Findings

| Question | Answer | Status |
|---|---|---|
| Why did the max norm become inf? | It is the pre-clip norm of step 1, whose unscaled gradient contained a non-finite value. Same seed, same initial weights, same first graph: the FP32 run's step 1 had norm 12.35, the largest in the whole experiment (`REPLICA__fp32__seed44`, `FULL__fp32__seed44`). At scale 65536, an FP16 gradient element in backward overflows when its true magnitude is above ≈ 1.0. With a total norm of 12.35 that is expected. This is the textbook GradScaler first-step calibration | Mechanism [C, code + local]. That step 1 is the skipped step: [S]. The JSON has one skip and one inf, an inf requires a non-finite gradient, and a non-finite gradient forces the skip. Which element overflowed is not measured |
| Did gradients become NaN/Inf? | Yes, in that one step: inf after unscale, then NaN after the ×0 in clipping (local reproduction: exactly the inf entries become NaN, everything else 0). They are discarded at the next `zero_grad` | [C, local] for the mechanism |
| Did parameters become NaN/Inf? | No mechanism for it: `optimizer.step()` did not run. The run went on to produce finite metrics, and its reloaded `last.pt` reproduced the logged validation metric to 3.0e-8 | [S]. Definitive check: finite `last.pt` parameters, Appendix A |
| Did the optimizer update the parameters at step 1? | No. Locally: parameters bit-identical, no AdamW state created. Across the run, AdamW `step` in `last.pt` should be **250** (251 − 1), against 251 for the FP32 replica | [C, local] for the mechanism; the count check needs `last.pt` (Appendix A) |
| Did GradScaler behave correctly? | Yes: skip, backoff to 32768, no further skips | [C, code] + [C, log] (skips = 1) |
| Was clipping applied correctly? | Yes, for its semantics. It ran on unscaled gradients, as AMP requires, and had no effect on parameters because the step was skipped. On applied steps, clipping is close to a no-op for the update itself at the first Adam step (F9) | [C, code] |
| Is the recorded metric misleading? | **Yes.** `max_grad_norm = inf` reports a gradient that was never applied. `clipped = True` marks a step that changed nothing. `history.csv` `grad_clipped_frac` also counts it (`trainer.py:202`), while `history.csv` `grad_norm_max` excludes it (finite-only). The honest summary is: max *applied* norm (finite), plus one calibration skip at step 1 | [C, code] |

### 6.3 Scientific consequence

From step 2 on, FP16 seed 44 trains from the **initial** weights on graph 2. FP32 seed 44 trains from weights that already took the 12.35-norm first update. The seed-44 FP16-vs-FP32 pair is therefore **not** a comparison of rounding effects on one trajectory. It differs by a whole optimizer step and by which graph AdamW's moment estimates start from. The one precision disagreement (FP16 not collapsed, FP32 collapsed at step 19) cannot be attributed to FP16 arithmetic [C].

---

## 7. Audit of all 18 runs (Task D)

From the experiment JSON. *Not in the JSON:* per-head prediction std, per-head losses, per-head death step, non-finite-loss rows, Adam step counts, checkpoint finiteness. All of these come from Drive via Appendix A.

`opt.step()` = logged steps − scaler skips. This assumes no non-finite-loss rows, which `history.csv` `skipped_nonfinite` confirms. Zero-gradient steps still call `optimizer.step()`, which moves parameters through momentum and weight decay only.

| # | Run (seed, precision) | Steps | opt.step() | Non-zero-grad steps | Collapse step (both heads) | Scaler skips | Max ‖g‖ pre-clip (step) | Cell raw≤0, min over steps | Grid raw≤0, max over steps | Val zero frac cell / grid | mm_a ρ cell / grid | End state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | REPLICA 42 fp16 | 251 | 251 | 14 | 15 | 0 | 0.992 (2) | 0.970 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 2 | REPLICA 43 fp16 | 251 | 251 | 250 | — | 0 | 1.108 (1) | 0.633 | 1.000 | 1.000 / 0.544 | undef / 0.146 | cell dead, grid alive |
| 3 | REPLICA 44 fp16 | 251 | **250** | 251* | — | **1** | **inf (1)** | 0.218 | 0.9997 | 1.000 / 0.602 | undef / 0.445 | cell dead, grid alive |
| 4 | REPLICA 42 fp32 | 251 | 251 | 14 | 15 | 0 | 0.985 (2) | 0.970 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 5 | REPLICA 43 fp32 | 251 | 251 | 251 | — | 0 | 1.395 (190) | 0.632 | 0.998 | 1.000 / 0.657 | undef / 0.415 | cell dead, grid alive |
| 6 | REPLICA 44 fp32 | 251 | 251 | 18 | 19 | 0 | 12.35 (1) | 0.218 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 7 | FULL 42 | 245 | 245 | 19 | 32 | 0 | 0.063 (3) | 0.977 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 8 | FULL 43 | 245 | 245 | 19 | 26 | 0 | 1.810 (1) | 0.615 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 9 | FULL 44 | 245 | 245 | 239 | — | 0 | 12.35 (1) | 0.218 | 1.000 | 1.000 / 0.671 | undef / 0.242 | cell dead, grid alive |
| 10 | DIVERSE6 42 | 246 | 246 | 246 | — | 0 | 0.328 (66) | 0.976 | 0.9993 | 1.000 / 0.665 | undef / 0.410 | cell dead, grid alive |
| 11 | DIVERSE6 43 | 246 | 246 | 3 | 4 | 0 | 0.708 (1) | 0.571 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 12 | DIVERSE6 44 | 246 | 246 | 246 | — | 0 | 2.039 (1) | 0.224 | 0.99998 | 1.000 / 0.955 | undef / −0.024 | cell dead, grid degenerate |
| 13 | DIVERSE36 42 | 252 | 252 | 2 | 3 | 0 | 0.069 (1) | 0.975 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 14 | DIVERSE36 43 | 252 | 252 | 3 | 4 | 0 | 0.639 (1) | 0.605 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 15 | DIVERSE36 44 | 252 | 252 | 88 | 95 | 0 | 2.237 (1) | 0.229 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 16 | SINGLE6 42 | 246 | 246 | 244 | — | 0 | 0.401 (229) | 0.773 | 1.000 | **0.810** / 0.762 | **−0.041** / 0.317 | both alive (cell weak) |
| 17 | SINGLE6 43 | 246 | 246 | 3 | 4 | 0 | 0.637 (1) | 0.618 | 1.000 | 1.000 / 1.000 | undef / undef | both dead |
| 18 | SINGLE6 44 | 246 | 246 | 246 | — | 0 | 2.882 (1) | 0.201 | 0.99989 | 1.000 / 0.691 | undef / 0.429 | cell dead, grid alive |

\* Includes the skipped step 1, whose norm was inf. 250 non-zero-gradient updates were applied.

**Column notes.**
- "undef" means the official code reported 0.0 for a constant prediction.
- The mm_a columns are the per-design metrics on the **same 6 matrix_mult_a graphs** in every run (validation rows 1–6). This is the only like-for-like validation comparison across all 18 runs. The training sets and step counts still differ (245–252), and the checkpoint is the last step, not the best.
- `collapse step` is identical on the full run and at the common 245-step budget in all 18 runs.

**Consistency checks run on the JSON [C]:**
- Collapse arithmetic: 18 of 18.
- Tail rule: 18 of 18.
- `final_val_pred_zero_frac == 1.0` ⇔ every per-design Spearman of that head is 0.0: 18 of 18, both heads.
- Plan: equal to the manifest.
- Reloaded `last.pt` against the history value on REPLICA, where pooled = mm_a per-design: |Δ| = 0, 1.2e-8, 3.0e-8, 0, 7.7e-8, 0. These are consistent with GPU scatter non-determinism, which confirms the checkpoint matches the logged end of training.
- For the 12-graph arms the pooled value cannot be re-derived from per-design numbers, so only the zero/non-zero consistency was checkable.

**Numerical stability [S].** The JSON shows one non-finite gradient norm in 18 runs (the FP16 calibration skip). Every reported validation metric is finite. Non-finite losses and parameter finiteness need Appendix A.

---

## 8. Discrepancies

| # | Between | Discrepancy | Effect |
|---|---|---|---|
| D1 | Task premise vs JSON | The premise says REPLICA FP16 s44 had zero scaler-skipped steps; the JSON says `scaler_skipped_steps: 1` | Removes the apparent paradox (§6) |
| D2 | `collapsed` vs per-head end state | 7 of 8 "not collapsed" runs have a dead cell head; `by_arm` reports only both-head collapse | Collapse counts understate failure: dead-cell-or-worse is 17 of 18 |
| D3 | Field name vs computation | `steps_both_heads_ge_0999` = either head (`max`), not both | The value over-states two-head near-death. Part B: 23 vs 0 |
| D4 | JSON `max_grad_norm` vs `history.csv` `grad_norm_max` | Non-finite included vs finite-only, under similar names | REPLICA16 s44: JSON inf; history will show a finite max plus `grad_norm_nonfinite = 1` |
| D5 | `steps.csv` `clipped`, history `grad_clipped_frac` | Count a scaler-skipped step as "clipped" | Overstates clipping |
| D6 | `provenance: "measured (this run)"` vs how the file was made | The file is a re-summary written in a later session (1.3–3.0 s per run, timestamped name). The first `07_collapse_experiment.json` exists on Drive and was not provided | Low risk (both read the same `steps.csv`), but the original should be compared (Appendix A) |
| D7 | `final_val_per_design` vs `final_val_*` | The former comes from the reloaded `last.pt` in the re-summary session, the latter from `history.csv` written at training time | Agree to ≤ 7.7e-8 where checkable |
| D8 | `COLLAPSE_EXPERIMENT_PLAN.md` hypothesis table vs the design | H2/H3 (size, diversity) are listed as decidable, but the collapses at steps ≤ 6 (4 of 10, at steps 3, 4, 4, 4) cannot depend on set size (§9) | The H2/H3 verdicts are void, not "inconclusive" |
| D9 | `tools/inspect_collapse.py` vs the 07 arms | The tool computes the zero-predictor baseline over the **split's** training graphs (251), not the run's training subset (`inspect_collapse.py:51-52, 66-73, 87-94`) | `history_minus_zero_predictor` is invalid for DIVERSE6/36 and SINGLE6 if used on them |
| D10 | `SMOKE_TRAINING_DIAGNOSIS.md` §2.4 ("one dead head can recover") vs 07 | Recovery is possible in principle but rare: the cell head is alive at the end in 1 of 18 runs | Expectations should be revised |

No discrepancy was found between the plan, the manifest and the executed runs, or between the JSON's own fields.

---

## 9. Scientific limitations and confounders (Task E)

**What the 18 runs do support:**
- **The cell head fails in essentially every configuration tested.** It is dead in 17 of 18 runs, under both precisions, all four training subsets and all three seeds. The failure is systematic, not a seed artifact [S].
- **Both-head collapse is frequent but not universal.** It happened in 10 of 18 runs. It is decided early (9 of 10 by step 32) and is sensitive to the seed, which here means initialization + sample order + dropout [S].
- **The outcome depends on the trajectory.** A bit-identical first step was followed by opposite outcomes, depending on the next graphs (FULL s44 vs REPLICA-FP32 s44) [S].
- **The official configuration reproduces the smoke collapse** [S, F11].
- **Neither precision prevents collapse.** FP32 collapsed in 2 of 3 replicas, FP16 in 1 of 3 [C].

**What they do not support:**
- **That dataset size causes collapse.** Logically excluded for the 4 collapses at steps 3–4 (C: a run's state at step k depends only on initialization, its first k graphs and dropout). Not separable from sample order for the rest. FULL (2 of 3) vs DIVERSE36 (3 of 3) vs DIVERSE6 (1 of 3) is also within 3-seed noise.
- **That diversity matters.** DIVERSE6 and SINGLE6 have identical seed-by-seed outcomes (42: alive, 43: collapse at step 4, 44: alive), which shows the seed dominating. SINGLE6 is also confounded with the specific design fft_a.
- **That FP16 causes or prevents collapse.** Two of three seeds agree. The third differs by a skipped first update (§6.3), not by rounding.
- **That the model cannot learn.** The grid head learns in several runs, with ρ up to 0.445 on 6 unseen-design graphs. The cell head's failure is untested against a learnable alternative (§11).
- **Anything about 200-epoch behaviour.** 245 steps are about 0.5 % of a 200-epoch run of roughly 50,000 steps. "Alive at step 245" does not mean alive later; DIVERSE36 s44 collapsed at step 95.

**Confounders and design limitations:**
1. **The seed couples three things.** Initialization, sample order and dropout masks all change together. Initialization is matched across arms within a seed (F10); order and dropout are not.
2. **Small samples.** 3 seeds per arm. The strongest possible split (3 of 3 vs 0 of 3) has a two-sided Fisher p of 0.10, so no arm contrast in 07 can reach conventional significance.
3. **The critical window precedes the manipulation.** The arms differ only in which graphs come first until a run has exhausted its smaller training set (6 or 36 graphs). Most collapses happen before that.
4. **Validation sets differ for pooled metrics.** REPLICA uses 6 graphs, the arms 12. Only the per-design mm_a numbers are comparable across all 18 runs, and they are taken at different step counts (245–252), from different training sets, at the last checkpoint. Six of the seven per-design entries rest on a single graph.
5. **REPLICA trains on the 6 graphs that the other arms hold out.**
6. **FULL is unbalanced** (pci_bridge32_a has 26 of 245 graphs); the DIVERSE arms are balanced.
7. **GPU non-determinism** (atomic scatter) separates nominally identical trajectories. The REPLICA16 s42 max norm differs from the smoke run's by 1.6e-4 relative.
8. **One variant only.** 07 tested official_seed42 / per_dataset / paper_eq3. The per_sample normalization and the pair-aware split are untested at A100 scale (the cell head collapsed under per_sample locally on CPU, `SMOKE_TRAINING_DIAGNOSIS.md` §8).
9. **The head-state measures are uneven.** The training-side measures are taken in train mode (with dropout) on the current graph only. The eval-mode measures come once per epoch, so once per run for REPLICA and FULL. There is no step-0 baseline.
10. **The official metric convention** reports undefined correlations as 0.0, which mixes "dead" with "uncorrelated" in every average.

---

## 10. Ranked hypotheses

| Rank | Hypothesis | Evidence for | Evidence against / gaps | Status |
|---|---|---|---|---|
| 1 | **Mechanism:** with the official `clamp(min=0)`, all-raw-≤0 is a zero-gradient region per head. For both heads together it is absorbing (gradient exactly 0). For the cell head alone it is "sticky": 18.8 % of the model is frozen and only passive revival through the shared layers is possible | F2, F6–F8; existing tests; 07 end states | — | **[C]** |
| 2 | **The cell head fails systematically under this setup**, independently of seed, precision and subset; "collapse" is the grid head dying on top of it | 17 of 18 dead at the end; Part B cell −0.5…−1.1 % vs zero predictor; it dies even from a 22 %-clamped initialization (seed 44) | Per-step order of head deaths in 07 not seen (needs `steps.csv`) | **[S]** |
| 3 | **Grid-head survival, and so full collapse, is a trajectory-dependent tipping event** in the first ~30 steps | Within-arm seed variation; identical-first-step divergence; 9 of 10 collapses ≤ step 32 | Only 18 runs; no step-level logs read | **[S]** |
| 4 | **Early-step dynamics push the cell head under.** AdamW's sign-like early steps of size lr on every parameter (F9) and the absence of warmup are large relative to targets with means of 0.0006–0.021. One step took cell raw ≤ 0 from 97.6 % to 99.98 % | Part B steps 1→2 | Untested; testing needs an optimizer deviation (not proposed first) | **[H]** |
| 5 | **The cell branch lacks learnable signal**, not just survivability. Its target is its tile's label, but it has no direct path to the tile (F5) and must infer it through net-mediated messages. Under weighted MSE, a near-constant prediction is then near-optimal and drifts into the clamp | F3, F5; the cell head's tiny gain over zero in Part B even while alive | Untested. Separating 4/1 (dynamics) from 5 (signal) is the purpose of §11 | **[H]** |
| 6 | Precision (FP16) causes collapse | — | 2 of 3 seeds identical; the third is explained by the calibration skip; FP32 collapses 2 of 3 | **not supported** |
| 7 | Training-set size / design diversity causes collapse | — | Untestable in 07 for early collapses (§9); 3-seed noise otherwise | **not supported** (not decidable with 07) |
| 8 | A single large-gradient step triggers collapse (plan H5) | — | The max-norm steps are at steps 1–3 in collapsed and non-collapsed runs alike (12.35 in both outcomes at seed 44); FULL s42 collapsed with max norm 0.063. The per-step test needs `steps.csv` | **not supported** by the summary |
| 9 | A label or pipeline defect makes the cell labels unlearnable (e.g. cell–tile misalignment) | — | F4 (bit-identical cell_density, consistent flattening); cell-location enrichment of non-zero targets; our step is bit-identical to official `train_one_epoch` (`SMOKE_TRAINING_DIAGNOSIS.md` §4.8) | **ruled out at code level [S]** |

---

## 11. Smallest defensible follow-up experiment (Task F; not started)

### 11.1 Prerequisite (no training, CPU, read-only)

Run Appendix A on the Drive run folders. It closes Tasks C and D:
- per-step, per-head states and death steps for all 18 runs;
- the Adam step count of 250 for REPLICA16 s44;
- checkpoint finiteness;
- JSON vs logs.

If it contradicts anything in §§6–7, this report must be revised before anything else.

### 11.2 Why one training-only deviation is unavoidable

The open question is whether the cell head dies because the clamp removes the gradient from an otherwise learnable head (dynamics, H4/H1), or whether the cell branch cannot fit its labels (signal, H5).

**No run of the unmodified official model can answer it.** Once the cell head is dead its gradient is exactly zero (F2, F8). An official run therefore yields no information about what the head *could* have learned. Lower learning rates, warmup or other initializations would change different things (optimizer and initialization), and still would not open the gradient path.

The smallest change that opens it:
- **Cell head, training forward only:** the `clamp(min=0)` is omitted, so the raw output enters the loss.
- **Grid head:** unchanged in both arms. It serves as a within-run official control.
- **Evaluation:** clamped, exactly as official, with the official metrics.

The change:
- adds no parameters, so the initialization stays identical by seed and the comparison is **paired**;
- consumes no RNG;
- lives in our wrapper, with `third_party/` untouched;
- is marked in `config.yaml` and the run id as a **deviation**.

It is a **diagnostic, not a reproduction** of the paper. It never replaces the official arm.

A straight-through gradient that keeps the forward pass bit-identical was considered and rejected. At an all-zero prediction, `variance_reg`'s std has a NaN gradient that only the clamp's mask hides (F2). A straight-through estimator would need a second modification.

### 11.3 Design ("HEADPATH")

| Item | Specification |
|---|---|
| Hypothesis H-dyn | The cell-head failure is caused by the train-time clamp: with the clamp removed in training, the cell head stays alive, beats the zero predictor and generalizes |
| Hypothesis H-sig | The cell branch lacks learnable signal: with the clamp removed, the cell head is alive but no better than predicting 0 |
| Arms | **O** = official (unchanged); **U** = cell-head clamp omitted in training forward passes only |
| Seeds | 42, 43, 44, 45, 46, paired across arms. Seeds 42–44 make arm O a logged replication of 07 FULL. Five is the smallest number for which a qualitative split (5/5 vs 0/5: Fisher two-sided p = 0.008; 4/5 vs 0/5: p = 0.048) can be decisive |
| Held fixed | FULL training set (245 graphs) and its 12-graph / 7-design validation set; per_dataset (c_max 3.465736); paper_eq3; **FP32** (removes the GradScaler confound, §6.3); TF32 off; AdamW 5e-4 / 1e-4; clip 1.0; dropout 0.15; batch 1; gradient checkpointing on; StepLR (no decay reached); patience off; the pre-registered manifest check from notebook 07 |
| Length | 2 epochs = 490 steps per run (covers the observed collapse window, ≤ 95 steps, five times over) |
| Logging (observation only, proven bit-identical by a test like `test_step_log_does_not_change_training`) | **Per step:** `cell_wmse`, `grid_wmse`, `var_cell`, `var_grid`; the per-head zero-predictor WMSE of that graph (α·E[y²]); pre-clip gradient L2 norm of each partition (cell-only, grid-only, shared; F7); the existing raw ≤ 0 / raw max per head. **Every 49 steps (incl. step 0):** eval-mode validation per head (pred std, zero fraction, official and independent Spearman/Pearson, pooled and per design), plus a *tile-projection reference*: the cell Spearman of `grid_pred[cell_tile]`, a free evaluation-only quantity showing how much cell ranking the grid branch already carries |
| Primary outcomes (per run) | **P1 alive:** validation cell zero fraction < 1.0 at step 490, and cell not dead on ≥ 90 % of the last 50 steps. **P2 learns:** cell WMSE over the last 98 steps ≥ 10 % below the zero-predictor WMSE (Σ over those steps). **P3 generalizes:** pooled validation cell Spearman (independent implementation) ≥ 0.10 at step 490 |
| Decision rule (pre-registered) | **H-dyn supported:** U meets P1–P3 in ≥ 4/5 seeds and O meets P1 in ≤ 1/5. **H-sig supported:** U meets P1 but its cell WMSE is within ±3 % of the zero predictor and validation cell ρ ≤ 0.05 in ≥ 4/5 seeds. Anything else: **inconclusive**, reported as such |
| Secondary | Grid head in both arms (alive, WMSE vs zero, ρ) as a descriptive non-inferiority check; both-head collapse frequency in O (5 more runs); reproducibility of O seeds 42–44 against 07 FULL (collapse at step 32 / 26 / none, with GPU non-determinism allowed) |
| Run validity | All losses and gradient norms finite; Adam `step` = logged steps; all parameters finite in `last.pt`; manifest check passes. A run that fails any of these is reported as invalid, not dropped |
| Cost | 10 runs × 490 steps ≈ 20 epoch-equivalents at 1–2 min each (FP32 estimated at 1–2 × the measured 53–60 s FP16 epoch), plus 11 small evaluations per run: **≈ 25–50 A100 minutes. This is an estimate, not a measurement** |
| What it will not decide | Whether to *adopt* the deviation for the main training (your methodology decision); FP16 behaviour; dataset size; 200-epoch behaviour |

**If no model deviation is acceptable at all**, the most informative fallback is:
- the forward-only tile-projection reference on the existing 07 checkpoints;
- plus an official-only 10-seed × 40-step logged replication of the early window.

It would estimate P(cell dead) and P(both dead) with per-head losses, but it **cannot** separate H-dyn from H-sig.

---

## 12. Is full training safe to proceed with?

**Engineering safety: yes [S].**
- No numerical failure beyond one normal GradScaler calibration skip.
- Checkpoints reload and reproduce the logged metrics.
- The step is bit-identical to official `train.py`.

**Scientific value now: no. Full training is not recommended yet.**
1. **A dead cell head is near-certain.** Under the official configuration the cell head was dead at the end of 17 of 18 short runs, and the 07 design gives no indication that more data or more steps revive it.
2. **Full collapse is likely.** Both heads died in 10 of 18 runs, mostly within 32 steps. Even a run that survives 245 steps has about 50,000 steps left in which to tip (DIVERSE36 s44 collapsed at step 95).
3. **Model selection cannot see it.** The official criterion, mean validation Spearman, scores a dead cell head as 0.0. Early stopping and `best.pt` will select dead-cell checkpoints without complaint. That criterion is official and should stay, but it cannot be relied on to flag the failure.
4. **The cost buys a predictable result.** About 200 epochs at 1 min or more each (several A100 hours) would most likely produce a cell result that is undefined, not reproduced.

**If you nevertheless want the official-configuration full run** as a documented negative result, that is legitimate. Run it only after the Appendix A audit, and report:
- the per-head end state (§3.3);
- dead-head correlations as **undefined**, never 0.0;
- the GradScaler event separately from gradient norms.

**Recommended order:**
1. Appendix A audit (CPU, read-only).
2. Your decision on whether the §11 diagnostic deviation may be run.
3. HEADPATH.
4. Only then, a decision about full training and in which configuration, with any deviation run alongside the official arm and never instead of it.

**Logging and reporting changes to propose** (not implemented; none affect training mathematics):
- the per-head classification (§3.3);
- per-head losses and partition gradient norms;
- "undefined" for constant predictions;
- separate "max applied norm" and "scaler events" fields;
- renaming `steps_both_heads_ge_0999`;
- a run-subset-aware zero-predictor in `inspect_collapse.py`.

---

## Appendix A. Read-only audit of the Drive run folders

**What it does.** It re-derives everything in §7 that the JSON does not contain:
- per-step head states (§3.3) and per-head death steps;
- scaler-skip and non-finite steps by step number;
- `optimizer.step()` calls compared with AdamW's `step` in `last.pt`;
- parameter and moment finiteness in `last.pt` and `best.pt`;
- head output-layer statistics;
- the 07 JSON compared field by field with the logs.

**What it never does.** No forward pass, no training, no GPU. It writes nothing except `--out`; point that at `/content`, not at Drive.

**How it was tested.** It was tested locally (torch 2.14.1, CPU) on a synthetic run folder: fabricated test logs and a toy checkpoint built from the official model's parameter layout. The test covered a step-1 overflow, a scaler skip, a dead cell head from step 3, two both-dead steps, and a deliberately wrong JSON field. All assertions passed, including the detection of the wrong field.

**Usage on Colab** (CPU runtime is enough; after the usual bootstrap cell). Save the script below as `/content/forensic_audit_07.py`, then:

```
!python /content/forensic_audit_07.py \
    --runs /content/drive/MyDrive/VeriHGN/runs/collapse_diag \
    --report /content/drive/MyDrive/VeriHGN/reports/07_collapse_experiment__20261019_191331.json \
    --project /content/verihgn-project --out /content/07_forensic_audit.json
```

Run it a second time with `--report .../07_collapse_experiment.json` (the original, first-session file) and compare the two outputs (D6). Send back `/content/07_forensic_audit.json` (both versions) and the printed table.

**What to read in the output:**

| Field | Expected if this report is right |
|---|---|
| `REPLICA__fp16__seed44` → `steps.scaler_skipped_steps` | `[1]` |
| `REPLICA__fp16__seed44` → `steps.nonfinite_grad_norm_steps` | `[(1, 'inf')]` |
| `REPLICA__fp16__seed44` → `adam_step_counts` | `[250]` |
| `REPLICA__fp16__seed44` → `adam_steps_match_log` | `True` |
| Every run → `last_pt.nonfinite_param_tensors`, `nonfinite_adam_moments` | `[]`, `0` |
| Every run → `steps.zero_grad_but_not_both_dead`, `both_dead_but_nonzero_grad` | `[]` and `[]` (F6) |
| Every run → `json_vs_logs` | `[]` |
| Runs 2, 3, 5, 9, 10, 12, 18 → `steps.cell.dead_tail` | large (cell dead on training graphs to the end) |
| Runs 2, 3, 5, 9, 10, 12, 18 → `steps.grid.dead_tail` | small |
| REPLICA16 s42: Σ `grad_norm` over its 14 non-zero steps | ≈ 0.0044485 × 251 = 1.117 (the smoke epoch-1 mean × steps) |

```python
"""Read-only forensic audit of the notebook-07 run folders (docs/POST_COLLAPSE_FORENSIC_REPORT.md, App. A).

For every runs/collapse_diag/<run>/ it re-derives, from steps.csv / history.csv / checkpoints only:
  * per-step head state (BOTH_ALIVE / CELL_DEAD / GRID_DEAD / BOTH_DEAD / INVALID), per-head dead
    counts, first-dead step and trailing dead run
  * scaler-skipped steps, non-finite-loss steps and non-finite gradient norms, by step number
  * optimizer.step() calls actually executed, checked against AdamW's per-parameter 'step' in last.pt
  * finiteness of every parameter and optimizer moment in last.pt / best.pt; head output-layer stats
and compares the collapse fields of the notebook-07 report JSON with the values re-derived here.
CPU only. No training, no forward pass, no GPU. Nothing is written except --out (if given).

usage: python forensic_audit_07.py --runs <.../runs/collapse_diag> --report <.../07_collapse_experiment*.json>
                                   [--project <verihgn-project>] [--no-checkpoints] [--out audit.json]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def rows(p: Path) -> list:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def true(x) -> bool:
    return x == "True"


def head_state(r) -> str:
    gn = r.get("grad_norm")
    if gn in ("", "None", None) or not math.isfinite(num(gn)) or true(r.get("step_skipped_by_scaler")):
        return "INVALID"
    cd, gd = num(r.get("cell_raw_le0")) >= 1.0, num(r.get("grid_raw_le0")) >= 1.0
    return "BOTH_DEAD" if cd and gd else "CELL_DEAD" if cd else "GRID_DEAD" if gd else "BOTH_ALIVE"


def tail(flags) -> int:
    n = 0
    for f in reversed(flags):
        if not f:
            break
        n += 1
    return n


def audit_steps(steps: list) -> dict:
    if not steps:
        return {"present": False}
    gs = [int(r["global_step"]) for r in steps]
    st = [head_state(r) for r in steps]
    zero = [true(r.get("grad_exactly_zero")) for r in steps]
    out = {"present": True, "rows": len(steps), "first_global_step": gs[0], "last_global_step": gs[-1],
           "global_steps_contiguous": gs == list(range(gs[0], gs[0] + len(gs))),
           "nonfinite_loss_steps": [g for g, r in zip(gs, steps) if r.get("grad_norm") in ("", "None")],
           "scaler_skipped_steps": [g for g, r in zip(gs, steps) if true(r.get("step_skipped_by_scaler"))],
           "nonfinite_grad_norm_steps": [(g, r["grad_norm"]) for g, r in zip(gs, steps)
                                         if r.get("grad_norm") not in ("", "None") and not math.isfinite(num(r["grad_norm"]))],
           "zero_grad_steps": sum(zero), "zero_grad_tail": tail(zero),
           "state_counts": {k: st.count(k) for k in ("BOTH_ALIVE", "CELL_DEAD", "GRID_DEAD", "BOTH_DEAD", "INVALID")},
           "state_counts_last50": {k: st[-50:].count(k) for k in ("BOTH_ALIVE", "CELL_DEAD", "GRID_DEAD", "BOTH_DEAD", "INVALID")}}
    out["optimizer_step_calls"] = len(steps) - len(out["nonfinite_loss_steps"]) - len(out["scaler_skipped_steps"])
    out["optimizer_step_calls_with_nonzero_grad"] = out["optimizer_step_calls"] - sum(zero)
    for h in ("cell", "grid"):
        dead = [num(r.get(f"{h}_raw_le0")) >= 1.0 for r in steps]
        le0 = [num(r.get(f"{h}_raw_le0")) for r in steps]
        out[h] = {"dead_steps": sum(dead), "dead_tail": tail(dead),
                  "first_dead_step": next((g for g, d in zip(gs, dead) if d), None),
                  "last_alive_step": next((g for g, d in zip(reversed(gs), reversed(dead)) if not d), None),
                  "raw_le0_step1": le0[0], "raw_le0_min": min(le0), "raw_le0_max": max(le0),
                  "raw_max_last": num(steps[-1].get(f"{h}_raw_max"))}
    # an exactly-zero gradient should only occur when both heads are fully clamped on that graph
    out["zero_grad_but_not_both_dead"] = [g for g, z, s in zip(gs, zero, st) if z and s != "BOTH_DEAD"]
    out["both_dead_but_nonzero_grad"] = [g for g, z, s in zip(gs, zero, st) if s == "BOTH_DEAD" and not z]
    fin = [(num(r["grad_norm"]), g) for g, r in zip(gs, steps) if math.isfinite(num(r.get("grad_norm")))]
    out["max_finite_grad_norm"] = max(fin) if fin else None
    out["step1"] = {k: steps[0].get(k) for k in ("sample", "loss", "grad_norm", "clipped", "scale_before",
                                                  "scale_after", "step_skipped_by_scaler",
                                                  "cell_raw_le0", "grid_raw_le0")}
    return out


HIST_KEYS = ("epoch", "train_loss", "skipped_nonfinite", "grad_norm_mean", "grad_norm_max", "grad_norm_nonfinite",
             "grad_zero_steps", "train_steps", "val_loss", "val_cell_pred_std", "val_cell_pred_zero_frac",
             "val_grid_pred_std", "val_grid_pred_zero_frac", "val_cell_spearman", "val_grid_spearman",
             "val_avg_spearman", "grad_scale")


def audit_history(hist: list) -> dict:
    if not hist:
        return {"present": False}
    keep = lambda r: {k: r.get(k) for k in HIST_KEYS}  # noqa: E731
    return {"present": True, "epochs": len(hist), "first": keep(hist[0]), "last": keep(hist[-1]),
            "sum_skipped_nonfinite": sum(int(num(r.get("skipped_nonfinite")) or 0) for r in hist),
            "sum_grad_norm_nonfinite": sum(int(num(r.get("grad_norm_nonfinite")) or 0) for r in hist),
            "epochs_cell_val_dead": [int(r["epoch"]) for r in hist if num(r.get("val_cell_pred_zero_frac")) >= 1.0],
            "epochs_grid_val_dead": [int(r["epoch"]) for r in hist if num(r.get("val_grid_pred_zero_frac")) >= 1.0]}


def audit_checkpoint(path: Path, load_checkpoint) -> dict:
    import torch
    if not path.exists():
        return {"present": False}
    s = load_checkpoint(path)
    model, opt = s["model"], s["optimizer"]
    nonfinite = [k for k, v in model.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
    steps, bad_moments = set(), 0
    for st in opt.get("state", {}).values():
        if "step" in st:
            steps.add(int(float(st["step"])))
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and not torch.isfinite(st[k]).all():
                bad_moments += 1
    head = {}
    for h in ("cell_head", "grid_head"):
        w, b = model[f"{h}.mlp.2.weight"].float(), model[f"{h}.mlp.2.bias"].float()
        head[h] = {"out_bias": float(b.item()), "out_weight_l2": float(w.norm()), "out_weight_max": float(w.max()),
                   "hidden_bias_mean": float(model[f"{h}.mlp.0.bias"].float().mean())}
    return {"present": True, "epoch": s["epoch"], "run_id": s["run_id"], "best_avg_spearman": s["best_avg_spearman"],
            "nonfinite_param_tensors": nonfinite, "nonfinite_adam_moments": bad_moments,
            "adam_step_counts": sorted(steps), "adam_state_tensors": len(opt.get("state", {})),
            "scaler": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (s.get("scaler") or {}).items()},
            "heads": head}


def compare(js: dict, a: dict) -> list:
    """JSON report fields vs values re-derived from steps.csv."""
    st = a["steps"]
    if not st.get("present"):
        return ["steps.csv missing"]
    out = []
    checks = {"steps_logged": st["rows"], "zero_grad_steps": st["zero_grad_steps"], "zero_tail_len": st["zero_grad_tail"],
              "scaler_skipped_steps": len(st["scaler_skipped_steps"]),
              "min_cell_raw_le0": st["cell"]["raw_le0_min"], "max_grid_raw_le0": st["grid"]["raw_le0_max"]}
    for k, v in checks.items():
        jv = js.get(k)
        if jv is None or (isinstance(v, float) and abs(float(jv) - v) > 1e-9) or (not isinstance(v, float) and jv != v):
            out.append(f"{k}: json={jv} logs={v}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--project", default="/content/verihgn-project")
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load_checkpoint = None
    if not a.no_checkpoints:
        sys.path.insert(0, str(Path(a.project) / "src"))
        from verihgn_repro.training.checkpointing import load_checkpoint
    rep = json.loads(Path(a.report).read_text())
    by_name = {r["name"]: r for r in rep["results"]}
    result = {"report": str(a.report), "runs": {}}
    for name in [r["name"] for r in rep["plan"]["runs"]]:
        d = Path(a.runs) / name
        res = {"dir_exists": d.exists(),
               "status": json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None,
               "steps": audit_steps(rows(d / "steps.csv")), "history": audit_history(rows(d / "history.csv"))}
        if load_checkpoint:
            res["last_pt"] = audit_checkpoint(d / "checkpoints" / "last.pt", load_checkpoint)
            res["best_pt"] = audit_checkpoint(d / "checkpoints" / "best.pt", load_checkpoint)
            lp, st, hi = res["last_pt"], res["steps"], res["history"]
            if lp.get("present") and st.get("present"):
                res["adam_steps_match_log"] = lp["adam_step_counts"] == [st["optimizer_step_calls"]]
            if lp.get("present") and hi.get("present"):
                res["last_pt_epoch_matches_history"] = lp["epoch"] == hi["epochs"]
        res["json_vs_logs"] = compare(by_name.get(name, {}), res) if name in by_name else ["run missing from report"]
        result["runs"][name] = res
        s = res["steps"]
        if s.get("present"):
            print(f"{name:26s} rows {s['rows']:3d} opt.step {s['optimizer_step_calls']:3d} "
                  f"(nonzero-grad {s['optimizer_step_calls_with_nonzero_grad']:3d}) skips {s['scaler_skipped_steps']} "
                  f"nonfinite-gn {s['nonfinite_grad_norm_steps']} states {s['state_counts']} "
                  f"cell dead-tail {s['cell']['dead_tail']} grid dead-tail {s['grid']['dead_tail']} "
                  f"adam-match {res.get('adam_steps_match_log')} json-diff {res['json_vs_logs'] or 'none'}")
        else:
            print(f"{name:26s} steps.csv missing")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str))
        print("wrote", a.out)
    return result


if __name__ == "__main__":
    main()
```

---

## Appendix B. Local checks performed (CPU, laptop; scratch directory only, outside the project)

| # | Check | Result |
|---|---|---|
| B.1 | Official 3-layer, 128-dim model on a random 400-cell / 100-tile synthetic graph with every edge type the model reads. One forward/backward each with the cell-only loss and the grid-only loss (official `weighted_mse` + `variance_reg`), recording which parameters receive non-zero gradient. Then the cell head was forced dead (output bias −10), followed by one backward with the official `verihgn_loss`. No optimizer | Partition: 190,598 cell-only / 140,417 grid-only / 681,738 shared / 0 unreached, out of 1,012,753. Dead cell head: sum \|grad\| over the cell head = 0.0; all cell-only tensors zero; 18 of 18 grid-only and 80 of 80 shared tensors non-zero; all gradients finite; cell WMSE = zero-predictor WMSE exactly |
| B.2 | The exact AMP statement order of `trainer.py:166-179` on a toy model, with one gradient element set to inf after `scale(loss).backward()`; torch `GradScaler("cpu")`, init scale 2¹⁶ | Logged row: `grad_norm=inf, clipped=True, scale 65536→32768, step_skipped_by_scaler=True`. Gradients after clipping: the inf entry becomes NaN, all others 0. Parameters bit-identical and finite; no AdamW state created. The next normal step: applied, Adam step counter 1 |
| B.3 | `clamp(min=0)` backward at raw [−1, 0, 1] with upstream gradient [NaN, 5, 7] | [0, 0, 7] |
| B.4 | Analytic zero-predictor loss α·E[y²] + λ(σ²_cell + σ²_grid) from the logged Part B target statistics | Equals the logged `zero_predictor_loss` on all 6 graphs to 6 digits |
| B.5 | Existing tests, without training (`-k summarize_run_collapse_rule or common_step_budget_rule or manifest or intersections or replica_order or plan_follows or rejects`; no cache written) | 10 passed |
| B.6 | Appendix A script on a synthetic run folder | All assertions passed; the deliberately wrong JSON field was detected |

The tests that train small models on CPU were not run, because the instructions exclude launching training.
