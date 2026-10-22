# Results analysis and offline dashboard: implementation plan (HEADPATH)

**Status (2026-10-22):**
- **Plan approved,** with these decisions:
  1. export on Colab with notebook 09;
  2. full-precision maps, with H and V overflow kept separate;
  3. pre-registered statistics primary; exploratory comparisons labelled as such;
  4. the threshold inconsistency is flagged, and records are not rewritten;
  5. notebook 07 is shown as a separate context page.
- **Stage 1 (export and validation) is implemented** (`docs/DASHBOARD_EXPORT.md`).
- **Stage 2 (dashboard) waits** for your notebook-09 results.

## 0. What I inspected

**Code and documentation:**
- `docs/HEADPATH_PLAN.md`, `docs/POST_COLLAPSE_FORENSIC_REPORT.md` (section list), `docs/PHASE_0_REPORT.md` (D8)
- `src/verihgn_repro/diagnostics/headpath.py`, `evaluation/head_metrics.py`, `evaluation/evaluate.py` (`save_predictions`), `model_deviations.py`, `training/trainer.py` (step, eval and history logging), `export/demo_bundle.py`
- `configs/headpath_manifest.json`, notebook 08 (via `tools/build_notebooks.py`)

**Experiment outputs available on this laptop:**
- `C:\Users\ragha\Downloads\08_headpath_experiment.json`: 123,000 B, sha256 `ce485b2569b5d355…`
- `08_headpath_plan.json` and `08_headpath_plan__20261020_192429.json`

## 1. Artifact inventory

| Artifact | Where | On this laptop? | Schema source |
|---|---|---|---|
| `reports/08_headpath_experiment.json` | Drive | **A copy in Downloads.** Whether it equals the Drive file is unverified: compare the sha256. | inspected (§2.1) |
| `reports/08_headpath_plan*.json` | Drive | copies in Downloads | same as `plan` in the report |
| `runs/headpath/<run>/steps.csv` (490 rows) | Drive | **no** | `trainer._log_step`, `_head_terms`, `_partition_norms` |
| `runs/headpath/<run>/evals.csv` (11 rows: steps 0, 49, …, 490) | Drive | **no** | `trainer._step_eval` → `head_metrics.eval_summary` |
| `runs/headpath/<run>/history.csv` (2 rows) | Drive | **no** | `trainer.run` |
| `runs/headpath/<run>/predictions_val_official/*.npz` (12) + `_COMPLETE.json` | Drive | **no** | `evaluate.save_predictions`, `headpath.export_predictions_atomic` |
| `runs/headpath/<run>/checkpoints/last.pt`, `best.pt` (~12 MB each) | Drive | no; **not needed** by the dashboard | `trainer._state` |
| `runs/headpath/<run>/{config.yaml, env.json, env_session_*.json, status.json}` | Drive | **no** | trainer |
| Processed graphs (inputs) | Drive | no; not needed | |

## 2. Schemas (exact)

### 2.1 `08_headpath_experiment.json` (top level: `plan`, `results[14]`, `run_status_counts`, `analysis`, `cell_feature_baseline`, `provenance`)

**`analysis`:**
- **Status fields:** `experiment_status` = VALID, `status_reasons` = [], `historical_replication` = PASSED, `verdict`, `run_statuses`
- **`category_counts`**, by arm and category
- **`final_activity`**, by arm (cell and grid lists, ordered by seed)
- **`primary_test`:**
  - name, endpoint
  - `pairs[6]`: seed, O, U, discordant
  - `excluded`, `n_valid_pairs`, `n_discordant`, `U_only`, `O_only`, `direction`
  - `p_two_sided_exact`, `alpha`, `significant`
- **`hsig`**, **`replication`** (seeds 42–44: expected vs observed), **`test_retest`** (per arm)
- **`conclusion`**, **`deviation_notice`**

**`results[i]`** (per run):
- **Identity:** name, arm, seed, retest, `cell_head_train_clamp`
- **Outcome:** valid, `invalid_reasons`, category, `hsig_criteria_met`
- **Criteria:**
  - **`P1`:** pass, final cell activity, std, zero fraction, dead fraction over the last steps
  - **`P2`:** outcome, and the clamped / raw / loss improvement vs the zero predictor
  - **`P3`:** pass, final validation cell Spearman (independent and official)
  - **`p2_zero_reference`:** sample sha256 and Σ zero-predictor WMSE
- **Grid head (`grid`):** final activity, WMSE improvement, final ρ (independent and official)
- **`per_design_and_tileproj_final`:** 59 keys: `design_<d>_{cell,grid}_{spearman, spearman_indep, pred_zero_frac, pred_std}` and `tileproj_*`
- **States:** `state_at_replication_step`, `both_head_collapse_07_rule`
- **Validity record:**
  - `optimizer_step_calls`
  - `checkpoint`: present, epoch, run id, finite parameters, finite Adam moments, step counts
  - `fresh_official_check`: max abs / relative Δ, tolerance ratio, self-repeat, atol, rtol, equivalent, bit_identical
- **Bookkeeping:**
  - `history_last` (values stored **as strings**)
  - `run_status`, `trainer_state`
  - `predictions_export`: status, dir, n_files, moved_aside
  - `seconds_this_session`

**`cell_feature_baseline`:** 136 keys: `_note`, `pooled_<feature>`, `design_<d>_<feature>`, `sample_<s>_<feature>`, and `design_<d>_<feature>_mean_of_samples`, for 5 features.

**Contains literal `NaN`** (e.g. grid ρ_indep for dead heads, and retest |Δ| when undefined). This is valid Python JSON but **invalid for a browser's `JSON.parse`**. It must be converted explicitly (§4).

### 2.2 `steps.csv`, one row per optimizer step

- **Basic columns:** epoch, global_step, step_in_epoch, graph_index, sample, loss, grad_norm (pre-clip), grad_exactly_zero, clipped, scale_before/after, step_skipped_by_scaler
- **Raw head outputs:** `{cell,grid}_raw_le0`, `{cell,grid}_raw_max`
- **HEADPATH columns:**
  - **Loss terms:** `cell_wmse_{loss,clamped,raw,zero}`, `grid_wmse_{loss,zero}`, `var_{cell,grid}_{loss,zero}`, `var_reg_loss`
  - **Stream statistics:** `{cell_loss, cell_clamped, cell_raw, grid_pred}_{std,zero_frac,mean}`
  - **Gradient norms:** `grad_norm_{cell_only,grid_only,shared}`

### 2.3 `evals.csv`, one row per evaluation (11 per run)

- epoch, global_step, val_loss, n_val
- **Per head** `h` ∈ {cell, grid}:
  - `val_<h>_{mae,rmse,pearson,spearman,kendall}` (official, pooled)
  - `val_<h>_{spearman,pearson}_indep`
  - `val_<h>_pred_{std,zero_frac,mean,max}`, `val_<h>_activity`
  - `val_<h>_evalraw_{le0_frac,std,max}`
- **Per design:** `design_<d>_<h>_{spearman,spearman_indep,pred_zero_frac,pred_std}`
- **Tile projection:** `tileproj_cell_spearman[_indep]`, `tileproj_mgc_matrix_mult_a_cell_spearman_indep`

### 2.4 `history.csv`, one row per epoch (2)

Training metrics on **clamped** predictions: `train_cell_*`, `train_grid_*`. Arm U also has `train_cell_raw_*`. Also: validation summary, timing, GPU memory, gradient summary.

### 2.5 `predictions_val_official/<sample>.npz` (12 per run) + `_COMPLETE.json` (`files: {name: sha256}`, `n_files`, `created_at`)

Arrays:
- **Grid maps, full grid at native resolution:** `grid_gt`, `grid_pred`, `grid_abs_err` [M, N]; `raw_h_overflow`, `raw_v_overflow` [M, N]
- **Cells, a 20,000-cell subsample per sample:** `cell_index`, `cell_gt`, `cell_pred`, `cell_tile`
  - The subsample is drawn by `np.random.RandomState(0)` in record order. It is **identical across runs** for the same sample, so it is paired across arms.

The predictions come from the official eval forward (clamped) in both arms.

## 3. What can be computed reliably, and what cannot

| Quantity | Source | Reliability |
|---|---|---|
| Categories, P1/P2/P3, H-sig, activity classes, replication, test–retest, validity, fresh-official check | report | **Authoritative** (pre-registered code). Re-derived during export as a consistency check (§4.3), never replaced |
| McNemar result (6 pairs, 2 discordant, p = 0.5) | report | **Authoritative.** p recomputed independently in the exporter (exact binomial) as a check |
| Validation trajectories (11 points), per-design ρ, activity over time | `evals.csv` | **Full-resolution** (computed on all cells and tiles at the time) |
| Training dynamics: losses, clamped WMSE / zero-predictor ratio, raw ≤ 0 fractions, partition gradient norms | `steps.csv` | **Full-resolution**, training data |
| Per-epoch training metrics | `history.csv` | Training data, clamped predictions (R6) |
| Grid metrics at step 490 | NPZ (full grid maps) | Exactly recomputable. Used to **verify** `evals.csv` at step 490 |
| Cell metrics at step 490 | NPZ | **Only on the 20,000-cell subsample.** Shown as "subsample", never in place of the full-cell values in the report |
| Spatial maps (label, H and V separately, prediction, \|error\|) | NPZ | Exact |
| Non-learned feature baseline | report | Descriptive, with the standardization caveat (`_note`) |
| **Test-set performance** | — | **Does not exist.** HEADPATH never loads the test split. The dashboard says so on every page |
| Comparison with the paper's Table 2 | — | **Not possible or meaningful.** 2 epochs, an adapted label (D1) and a diagnostic validation set; stated explicitly |
| Confidence intervals / extra tests | — | **Not planned.** The only confirmatory statistic is the pre-registered McNemar test. Paired per-seed values are shown as points (n = 6), with no error bars implying precision. Any further statistic would need your approval and an "exploratory" label |

## 4. Architecture (offline, Windows 11)

Three stages: export → bundle → static app.

```
Google Drive (read-only)                       Laptop (offline)
reports/08_*.json  ─┐
runs/headpath/*  ───┼─► [A] export (Colab CPU) ─► export/headpath_dashboard_<ts>.zip ─► [B] unzip into app/public/data
                    │       validate + convert        (one download, ~30–80 MB est.)         │
                    └─► never written                                                       ▼
                                                                    [C] React + Plotly static app, built once,
                                                                        served by `python -m http.server` (stdlib)
```

### 4.1 [A] Exporter: `tools/export_headpath_dashboard.py` (+ notebook `09_results_export.ipynb`, CPU only)

**Read-only by construction:**
- opens sources with `"rb"` only;
- writes only to a new `MyDrive/VeriHGN/export/headpath_dashboard_<ts>/` directory, or a local path, and refuses any output path under `runs/` or `reports/`;
- reuses `headpath.protected_artifacts` fingerprinting before and after, so the report shows `runs/headpath` and `reports/08_*` are byte-identical.

It runs on Colab, so the 14 run folders and checkpoints never need to be downloaded to the laptop. It can also run locally on a copied folder.

**Validation gates.** The export fails loudly, and writes nothing usable, unless:
- 14 runs are present, and `run_statuses` and `experiment_status` are read from the report, never assumed;
- every `predictions_val_official/` passes `headpath.predictions_complete` (marker + sha256);
- every run has 490 `steps.csv` rows and `evals.csv` rows at exactly the 11 pre-registered steps.

### 4.2 Bundle format (`schema_version` 1)

- **`manifest.json`:**
  - schema version, export time
  - code version (bundle sha, official commit)
  - **sha256 and size of every source file**
  - source paths, and the list of verification results
- **`report.json`:** the experiment report with every non-finite number replaced by `null`. A parallel `undefined_paths` list records where a NaN or Inf was, so the UI shows "undefined (constant prediction)" and never 0.
- **`runs/<run>/`:**
  - `steps.json`, `evals.json`, `history.json`: column-oriented, numbers as numbers, the same NaN handling
  - `meta.json`: arm, seed, retest flag, deviation flag, run id, config deviations from `config.yaml`
- **`maps/<sample>/`:**
  - `grid_gt`, `raw_h`, `raw_v` as `.f32.bin.gz`, stored **once per sample** (identical across runs; verified by hash)
  - `cell_index`, `cell_gt`, `cell_tile` (once)
  - `shape.json`
- **`runs/<run>/maps/<sample>/`:** `grid_pred.f32.bin.gz` and `cell_pred.f32.bin.gz`. |error| is computed in the browser.
  - Float32, gzip; the browser decompresses with `DecompressionStream`, which is built into Edge and Chrome.
  - Size estimate: 14 runs × 1.67 M tiles × 4 B ≈ 94 MB before compression. Dead heads (all zeros) compress to almost nothing.
- **`verification.json`:**
  - categories re-derived from the stored P1/P2/P3 with the **effective** thresholds (0.08 / 0.12)
  - McNemar p recomputed
  - grid ρ at step 490 recomputed from the NPZ vs `evals.csv`
  - the zero-predictor reference identical within each pair
  - retest runs excluded from the primary test
  - **Any mismatch is shown as a red banner** in the app.

### 4.3 Analysis module: `src/verihgn_repro/analysis/headpath_results.py` (pure functions, used by the exporter, unit-tested)

- `load_report`, `sanitize` (NaN → null plus paths)
- `rederive_categories`, `mcnemar_exact`
- `paired_table` (seeds 42–47 only; retest separately)
- `rolling_ratio(steps, 'cell_wmse_clamped', 'cell_wmse_zero', window=98)`, which reproduces the P2 window at step 490
- `grid_metrics_from_npz`
- `activity_timeline`

**No new statistics** beyond the pre-registered ones.

### 4.4 [C] Dashboard app: `app/` (D8, approved)

**Technology:**
- React 18 + Plotly (`plotly.js-dist-min`), built with Vite 5 into a static `app/dist/`.
- `npm install` needs the internet **once**, at build time; the built app needs none.
- **Launcher:** `run_dashboard.bat` runs `python -m http.server --directory app/dist 8765` and opens the browser. `fetch` does not work from `file://`, so a local server is used; it is standard-library only, with no new Python dependency.

**Pages, all values read from the bundle:**
1. **Overview and integrity**
   - status banners: `experiment_status`, verdict, replication, verification results, protected-artifact check
   - a permanent notice: *validation only — no test results exist*; *not a reproduction of the paper*; *arm U = diagnostic deviation* (verbatim `deviation_notice`)
2. **Primary outcome**
   - the 6 paired seeds (O category | U category | discordant)
   - discordant counts, direction, exact p, and the significance rule: with 6 pairs only 6/6 discordant reaches p ≤ 0.05
   - the conclusion text from the report, unedited
   - retests in a separate, greyed "excluded from inference" box
3. **Per-seed criteria:** paired dot plots (O → U lines per seed), with the pre-registered thresholds drawn and taken from the report:
   - P2 improvement, with the 0.08 / 0.12 band
   - P3 cell ρ, with the 0.10 line
   - P1 activity
4. **Training dynamics** (labelled TRAINING):
   - rolling clamped cell WMSE / zero predictor
   - raw ≤ 0 fractions for the cell and grid heads
   - partition gradient norms (log scale; zero shown as a gap with an annotation)
   - both-head-collapse markers
   - arm U's raw-stream loss is never plotted against arm O's loss (R6)
5. **Validation trajectories** (labelled VALIDATION, diagnostic set):
   - pooled cell and grid ρ (independent; undefined shown as gaps, not 0) over the 11 points
   - activity class timeline
   - official vs independent metric agreement
6. **Per design:**
   - final ρ heatmap (design × run)
   - matrix_mult_a (cross-design) separated from the 6 held-out within-design samples
   - the non-learned feature baseline beside it, with the standardization note
7. **Spatial maps:**
   - sample and run selectors
   - label (normalized log1p of the mean of H and V), raw H and V overflow **separately**, prediction, |error|
   - shared colour scale per sample
   - cell scatter (subsample, labelled)
8. **Reproducibility:** fresh-official check, checkpoint checks, test–retest |Δ| (NaN shown as undefined), seconds per run
9. **Methods and limitations:** deviations, decisions R1–R9 / D1–D7, §5 limitations, and provenance (hashes)

**Visual rules:**
- one fixed colour per arm; arm U always labelled "U (diagnostic, train-time clamp removed)"
- retest runs always visually distinct and never pooled with seeds 42–47
- no bar charts with error bars for n = 6
- undefined values never drawn as 0

## 5. Limitations the dashboard must state

1. **No test-set evaluation exists.** All performance numbers are on the 12-graph diagnostic validation set: 6 matrix_mult_a graphs (the official validation design) and 6 held-out samples of training designs, which measure within-design generalization only.
2. **2 epochs (490 steps)** per run. This is a diagnostic, not full training. It is not comparable to the paper and is **not a reproduction**.
3. **n = 6 paired seeds.** The pre-registered test cannot reach p ≤ 0.05 unless 6/6 pairs are discordant in one direction. The observed p = 0.5 (2/2 discordant, U > O) is **not significant**.
4. **Arm U deviation** (train-time cell clamp removed; the loss surface also changes). The grid head shares 67 % of the parameters, so grid differences between arms are confounded.
5. **Grid head DEAD at step 490** in 5/6 arm-O and 4/6 arm-U runs. Grid ρ is then undefined; the official code reports 0.0.
6. **Cell values in the NPZ files** are a 20,000-cell subsample per sample. Full-cell metrics come only from `evals.csv` and the report.
7. **GPU non-determinism:** a self-repeat Δ of about 4e-8 is recorded per run.
8. **Report inconsistency:** `plan.thresholds.p2_min_improvement = 0.10` is recorded but **not used**. `headpath.p2_outcome` applies FAIL ≤ 0.08 < BORDERLINE < 0.12 ≤ PASS. The dashboard shows the effective rule and lists this as a documentation inconsistency. The exporter's re-derivation of every category checks that the stored categories follow the effective rule.

## 6. Files

**New:**

| File | Purpose |
|---|---|
| `src/verihgn_repro/analysis/__init__.py`, `analysis/headpath_results.py` | pure analysis and consistency checks |
| `tools/export_headpath_dashboard.py` | read-only exporter → bundle |
| `notebooks/09_results_export.ipynb` (generated by `tools/build_notebooks.py`) | Colab, CPU only: mount, export, fingerprint check, write the zip to `MyDrive/VeriHGN/export/` |
| `tests/test_dashboard_export.py` | exporter on synthetic HEADPATH run folders (the existing test helpers create them on the CPU) |
| `app/` (`package.json`, `vite.config.js`, `index.html`, `src/**`) | the React + Plotly dashboard |
| `app/src/**/*.test.jsx` (Vitest + jsdom) | render tests with a **synthetic** bundle with deliberately different values, proving nothing is hard-coded |
| `run_dashboard.bat`, `tools/install_dashboard_bundle.py` | Windows launcher; installs a downloaded bundle into `app/public/data` after checking its hashes |
| `docs/DASHBOARD.md` | user guide |

**Modified (small):**

| File | Change |
|---|---|
| `tools/build_notebooks.py` | + notebook 09 |
| `tools/make_colab_bundle.py` | include `analysis/` and the exporter; exclude `app/node_modules` and the data |
| `README.md` | status row and dashboard section |

**Not touched:**
- `third_party/`
- the experiment, trainer and evaluation code (unless you approve fixing the `p2_min_improvement` record, a reporting-only field)
- every Drive report, run folder and checkpoint

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The local report copy differs from the Drive file | The exporter reads the Drive file. The dashboard shows its sha256; compare it with the local copy |
| NaN or Inf in JSON breaks the browser, or shows up as 0 | explicit sanitize + `undefined_paths`; UI tests with undefined values |
| Accidental write into `runs/` or `reports/` | path guard + before/after fingerprint (existing `protected_artifacts`) |
| Bundle too large | labels de-duplicated, gzip, maps optional (`--no-maps` gives a few MB) |
| Hard-coded numbers creep into the UI | UI tests with a synthetic bundle; no numeric result literals allowed in `app/src` (lint test) |
| Validation shown as "performance" | fixed page badges (TRAINING / VALIDATION — diagnostic set); a global "no test results" notice |
| Over-interpretation of n = 6 | only the pre-registered test; per-seed points; the significance rule shown beside p |
| npm needs internet | only at build time. Alternatively I can commit the built `app/dist` so you only need Python to view it |
| Edge / Chrome `DecompressionStream` missing | supported in current Edge and Chrome; fall back to an uncompressed export flag |

## 8. Order of work and estimates (my time; your Colab time is separate)

| # | Step | Estimate |
|---|---|---|
| 1 | Analysis module + exporter + tests (synthetic runs) | ~2–3 h |
| 2 | Notebook 09 + bundle rebuild. **You** run it on Colab: CPU only, ~5–15 min; download one zip | — |
| 3 | Dashboard app + UI tests, first developed against a synthetic bundle | ~4–6 h |
| 4 | Install your real bundle, verification banner green, walkthrough | ~30 min |

A minimal first deliverable is possible if you prefer: steps 1–2 plus pages 1–3 only.

## 9. Decisions needed from you

1. **Where to export.** On Colab via notebook 09 (recommended; avoids downloading the run folders), or do you have `runs/headpath` synced locally?
2. **Spatial maps.** Include them, at full float32 (~30–80 MB bundle, est.), or export without maps first?
3. **Statistics.** Keep the dashboard to the pre-registered statistics only (recommended), or allow clearly labelled exploratory per-seed paired summaries?
4. **The `p2_min_improvement` record.** Leave the report untouched and only flag the inconsistency (recommended), or also correct the field for future runs in `headpath.py`?
5. **Include notebook 07** collapse-experiment results as a context page (its report is in Downloads), or HEADPATH only?
