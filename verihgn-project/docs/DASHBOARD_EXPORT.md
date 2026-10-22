# Stage 1: results export (notebook 09), how to run it and what it checks

**Scope:** export and validation only. The dashboard itself (Stage 2) is built only after you have run notebook 09 and shared its integrity report.

## 1. What notebook 09 does

It reads the experiment files on Drive and builds `headpath_dashboard_<timestamp>.zip` for the offline dashboard. Nothing in the experiment is re-run.

| Reads (never writes) | Purpose |
|---|---|
| `reports/08_headpath_experiment.json` (authoritative), `reports/08_headpath_plan*.json` | report, plan; other `08_headpath_experiment*` versions are listed, not used |
| `runs/headpath/<14 runs>/steps.csv`, `evals.csv`, `history.csv`, `config.yaml`, `status.json`, `env.json` | training and validation records |
| `runs/headpath/<run>/checkpoints/last.pt` | re-checked (epoch, Adam step count, finite values) and compared with the report; **not exported** |
| `runs/headpath/<run>/predictions_val_official/*.npz` + `_COMPLETE.json` | spatial prediction maps (full precision) |
| `processed/ispd15/<hash>/<design>/<sample>/sample.npz` (the 12 validation samples) | label provenance |
| `reports/07_collapse_experiment__*.json`, `07_collapse_plan.json`, `07_forensic_audit.json`, `runs/collapse_diag/<run>/{steps,history}.csv` | the notebook-07 **context** page (a separate, earlier experiment) |

**Writes:**
- the bundle, on the Colab VM disk (`/content/dashboard_export/`);
- then **only** into `MyDrive/VeriHGN/export/`:
  - the zip;
  - `<name>_INTEGRITY_REPORT.md`;
  - `<name>_integrity_report.json`.

**Refuses** to write under `runs/`, `reports/`, `processed/`, `splits/` or `code/`, and never overwrites a file.

## 2. Checks (all recorded in the integrity report; any FAIL means no zip)

| Group | Check |
|---|---|
| Sources | every file under `runs/headpath`, `runs/collapse_diag`, `reports/07_*` and `08_*`, the earlier protected 05–06 artifacts and the 12 processed samples, **content-hashed before and after**. The export fails if anything was added, removed or changed |
| Report | 14 planned runs present, all COMPLETED, `experiment_status` VALID, the plan files agree with the report, notebook 08's `protected_artifacts` record |
| Per run | 490 steps; evaluations at the 11 pre-registered steps; losses and gradient norms finite |
| Per run (re-derivation) | P1 / P2 / P3, H-sig, category, grid summary, per-design values, zero-predictor reference and optimizer-step count **re-derived from the CSVs** and compared with the report |
| Per run (checkpoint) | `last.pt` re-read: equals the report's checkpoint record; epoch 2; Adam steps = optimizer steps; all values finite |
| Predictions | `_COMPLETE.json` sha256 of every file; exactly the 12 validation samples |
| Predictions (per file) | keys and shapes; finite; official clamped outputs ≥ 0; `grid_abs_err = \|pred − gt\|` bit-exact; `cell_gt = grid_gt[cell_tile]` |
| Predictions (grid metrics) | grid Spearman, zero fraction and std **recomputed from the full maps** vs `evals.csv` at step 490 (tolerance 1e-4: the NPZ comes from a second eval pass, so GPU non-determinism applies) |
| Labels | `grid_gt = log1p((H+V)/2)/c_max`, raw H and raw V (kept separate) and the cell tiles are **bit-identical to the processed samples**; label arrays identical across all runs (stored once) |
| Statistics | the exact McNemar test re-derived from the stored categories (seeds 42–47 only, retests excluded), and checked against scipy |
| Thresholds | **WARN (flag only):** the recorded `p2_min_improvement = 0.10` is not used; the implemented rule is FAIL ≤ 0.08 < BORDERLINE < 0.12 ≤ PASS. Runs whose improvement lies between the two readings are listed (for the local report copy: none). The report is not modified |
| Notebook 07 | report present (one result per planned run); collapse flags re-derived from each run's `steps.csv` |
| Bundle | every array decoded again and compared (lossless); every JSON file parsed with a strict parser (no NaN / Infinity); the zip re-read and every member checked against the manifest's size and sha256; the Drive copy re-verified |
| JSON normalization | each JSON copied from a source is re-serialized strictly (`allow_nan=False`): NaN, +Infinity and −Infinity become `null`, nothing else changes. One integrity entry per file gives the count and kinds; the source sha256 is checked unchanged before and after. A `null` from this step means **undefined**, never 0 |

## 3. Bundle format (`schema_version` 1)

```
headpath_dashboard_<ts>/
  manifest.json              sources (path, size, sha256 of the untouched originals), files (size, sha256), array
                             encoding, json_normalization summary, status
  normalization_index.json   every JSON copied from a source: count and JSON-pointer paths of NaN / +Inf / -Inf -> null
  integrity_report.json      every check (PASS / WARN / INFO / FAIL) with details
  INTEGRITY_REPORT.md
  headpath/report.json       report with non-finite numbers → null, plus `undefined_paths` (never 0)
  headpath/plan.json
  headpath/analysis.json     paired table (seeds 42–47), re-derived McNemar, retests (excluded),
                             threshold note, EXPLORATORY per-seed U − O differences (labelled, no tests),
                             NPZ-vs-evals grid comparison
  headpath/sources/          the report and plans as STRICT JSON (non-finite -> null; see normalization_index.json)
  headpath/runs/index.json   arm, seed, retest flag, primary_inference flag, checkpoint record, map files
  headpath/runs/<run>/{steps,evals,history}.json   column-oriented; undefined cells listed
  headpath/runs/<run>/sources/                     verbatim CSV and config.yaml; status, env, _COMPLETE as strict JSON
  headpath/runs/<run>/maps/<sample>/{grid_pred,cell_pred}.bin.gz
  headpath/maps/index.json   per sample: design, role (validation design / held-out training design), shape
  headpath/maps/<sample>/{grid_gt,raw_h_overflow,raw_v_overflow,cell_index,cell_gt,cell_tile}.bin.gz
  context07/                 notebook-07 report, plan, forensic audit and per-run CSVs (separate experiment)
```

**Array encoding (lossless):**
- float32 is kept bit-identical; integer indices are checked to fit int32.
- The data is little-endian, byte-shuffled into `itemsize` planes, then gzip-compressed.
- `|error|` is not stored; the dashboard computes `|pred − gt|` in float32, which the exporter checked to be bit-exact against the NPZ.
- Cell arrays are the 20,000-cell subsample per sample saved by the experiment. The same cells appear in every run, so the arms are paired.

## 4. How to run notebook 09

1. **Laptop:** upload the rebuilt `dist\verihgn_code.zip` to `MyDrive/VeriHGN/code/verihgn_code.zip`, replacing the old one. Upload `notebooks\09_results_export.ipynb`. Nothing else is needed. The zip changes no experiment code: notebooks 01–08 and the pipeline are byte-identical to the previous bundle.
2. **Colab:** a **CPU runtime is enough** (*Runtime → Change runtime type → CPU*). Notebook 02 is not needed.
3. **Run the bootstrap cell** and authorize Drive in the popup.
4. **Run the export cell** (≈ 10–25 min, an estimate; mostly hashing about 1 GB of Drive files twice). It prints the check summary and the integrity report.
   - On any FAIL it stops with `ExportValidationError`. It still copies `…_FAILED_INTEGRITY_REPORT.md` and `.json` into `MyDrive/VeriHGN/export/`.
   - No zip is produced. **Send me those files.**
5. **Run the last cell.** It copies the zip and its integrity reports to `MyDrive/VeriHGN/export/`, re-verifies the zip on Drive, then flushes Drive (unmount + remount + re-hash). Nothing is written after the flush.
6. **Send back:**
   - `<name>_INTEGRITY_REPORT.md` (or the `.json`)
   - the printed zip sha256
   - the full output of both cells

   Download the zip itself to the laptop now or later. Stage 2 installs it after verifying its sha256 locally (`python tools/export_headpath_dashboard.py --verify <zip>`).

## 5. What the export does not claim

- **No test-set results exist.** HEADPATH evaluated only the 12-graph diagnostic validation set: 6 matrix_mult_a graphs plus 6 held-out samples of training designs.
- **Arm U is a diagnostic deviation** (the train-time cell-head clamp is removed). It is not a reproduction of the paper.
- **Retest runs** are excluded from the primary statistic.
- **Per-seed U − O differences** are exploratory.

## 6. Fix log

**2026-10-22, first Colab run of notebook 09.**
- **What happened:** the export's own checks passed, but the final ZIP verification rejected `headpath/sources/08_headpath_experiment.json` (NaN) and `context07/sources/07_collapse_experiment__20261019_191331.json` (Infinity).
- **Cause:** those two source copies were written byte for byte. The originals contain non-finite numbers, which Python writes as `NaN` / `Infinity` and which strict JSON does not allow. The synthetic test data had none, so the tests missed it.
- **Fix:**
  - every JSON copied from a source now goes through `Bundle.copy_json`, which writes strict JSON (non-finite → `null`) and records each replacement;
  - `Bundle.copy` refuses `.json`;
  - an extra pre-zip check parses every bundle JSON strictly.
- **Not changed:** the ZIP verifier, and the originals on Drive, which are only read and whose sha256 values are recorded.
- **On the local copies of the real reports,** the normalization replaces 145 NaN (08 report) and 1 Infinity (07 report).
