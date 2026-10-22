# Runbook — Colab validation and A100 benchmarking (approved stage)

Scope: the official 9 designs / 383 samples. **No full-scale training** in this stage. All results come from your Colab session; nothing in this file is a result.

## 0. What changed since the Phase 1 report (engineering only)

None of these changes touches the model, the features, the labels or the training mathematics.

| Change | Why |
|---|---|
| **Graphs are no longer cached on Drive.** They are rebuilt in RAM from the processed files (0.2–1.7 s each). | Measured on real samples: a large graph is ~140 MB, so one variant over 383 samples is ~30 GB. Four variants would exhaust Drive. |
| Processed data goes to `processed/ispd15/<prep_hash>/` | Changing an assumption creates a new folder; existing Drive data is never overwritten |
| Trainer collates one graph per step instead of holding a second copy | Halves host RAM (official DataLoader behaviour) |
| History logs pre-clip gradient norm (mean/max), clipped fraction, non-finite count | Required for the "gradient behaviour" comparison |
| The benchmark adds a repeatability check (same seed twice, 3 epochs) and never touches the test split | Reproducibility comparison; no test leakage into the precision choice |
| `pin_offset_norm` is configurable, and every unresolved assumption is listed in the preprocessing hash and report | Stage 5: keep assumptions explicit and ablatable |
| Smoke config: 3 epochs, first session stops at 2 | The dry run showed the old 2-epoch config could not demonstrate resume |
| New tool `tools/dry_run_notebooks.py` | Runs notebooks 03–05 on CPU with real fixture samples. It passed before this bundle was built |

## 0b. Resume fix (bundle rebuilt 2026-10-19)

The smoke-test resume failed on the A100 with `TypeError: RNG state must be a torch.ByteTensor`. The cause and the fix are in [`INCIDENT_NB05_RESUME.md`](INCIDENT_NB05_RESUME.md). What it means for you:

- **Checkpoints live at `runs/<run_id>/checkpoints/last.pt`**, not `runs/<run_id>/last.pt`. Use cell 4a of notebook 05 (`inspect_run_dir`) to see what is actually on Drive.
- **Your epoch-1–2 smoke run is reused.** If it is intact, cell 4c resumes epoch 3 from it, and `05_smoke.json` records `session1_source: previous_session`.
- **Nothing on Drive is deleted or overwritten.** If the run directory is in an unexpected state, a new `__rerun_<time>` directory is used instead.
- **The precision benchmark is off by default** (`RUN_PRECISION_BENCHMARK = False`). Set it to `True` only once it is approved.
- **Notebooks 01–04 and the processed data are unaffected.** You do not need to re-run notebooks 03 or 04. You do need to upload the new zip and the new notebook 05, then run notebook 02 (raw data on `/content`), then notebook 05.

## 0c. Smoke-test training diagnosis (bundle rebuilt 2026-10-19)

The smoke test logged a gradient norm of exactly 0 and correlations of exactly 0 from epoch 2 on. [`SMOKE_TRAINING_DIAGNOSIS.md`](SMOKE_TRAINING_DIAGNOSIS.md) traces this to prediction collapse at the official heads' `clamp(min=0)`.

- **Do not run the precision benchmark yet.**
- Run notebook 06 Part A, which is read-only. Run Part B only if you authorise it.
- Send back `reports/06_*.json` and the full `reports/05_smoke.json`.

## 0d. Controlled collapse experiment (proposed, not approved)

Notebook 06 Parts A and B have been analysed; see [`COLLAPSE_EXPERIMENT_PLAN.md`](COLLAPSE_EXPERIMENT_PLAN.md). Notebook 07 holds the proposed experiment.

- Cell 1 of notebook 07 only plans; it writes `reports/07_collapse_plan.json`.
- The run cell stays off (`RUN_COLLAPSE_EXPERIMENT = False`) until the experiment is approved.

## 1. Upload (once)

1. The bundle is already built at `verihgn-project\dist\verihgn_code.zip`. To rebuild it, run `.venv\Scripts\python tools\make_colab_bundle.py`.
2. Open <https://drive.google.com> and go to **My Drive**.
   - If `VeriHGN` does not exist, create it (*New → New folder*), then create `code` inside it.
   - If `MyDrive/VeriHGN/` already exists, **do not delete anything** in it.
3. Upload `verihgn_code.zip` into `MyDrive/VeriHGN/code/`. If an older zip is there, Drive asks; choose **Replace existing file**. That replaces only the code zip; data folders are untouched. Each notebook re-extracts the code when the zip changes.
4. Open the [CircuitNet folder](https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB). Click the folder name, then *Organize → Add shortcut*, and choose **My Drive**. A shortcut uses no storage.
5. Upload the five notebooks from `verihgn-project\notebooks\` to Colab (*File → Upload notebook*), or put them in `MyDrive/VeriHGN/notebooks/`.

## 2. Run order

In every notebook, first choose *Runtime → Change runtime type → A100 GPU* and run cells top to bottom. The first cell asks you to authorise Drive; that is your own Google login popup.

| # | Notebook | Stop and report back if |
|---|---|---|
| 1 | `01_environment_setup` | "Drive route not available" **and** the gdown test fails |
| 2 | `02_dataset_fetch_and_audit` | the audit assertion fails |
| 3 | `03_preprocessing` | any assertion fails (failed samples, missing samples, audit, splits) |
| 4 | `04_graph_construction_validation` | "validation failed" |
| 5 | `05_gpu_profiling_and_precision_benchmark` | any error; the benchmark cell can simply be re-run after a disconnect |

Notebooks 02–05 need the raw files on `/content`. If the runtime restarts between notebooks, re-run notebook 02 first; it skips work that is already done.

## 3. What to send back

Download `MyDrive/VeriHGN/reports/` (right-click → Download) or at least:

- `STAGE_REPORT.md` (written by the last cell of notebook 05)
- `01_environment.json`
- `03_preprocessing_official9.json`
- `03_processed_audit_official9.json`
- `04_validation_official9.json`
- `05_peak_memory.json`
- `05_smoke.json`
- `05_precision_benchmark.json`
- `05_persistence.json`

Include any error traceback verbatim. The Stage 6 report (preprocessing counts, Drive access, storage, GPU, peak memory, smoke test, the FP32/FP16/BF16 comparison and a precision recommendation, failures, persistence) will be written **only from those files**.

## 4. Estimates

- Disk on Colab: ~25 GB local.
- RAM: ~30–40 GB peak while graphs for one variant are in memory. The A100 runtime normally has ~80 GB; notebook 01 reports the actual figure.
- Drive: processed data ~3–4 GB; reports and benchmark checkpoints are small.
- GPU time: ~2.5–4.5 h, mostly the benchmark: 3 precisions × 2 seeds × 20 epochs + 3 × 2 × 3 repeat epochs.

All of these are estimates until measured.
