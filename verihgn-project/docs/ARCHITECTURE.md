# Architecture & Infrastructure Plan (Phase 0, updated 2026-10-19)

Status: **proposal, waiting for approval.** Nothing below has been implemented.

**Fidelity rule:**

- The model, graph definition, features, hierarchy, loss, training method and evaluation method follow `RESEARCH_TO_CODE_MAPPING.md`.
- Infrastructure choices below may change **only** where results are stored and how fast they are computed.
- Every option that could change numerics is a config flag, default = official behaviour, and is recorded in the run metadata.

---

## 1. Three-tier topology

```
 ┌──────────────────────────┐        public link / gdown        ┌───────────────────────────┐
 │ Shared Drive folder      │ ─────────────────────────────────▶│ Colab VM (A100-40GB)      │
 │ "CircuitNet" (read-only) │   archives → /content (local SSD) │  preprocess · build graphs│
 │ 1GjW-1LBx1563bg3pHQGv... │                                   │  train · eval · export    │
 └──────────────────────────┘                                   └─────────────┬─────────────┘
                                                                  writes     │ drive.mount
                                                                             ▼
                                                     ┌────────────────────────────────────┐
                                                     │ MyDrive/VeriHGN/ (persistent)      │
                                                     │ cache · runs · checkpoints · export│
                                                     └─────────────────┬──────────────────┘
                                         only export/demo_bundle (MBs) │ Drive for desktop sync
                                                                        │ or manual zip download
                                                                        ▼
                                                     ┌────────────────────────────────────┐
                                                     │ Windows laptop                     │
                                                     │ Claude Code dev · dashboard (static│
                                                     │ site + optional CPU inference API) │
                                                     └────────────────────────────────────┘
```

**How the tiers communicate:** through files on Drive only. There is no network link between Colab and the laptop.

- Colab writes versioned artifacts to `MyDrive/VeriHGN/`.
- The laptop reads only the small `export/demo_bundle_<run>/` folder.
  - **Route A:** Google Drive for desktop syncs just that folder.
  - **Route B:** download it as a zip from the Drive web UI.
- The dashboard loads that bundle from disk, so it works with Colab **off**.
- A *temporary* live link is optional: a Gradio `share=True` URL from notebook 07. It is labelled "live, temporary" in the UI.

## 2. Google Drive storage layout

```
MyDrive/
├── CircuitNet  → shortcut to the shared folder (0 bytes of quota)
└── VeriHGN/
    ├── raw_cache/ispd15/          # OPTIONAL copies of needed archives (≈ 10 GB) to skip re-download
    ├── processed/ispd15/<prep_hash>/<design>/<sample>/   # official per-sample format (.npy, layout.csv, metadata.txt)
    │   └── preprocess_manifest.json   # inputs, constants, c_max (train-only), git hash, timestamps
    ├── graphs/ispd15/<prep_hash>/{train,val,test}.pt    # cached HeteroData lists (official builder output)
    ├── splits/ispd15_seed42.json  (+ ispd15_twinaware.json)
    ├── runs/<run_id>/
    │   ├── config.yaml            # full resolved config + flags + deviations list
    │   ├── env.json               # GPU, VRAM, CUDA, torch/PyG versions, git hash of src + third_party
    │   ├── status.json            # {state: running|completed|failed|early_stopped, epoch, best, updated_at}
    │   ├── history.csv            # per-epoch loss, metrics, lr, time, peak VRAM, GPU util
    │   ├── checkpoints/last.pt, best.pt, epoch_XXX.pt (every k epochs, rotated)
    │   ├── eval/test_metrics.json, per_design.json, per_sample.json
    │   └── predictions/<sample>.npz   # grid gt/pred (L0, native M×N), cell gt/pred, cell xy
    └── export/demo_bundle_<run_id>/   # small, laptop-facing (see §6)
```

**Where the original dataset lives.**

- The **original dataset stays in the shared folder** and is referenced through the shortcut. Copying all of ISPD15 (≈ 24 GB) needs that much free quota. The free plan has 15 GB in total.
- `raw_cache/` holds only the archives we actually use, and only if you choose that (see Decision I-2).
- Processed data, graphs, checkpoints, logs, configs, results and visualizations all live under `VeriHGN/`, as you asked.

## 3. Google Drive access: what is verified and what is not

| Check | Status |
|---|---|
| Folder is publicly link-readable | **Verified.** Listed via the public embedded view; downloaded files via `drive.usercontent.google.com` with curl, no credentials |
| Folder title and owner | Title "CircuitNet". Owner not visible in the public view. It is **not** in your My Drive by default; at most it would appear under *Shared with me* |
| Access via `drive.mount` in Colab | **Not verified.** I cannot run your Colab session. Notebook 01 will test it |
| Access via `gdown` in Colab | **Not verified in Colab** (verified only with curl locally). Notebook 01 will test it |
| Download quota | **Unknown.** Popular public Drive files can hit "too many users downloaded recently" for a while. Fallbacks: (a) mount + copy from the shortcut; (b) `rclone` / Drive API with *your* OAuth inside Colab |

**Notebook 01 access procedure.** The only authentication is `google.colab.auth` or `drive.mount`, done by you in the browser.

1. Mount Drive (you authorize in the popup).
2. Look for a shortcut, by **folder ID**, in this order:
   - `MyDrive/CircuitNet`
   - a scan of `MyDrive` for shortcuts whose target ID = `1GjW-1LBx…`
   - via `drive.mount`'s `.shortcut-targets-by-id/<id>` path
   - Print exactly which method succeeded.
3. If no shortcut is found, print the instructions: open the folder link → *Organize → Add shortcut → My Drive*. Then re-run the cell.
4. Independently, test `gdown` on the 124 MB `graph_infromation.tar.gz`. Record the speed and any quota error.
5. Verify the expected file IDs and sizes (from `DATASET_AUDIT.md`) before extracting.

## 4. Colab strategy

**Notebooks:** sequential and idempotent. Each one starts with the same bootstrap cell: mount, detect, install, set paths.

| NB | Purpose | GPU? | Re-run cost |
|---|---|---|---|
| 01_environment_setup | GPU/CUDA/torch/PyG checks, Drive mount + dataset discovery, disk space, seed | no | seconds |
| 02_dataset_audit | Verify archive IDs/sizes, extract needed members to `/content`, per-design stats | no | skip if done |
| 03_data_preprocessing | Raw → official per-sample format on Drive; train-only c_max; manifest | CPU (parallel) | skip by `prep_hash` |
| 04_graph_construction | Official `build_verihgn_graph` → cached `.pt`; validation report (types, dims, index ranges, NaN/Inf, hierarchy) — **hard stop on failure** | optional | skip by hash |
| 05_model_training | Smoke test → full training with auto-resume | **A100** | resumes |
| 06_evaluation | Test metrics, per-design, baselines, ablations | A100 | per run |
| 07_demo | Export predictions + demo bundle; Gradio live demo | small | seconds |

**Dependencies.** PyTorch is preinstalled in Colab. PyG and torch-scatter are installed from wheels that match the **detected** torch/CUDA version. The cell prints versions and fails loudly on a mismatch. The paper used torch 2.1 / PyG 2.4. We use Colab's current torch and log the versions as an environment deviation; this does not change the maths.

## 5. Hardware profile and A100 plan

**Confirmed hardware (from you):**

- NVIDIA A100-SXM4-40GB
- 40,960 MiB VRAM
- CUDA 13.0 (driver)
- 0 MiB in use at connect time

No training has been run.

**Auto-detection** at the start of every GPU notebook (`src/utils/hardware.py`):

- Reads `torch.cuda.get_device_properties`: name, total VRAM, compute capability, bf16 support.
- Picks a **profile**.
- Writes the result to `env.json`.

| Setting | A100 (≥ 35 GB, sm80) | L4 (≈ 22 GB, sm89) | T4 (≈ 15 GB, sm75) | CPU |
|---|---|---|---|---|
| Batch size (graphs) | **1 (official)** | 1 | 1 | 1 |
| Precision | **Decided by benchmark (I-3):** FP32 (reference) vs fp16-AMP + GradScaler (official) vs bf16-AMP | fp16-AMP or FP32 | fp16-AMP or FP32 | FP32 |
| TF32 matmul | on (see note) | on | n/a | n/a |
| Gradient checkpointing | **official = on**; `off` allowed if measured peak < 30 GB | on | on | on |
| Scatter chunk size | 500 K (official) | 500 K | 250 K | 100 K |
| Keep dataset resident on GPU | yes, if measured total fits (avoids per-step H2D copies) | partial | no | — |
| Subgraph fallback | not expected | if OOM | likely for large designs | smoke tests only (1 small sample) |

**Fidelity notes on the A100 optimizations:**

- **Batch size stays 1**, as in the official code. Batching several graphs would change optimization dynamics, so it is not used to fill the 40 GB.
- **Gradient checkpointing and chunk sizes** trade memory for speed. They are mathematically identical, so they are free to tune.
- **TF32 and bf16 change floating-point rounding only.** They stay opt-in, are logged, and are cross-checked once against fp16 on validation metrics.
- **Keeping the data on the GPU.** The whole cached train/val set (~ 380 mgc graphs, ≤ 150 K cells each) is moved to the GPU once if it fits. That removes repeated CPU→GPU copies. The size is measured first, not assumed.

**Monitoring.** Every epoch logs to `history.csv`:

- `torch.cuda.max_memory_allocated` and `max_memory_reserved` (reset per epoch)
- `nvidia-smi` utilization and memory, sampled by a background thread
- wall time

The smoke test records peak VRAM **per design**, which settles the memory question with real numbers.

## 6. Checkpointing and recovery

- **What is saved:**
  - model, optimizer, LR scheduler and GradScaler state
  - epoch, best metric, patience counter
  - Python / NumPy / torch / CUDA RNG states, DataLoader generator state
  - resolved config, git hashes
- **When:**
  - `last.pt` every epoch (configurable `ckpt_every`)
  - `best.pt` when mean validation Spearman improves (official criterion)
  - `epoch_XXX.pt` every `keep_every` epochs, rotated to N copies
- **Atomic writes:** write to `*.tmp` on Drive, `fsync`, then `os.replace`. A disconnect mid-write leaves the previous file intact. On load, files that fail to deserialize are skipped and the next newest is tried.
- **Resume:** re-running the training cell finds `runs/<run_id>/checkpoints/last.pt`, restores everything, and continues at `epoch+1`. A `--fresh` flag starts a new run id.
- **Run state:** `status.json` has `running` (with a heartbeat timestamp), `completed`, `early_stopped` or `failed`. A run marked `running` whose heartbeat is stale is shown as **"interrupted — resumable"**, never as "training".
- **Determinism caveat:** scatter operations on GPU are not bit-deterministic, so a resumed run is statistically equivalent, not bit-identical. This is documented.

## 7. Saved-artifact dashboard (works with Colab off)

**Demo bundle.** Notebook 07 exports `export/demo_bundle_<run>/` (target < 100 MB):

- `manifest.json`: run id, model hash, config, dataset version, export time, a **provenance flag on every item**
- `metrics/`: test metrics (cell and grid, overall and per design), training history, baselines and ablations — **copied from evaluation outputs, never typed in**
- `predictions/`: per test sample, L0 grid GT / prediction / |error| as float16 arrays (native M×N), plus a cell-level subsample (xy, gt, pred)
- `dataset/`: per-design statistics, split manifest, graph statistics
- `model/best_state_dict.pt` (~ few MB) + config, used for CPU inference on the laptop
- `inputs/`: preprocessed inputs for 2–3 small test samples (e.g. fft / pci_bridge, a few MB), so the laptop can **re-run the real model on CPU** and show it reproduces the saved predictions

**Dashboard.**

- A **static web app**: React + Vite + TypeScript, with Plotly or deck.gl for heatmaps. It reads the bundle from `public/data/`.
- It runs with `npm run dev` or as a built static site (GitHub Pages / Netlify), with no backend required.
- An **optional local FastAPI** (CPU torch + PyG) provides "Live inference" on the bundled samples. The status badge shows **Saved prediction**, **Live inference (local CPU)** or **Live (Colab, temporary URL)**, and a mode is shown only if its backend answers a health check.

**Provenance labels on every chart:**

- *Model output (run X)*
- *Measured result*
- *Paper-reported value (Table N)*: always visually distinct and never mixed into our metrics
- *Illustrative*: avoided where possible

When an artifact is missing, the UI shows "Not evaluated".

**Laptop footprint:** only the demo bundle plus node/npm and a CPU Python environment for the optional API. No raw dataset or full checkpoints history.

## 8. Planned repository layout (refined from the brief)

```
verihgn-project/
├── third_party/VeriHGN/        # official code, MIT, untouched (git submodule-like, pinned 28a6f1e)
├── src/verihgn_repro/
│   ├── config.py               # dataclass config + YAML, profile merge, deviation registry
│   ├── paths.py                # Drive/local path resolution (Colab vs laptop)
│   ├── data/{circuitnet_io.py, preprocessing.py, feature_extraction.py, splits.py, validate.py}
│   ├── models/{official.py (re-export), variants.py (ablation flags)}
│   ├── training/{trainer.py, checkpointing.py, losses.py (re-export), metrics.py (re-export)}
│   ├── evaluation/{evaluate.py, baselines.py, ablations.py}
│   ├── visualization/{congestion_maps.py, plots.py}
│   ├── export/demo_bundle.py
│   └── utils/{hardware.py, seed.py, monitor.py}
├── notebooks/01..07_*.ipynb
├── app/  (web/ React dashboard, api/ FastAPI optional)
├── artifacts/ (git-ignored; laptop copy of demo bundle)
├── tests/
└── docs/
```

This departs from the brief's `src/models/{message_passing,gating,...}.py` split on purpose. The model is **imported from the official code** rather than re-written, which is the strongest fidelity guarantee. The per-component explanations go in docs and the dashboard's architecture page instead.

---

## 9. Phase 1 implementation notes (2026-10-19)

- **Code delivery to Colab.**
  - `tools/make_colab_bundle.py` writes `dist/verihgn_code.zip` (code only, ~0.1 MB).
  - You upload it to `MyDrive/VeriHGN/code/`. Every notebook's bootstrap cell unpacks it when its modification time changes.
  - The laptop never needs the dataset.
- **Raw data in Colab.**
  - Archives are read from the Drive shortcut (or fetched with `gdown`).
  - Extraction covers only the 9 official-scope designs and 6 routability folders, into `/content/circuitnet_ispd15` (local disk, rebuilt each session).
- **Storage format.**
  - `processed/ispd15/<design>/<sample>/sample.npz` + `meta.json`. The official per-sample `layout.csv` would require H/V MacroMargin columns that CircuitNet lacks.
  - H and V labels are stored separately. The (H+V)/2 aggregation happens at graph build (official aggregation).
- **Graph cache.** `graphs/ispd15/graphs_<key>_<split>.pt`, where the key hashes the preprocessing hash, split, grid features, normalization and c_max.
- **Reports.** Each notebook writes JSON to `MyDrive/VeriHGN/reports/`. Notebook 05 refuses to train unless `04_validation_summary.json` says every variant passed.
- **torch_scatter.** If no wheel exists for Colab's torch build, the PyG-backed shim is used (same maths, tested). This is recorded in `env.json`.
