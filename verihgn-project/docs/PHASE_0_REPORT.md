# Phase 0 Report — VeriHGN Reproduction

Date: 2026-10-19 · Status: **Phase 0 complete. Waiting for approval. No model, training or dashboard code has been written.**

Companion documents:

- `RESEARCH_METHODOLOGY.md`
- `DATASET_AUDIT.md`
- `RESEARCH_TO_CODE_MAPPING.md`

## 1. Environment (local)

| Item | Finding |
|---|---|
| OS / shell | Windows 11, PowerShell + Git Bash |
| Python | 3.12.8 (bash `python`), 3.14 (py launcher). **No numpy/torch/PyG installed**; a throw-away venv with numpy was created in the session scratchpad for inspection only |
| GPU (local) | **None** (no `nvidia-smi`). All training runs on Colab |
| GPU (Colab) | **NVIDIA A100-SXM4-40GB, CUDA 13.0, confirmed by the user on 2026-10-19.** No training has been run yet. VRAM headroom versus the paper's ≈ 30 GB ISPD2015 peak is about 10 GB |
| Roles | **Google Drive** = persistent storage (cache, checkpoints, artifacts). **Colab** = GPU compute. **Windows laptop** = development + dashboard |
| Disk | 585 GB free |
| Network | GitHub, Zenodo and Google Drive reachable. Python `urllib` fails TLS verification (Cloudflare WARP interception); curl and pip work |
| Existing work | `Desktop/ai for electronics project/` held only `Google.pdf` (the paper). Nothing was overwritten. New: `verihgn-project/docs/*` and `verihgn-project/third_party/VeriHGN` (official clone) |

## 2. What was successfully inspected

| Source | Status | Method | Notes |
|---|---|---|---|
| Paper (arXiv v3 PDF, local) | **Inspected** (all 10 pages) | `pdftotext` | Math glyphs lost in extraction; equations reconstructed from the prose and the code |
| Official GitHub repo | **Inspected** (every file) | `git clone` | 7 Python files, 1 config. Tags v1.0.0 = v2.0.0 = HEAD |
| Zenodo record | **Inspected** | REST API | Only the 20 KB code zip |
| Drive dataset: ISPD15 | **Partially inspected** | Public folder view + curl | Full listings of the gcell placement and routability archives. Graph information fully loaded. Every routability map loaded for 3 samples. 1 DEF opened |
| CircuitNet code and docs | **Inspected** (relevant parts) | GitHub raw + docs site | `feature_extraction/src/read.py` (label formulas), `build_graph_demo/build_graph.py` (pin_attr semantics), routability-feature docs |
| Drive dataset: N14 / N28 | **Listed only** | Public folder view | Files not opened or sized |

## 3. What could not be inspected

- The **preprocessing code** that produced the official input format (`*_features.npy`, `layout.csv`, `metadata.txt`). It is not published anywhere I could find.
- Full contents of `DEF.tar.gz` (11.4 GB): whether a LEF is included, and pin directions.
- Label arrays for the other 530 samples. Only 3 samples' arrays were loaded; the names and counts of all 533 were verified.
- Which congestion map the authors used. This is **unknowable** from the published material; see D1.
- The origin of the code's `H_MacroMargin` / `V_MacroMargin` / `PinRUDY` columns. A web search found nothing.
- Any pretrained checkpoint. **None exists.**

## 4. Verified architectural facts (paper and code agree)

- **Graph:** cell, net, grid L0 and grid L1 nodes. Grids form 2×2 hierarchical blocks with **2 levels**.
- **Feature sizes:** cell 13, net 6, grid 5, pin edge 3, geometric edge 4.
- **Message passing:**
  - pin-aware MLP for cell→net
  - linear + mean for grid→net, cell→grid and net→cell
  - scalar-gated geometric cell–cell messages
  - residual cell update
  - fine→coarse φ_fc, coarse→fine φ_cf
  - vector-gated hierarchical integration
- **Heads:** separate cell and grid heads.
- **Hyperparameters:** d = 128, L = 3, AdamW (5e-4, 1e-4), StepLR (50, 0.5), ≤ 200 epochs, patience 20 on mean validation Spearman.
- **Loss:** WMSE (α = 10 for y > 0) + λ = 0.5 · ReLU(std y − std ŷ)², for cell and grid.
- **Metrics:** MAE, RMSE, Pearson, Spearman, Kendall, for cell and grid separately.
- **Dataset:** ISPD15 has 533 placement samples (matches Table 1). The official code uses 9 designs / 383 of them.
- **Labels:** GR congestion maps are co-registered **exactly** with the placement gcell grid (`cell_density` reproduced bit-for-bit).

## 5. Paper ↔ official code conflicts (resolution needed or proposed)

| # | Topic | Paper | Code | Proposed resolution |
|---|---|---|---|---|
| C1 | Grid features | densities + mean area + mean HPWL | RUDY, PinRUDY, MacroRegion, H/V MacroMargin | D3 |
| C2 | Label normalization | per-dataset c_max | per-sample max (the c_max argument is unused) | D4 |
| C3 | L0 grid adjacency | used | built but unused | Follow the code; L0 adjacency as an ablation |
| C4 | Readout | Fig. 5: concatenated readout | separate heads | Follow the code (matches the §3.5 text) |
| C5 | L1 features | used at every level | computed but never reach the output | Follow the code; flag it in the docs |
| C6 | Dataset scope | 20 designs, 100K–1.2M cells | superblue excluded → 9 usable designs, 29K–150K cells | Follow the code; report the discrepancy |

## 6. Unverified assumptions (each carries a fidelity risk)

1. **Geometric edge rule.** Proposed: symmetric kNN, k = 8, on cell centres. *Impact:* changes graph density and geometric-message-passing behaviour.
2. **Pin direction.** Pin names (`o` = output) unless a LEF is found. *Impact:* the input-pin ratio and the direction bit in the pin feature.
3. **Cell feature column order** (cols 2–12). Taken from the order of paper Eq. 1. *Impact:* none on the maths, as long as we are self-consistent.
4. **H/V congestion source map.** GR overflow-based is the best-supported choice but unproven; this is D1, see the audit §3.5.
5. **L0 resolution.** Native gcell grid, measured: fft_a 275×274, pci_bridge32_a 105×103, matrix_mult_a 507×507. The paper's downsample target is unknown. *Impact:* graph size and memory.
6. **HPWL** is computed from cell centres (no pin offsets in the data).

## 7. Dataset access you need to provide

**Nothing secret.** The folder is publicly link-shared. You need:

- A Google account for Colab.
- Optionally, a Drive shortcut to the CircuitNet folder.
- About **1–3 GB of free Drive space** for cached preprocessed graphs, checkpoints and artifacts.

The raw archives (≈ 10–24 GB) are downloaded to Colab's temporary disk each session, or once into Drive if you have the space.

## 8. Implementation risks

| Risk | Severity | Mitigation |
|---|---|---|
| **GPU memory.** The paper reports ≈ 30 GB peak on ISPD2015 (L40S 48 GB). **The A100-40GB is confirmed**, so the paper configuration should fit. Smaller GPUs (T4 15 GB, L4 22.5 GB) may not | **Medium** (was High) | On the A100, run the paper settings unchanged. On every GPU: detect it automatically (name, VRAM, bf16 support), **measure** peak memory per design in the smoke test, and choose a profile. A100 → AMP (bf16 when supported), batch size 1, no subgraphing. Smaller GPUs → keep gradient checkpointing and lower the chunk sizes; if still out of memory, use tile-based subgraph training, logged as a deviation in run metadata |
| No released preprocessing → our features ≠ authors' | High | Emit the official format, reuse the official model and graph builder unchanged, label it "adapted reproduction", and do not claim to match Table 2 |
| Tiny split (6 train / 1 val / 2 test designs) | Medium | Report per-design metrics; optional design-level k-fold as an additional, clearly separated experiment |
| Netlist twins across splits (fft_a/b, matrix_mult_b/c) | Medium | D5 |
| Post-route DEF contains wiring | Medium | Never read the ROUTED geometry; inputs come only from placement + netlist |
| Colab session limits (≈ 12 h free; disconnects) | Medium | Resumable checkpoints on Drive every epoch; cached graphs |
| Download time (≈ 7 MB/s observed locally; 2 GB in 283 s) | Low–Med | Download only the needed archives; cache extracted per-sample arrays on Drive |
| CircuitNet-N14/N28 are much larger (multi-part archives) | Medium | Phase 1 scope = ISPD2015 only (official config); N28 optional later |

## 9. Recommended architecture

> **Updated 2026-10-19:** the full infrastructure plan now lives in `ARCHITECTURE.md`. It covers:
> - the Drive / Colab / laptop topology
> - the Drive layout
> - the access-verification procedure
> - A100 profiles
> - checkpoint and recovery
> - the demo bundle
>
> The summary below is still valid.

- **Code reuse.** Vendor `third_party/VeriHGN` (MIT) **unchanged** and import its `graph_builder`, `VeriHGN` model, loss and metrics. Our code wraps it and does not fork it. Any variant (ablations, a c_max fix) lives in subclass or adapter modules with a config flag, so "official behaviour" stays one switch away.
- **Our new code** (`src/`):
  - `data/circuitnet_ispd15.py`: archive download and extraction, safe loading.
  - `data/feature_extraction.py`: raw → official per-sample format.
  - `data/splits.py`: manifest.
  - `training/trainer.py`: resumable wrapper around the official loss and metrics.
  - `evaluation/`: baselines and ablations.
  - `visualization/`: heatmaps from real predictions.
  - `tests/`.
- **Colab notebooks (7):**
  1. setup (GPU check, Drive mount, data discovery)
  2. dataset audit
  3. preprocessing (cached to Drive)
  4. graph construction and validation
  5. training (resumable)
  6. evaluation, baselines and ablations
  7. predictions export + Gradio demo
- **Artifacts.**
  - Training writes to `MyDrive/verihgn_artifacts/`: `checkpoints/`, `metrics/*.json`, `history.csv`, `predictions/<sample>.npz` (grid GT, prediction and error at L0; cell GT and prediction), and `dataset_meta.json`.
  - A small "demo bundle" (a few test samples downsampled + all metrics) is exported to the repo for the dashboard.
- **Dashboard.**
  - Static web app (React + Vite or plain TS + Plotly) that reads only the exported JSON / compact binary artifacts.
  - It works offline and is hostable free on GitHub Pages, Netlify or Vercel, which gives saved-artifact demo mode with no Colab dependence.
  - Live inference is a separate optional FastAPI + Gradio backend: Colab `share=True` for temporary links, or Hugging Face Spaces (CPU) for a persistent small-sample backend.
  - The dashboard shows "Live / Offline" status honestly.

## 10. Final decision table (Phase 0 closure)

**Status meanings:**

- **Verified:** fully supported by the paper, the official code or measured data.
- **Adaptation:** best-supported choice, documented as a deviation.
- **Approval:** needs your sign-off.
- **Open:** can only be settled in Colab.

| ID | Decision | Proposal | Evidence | Status |
|---|---|---|---|---|
| **D1** | Congestion label | (GR_H_overflow + GR_V_overflow)/2 → log1p → normalize; native gcell grid | Paper: "global-routing" maps, sparse congestion. Official: WMSE weight on `y > 0` (needs a sparse label). CircuitNet: identical label in its own task. Measured: 1–12 % non-zero, grid-aligned. **No source states the map type** | **Adaptation**, needs **Approval** |
| **D2** | L0 resolution | Native GR gcell grid (fft_a 275×274; pci_a 105×103; mm_a 507×507). No resampling | Alignment with placement **verified exactly**. The paper's downsample target is unknown | **Adaptation**, needs **Approval** |
| **D3** | Grid features | Primary: paper Eq. 3 features, computed by us. Ablation: the official code set, but only 3 of its 5 maps exist (RUDY, RUDY_pin, macro_map); H/V MacroMargin is absent from CircuitNet, so that set cannot be reproduced in full | P≠C; dataset listing | **Adaptation**, needs **Approval** |
| **D4** | Label normalization | **Both** modes are run and reported: `per_dataset` (paper; c_max from train samples only) and `per_sample` (official code). No selection by metric. The paper-stated mode is labelled "paper", the other "official-code" | Paper §4.1 vs `graph_builder.py:236-239` | **Verified conflict**; plan needs **Approval** |
| **D5** | Split | Official seed-42 split (train 251 / val 45 / test 87 samples) **plus** a twin-aware split. Both reported | Twin leakage **measured**: both test designs have a netlist twin in train | **Verified** (split computed); plan needs **Approval** |
| **D6** | Scope | Primary: official 9 designs / 383 samples. Optional `ISPD2015-all` (14 designs / 533 samples) on the A100 as a labelled extra. N28 later, optional | Table 1 (533) = full set; official code = 9-design subset; Table 2's subset is unknowable | **Verified** facts; scope needs **Approval** |
| **D7** | Compute | A100-40GB primary; auto-profile for L4 / T4 / CPU | Your report | **Verified** (user-confirmed; no run yet) |
| **D8** | Dashboard | Static React + Plotly site from the saved demo bundle; optional local CPU API + temporary Gradio | Architecture doc | needs **Approval** |
| **I-1** | Drive access from Colab | Notebook 01 tests mount + shortcut, `.shortcut-targets-by-id`, and `gdown`. Public link access already verified locally | curl | **Open** (testable only in your session) |
| **I-2** | Raw-archive cache on Drive | Cache the ~10 GB of needed archives if your Drive has space; otherwise re-download per session | — | needs **Approval** (tell me your free Drive space) |
| **I-3** | Precision | Run a precision benchmark before fixing a default (see below) | Official code = fp16 AMP for training and validation; `test.py` = FP32 | needs **Approval** |

**Note on I-3.** You asked for FP32 as the faithful baseline "unless the official implementation establishes another precision mode". It does: `train.py` trains and validates under fp16 autocast with GradScaler. I therefore propose:

- **Smoke-scale benchmark on the A100** (same seed, same split, ~20 epochs, 2 seeds each), comparing **FP32 (TF32 off)**, **FP16-AMP (official)** and **BF16-AMP**.
- **Record for each:** validation and test metrics, NaN/overflow events and GradScaler skips, s/epoch, peak VRAM.
- **Then present** the table and let you pick the default for full training.
- **Until then:** FP16-AMP is labelled "official", FP32 "numerical reference".
- **Evaluation** is always also run in FP32, as in `test.py`.

**Research-fidelity statement.** Because the authors' preprocessing is unpublished and several map/feature choices are unstated, this project will be presented as an **adapted reproduction**: the official VeriHGN model and training code run on CircuitNet data that we preprocess ourselves. It will **not** claim to reproduce the numbers in Tables 2–6. Paper values appear only as labelled reference values.

## 11. Time estimate for the remaining phases (rough)

| Phase | Your time | Compute |
|---|---|---|
| Phase 1–4: structure, preprocessing, graphs | — | 3–6 h of my work + ≈ 1–2 h of your Colab preprocessing runs |
| Phase 5–6: model wiring, smoke tests, training | — | ≈ 2–4 h of Colab GPU per full run (paper: 97 min on L40S) |
| Phase 7–8: evaluation, ablations, visualization | — | Ablations multiply GPU time (×7) |
| Phase 9–10: dashboard + deployment | — | ≈ 1 day of my work |

**Smaller option.** If you need something to show quickly, a **"vertical slice"** is possible:

- 2 designs, ≈ 20 samples, preprocessing in about 30 min
- a short Colab training run (≈ 30 min on T4)
- real predictions in a minimal dashboard

It can be ready well before the full reproduction.
