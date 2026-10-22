# Phase 1 Report — Data pipeline, graphs, validation, training engine

Date: 2026-10-19 · Status: **Phase 1 complete. No training has been run on the A100. Waiting for approval.**

Everything below was executed on the Windows laptop (CPU) with the official code at commit `28a6f1e`. Real-data results come from **3 real CircuitNet-ISPD15 samples** (one each of fft_a, pci_bridge32_a and matrix_mult_a), held only in the session scratchpad. The full 383-sample run happens in Colab (notebooks 02–04).

## 1. Implemented components

| Component | File(s) | Notes |
|---|---|---|
| Official-code access | `official.py`, `compat/torch_scatter_shim.py` | Imports the unmodified official modules. `torch_scatter` has no wheel for torch 2.14 (laptop), so a shim delegating to PyG's pure-torch `scatter` is used automatically and recorded in `env.json` |
| Dataset access | `data/circuitnet.py`, `colab.py` | Verified file IDs and sizes. Drive discovery: shortcut, `.shortcut-targets-by-id`, MyDrive scan. `gdown` fallback. Streaming extraction with tarfile's `data` filter |
| Safe loading | `utils/safe_npy.py` | CircuitNet's pickled `.npy` files are read with a restricted unpickler; a test proves code execution is blocked |
| Netlist | `data/netlist.py` | graph_information semantics; macro flag; pin direction (std cells: data-derived; macros: config); netlist signature for twin detection |
| Features | `data/features.py` | Cell 13, net 6, pin 3, geometric 4, grid Eq. 3 (5), code-ablation grid (3). Column names and provenance stored per sample |
| Preprocessing | `data/preprocess.py` | Per sample: `sample.npz` (H, V labels separate) + `meta.json`. Atomic writes, hash-based skipping, parallel workers, manifest |
| Graph assembly | `data/graph.py` | Statement-by-statement mirror of `build_verihgn_graph` using the official helpers. Configurable grid features and normalization. Raw H/V kept on the graph for visualization |
| Splits | `data/splits.py` | Official seed-42 split (tested equal to the official method); pair-aware split; leakage report |
| Validation gates | `data/validate.py` | 25+ checks per graph, cross-split leakage, official-model forward/backward |
| Training engine | `training/trainer.py`, `training/checkpointing.py` | Official optimization; precision modes; FP32 validation and test; atomic checkpoints; exact resume; heartbeat status; history with VRAM |
| Precision benchmark, VRAM probe | `training/benchmark.py` | I-3 measurement tooling; nothing is chosen automatically |
| Evaluation and export | `evaluation/evaluate.py`, `export/demo_bundle.py` | Official metrics (pooled, per design, per sample); prediction files; saved-artifact bundle with provenance tags |
| Config | `config.py`, `configs/*.yaml` | `base.yaml` = official hyperparameters (tested); deviation registry written into each run |
| Notebooks | `notebooks/01–05` (from `tools/build_notebooks.py`) | Shared bootstrap; validation gate enforced before training; every expensive step opt-in |

The dashboard (D8) is not built yet. Phase 1 delivers its data side: the demo-bundle exporter and its format, which is tested.

## 2. Directory structure

See `README.md` § Layout. New since Phase 0: `src/`, `configs/`, `notebooks/`, `tests/`, `tools/`, `pyproject.toml`, `requirements.txt`, `README.md`, `.gitignore`. Phase 0 documents and `third_party/` are preserved.

## 3. Dataset and preprocessing validation (real data)

| Check | Result |
|---|---|
| Samples processed | 3 / 3; 0 failures; 7.2 s total (0.8–5.3 s per sample, CPU) |
| Our Eq. 3 cell-density feature vs CircuitNet `cell_density` | **Exactly equal** on all 3 samples (bit-identical) |
| Placement inside the label grid | yes, all 3 |
| Label shapes | H and V maps = M×N = placement gcell grid: 275×274, 105×103, 507×507 |
| Std-cell pin semantics | every standard cell has exactly one output pin (real-data test) |
| Feature dims and order | cell 13 / net 6 / pin 3 / geom 4 / grid 5 or 3; column names stored in each `meta.json`; order checked by a hand-computed unit test |
| Save → reload | sample round trip and graph cache round trip are identical (tests) |

## 4. Graph construction validation (real data, all 4 variants)

All gates **passed** for each combination of {paper_eq3, official_code_available} × {per_dataset, per_sample}. The unmodified official model's forward, loss and backward passes ran on every split with finite loss and gradients.

| Sample | Grid | Cells | Nets | L0 / L1 tiles | Pin edges | Geom edges | Grid–net edges | Label non-zero |
|---|---|---|---|---|---|---|---|---|
| 380-mgc_matrix_mult_a-util_0.75 | 507×507 | 149,655 | 154,284 | 257,049 / 64,516 | 439,891 | 1,312,028 | 740,288 | 12.5 % |
| 3-mgc_pci_bridge32_a-util_0.85 | 105×103 | 29,521 | 29,985 | 10,815 / 2,756 | 85,112 | 257,742 | 154,012 | 7.9 % |
| 68-mgc_fft_a-util_0.7 | 275×274 | 30,631 | 32,088 | 75,350 / 18,906 | 97,138 | 270,898 | 155,490 | 1.8 % |

**Checks run on every graph:**

- **Types:** exactly 4 node types and 11 relation types.
- **Dimensions and values:** feature dimensions, plus NaN/Inf checks.
- **Indices:** `long` dtype and in range, for every relation.
- **Reverse relations:** each is the exact reverse of its forward relation.
- **Geometric edges:** symmetric, no self-loops.
- **Grid edges:** 4-connected adjacency count; every L0 tile has exactly one L1 parent, which is its 2×2 block.
- **Cell–grid edges:** one per cell.
- **Labels:** cell labels equal the label of their tile; label support equals the raw H+V support; per-sample labels peak at 1.
- **Splits:** no sample or design appears in two splits.

**Normalization observation (expected, not altered).** With per-dataset c_max taken from the training split (1.985 on these 3 samples), held-out labels can exceed 1 (fft_a reaches 1.404). They are not clipped. The paper says labels are "approximately within [0, 1]".

## 5. Test results

`34 passed` (`pytest`, 4 min 6 s on CPU, with the real-data fixtures enabled). Without fixtures, 30 pass and 4 real-data tests skip.

| Area | Tests |
|---|---|
| Official equivalence | `assemble` == official `build_verihgn_graph` tensor for tensor (even and odd grids); official split == official `_split_designs`; `base.yaml` == official `ispd2015.yaml` |
| Features | hand-computed cell/net/pin values; difference-array coverage vs brute force; kNN symmetric; CircuitNet density convention; direction rules |
| Splits | official membership + measured twin leakage; pair-aware leak-free, complete, deterministic |
| Pipeline | preprocess round trip and skip; graphs pass the gates for all 4 variants + cache round trip; corruption detected |
| Training | **resumed run (2+1 epochs) bit-identical to an uninterrupted 3-epoch run (CPU, FP32)**; corrupted `last.pt` falls back; stale heartbeat shows "interrupted"; FP32 test evaluation; prediction export; demo bundle decodes back to exactly the saved predictions |
| Safety | the shim matches reference scatter (including empty segments); malicious pickle blocked |
| Real data | exact cell-density match; pin semantics; all gates, both normalizations, both feature sets |

## 6. Official split, exact membership (computed and tested)

| Split | Official seed 42 | Pair-aware (seed 42, twins kept together) |
|---|---|---|
| train | pci_bridge32_a, edit_dist_a, matrix_mult_b, des_perf_a, pci_bridge32_b, fft_a (251 samples) | computed in Colab from real signatures (notebook 03) |
| val | matrix_mult_a (45) | — |
| test | fft_b, matrix_mult_c (87): **both twins of train designs** | 2 designs, leak-free by construction (tested) |

## 7. Remaining research uncertainties

| # | Item | Class |
|---|---|---|
| 1 | Which congestion map the authors used (we use GR overflow) | A / D |
| 2 | Geometric edge rule: kNN, k = 8, cell centres, micrometre features unscaled | U |
| 3 | Macro pin direction: no LEF anywhere in the dataset (DEF archive fully scanned); default input | U |
| 4 | Pin offset normalization (by net bbox width/height) | U |
| 5 | HPWL and net bboxes from cell centres (no pin offsets in data) | A |
| 6 | Paper's label downsampling target resolution (we use the native grid) | A |
| 7 | Origin of the official H/V MacroMargin features (not in CircuitNet; omitted, not approximated) | unresolved |
| 8 | Which dataset subset produced Table 2 | unresolved |

**Corrected since Phase 0.** The "IO-port pins attached to the first macro" were a misreading. They are the macros' own pins, which are named after their nets (DEF-confirmed). IO ports are not in graph_information at all.

## 8. Colab instructions

See `README.md` § Colab. In short: upload `dist/verihgn_code.zip` → add the CircuitNet shortcut → run notebooks 01 → 02 → 03 → 04. Notebook 05 checks the validation gate, then runs only what you enable.

**I-1 and I-2 remain open** until you run notebook 01. It writes `MyDrive/VeriHGN/reports/01_environment.json` with the access result and free space.

## 9. Ready for model training?

**Ready, once notebook 04 passes on the full 383-sample dataset:**

- data pipeline
- graph construction
- validation gates
- the official model on generated graphs
- checkpoint and resume
- precision benchmark
- VRAM probe
- FP32 evaluation
- prediction export

**Not yet done:**

- any A100 run (I-1/I-2 access and storage are unverified)
- the full-dataset audit and validation (notebooks 02–04)
- the precision benchmark and the precision choice
- training
- baselines and ablation runs
- the dashboard UI

**Proposed next step (needs your approval):**

1. Run notebooks 01–04 on the A100.
2. Run notebook 05's VRAM probe and smoke run, then the precision benchmark.
3. Bring back the reports (`MyDrive/VeriHGN/reports/*.json`). Full training starts only after you choose the precision.
