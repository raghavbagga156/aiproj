# Research-to-Code Mapping (Phase 0, final)

**References:**

- Official code: `third_party/VeriHGN` at tag v2.0.0 = commit `28a6f1e` (v1.0.0 = v2.0.0 = HEAD; identical to Zenodo 10.5281/zenodo.20496640).
- "Our implementation" = the **plan** for Phase 1+. Nothing is implemented yet.

## Fidelity classes (one per row)

| Class | Meaning |
|---|---|
| **P** | Directly supported by the paper |
| **C** | Verified in the official implementation |
| **D** | Derived from dataset documentation / dataset inspection (CircuitNet code, docs, measured arrays) |
| **A** | Engineering adaptation, required because the author preprocessing code or details are unavailable |
| **U** | Unresolved assumption, with no supporting source |

Combinations such as "P+C" mean every listed source agrees. "P≠C" marks a conflict; the row says what we follow. **No row is "exact reproduction of published results"**: the preprocessing (A rows) makes the whole pipeline an *adapted reproduction*.

## Mapping table

| Component | Paper description | Official repository implementation | Our implementation (planned) | Fidelity class |
|---|---|---|---|---|
| Node types | Cell, Net, Grid (multi-level) §3.1 | `cell`, `net`, `grid_l0`, `grid_l1` (`data/graph_builder.py:245-259`) | Same (official builder reused) | **P+C** |
| Grid hierarchy / #levels | 2×2 blocks; K = 2 optimal (Fig. 6c) | Exactly L0 + L1, `ceil` sizes, zero-pad pooling | Same | **P+C** |
| Grid L0 resolution | "design-specific global-routing resolution", downsampled to an unstated target | Whatever `metadata.txt` says | Native CircuitNet GR gcell grid (fft_a 275×274), proven aligned with placement | **D** (alignment) + **A** (no downsample; target unknown) |
| Cell features (13) | Eq. 1 list | Loads `cell_features.npy`; cols 0–1 = raw x, y; per-graph z-score | Compute Eq. 1 from `instance_placement_micron` + `graph_information`, in Eq. 1 order | **P** (definition), **C** (dim, x/y position), **A** (computation), **U** (cols 2–12 order) |
| Pin direction (for the in-pin ratio and pin-edge feature) | dir ∈ {0, 1} | Precomputed | Std cells: pin `o` = output (the only output pin of all 16 masters; tested on real data). Macro pins: config, default input | **D** (std cells) / **U** (macro pins: no LEF in the 11.4 GB DEF archive, DEF gives directions only for IO ports) |
| Net features (6) | Eq. 2 | `net_features.npy`, z-score | Compute from cell-centre bboxes | **P** + **C** (dim) + **A** (centre-based HPWL; no pin offsets in the data) |
| Grid features (5) | Eq. 3: cell/pin/net density, mean cell area, mean net HPWL | RUDY, PinRUDY, MacroRegion, H/V MacroMargin | **D3**: Eq. 3 computed by us (primary) | **P≠C** → follow **P**, computed **A**. The code set is **only 3/5 reproducible** (RUDY, RUDY_pin, macro_map exist; MacroMargin does not) |
| Pin-edge features (3) | Eq. 4 | `pin_features.npy` | Compute | **P** + **C** (dim) + **A** |
| Geometric edges | "spatially nearby cells" | Precomputed, rule unknown | Symmetric kNN, k = 8, on cell centres | **U** |
| Geometric edge features (4) | Eq. 5 | `geom_edge_features.npy` | Compute | **P** + **C** (dim) + **A** |
| Pin edges (cell ↔ net) | Netlist incidence | `pin`, `pin_rev` | Built from `pin_attr` (index semantics confirmed by CircuitNet's `build_graph.py`) | **P+C+D** |
| IO ports | Not discussed | Not visible | Not in graph_information (DEF `PINS` only); nothing to drop. Net-named pins belong to macros (DEF-confirmed) | **D** |
| Cell–grid edges | Fig. 2 ④ | One tile per cell, by raw x/y | Tile of the cell (gcell coordinates, same convention as `cell_density`) | **P+C+D** |
| Grid–net edges | Fig. 3 / ablation | Net tile-bbox; > 100 tiles dropped; 5 M budget | Same (official builder) | **P** (existence) + **C** (rule) |
| Grid adjacency | 4-connected, all levels | Built at L0 + L1; only L1 used | Follow C; L0-adjacency variant as an ablation | **P≠C** → **C** |
| Hierarchical edges | parent ↔ 2×2 children | `child_of` / `parent_of` | Same | **P+C** |
| Input encoders | — | Linear-ReLU-Linear per type; shared L0/L1 grid encoder | Same | **C** |
| Cell→Net msg | Eq. 6 | `mlp_pin` + mean | Official module | **P+C** |
| Grid→Net msg | Eq. 7 | Linear + chunked mean | Official module | **P+C** |
| Net update | "analogous residual" | LN(ReLU(W[·]) + h) + dropout | Official module | **C** |
| Cell→Grid msg / grid update | Eq. 7 + "grid-level messages" | No L0 grid messages | Official module | **P≠C** → **C** |
| Net→Cell msg | Eq. 7 | Uses the updated net state | Official module | **P+C** |
| Geometric gated msg | Eqs. 8–9 | Scalar gate, chunked mean | Official module | **P+C** |
| Cell update | Eq. 10 | + LayerNorm, dropout | Official module | **P+C** |
| Fine→coarse / L1 spatial / coarse→fine | Eqs. 11–12 | `phi_fc`, `W_l1_spatial`, `phi_cf` | Official module | **P+C** (L1 spatial: **C**) |
| Hierarchical gate | Eqs. 13–14 | Vector gate | Official module | **P+C** |
| L1 input features | Same definition at every level | Computed but never reach the output | Reproduce; flagged in docs | **C** (observed) |
| Prediction heads | Separate heads (§3.5); Fig. 5 concatenated | Two MLP heads, clamp ≥ 0 | Official module | **P (text) + C**; Fig. 5 **P≠C** → **C** |
| Cell-head clamp in training, HEADPATH arm U only (`docs/HEADPATH_PLAN.md`) | — | `clamp(min=0)` in every forward pass | **Diagnostic deviation, not the reproduction:** clamp omitted in training forward passes only (`model.cell_head_train_clamp: false`, run-id suffix `__DEV-cellhead-train-unclamped`); official clamped forward for every evaluation; the official loss then sees raw cell outputs (weighted MSE and variance regulariser) | **DEV** (neither P nor C; results never reported as reproduction results) |
| Grid label source | GR "congestion maps" (map type unstated) | `H_Congestion`, `V_Congestion` (origin unpublished) | (GR_H_overflow + GR_V_overflow)/2; semantics from CircuitNet `read.py` | **D** + **A** (best-supported, not proven identical) |
| Label transform | log(1+c) | log1p | Same | **P+C** |
| Label normalization | ÷ **per-dataset** c_max | ÷ **per-sample** max (c_max argument ignored) | **Both run and reported.** `per_dataset` (c_max from **train** samples only) and `per_sample` (official code) | **P≠C** → both kept; neither chosen by metric |
| Cell label | Undefined | `grid_y[tile(cell)]` | Same | **C** |
| Loss | Eq. 15 + variance regularization (no formula) | WMSE w = 10·[y > 0] + 1·[y = 0]; ReLU(σ_y − σ_ŷ)²; λ = 0.5 | Official functions | **P** (form) + **C** (exact) |
| Optimizer / scheduler | AdamW 5e-4 / 1e-4; StepLR 50 / 0.5 | Same | Same | **P+C** |
| Epochs / early stopping / selection | 200 / 20 / mean val Spearman | Same | Same | **P+C** |
| d, L | 128, 3 | 128, 3 | Same | **P+C** |
| Dropout, grad clip, batch size | — | 0.15, 1.0, 1 graph | Same | **C** |
| Numeric precision | Not stated | Train + val under **autocast fp16 + GradScaler**; `test.py` in FP32 | Benchmark FP32, FP16-AMP (official) and BF16-AMP on the A100; default chosen from evidence (I-3) | **C** (official = fp16 AMP); default **pending** |
| Gradient checkpointing, chunked scatter | — | On; 500 K chunks | On (official). Off only if measured memory allows; the maths are identical | **C** |
| Dataset scope | 533 samples; "20 designs" | 9 designs / 383 samples (superblue filtered) | Official 9-design subset primary; `ISPD2015-all` optional | **C** + **D** |
| Split | Not described | Design-level 70/10/20, seed 42 | Official split (exact membership computed) + twin-aware split | **C** + **D** (twin leakage measured) |
| Training loop | — | No resume; best-only checkpoint | Resumable wrapper; same optimisation | **C** + **A** (engineering only) |
| Metrics | 5 metrics, cell/grid | Pooled over test; Kendall on a 50 K subsample | Official functions + per-design + per-sample | **P+C** |
| Baselines | 6 published models | None | Simple baselines (ours); paper baselines only if feasible | **A** / not implemented |
| Ablations | 7 variants (N28) | None | Flags (on ISPD2015 unless N28 is added) | **A** |
| Storage format | — | Per-sample `*.npy` + `layout.csv` + `metadata.txt` | `sample.npz` + `meta.json` per sample, with H and V labels kept separately. The official `layout.csv` cannot be written honestly (needs H/V MacroMargin) | **A** |
| Preprocessing (DEF → features) | "parse placement DEF…" | Not released | CircuitNet `instance_placement_*` + `graph_information` + routability maps (no DEF parsing) | **A** |
