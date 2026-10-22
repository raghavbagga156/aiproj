# VeriHGN — Research Methodology (Phase 0 extraction)

**Paper:** Runbang Hu, Bo Fang, Bingzhe Li, Yuede Ji. *VeriHGN: Heterogeneous Graph-Based Congestion Prediction for Chip Layout Verification.* KDD '26 (Proc. 32nd ACM SIGKDD, V.2), Jeju, Aug 2026. DOI 10.1145/3770855.3818099. arXiv:2603.11075v3 (4 Jun 2026).

**Source used:** the local PDF `Desktop/ai for electronics project/Google.pdf`, which is arXiv v3. Its text was extracted with `pdftotext`. The extraction **dropped most Greek letters and math symbols** in Eqs. 1–15. Every equation below was reconstructed from the surrounding prose **and checked against the official code** (`third_party/VeriHGN`, tag v2.0.0 = commit `28a6f1e`). The label at the end of each item says which source supports it:

- **[P]** — paper
- **[C]** — official code
- **[P+C]** — both agree
- **[P≠C]** — they disagree; see `RESEARCH_TO_CODE_MAPPING.md`

---

## 1. Research problem

Predict **routing congestion** for a *placed* (pre-routing) VLSI design. Exact congestion is normally available only after (global/detailed) routing, which is expensive and iterative. The model predicts it at two granularities [P]:

- **Cell level:** one value per standard cell or macro instance.
- **Grid level:** one value per spatial bin (gcell tile) of the placement area.

## 2. Motivation

- Geometry-based predictors (RouteNet, ST-FPN, Lay-Net) rasterize the layout into images. They miss netlist topology and long-range routing demand [P].
- Topology-based predictors (CongestionNet, CircuitGNN, MIHC) keep connectivity but lack fine spatial context. Their multi-view variants fuse the two views late [P].
- Congestion emerges from **logical connectivity × physical placement × multi-scale resource competition**. That calls for one unified relational representation [P].

## 3. Proposed solution

One heterogeneous graph with **cell, net and multi-resolution grid nodes**. Relation-specific message passing runs over it, plus hierarchical fine↔coarse grid propagation with gated integration, two regression heads, and a congestion-weighted loss with variance regularization [P].

## 4. Graph representation

### 4.1 Node types [P+C]

| Node | Meaning | Prediction target |
|---|---|---|
| Cell | standard cell or macro instance | cell congestion |
| Net | electrical net | — |
| Grid (L0, L1, …) | spatial bin at a resolution level | grid congestion (L0 only) |

### 4.2 Hierarchical grid [P+C]

- **L0** is the finest grid, M×N.
- Each level-(l+1) node is a non-overlapping **2×2 block** of level-l nodes.
- The text says "all grid levels are included".
- The sensitivity study (§4.4) finds **K = 2 levels** optimal; K = 3 gives no gain.
- The official code implements exactly **two levels (L0, L1)**. At odd sizes the L1 size is `ceil(M/2)×ceil(N/2)`, with zero-padding before pooling.
- Fig. 2 illustrates three levels, but that is illustrative only.

### 4.3 Node features

**Cell — 13 dims (Eq. 1) [P+C dim; content P only]**

- **Geometry:** x, y (placement coordinates), w, h, area = w·h.
- **Topology:** incident-net count, pin count, mean and max HPWL of incident nets, mean incident-net degree.
- **Type:** macro indicator, input-pin ratio, normalized master-type frequency (fraction of cells sharing the same master).
- Continuous features are standardized.
- The code reads `cell_features.npy (num_cells, 13)` and uses columns 0 and 1 as raw x, y for grid mapping. It cannot confirm the order of the other 11 columns, because the file that writes them is not released.

**Net — 6 dims (Eq. 2) [P+C dim]**

- HPWL, bbox width, bbox height, degree |e|, log|e|, bbox area.

**Grid — 5 dims (Eq. 3) [P≠C]**

- **Paper:** normalized cell density, pin density and net density; mean cell area; mean HPWL of overlapping nets. The same definition holds at every level.
- **Code:** columns `RUDY, PinRUDY, MacroRegion, H_MacroMargin, V_MacroMargin` read from `layout.csv`.
- **Code, L1:** 2×2 mean pooling of the raw L0 features, then standardization.

### 4.4 Edge types and features

| Paper edge | Code relation(s) | Features |
|---|---|---|
| Pin edge cell–net [P+C] | `cell-pin->net`, `net-pin_rev->cell` | 3-d (Eq. 4): signal direction ∈{0,1}, normalized (dx, dy) of the cell relative to the net bbox centroid |
| Geometric cell–cell [P+C] | `cell-geom->cell` | 4-d (Eq. 5): Δx, Δy, Manhattan distance, Euclidean distance. **The neighbourhood rule ("spatially nearby") is not specified anywhere.** |
| Grid adjacency, 4-connected within a level [P+C] | `grid_l0-adj`, `grid_l1-adj` | none |
| Hierarchical parent–child [P+C] | `grid_l0-child_of->grid_l1`, `grid_l1-parent_of->grid_l0` | none ("unweighted, structural") |
| Cell–grid (Fig. 2 ④; "Cro edge") [P+C] | `cell-in->grid_l0`, `grid_l0-contains->cell` | none |
| Grid–net (Fig. 3; ablation "Grid-Net MP") [P+C] | `net-spans->grid_l0`, `grid_l0-spans_rev->net` | none. **Code-only rule:** a net connects to every L0 tile in the bbox of its cells' tiles. Nets spanning more than 100 tiles are dropped, and the total is capped at 5 M edges (smallest nets first). |

## 5. Message passing (one VeriHGN layer)

Hidden size d = 128; L = 3 layers [P+C]. Symbols:

- h = node embeddings, after a 2-layer MLP input projection per node type [C]
- e = edge features
- mean = mean over neighbours

1. **Cell → Net, pin-aware (Eq. 6) [P+C]**
   - m_cn = MLP_pin([h_c ‖ e_cn])
   - m_net^cell = mean_c m_cn
   - MLP_pin = Linear(d+3, d) → ReLU → Linear(d, d).
2. **Grid → Net (Eq. 7) [P+C]**
   - m_net^grid = mean_g W_gn h_g over `spans` edges.
3. **Net update [C; paper only says "analogous residual"]**
   - h_n' = LayerNorm(ReLU(W[m_net^cell ‖ m_net^grid ‖ h_n]) + h_n), then dropout.
4. **Cell → Grid (Eq. 7) [P+C]**
   - m_g^cell = mean_c W_cg h_c
   - h_g' = LayerNorm(ReLU(W[h_g ‖ m_g^cell]) + h_g).
   - The paper adds "grid-level messages" here. **The code uses no L0 adjacency messages.**
5. **Net → Cell (Eq. 7) [P+C]**
   - m_c^net = mean_n W_nc h_n' (uses the *updated* net embedding).
6. **Geometric cell–cell, gated (Eqs. 8–9) [P+C]**
   - α_ij = σ(W_α[h_i ‖ h_j ‖ e_ij]). In the code the gate is a **scalar** per edge.
   - m_c^geom = mean_j α_ij · W_geo h_j.
   - (The code's naming is reversed: the gate sees h_src ‖ h_dst and aggregates to dst.)
7. **Cell update (Eq. 10) [P+C, plus LayerNorm in C]**
   - h_c' = LayerNorm(ReLU(W[h_c ‖ m_c^net ‖ m_c^geom]) + h_c), then dropout.
8. **Hierarchical grid (§3.4) [P+C]**
   - **Fine→coarse (Eq. 11):** h^(1) = φ_fc(mean_children h'^(0)). φ_fc = Linear+ReLU.
   - **L1 spatial [C]:** h^(1) += dropout(mean_adj W h^(1)). The paper mentions adjacency edges but gives no equation.
   - **Coarse→fine (Eq. 12):** h̃^(0) = φ_cf(h^(1)_parent). φ_cf = Linear+ReLU.
   - **Gated integration (Eqs. 13–14):** γ = σ(W[h'^(0) ‖ h̃^(0)]) (vector gate); h_out^(0) = γ⊙h̃^(0) + (1−γ)⊙h'^(0).
   - **[C]:** the L1 embedding passed in from the previous layer is **ignored**; L1 is rebuilt from L0 each layer. The `grid_l1.x` input features therefore have no effect on the output.

## 6. Prediction heads [P≠C on readout]

- **§3.5 text and code:** separate heads on the final cell and L0-grid embeddings. Each head is Linear(d, 64) → ReLU → Linear(64, 1), clamped at ≥ 0 [C].
- **Fig. 5 caption:** embeddings of cell, geom, net and grid are "concatenated and mapped through a readout layer". **The code does not do this.**

## 7. Labels and normalization

- **Grid label [C]:** (H_Congestion + V_Congestion)/2 per L0 tile, then log(1+c), then divided by a max.
  - **Paper:** the max is a **per-dataset** c_max.
  - **Code:** `compute_dataset_cmax` is computed, but `build_verihgn_graph` ignores it and divides by the **per-sample** max [P≠C].
  - **This project:** both modes are run and reported side by side (decision D4), labelled "paper" and "official-code". Neither is selected by metric.
  - **Measured effect:** per-sample log1p maxima of 1.52, 1.99 and 2.79 on three real samples, so the two modes give different label scales.
- **Cell label [C only]:** the grid label of the L0 tile containing the cell's (x, y). The paper never defines cell-level ground truth. In the code, **cell labels are a projection of grid labels**, so cell- and grid-level metrics are not independent.
- **Resampling [P only]:** label maps are "area-averaged to the target grid resolution using bilinear downsampling". The target resolution is not stated. The code consumes `layout.csv` at whatever M×N `metadata.txt` declares.
- **H/V congestion definition:** not stated in the paper (overflow-based vs utilization-based; global routing vs early global routing). Which map VeriHGN used is **unverifiable**. Full analysis, with real arrays, in `DATASET_AUDIT.md` §3.5. In short:
  - **CircuitNet definition** (`feature_extraction/src/read.py::read_route_congestion`), per gcell (x, y), from the Innovus global-routing congestion report:
    - `GR_*_overflow = −overflow/capacity` if the router reports overflow (< 0), else 0.
    - `GR_*_util = 1 − used/capacity` (remaining-track fraction) if no overflow, 1 if overflow, 0 if capacity = 0.
    - Arrays are `[x, y]` (Fortran-order reshape).
  - **CircuitNet's own congestion task** (`generate_training_set.py`) uses **GR H overflow + GR V overflow** as its label.
  - **Indirect evidence for overflow-based in VeriHGN:**
    - The official WMSE weights targets with `y > 0`. That is meaningful only for a sparse label: overflow is 0 almost everywhere, while utilization is mostly non-zero.
    - The paper says "most cells and grid regions have near-zero congestion".
    - Classification: **Derived from dataset documentation + inferred; not verified.**

## 8. Loss [P described, C exact]

- **Weighted MSE:** L_wmse = mean_i w_i (ŷ_i − y_i)², with **w_i = α = 10 if y_i > 0, else 1** [C]. The paper only says "congestion-dependent weight".
- **Variance regularization [C]:** R = ReLU(std(y) − std(ŷ))², applied separately to cell and grid.
- **Total [C]:** L = L_wmse^cell + L_wmse^grid + λ(R^cell + R^grid), with **λ = 0.5** (config). `train.py` falls back to α = 5, λ = 0.1 if the config keys are missing.

## 9. Training [P+C]

- **Optimizer:** AdamW, lr 5×10⁻⁴, weight decay 10⁻⁴.
- **Scheduler:** StepLR, step 50, γ 0.5.
- **Epochs and stopping:** at most 200 epochs, early stopping with patience 20.
- **Model selection:** best mean validation Spearman, (ρ_cell + ρ_grid)/2.
- **Code-only details [C]:**
  - batch size 1 graph
  - dropout 0.15
  - gradient clip 1.0
  - AMP mixed precision. Details from the code:
    - `train.py` trains under `torch.amp.autocast('cuda')` (default dtype **fp16**) with `GradScaler`.
    - `validate()` (also used for model selection and the test pass at the end of `train.py`) runs autocast fp16.
    - The standalone `test.py` runs in **full FP32** (no autocast).
    - The paper does not mention precision.
  - gradient checkpointing per layer
  - chunked scatter (500 K edges per chunk)
- **Split [C]:** design-level split, 70/10/20 with `np.random.seed(42)` shuffle. The ISPD2015 variant **excludes `superblue*` designs**.
- **Hardware [P]:** 1× NVIDIA L40S (48 GB), CUDA 11.8, PyTorch 2.1, PyG 2.4.

## 10. Evaluation metrics [P+C]

- **Metrics:** MAE, RMSE, Pearson, Spearman, Kendall τ. Each is computed separately for cells and L0 grids.
- **How the code pools them [C]:**
  - Metrics are computed over the **concatenation of all test samples**, not averaged per sample.
  - Kendall is computed on a random 50 K subsample (seed 0).
  - A metric returns 0 if either vector has std < 1e-8.
  - `test.py` also reports per-design metrics.

## 11. Experimental setup and reported results [P]

**Datasets (Table 1):**

| Dataset | Tech | #Designs | #Cells | #Nets | #Samples |
|---|---|---|---|---|---|
| ISPD2015 | — | 20 | 100K–1.2M | 100K–1.3M | 533 |
| CircuitNet-N28 | 28 nm | 10 | 35K–1.5M | 38K–1.6M | 3K |
| CircuitNet-N14 | 14 nm | 8 | 35K–1.5M | 38K–1.6M | 2K |

(Table 1's ISPD2015 row conflicts with the actual data; see `DATASET_AUDIT.md`.)

**Baselines (Table 2):**

- GCN and GAT
- CircuitGNN
- MIHC
- ST-FPN (re-implemented by the authors)
- Lay-Net (open source)

**Headline ISPD2015 results for VeriHGN (Table 2):**

| Level | MAE | RMSE | Pearson | Spearman | Kendall |
|---|---|---|---|---|---|
| Cell | 0.025 | 0.031 | 0.663 | 0.692 | 0.465 |
| Grid | 0.029 | 0.035 | 0.677 | 0.308 | 0.211 |

**Ablations (Table 3, N28 only):** each removes one of:

- hierarchical grid
- grid–net message passing
- geometric cell–cell message passing
- gated aggregation
- enriched features
- weighted loss
- variance regularization

**Other experiments:**

- **Sensitivity (Fig. 6):** L ∈ {1..4}, d ∈ {32..256}, K ∈ {1, 2, 3}. The optimum is L = 3, d = 128, K = 2.
- **Cross-node transfer:** N14↔N28, zero-shot and few-shot (Tables 4–5).

**Runtime and memory (Table 6, as parsed from the PDF layout; column alignment is slightly ambiguous):**

| Dataset | Train time | Time per epoch | Inference | Peak GPU memory |
|---|---|---|---|---|
| ISPD2015 | ≈ 97 min | ≈ 22 s | ≈ 9.7 s | **≈ 30 GB** |
| N14 | ≈ 918 min | — | — | ≈ 37 GB |
| N28 | ≈ 846 min | — | — | ≈ 39 GB |

## 11a. Diagnostic experiments that deviate from the paper (NOT part of the reproduction)

The reproduction runs only the unmodified official model (§§5–9). Separately, investigating the prediction collapse required one **diagnostic deviation**, which is never part of the reproduction:

- **HEADPATH arm U** (`docs/HEADPATH_PLAN.md`): the cell head's `clamp(min=0)` is omitted in training forward passes only, so the official loss sees raw cell outputs. Evaluation stays official.
- **How it is marked:** flag `model.cell_head_train_clamp = false`, run-id suffix `__DEV-cellhead-train-unclamped`, and an entry in every such run's `deviations`.
- **How it may be used:** its results answer a diagnostic question, namely whether the train-time clamp is what kills the cell head. They are **not** VeriHGN results, may not be compared with the paper's tables, and are reported separately from every reproduction run.

## 12. What the paper does NOT specify (needs a decision)

1. The geometric-edge neighbourhood rule: kNN, radius, or same-row neighbours, and the value of k or r.
2. The L0 grid resolution M×N for ISPD2015.
3. Which CircuitNet congestion maps form H/V congestion: GR or eGR, overflow or utilization.
4. The exact feature order inside `cell_features.npy`, and how pin direction is obtained (LEF? name heuristics?).
5. How IO ports, fixed macros and nets of degree 1 are handled.
6. The preprocessing code that produces the official input format (`*.npy`, `layout.csv`, `metadata.txt`). **It is not released.**
