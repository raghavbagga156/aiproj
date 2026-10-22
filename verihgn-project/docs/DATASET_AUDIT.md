# Dataset Audit (Phase 0)

Audit date: 2026-10-19. Everything below was **observed directly**: Drive folder listings, HTTP byte ranges for file sizes, full streamed `tar -t` listings, and loading of real files. Items marked *Not inspected* were not opened.

## 1. Sources and access status

| Source | Access method | Status |
|---|---|---|
| Google Drive folder `1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB` (title **"CircuitNet"**) | Public "anyone with link" embedded folder view and `drive.usercontent.google.com` downloads via curl. **No login was needed and no credentials were used.** | Partially inspected (ISPD15 fully listed, partly loaded; N14/N28 listed only) |
| Zenodo 10.5281/zenodo.20496640 | Zenodo REST API | Successfully inspected. **It contains only `VeriHGN-v2.0.0.zip` (20 KB), the source code.** No data, no checkpoints. |
| Official GitHub `SC-Lab-Go/VeriHGN` | `git clone` | Successfully inspected. **It contains no data, no preprocessing scripts and no pretrained weights.** |

**Key finding:** the Drive folder is the **raw public CircuitNet release** (PKU, BSD-3-Clause). It is **not** the preprocessed per-sample format that the official VeriHGN code expects:

- `cell_features.npy`
- `net_features.npy`
- `pin_*.npy`
- `geom_*.npy`
- `layout.csv`
- `metadata.txt`

The script that converts CircuitNet into that format is not published.

## 2. Drive folder structure (observed)

```
CircuitNet/                                   (root, shared folder)
├── LICENSE                                   BSD 3-Clause
├── CircuitNet-ISPD15/
│   ├── DEF.tar.gz                            11,397,704,623 B (11.4 GB)
│   ├── routability_features.tar.gz            7,588,018,472 B ( 7.6 GB)
│   └── graph_features/
│       ├── graph_infromation.tar.gz             123,753,781 B (124 MB)   [sic: "infromation"]
│       ├── instance_placement_gcell.tar.gz    1,955,989,756 B (2.0 GB)
│       └── instance_placement_micron.tar.gz   2,724,127,925 B (2.7 GB)
├── CircuitNet-N14/   graph_features/ (split archives _00.._10), IR_drop_features/, preprocess_scirpts/,
│                     routability_features/{cell_density, macro_region, RUDY/*, congestion/, DRC/}, timing_features/
└── CircuitNet-N28/   graph_features/, IR_drop_features/, preprocess_scripts/, raw_data/{netlist.tar.gz, LEF&DEF/},
                      routability_features/{cell_density, macro_region, RUDY/*, congestion/, DRC/}, timing_features/
```

The total ISPD15 download is **≈ 23.8 GB compressed**. N14 and N28 are considerably larger; their archive sizes were not measured.

## 3. ISPD15 contents (observed)

### 3.1 `graph_infromation.tar.gz`: downloaded and fully loaded

It holds three object (pickled) `.npy` arrays per design, for **16 designs**:

| File | Shape | Content |
|---|---|---|
| `node_attr/<design>_node_attr.npy` | (2, 1 + #cells) | row 0 = instance names (index 0 is the top module, e.g. `fft`); row 1 = master/cell-type names (index 0 = `module`) |
| `net_attr/<design>_net_attr.npy` | (1, #nets) | net names |
| `pin_attr/<design>_pin_attr.npy` | (3, #pins) | row 0 = pin name (`o`, `a`, `b`, `ck`, …); row 1 = **net index** (0-based); row 2 = **node index** (into `node_attr`, ≥ 1) |

Per-design statistics (computed from these files):

| Design | Cells | Macros (name == master) | Masters | Nets | Pins | Max net degree |
|---|---|---|---|---|---|---|
| mgc_des_perf_a | 108,292 | 4 | 16 | 110,281 | 310,036 | 8,517 |
| mgc_des_perf_b | 112,644 | 0 | 12 | 112,878 | 318,745 | 8,802 |
| mgc_edit_dist_a | 127,419 | 6 | 20 | 131,134 | 373,180 | 5,504 |
| mgc_fft_2 | 32,281 | 0 | 10 | 33,307 | 101,375 | 1,984 |
| mgc_fft_a | 30,631 | 6 | 16 | 32,088 | 97,138 | 1,913 |
| mgc_fft_b | 30,631 | 6 | 16 | 32,088 | 97,138 | 1,913 |
| mgc_matrix_mult_a | 149,655 | 5 | 19 | 154,284 | 439,891 | 2,818 |
| mgc_matrix_mult_b | 146,442 | 7 | 21 | 151,612 | 431,905 | 2,746 |
| mgc_matrix_mult_c | 146,442 | 7 | 21 | 151,612 | 431,905 | 2,746 |
| mgc_pci_bridge32_a | 29,521 | 4 | 16 | 29,985 | 85,112 | 3,253 |
| mgc_pci_bridge32_b | 28,920 | 6 | 18 | 29,417 | 83,583 | 3,186 |
| mgc_superblue11_a | 927,074 | n/a* | 5,677 | 935,613 | 3,043,447 | 3,602 |
| mgc_superblue12 | 1,287,037 | n/a* | 2,081 | 1,293,413 | 4,767,082 | 8,488 |
| mgc_superblue14 | 608,914 | n/a* | 5,093 | 619,815 | 2,026,930 | 8,864 |
| mgc_superblue16_a | 680,869 | n/a* | 4,634 | 697,303 | 2,262,149 | 2,368 |
| mgc_superblue19 | 506,178 | n/a* | 4,545 | 511,685 | 1,697,277 | 9,155 |

\*In superblue designs the macro masters have their own names, so the `name == master` heuristic cannot find macros. A LEF `CLASS BLOCK` lookup (or a size threshold) is required.

**Quirks observed:**

1. **Macro pins (corrected in Phase 1).** In designs with macros, pins whose names equal net names (e.g. `ispd_clk`) belong to the macros themselves. ISPD15 mgc macros name each pin after its net; the DEF confirms it (`( h2 ispd_clk )`), with pin counts identical to graph_information (154 for `h2`, 140 for `hh4` in pci_bridge32_b). Top-level IO ports (the DEF's `PINS 361`) are **not part of graph_information** at all. The Phase 0 reading of this as an "indexing artefact" was wrong.
2. **Pin direction.**
   - **Standard cells:** all 16 masters in the 9 official-scope designs have exactly one output pin, named `o`; every other pin (`a`–`d`, `ck`) is an input. This rule is derived from the data (and tested on real samples).
   - **Macro pins:** direction is **not recoverable**. The full 11.4 GB `DEF.tar.gz` was scanned and contains **no LEF file**, and the DEF gives directions only for top-level IO ports. Macro pins are set to *input* by default (`macro_pin_direction` config) and classified **U**.
3. **Single-pin nets.** Net degree ranges from 1 to 9,155.

### 3.2 `instance_placement_gcell.tar.gz`: full listing streamed; one sample loaded

- **533 files.** That is 533 placement samples, which **matches the paper's Table 1 "533 samples".**
- File naming: `<id>-<design>-util_<u>`.
- Each file is a pickled `dict{instance_name: [x_lo, y_lo, x_hi, y_hi]}` in **gcell units**. For `68-mgc_fft_a-util_0.7`, coordinates range over x ∈ [0, 274] and y ∈ [0, 273], i.e. about a 275×274 gcell grid.
- The dict covers every cell and macro. Its keys equal `node_attr` minus the top module.

Samples per design (observed):

| Design | Samples | Design | Samples |
|---|---|---|---|
| mgc_des_perf_a | 45 | mgc_pci_bridge32_a | 27 |
| mgc_edit_dist_a | 45 | mgc_pci_bridge32_b | 45 |
| mgc_fft_a | 45 | mgc_superblue11_a | 30 |
| mgc_fft_b | 43 | mgc_superblue12 | 30 |
| mgc_matrix_mult_a | 45 | mgc_superblue14 | 30 |
| mgc_matrix_mult_b | 44 | mgc_superblue16_a | 30 |
| mgc_matrix_mult_c | 44 | mgc_superblue19 | 30 |

- **14 designs and 533 samples in total.**
- `mgc_des_perf_b` and `mgc_fft_2` have graph information but **no placement samples**.
- **Utilization targets:** 0.7 (111), 0.75 (109), 0.8 (103), 0.85 (109), 0.9 (101).

### 3.3 `instance_placement_micron.tar.gz`: first 40 MB streamed; one sample loaded

The format is the same dict, with coordinates in **microns**. For `fft_a` they span about 0–747 µm. This gives true cell geometry (w, h, area) for the cell features.

### 3.4 `DEF.tar.gz`: first 40 MB streamed; one file opened

- Files are named `DEF/<id>-<design>-util_<u>.def.gz`, one per sample.
- In `1-mgc_pci_bridge32_b-util_0.9`: DEF 5.8, `UNITS DISTANCE MICRONS 1000`, `DIEAREA (0 0) (602000 598000)`, `COMPONENTS 28920`, `NETS 29417`.
- **The NETS section contains `ROUTED` wiring.** These DEFs are **post-route**, so routing geometry must **never** be used as an input feature (label leakage).
- Whether LEF files are inside the archive is **Not inspected**. That needs the full 11.4 GB listing.

### 3.5 `routability_features.tar.gz`: full listing streamed

**Full listing (streamed 7.6 GB in 969 s, 2026-10-19; 9,088 entries).** There are 17 folders with exactly **533 files each**, one per sample, named like the placements. The archive is **47.97 GB uncompressed**.

```
routability_features/
├── cell_density
├── macro_map                                   (= "macro_region" in N14/N28)
├── net_density/{net_density_horizontal, net_density_vertical}
├── RUDY/{RUDY, RUDY_long, RUDY_short, RUDY_pin, RUDY_pin_long}
└── congestion/
    ├── congestion_global_routing/
    │   ├── overflow_based/{congestion_GR_horizontal_overflow, congestion_GR_vertical_overflow}
    │   └── utilization_based/{congestion_GR_horizontal_util, congestion_GR_vertical_util}
    └── congestion_early_global_routing/
        ├── overflow_based/{congestion_eGR_horizontal_overflow, congestion_eGR_vertical_overflow}
        └── utilization_based/{congestion_eGR_horizontal_util, congestion_eGR_vertical_util}
```

**Mapping to the official code's `layout.csv` columns:**

| Official column | Likely CircuitNet source | Status |
|---|---|---|
| `RUDY` | `RUDY/RUDY` | available; name match, **inferred** |
| `PinRUDY` | `RUDY/RUDY_pin` | available; **inferred** |
| `MacroRegion` | `macro_map` | available; **inferred** |
| `H_MacroMargin`, `V_MacroMargin` | none | **absent from CircuitNet**; definition unknown |
| `H_Congestion`, `V_Congestion` | `GR_{horizontal,vertical}_overflow` (see below) | **inferred** |

#### Label maps: measured on 3 real samples

Samples:

- `68-mgc_fft_a-util_0.7`
- `3-mgc_pci_bridge32_a-util_0.85`
- `380-mgc_matrix_mult_a-util_0.75`

Each map was loaded with a restricted unpickler; they are plain `float64` arrays.

| Map | fft_a shape | pci_a shape | mm_a shape | Value range | Non-zero tiles |
|---|---|---|---|---|---|
| `cell_density` | 275×274 | 105×103 | 507×507 | 0–15 (integer counts) | 80–89 % |
| `GR_horizontal_overflow` | 275×274 | 105×103 | 507×507 | 0–30.5 | 1.3 %, 5.9 %, 4.6 % |
| `GR_vertical_overflow` | 275×274 | 105×103 | 507×507 | 0–14.75 | 0.6 %, 2.2 %, 8.0 % |
| `GR_horizontal_util` / `GR_vertical_util` | same | same | same | 0–1 | 30–65 % |
| `eGR_*_util` | **412×411** | **156×154** | **760×758** | 0–1 | 27–63 % |

**Semantics.** Source: CircuitNet `feature_extraction/src/read.py::read_route_congestion`, applied to the Innovus global-routing congestion report. For gcell (x, y), with capacity c, used u and overflow o (the report gives o < 0 when demand exceeds capacity):

- **Overflow-based:** `GR_overflow = −o / c` if o < 0; `0` otherwise (and 0 if c = 0). This is the *excess demand as a fraction of capacity*, unbounded above.
- **Utilization-based:** `GR_util = 1 − u / c` if o ≥ 0 (the remaining-track fraction); `1` if overflowed; `0` if c = 0. Despite its name, this is **not** "used/capacity".
  - Measured on all three samples: every tile with overflow > 0 has util == 1, which confirms the code path.
- **Directions:** H and V are separate routing directions. Per the CircuitNet docs the tech LEF is "HVH": M1 horizontal, M2 vertical, and so on.
- **eGR vs GR:** eGR = early/trial global routing; GR = global routing, "more accurate" (CircuitNet docs). eGR maps are on a **different, ~1.5× finer grid** and are not aligned with the placement gcells.

**Grid alignment, verified exactly.** Recomputing an occupancy count from `instance_placement_gcell`, covering tiles `[x0, max(x1, x0+1)) × [y0, max(y1, y0+1))` for every instance including macros, reproduces `cell_density` **exactly (bit-identical)** on both samples tested.

- Therefore the placement gcell coordinates, `cell_density` and all GR maps share one grid.
- That grid is indexed **[x, y]** (axis 0 = x), with no offset.
- This is the native routing gcell grid (fft_a: 275×274 gcells over ~747 µm, ≈ 2.7 µm per gcell).

**VeriHGN-style label on these samples.** (H_ovf + V_ovf)/2:

- non-zero on 1.8 % (fft_a), 7.9 % (pci_a) and 12.5 % (mm_a) of tiles
- per-sample log1p max = 2.79 / 1.52 / 1.99

So **per-sample vs. per-dataset normalization changes label scale by up to ~1.8× between samples**; this is relevant to D4.

**Equivalence with the paper's target — classification.**

| Question | Answer | Basis |
|---|---|---|
| Does the paper say which map? | No. Only "ground-truth congestion maps … provided at design-specific global-routing resolutions" | Paper §4.1 |
| Does the official code say? | It reads `H_Congestion`, `V_Congestion` from a `layout.csv` whose generator is unpublished | Official code |
| "Global-routing resolution" | Matches **GR** maps (native gcell grid), not eGR (different grid) | Paper + dataset |
| Sparse / "near-zero" congestion, WMSE weight on `y > 0` | Matches **overflow-based** (1–12 % non-zero); utilization-based is dense and bounded at 1 | Paper + official code + dataset (measured) |
| CircuitNet's own congestion label | GR H overflow + GR V overflow | CircuitNet `generate_training_set.py` |
| Bilinear / area-averaged "downsampling to the target resolution" | Not reproducible: target resolution unknown. The official code consumes a pre-sized grid | Unresolved |

**Conclusion:** using **(GR_H_overflow + GR_V_overflow)/2** is the best-supported choice. It is **not verified as identical** to the authors' target, so it is classified as an **adaptation derived from dataset documentation and inference**.

## 4. Feature and label availability vs. what VeriHGN needs

| VeriHGN input | Available in CircuitNet-ISPD15? | How |
|---|---|---|
| Cell x, y, w, h, area | Yes | `instance_placement_micron` |
| Cell–net incidence, pin count, net degree | Yes | `pin_attr` |
| Incident-net HPWL stats, net bbox / HPWL | Yes (derived) | Cell centres from placement + incidence. Pin offsets inside cells are not available, so cell-centre HPWL is an approximation |
| Macro flag | Partly | `name == master` for mgc designs; superblue needs LEF |
| Input-pin ratio, pin direction | Inferred | Pin-name heuristic (LEF not verified) |
| Type frequency | Yes | `node_attr` row 1 |
| Pin offset (dx, dy) to the net bbox centroid | Yes (cell-centre level) | Derived |
| Geometric cell–cell edges | Derived | Neighbourhood rule must be chosen (unspecified) |
| Grid features, paper Eq. 3 (cell/pin/net density, mean cell area, mean net HPWL) | Derived | Computed by us from placement + netlist |
| Grid features, official code (RUDY, PinRUDY, MacroRegion, H/V MacroMargin) | 3 of 5 | `RUDY`, `RUDY_pin` and `macro_map` exist (533 each). **H/V MacroMargin does not exist in CircuitNet**; its definition is unknown |
| Grid congestion label | Yes (adapted) | (GR_H_overflow + GR_V_overflow)/2, on the native gcell grid; see §3.5 |
| Cell congestion label | Derived | Grid label at the cell's tile (official code's definition) |

## 5. Train / validation / test splits

- **No official split file exists** in the dataset or the repo.
- The official code makes a design-level split: sorted design folders → exclude names containing `superblue` → `np.random.seed(42)` shuffle → 70 / 10 / 20.
- On the observed data that gives **9 designs and 383 samples**. Computed exactly, assuming folders are named after the CircuitNet design names (`mgc_*`):

| Split | Designs | Samples |
|---|---|---|
| train | pci_bridge32_a, edit_dist_a, matrix_mult_b, des_perf_a, pci_bridge32_b, fft_a | 251 |
| val | matrix_mult_a | 45 |
| test | **fft_b**, **matrix_mult_c** | 87 |

- **Netlist leakage is certain in the official split.**
  - `fft_a`/`fft_b` and `matrix_mult_b`/`matrix_mult_c` have identical netlists (identical cell, net, pin and master counts).
  - **Both test designs have their twin in train.** Placements differ, but connectivity and cell features do not.
  - Comparability with the official code requires this split. A twin-aware split is reported alongside it (D5).
- With a single validation design, early stopping is noisy.

## 6. Discrepancies with the paper

1. **Dataset scope (D6).**

| Quantity | Paper Table 1 | CircuitNet-ISPD15 (observed) | Official code (`split_variant: ISPD2015`) |
|---|---|---|---|
| Designs | 20 | 16 with graph info; **14 with placements and labels** | **9** (superblue excluded) |
| Samples | **533** | **533** | **383** |
| Cells | 100K–1.2M | 28.9K–1.29M | 28.9K–150K |
| Nets | 100K–1.3M | 29.4K–1.29M | 29.4K–154K |

   **Interpretation:**
   - "533" equals the **full** CircuitNet-ISPD15 sample set *including* superblue.
   - "20 designs" most likely refers to the ISPD 2015 contest suite rather than the data actually used. This is **inferred**: only 14 designs have samples here.
   - The cell range matches none of the three columns exactly.
   - Table 1 therefore describes the source data, while the **released code trains and tests on the 9-design / 383-sample subset**.
   - Which subset produced Table 2 cannot be determined from the published material.

   **Excluded designs and reasons:**

| Excluded | Reason | Source |
|---|---|---|
| `mgc_des_perf_b` | No placement or label samples in CircuitNet-ISPD15 | Dataset |
| `mgc_fft_2` | No placement or label samples in CircuitNet-ISPD15 | Dataset |
| `mgc_superblue11_a`, `superblue12`, `superblue14`, `superblue16_a`, `superblue19` (150 samples, 0.5–1.3M cells) | Hard-coded filter `'superblue' not in d` | Official code |
| Remaining 4 contest designs not in CircuitNet | Not distributed | Dataset |

   **Effect on comparability:**
   - Results on the 9-design subset are comparable to the **official code's** protocol, but **not provably** to Table 2.
   - Superblue graphs are 4–9× larger. Including them would change both the difficulty and the memory profile.
   - An "all 14 designs" variant (`ISPD2015-all`, 533 samples) can be run on the A100 as a clearly labelled additional experiment. Its feasibility is to be measured.

2. **Macro identification in superblue.** It needs LEF `CLASS BLOCK`, since the `name == master` heuristic fails there. This matters only for `ISPD2015-all`.
3. **Input format.** The official input format is not produced by anything published. We must write our own preprocessing, so results will be an **adapted reproduction**.
4. **Label source.** The official code's `H_Congestion` / `V_Congestion` and `H/V_MacroMargin` columns have no documented origin. A web search found no match.

## 7. Compatibility with the official implementation

- `graph_builder.py`, `models/verihgn.py`, `utils/*` and `train.py` can be **reused unchanged**, as long as we emit the documented per-sample format.
- Our preprocessing therefore targets **exactly** that format (`cell_features.npy`, … `layout.csv`, `metadata.txt`). The official graph builder and model then run on it as-is.

## 8. Required preprocessing (Phase 3 plan)

1. Download the ISPD15 archives into Colab **local disk** `/content`. They do not fit in a free 15 GB Drive, and nothing is copied into the user's quota.
   - `graph_infromation` and `instance_placement_micron` are needed (≈ 2.8 GB).
   - Only the needed folders of `routability_features` are stream-extracted: GR overflow H/V, RUDY, RUDY_pin, macro_map, and cell_density for alignment checks. That is about 6 of 17 folders, roughly 17 GB of the 48 GB uncompressed; nothing else is written to disk.
   - `DEF.tar.gz` (11.4 GB) is needed **only** if LEF/pin-direction or die-area information is required.
2. Per sample: build the cell / net / pin / geom arrays, the grid features and the labels. Write the official per-sample format to **Drive** (compact; a few MB per mgc sample).
3. Compute the label normalization on the training split only. Record every constant in `preprocess_manifest.json`.

## 9. Colab access procedure (no credentials in code)

The folder is link-shared publicly, so either method works:

- **A. Recommended — direct download.** Use `gdown --folder` or `gdown <file-id>` with the file IDs listed above. Nothing touches the user's Drive quota.
- **B. Drive mount.**
  1. In the Drive web UI, open the folder link and choose *Organize → Add shortcut to Drive* (for example into `MyDrive/CircuitNet`).
  2. In Colab, run `from google.colab import drive; drive.mount('/content/drive')`.
  3. The data appears under `/content/drive/MyDrive/CircuitNet/…`.
  4. Notebook 01 will search `/content/drive` for the folder ID or name rather than assuming a path.
  5. Large tarballs should still be **copied to `/content`** before extraction, because Drive-FUSE is slow for random I/O.
