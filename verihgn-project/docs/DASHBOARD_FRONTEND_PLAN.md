# Offline dashboard (Stage 2): audit and frontend plan

**Status (2026-10-22):** audit and plan only. **No frontend code exists yet.** Nothing has been built, run or visually inspected. Waiting for approval.

## 1. Audit: what exists

### 1.1 Frontend: none

| Evidence | Result |
|---|---|
| Search of the whole project (excluding `.venv/`, `third_party/`) for `*.html`, `*.htm`, `*.js`, `*.jsx`, `*.ts`, `*.tsx`, `*.css`, `*.scss`, `*.vue`, `*.svelte`, `package.json`, `vite.config.*`, `*.bat`, `node_modules` | **zero files** |
| `verihgn-project/app/` | **empty** directory, created 2026-10-19 13:02 (Phase 1 scaffold) and never filled |
| `verihgn-project/artifacts/` | empty (same scaffold) |
| `README.md:16` | "Dashboard: not started (decision D8: React + Plotly, saved-artifact mode)" |
| `docs/DASHBOARD_PLAN.md` status | "Stage 2 (dashboard) waits for your notebook-09 results" |
| `docs/DASHBOARD_EXPORT.md:3` | "The dashboard itself (Stage 2) is built only after you have run notebook 09" |
| Files the plan lists for Stage 2 (`app/package.json`, `app/vite.config.js`, `app/index.html`, `app/src/**`, `run_dashboard.bat`, `tools/install_dashboard_bundle.py`, `docs/DASHBOARD.md`) | **none exist** |

### 1.2 Why the ZIP has no frontend

The ZIP was **designed as a data-only bundle**, to be consumed by a separate static frontend. The frontend was **not abandoned**: it was **never started**, because Stage 2 was gated on your approval of the Stage 1 export.

The evidence is in `docs/DASHBOARD_PLAN.md` §4:
- the architecture diagram: "[B] unzip into app/public/data" → "[C] React + Plotly static app, built once";
- §4.4: "[C] Dashboard app: `app/`".

The exporter (`src/verihgn_repro/export/dashboard_bundle.py`) writes only data, integrity and provenance files by construction. That is correct: the application code belongs in the repository, not in a results bundle.

### 1.3 The verified data bundle (inspected locally, read-only)

**File and verification:**
- **File:** `C:\Users\ragha\Downloads\headpath_dashboard_20261022_110005.zip`
- **Size:** 16,675,322 B
- **sha256:** `0e2b95ffbdc2ec6df3cea5ac9b4bd750af9308e0c091686ed7655a9037106969` (equals the value you gave)
- **Local `verify_bundle`:** PASSED. 636 listed files + the manifest; all 408 arrays decoded losslessly; manifest and integrity status PASSED.
- **Checks:** 352 PASS · 2 WARN · 4 INFO · 0 FAIL.
- **Sources:** 477 fingerprinted before and after, unchanged.
- **JSON normalization:** 146 non-finite values → null, in 47 files.
- **Entries:** 637 = 408 `.gz` arrays, 136 `.json`, 78 `.csv`, 14 `.yaml`, 1 `.md`.

**The two WARNs, which the dashboard must show:**
1. **No protected-artifact record in the report.** `report.protected_artifacts`: the authoritative Drive report (sha256 `ce485b25…`, identical to your Downloads copy) has no `protected_artifacts` record, which notebook 08 normally writes last.
2. **Threshold inconsistency.** The recorded `p2_min_improvement` (0.10) is not used; the implemented rule is 0.08 / 0.12. No run lies in the interval where the two readings differ.

**Contents usable by the dashboard:**

| Area | Content | Size |
|---|---|---|
| `headpath/report.json` | sanitized report (null + `undefined_paths`) | 14 runs |
| `headpath/analysis.json` | paired table, re-derived McNemar, retest, threshold note, **exploratory** U − O, NPZ-vs-evals grid comparison | |
| `headpath/runs/<run>/steps.json` | 490 rows × 42 columns per run (training) | 14 runs |
| `headpath/runs/<run>/evals.json` | 11 rows × 93 columns per run (validation) | 14 runs |
| `headpath/runs/<run>/history.json` | 2 rows per run (per-epoch; training metrics on clamped predictions) | |
| `headpath/maps/<sample>/…` | label, raw H, raw V (shared); cell subsample indices / labels / tiles | 12 samples |
| `headpath/runs/<run>/maps/<sample>/…` | `grid_pred`, `cell_pred` (float32, lossless) | 14 × 12 |
| `context07/…` | notebook-07 report, plan, forensic audit, 18 runs' steps / history, field caveats A1–A6 | 18 runs |
| `manifest.json`, `integrity_report.json`, `normalization_index.json`, `INTEGRITY_REPORT.md` | provenance and integrity | |

- **Map sizes:** 105 × 103 (pci_bridge32_a) up to 524 × 524 (matrix_mult_a); 1,938,972 tiles in total.
- **Arrays:** 13.0 MB compressed, 148.2 MB raw. Loaded **per selected sample and run**, never all at once.

### 1.4 What the data cannot support, and the dashboard will not show

| | |
|---|---|
| **Test-set results** | none exist; the HEADPATH runs never loaded the test split |
| Comparison with the paper's Table 2 | not meaningful: 2-epoch diagnostic, adapted label (D1), diagnostic validation set |
| Full-cell maps or scatter | the NPZ files hold a 20,000-cell subsample per sample (identical across runs). Full-cell metrics come only from `evals.csv` and the report |
| Inferential statistics beyond the pre-registered McNemar test | none; exploratory per-seed differences are descriptive only |
| Validation curves finer than 11 points | evaluations were at steps 0, 49, …, 490 only |
| Checkpoint-derived quantities | checkpoints were deliberately not exported (checked, not copied) |

## 2. Architecture

```
headpath_dashboard_<ts>.zip ──► run_dashboard.bat  (Python stdlib only)
                                  1. sha256 + verify_bundle (reuses the Stage 1 verifier)
                                  2. extract to %LOCALAPPDATA%\VeriHGN-dashboard\<zip-sha256>\  (never into the project)
                                  3. serve app/dist + data on http://127.0.0.1:8765 (local only) and open the browser
                                         │
                                         ▼
                       React 18 + TypeScript + Plotly (static build in app/dist, no backend, no external requests)
                         data layer: typed loaders + schema guards + lossless array decoder (gzip → unshuffle → typed array,
                                     sha256_raw re-checked in the browser) + null-safe value model
                         pages: 12 sections (§4), hash routing (works from a static server)
```

**Key decisions:**

| Decision | Choice | Why |
|---|---|---|
| Runtime | static build + Python's `http.server` on `127.0.0.1` | no backend logic; `fetch` and ES modules don't work from `file://`; Python is already installed |
| Internet | needed **once** for `npm ci` / build. Optionally I commit the built `app/dist`, so viewing needs Python only | offline after installation |
| Language | TypeScript (strict) | the bundle schema becomes compile-time types; a null can't silently become 0 |
| Charts | Plotly (`plotly.js-dist-min`, pinned) through a thin `<Chart>` wrapper holding all styling | one place for the chart conventions (§3.5) |
| Array decoding | `fflate` (gzip, pinned, about 8 kB) + byte unshuffle + `crypto.subtle` sha256 check | identical in the browser and in tests; lossless |
| Routing / state | small hash router; selection state in the URL (`#/maps?sample=…&seed=…`) | shareable views; no extra dependency |
| Fonts | system UI stack (Segoe UI Variable on Windows 11); `ui-monospace` for hashes | offline, crisp on Windows, no font download |
| Pinned dependencies (exact versions, lockfile committed) | `react@18.3.1`, `react-dom@18.3.1`, `plotly.js-dist-min@2.35.2`, `fflate@0.8.2`, `vite@5.4.11`, `@vitejs/plugin-react@4.3.4`, `typescript@5.6.3`, `vitest@2.1.8`, `jsdom@25.0.1`, `@testing-library/react@16.1.0` | reproducible builds |

**Integrity in the app:**
- The provenance page shows the ZIP sha256 computed by the launcher, the manifest status, every WARN, and the normalization index.
- If the integrity status is anything but PASSED, a **blocking red banner** is shown on every page.

## 3. Design system

### 3.1 Principles

1. **Quiet chrome, loud data.** Neutral warm surfaces and hairline structure; colour is reserved for the two arms and for status.
2. **Evidence types are visibly different** (§3.6). An observed value, a pre-registered criterion, a statistical conclusion, an exploratory description and a limitation each have a fixed treatment, so they can't be confused.
3. **Nothing implies significance that wasn't shown.**
   - No "winner" styling; no green or red on arm comparisons.
   - p = 0.5 is shown with its rule ("significant only if p ≤ 0.05; with 6 pairs the smallest attainable p is 0.03125").
   - All of these numbers are computed from the bundle, never written into the code.
4. **Undefined is visible:** a "undef." chip, a gap in a line, a hatched cell. Never 0, and never silently dropped.

### 3.2 Colour tokens

Arm colours were validated with the data-viz palette validator against these surfaces:
- **light, on `#fbfaf7`:** worst colour-blind ΔE 24.7, normal-vision ΔE 33.6, contrast ≥ 3:1 — all pass;
- **dark, on `#1a1a19`:** worst colour-blind ΔE 26.8, normal-vision ΔE 31.8 — all pass.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--page` | `#f6f4ef` | `#121211` | app background (warm off-white) |
| `--surface` | `#fbfaf7` | `#1a1a19` | cards, chart surface |
| `--ink` | `#1d1c1a` | `#f0efec` | primary text |
| `--ink-2` | `#55534e` | `#c3c2b7` | secondary text |
| `--ink-3` | `#898781` | `#898781` | axis labels, captions |
| `--hairline` | `#e4e1d8` | `#2c2c2a` | borders, gridlines |
| `--arm-O` | `#2a78d6` | `#3987e5` | **Arm O (official)**, everywhere |
| `--arm-U` | `#eb6834` | `#d95926` | **Arm U (diagnostic deviation)**, everywhere |
| status good / warning / serious / critical | `#0ca30c` / `#fab219` / `#ec835a` / `#d03b3b` | same | **integrity and validity only**; always icon + label, never alone |

Map colour scales must not be confused with the arm colours:
- **Sequential** (congestion, |error|): a single-hue warm-graphite ramp from `--surface` to `--ink`. It is perceptually monotone, and its lightness steps will be checked in implementation.
- **Diverging** (ρ heatmap, centred on 0): teal ↔ magenta, with a neutral grey midpoint, validated before use. Blue ↔ red is not used because blue is Arm O.

### 3.3 Typography

- **Stack:** `"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, sans-serif`. Hashes use `ui-monospace, "Cascadia Mono", Consolas`.
- **Scale (px):**

| Size | Weight | Use |
|---|---|---|
| 28 | 600 | page title, one per page |
| 20 | 600 | section |
| 16 | 600 | card title |
| 14 | 400 | body, line height 1.55 |
| 13 | 400 | table |
| 12 | 400 | caption / axis, `--ink-3` |

- **Figures:** proportional for prose and large numbers; `tabular-nums` only in tables and axis ticks.
- **No** serif or display faces, all-caps headings, or text in series colours. Values stay in ink; a coloured mark beside them carries the arm identity.

### 3.4 Space, shape and layout

- **Space:** 4-px grid. Page gutters 32 px (desktop), 16 px (phone). Card padding 20 px. Section gap 32 px.
- **Shape:** cards 10 px radius with a 1-px hairline; **no shadows, gradients, glass or glow**. Controls 6 px radius.
- **Layout:**
  - **Sidebar:** 248 px, fixed, grouped navigation.
  - **Top bar:** experiment identity, a validation-only notice, the integrity badge, and a light/dark toggle.
  - **Content:** maximum width 1180 px.
- **Responsive:**
  - ≥ 1200 px: two-column card grids.
  - 768–1199 px: one column, sidebar still visible.
  - < 768 px: the sidebar collapses into a top menu; charts are full width; tables scroll horizontally inside their card, never the page; heatmaps keep their aspect ratio.

### 3.5 Chart conventions (one `<Chart>` wrapper applies them all)

- **Plot:** hairline solid gridlines on y only; 2-px lines; 8-px markers; no dual axes; no 3-D; no pies.
- **Legend:** shown for every chart with ≥ 2 series, plus direct "O" / "U" end labels.
- **Arm encoding:** Arm O = solid line and filled circle. Arm U = solid line and filled diamond, so arms differ in shape as well as colour. Retest runs = hollow markers, dashed, labelled "retest (excluded)".
- **Tooltips:** custom templates with the metric name, unit or definition, the run and seed, and the value, or **"undefined (constant prediction)"**.
- **Undefined values:** gaps (`connectgaps: false`) plus an annotation.
- **Accessibility:** every chart has a "table" toggle listing the plotted numbers.
- **Toolbar:** Plotly's toolbar reduced to *download PNG* and *reset*; no Plotly logo.
- **Axis titles:** always include the definition, e.g. "pooled Spearman ρ (independent), validation, 12 graphs".

### 3.6 Evidence types (consistent visual grammar)

| Type | Treatment | Example |
|---|---|---|
| **Observed** | plain card | per-run values, curves, maps |
| **Pre-registered criterion** | thin `--ink-2` left rule + "Pre-registered" label | P1 / P2 / P3 thresholds, the McNemar rule |
| **Statistical conclusion** | bordered note with "Pre-registered test" label; text taken verbatim from `analysis.conclusion` | "Not statistically conclusive: …" |
| **Exploratory** | dotted-outline card with "Exploratory — not pre-registered, no inference" label | per-seed U − O |
| **Limitation / caveat** | `--surface` card with ⓘ and a "Limitation" label | validation-only, n = 6, subsample |
| **Integrity status** | status colour + icon + label | PASSED, WARN |

## 4. Pages

Global, on every page:
- top-bar notices: "Validation set only (diagnostic, N graphs) · No test-set results · Arm U is a diagnostic deviation". N is read from the bundle.
- the page's data scope: TRAINING, VALIDATION or CONTEXT (notebook 07).

| # | Page | Content (all computed from the bundle) | Form |
|---|---|---|---|
| 1 | **Overview** (landing) | research question; status strip (experiment status, verdict, integrity, replication, number of WARNs); **primary result**: the 6-pair matrix (seed × arm, category glyph) with the McNemar sentence built from data; three quiet facts (valid pairs, discordant pairs, exact p with its rule); "what this does / does not show"; links | matrix + prose; **no KPI wall** |
| 2 | **Methodology and caveats** | arms (verbatim `deviation_notice`); design (runs, seeds, steps, eval points); validation set composition (12 samples: role, design, grid size, from `maps/index.json`); pre-registered criteria table (thresholds from `plan.thresholds`); the threshold inconsistency (flag); retest policy; what is not claimed | tables + evidence notes |
| 3 | **Arm O vs Arm U** | category × arm count matrix (6 categories, ink only); final cell and grid activity per arm (dot strip, DEAD / DEGENERATE / HEALTHY); P2 improvement and P3 ρ per arm, with threshold rules drawn | strip plots with criterion lines; table toggle |
| 4 | **Seed-wise paired results** | per seed 42–47: O → U slope (dumbbell) for P2 improvement, final cell ρ, final grid ρ (undefined = hollow marker in an "undefined" band); retests in a separate greyed panel | dumbbells; **Exploratory** card for U − O differences |
| 5 | **Training dynamics** (TRAINING) | rolling clamped cell WMSE / zero-predictor ratio (98-step window = the P2 window); cell and grid raw ≤ 0 fraction; cell-only / grid-only / shared gradient norms (log y; zeros as gaps + note); both-head-collapse marker. Seed selector, or "all seeds" as faint lines with the selected seed emphasized. Arm U's raw-stream loss is never plotted against Arm O's (R6) | line small multiples, shared x = global step |
| 6 | **Validation curves** (VALIDATION) | 11 evaluation points: pooled cell and grid ρ (independent), zero fraction and prediction std, validation loss, activity-state band | lines with markers; gaps for undefined |
| 7 | **Cell- and grid-level metrics** | final (step-490) metric table per run (official and independent ρ, Pearson, Kendall, MAE, RMSE, std, zero fraction, activity); tile-projection reference; non-learned feature baseline (5 features, pooled / per design / mean of samples, with the standardization note) | sortable table + strip plots |
| 8 | **Per-design comparison** | design × run ρ heatmap (diverging teal–magenta, centred on 0; hatched = undefined); matrix_mult_a (6 graphs, cross-design) separated from the 6 held-out (1 graph each, within-design); baseline column | heatmap + table |
| 9 | **Prediction maps** | sample selector (12), seed selector, paired arms: label · O prediction · U prediction · \|error\| O · \|error\| U; raw H and raw V overflow shown separately; shared colour scale per sample; "DEAD: all predictions 0" overlay where it applies; cell scatter (pred vs label) labelled "20,000-cell subsample" | heatmaps (full native resolution) + scatter |
| 10 | **Collapse investigation** (CONTEXT, notebook 07) | separation banner: *earlier experiment, official model, 1 epoch, different arms*; per-run table with caveat footnotes A1–A5 from `field_caveats`; collapse timelines from `steps.csv` (small multiples by run group, ink only, so the arm colours never mean something else here) | small multiples + table |
| 11 | **Statistics and limitations** | McNemar details (pairs, discordant, U-only / O-only, direction, exact p, α, smallest attainable p); H-sig counts; historical replication check; test–retest \|Δ\| (undefined shown as such); full limitation list | evidence notes + tables |
| 12 | **Provenance and integrity** | ZIP sha256 (from the launcher), manifest / integrity status, check counts, filterable table of all checks (WARNs pinned on top), source fingerprints, normalization index (146 replacements, per file), exporter version, official commit | tables |

**Overview layout (desktop):**
```
┌──────────────┬──────────────────────────────────────────────────────────────────────┐
│ VeriHGN      │ HEADPATH · cell-head clamp diagnostic   [Validation only] [Integrity: status · n WARN] ◐ │
│ diagnostic   ├──────────────────────────────────────────────────────────────────────┤
│              │ Overview                                                             │
│ OVERVIEW     │ Research question (one paragraph)                                    │
│ • Summary    │ ┌ status strip: experiment status · replication · runs completed   ┐  │
│ RESULTS      │ └──────────────────────────────────────────────────────────────────┘  │
│ • O vs U     │ ┌ Primary result (pre-registered) ──────────────────────────────────┐ │
│ • Seed pairs │ │  seed   s1  s2  s3  s4  s5  s6     ← seeds read from the bundle   │ │
│ • Training   │ │  O      ·   ·   ·   ·   ·   ·      ← one category glyph per run   │ │
│ • Validation │ │  U      ·   ·   ·   ·   ·   ·        (legend lists all categories) │ │
│ • Metrics    │ │  sentence generated from analysis.primary_test + conclusion       │ │
│ • Designs    │ └────────────────────────────────────────────────────────────────────┘ │
│ • Maps       │ ┌ What this shows ───────────┐ ┌ What it does not show ───────────┐  │
│ CONTEXT      │ └────────────────────────────┘ └───────────────────────────────────┘  │
│ • Collapse   │                                                                      │
│ EVIDENCE     │                                                                      │
│ • Statistics │                                                                      │
│ • Methods    │                                                                      │
│ • Provenance │                                                                      │
└──────────────┴──────────────────────────────────────────────────────────────────────┘
```
Every value in this sketch is a placeholder; in the app each one is read from the bundle.

## 5. Research integrity in the code

- **No result literals.**
  - All numbers, counts, categories and conclusion sentences are computed from the bundle.
  - Static text is limited to definitions and methodology; it contains no numbers.
  - The pre-registered thresholds are read from `plan.thresholds`.
- **Null-safe value model.** A `Num = number | null` type is used throughout. The `fmt()` helper renders null as "undefined". Chart adapters convert null to a gap, never to 0. A unit test feeds nulls through every adapter.
- **Retests** are filtered out of every primary computation by `run.primary_inference`, from the bundle. They appear only in their own labelled panels.
- **The official and independent metric streams** are labelled and never mixed. Where they differ (0.0 vs undefined for constant predictions), both are shown, with the explanation.

## 6. Testing and visual QA

| Layer | Test |
|---|---|
| Data layer (Vitest) | schema guards; the array decoder against the Stage 1 Python encoder (golden vectors including −0.0, NaN, subnormals, int32); sha256_raw re-check; null never 0 |
| Synthetic fixture bundle | generated by the **real Stage 1 exporter** on the synthetic CPU pipeline (`tools/make_dashboard_fixture.py`), with values deliberately different from the real experiment. All UI tests run on it; a test asserts real-experiment values (e.g. the real p and run counts) do **not** appear anywhere |
| UI (Vitest + Testing Library + jsdom) | every page renders from the fixture; displayed numbers equal the fixture's; the validation-only notice and the deviation notice appear on every page; a fixture with a FAIL integrity status shows the blocking banner; retest values never enter the primary panels |
| No-literal lint | `app/src/**` scanned for numeric literals outside `theme.ts` and layout constants |
| Launcher (pytest) | refuses a tampered ZIP, never extracts into the project or Drive paths, binds to 127.0.0.1 only |
| Visual QA | headless **Microsoft Edge** (installed; no download) screenshots of every page at 1440 × 900 and 390 × 844, in light and dark. I inspect them, fix spacing, labels, overflow and contrast, and repeat. The screenshots are saved under `app/qa/` |
| Final check on the real bundle | `run_dashboard.bat <your zip>`, every page screenshotted and reviewed |

## 7. Files to create

| Path | Purpose |
|---|---|
| `app/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `index.html` | build (pinned) |
| `app/src/main.tsx`, `App.tsx`, `router.ts` | shell, hash routing |
| `app/src/theme/{tokens.css,theme.ts}` | design tokens (§3) |
| `app/src/data/{schema.ts,load.ts,arrays.ts,values.ts,derive.ts}` | typed loaders, decoder, null-safe values, pure derivations (rolling P2 window, pairs) |
| `app/src/components/{Shell,Sidebar,TopBar,Card,EvidenceNote,StatusBadge,Chart,DataTable,Undefined,ArmLegend,Heatmap}.tsx` | design-system components |
| `app/src/pages/*.tsx` | the 12 pages of §4 |
| `app/src/**/*.test.ts(x)`, `app/test-fixtures/synthetic_bundle/` | tests + synthetic fixture |
| `tools/make_dashboard_fixture.py` | builds the synthetic fixture with the real exporter |
| `tools/run_dashboard.py`, `run_dashboard.bat` | verify → extract → serve → open |
| `tests/test_dashboard_launcher.py` | launcher tests |
| `docs/DASHBOARD.md` | install, build and launch guide (exact commands), troubleshooting |

**Modified:**
- `README.md`: status row and dashboard section.
- `.gitignore`: `app/node_modules`, `app/dist` (unless you want the build committed).
- `tools/make_colab_bundle.py`: exclude `app/`.

**Untouched:** the experiment pipeline, training and evaluation code, checkpoints, every Drive artifact, and the verified ZIP, which is only read.

## 8. Milestones (my working time, estimates)

| # | Milestone | Done when | Estimate |
|---|---|---|---|
| M1 | Scaffold, design tokens, data layer + decoder + fixture, shell (sidebar / top bar / theme), **Overview** and **Provenance** pages | runs on the synthetic fixture and on your real ZIP; screenshots reviewed | 4–5 h |
| M2 | O vs U, Seed pairs, Training, Validation, Metrics, Per-design, Statistics, Methodology | tests green; screenshots reviewed | 5–7 h |
| M3 | Prediction maps, Collapse context, responsive and dark-mode polish, launcher, `docs/DASHBOARD.md` | every page inspected at two widths and in both themes on the real ZIP | 3–4 h |

I'd stop after M1 for you to review the look and the data wiring, before M2.

## 9. Risks

| Risk | Mitigation |
|---|---|
| A visual choice implies significance or a winner | evidence-type grammar (§3.6); no good/bad colours on arm comparisons; the significance rule always shown beside p; review checklist per page |
| Large maps (524 × 524) slow in the browser | load per selection; Plotly heatmap at native resolution (≈ 275 k cells is fine); I'll measure render times |
| npm registry access | needed once; afterwards fully offline. Optionally commit `app/dist` |
| Colour-scale confusion with the arm colours | maps use graphite (sequential) and teal–magenta (diverging), never blue / orange |
| Notebook-07 fields misread (A1–A6) | caveat footnotes from `field_caveats`, shown beside every affected field |
| Dark mode readability | separate validated dark steps; screenshots in both themes |

## 10. Decisions needed from you

1. **Approve M1** and stop there for a design review, or approve M1–M3 in one go?
2. **Commit the built `app/dist`** so viewing needs only Python, or build locally with npm?
3. **Fonts:** the system font (Segoe UI Variable on Windows; recommended, fully offline), or bundle Inter (open licence) for identical rendering on every machine?
