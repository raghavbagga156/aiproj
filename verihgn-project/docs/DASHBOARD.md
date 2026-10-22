# Offline dashboard: install, launch, develop

**Status: milestone M3 (all planned sections built).**
- **Built (M1):** foundation, design system, data layer, Overview and Provenance pages.
- **Built (M2):** Arm O vs Arm U, Seed-wise pairs, Training dynamics, Validation curves, Cell and grid metrics, Per-design comparison, Statistics and limitations.
- **Built (M3):** Prediction maps, Collapse investigation (notebook 07), Methodology and caveats; global caveat chips and a limitations footer on every page; launcher re-checks cached extractions; end-to-end test.
- **Plan:** `docs/DASHBOARD_FRONTEND_PLAN.md`.

## 1. Launch (Windows 11; Python only, no Node.js, no internet)

```bat
cd "C:\Users\ragha\Desktop\ai for electronics project\verihgn-project"
run_dashboard.bat "C:\Users\ragha\Downloads\headpath_dashboard_20261022_110005.zip"
```

- With no argument, the newest `headpath_dashboard_*.zip` in your Downloads folder is used.
- Paths with spaces work; keep the quotes.
- The browser opens `http://127.0.0.1:8765/#/overview`. Press **Ctrl+C** in the window to stop.
- **Requirements:** Python 3.10 or newer, standard library only. The project's `.venv` Python is used if present. The prebuilt app (`app/dist/`) is part of the repository.

**What the launcher does** (`tools/run_dashboard.py`):
1. **Verifies the ZIP before showing anything:**
   - zip CRCs;
   - every file listed in `manifest.json`, with the listed size and sha256, and no unlisted files;
   - every `.json` file parses as strict JSON (no NaN / Infinity);
   - every array decodes (gzip → byte un-shuffle) to its recorded sha256;
   - manifest and integrity status are PASSED.
2. **Extracts the ZIP once per sha256** into `%LOCALAPPDATA%\VeriHGN-dashboard\bundles\<sha16>\`:
   - It never extracts into the project, a Drive folder, or the folder that holds the ZIP.
   - It never overwrites: an incomplete earlier extraction is moved aside, not deleted.
3. **Serves the app and the bundle on 127.0.0.1 only.** There is no backend logic and no external request.
4. **If verification fails,** the app still opens, but in its integrity-failure state, and shows **no results**.

**In the browser:** every bundle file is re-hashed against the manifest before use, and every array is decoded and re-hashed. Any mismatch blocks the results.

**Options:**
- `--port N`;
- `--no-browser`;
- `--cache-dir "<dir>"`;
- `--check-only` (verify and extract only; exit code 0 if verified).

## 2. What M1 shows

| Page | Content (every value is read from the bundle) |
|---|---|
| Executive summary | status strip; the pre-registered primary result (seed × arm matrix, discordant pairs, exact McNemar p with its significance rule, the verbatim conclusion); P2 and P3 criterion strips with the thresholds from the plan; what the data do and do not show; verbatim deviation notice; retest runs (excluded); integrity warnings |
| Provenance and integrity | launcher and in-browser verification; export check counts; bundle identity (export sha256, manifest, exporter, official commit); pinned WARN / FAIL checks; filterable table of all export checks; the JSON-normalization record (undefined values); array encoding; source fingerprints; bundle contents |

**Shown on every page:**
- "Validation only · N graphs";
- "No test-set results";
- "Arm U: diagnostic deviation".

## 2b. What M2 shows

Every page carries a data-scope tag (training, validation, training + validation, or pre-registered analysis).

| Page | Content (every value is read from the bundle or recomputed from its logs in the browser) |
|---|---|
| Arm O vs Arm U | run categories per arm (retests excluded); P2 and P3 criterion strips; grid-head WMSE improvement and validation ρ; H-sig counts |
| Seed-wise pairs | per-seed dumbbells for P2, P3, grid WMSE improvement and grid ρ; **exploratory** per-seed differences (U − O), labelled not pre-registered, no inference; test–retest control |
| Training dynamics | rolling P2 window (its final value is checked against the recorded P2); fraction of raw cell / grid outputs at or below zero; gradient norms on a log axis (zero-gradient steps are counted, not drawn) |
| Validation curves | validation loss; pooled cell and grid Spearman ρ (undefined ρ of a constant prediction stays a gap and is counted); fraction of predictions exactly zero; head activity at each evaluation; official vs independent correlation at the final step |
| Cell and grid metrics | validation metrics at the final step for all runs (retests marked "excluded"); training-set metrics at the final epoch; run validity; tile-projection reference; non-learned input-feature baseline |
| Per-design comparison | heatmap of final-step ρ per validation design and run (diverging, zero-centred; "undef." labelled); table instead of the heatmap on narrow screens |
| Statistics and limitations | the pre-registered exact McNemar test (re-derived from run categories and checked against the record), the pairs, H-sig counts, historical replication check, limitations, and the threshold documentation inconsistency (flagged, record not modified) |

Validation results are never presented as test performance; there are no test-set results.

## 2c. What M3 shows

| Page | Content |
|---|---|
| Prediction maps | one validation graph and seed at a time: label, Arm O and Arm U grid predictions, \|error\| of each (computed in the browser in float32), raw H and raw V overflow kept separate; all at native gcell resolution. Colour scale **shared** across compared panels by default (per-panel option, with a warning); optional square-root colour mapping (colour bar and hover stay in data units). "DEAD" marks an all-zero prediction map. Cell head: pred-vs-label density of the experiment's 20,000-cell subsample (no full-cell map exists). In-browser checks: label recomputed from H and V, cell labels equal their tile's label; exporter's maps-vs-log comparison. Selection is kept in the URL (`#/maps?sample=…&seed=…`) |
| Collapse investigation | notebook 07, labelled a **separate earlier experiment** (own scope tag, ink-only charts, never pooled with HEADPATH); per-run summary as recorded, with caveats A1–A5 quoted from the bundle and A6 (prediction spread read from history.csv); findings re-derived from the records (end-state of the cell head, timing of both-head collapses, FP16 vs FP32, the non-finite gradient norm); collapse timelines per group; how HEADPATH's replication check refers to it |
| Methodology and caveats | what the experiment is and is not; fidelity classes P / C / D / A / U; every run's recorded deviation list (config.yaml) quoted verbatim and grouped into sourced facts, adaptations and unresolved assumptions; key-by-key Arm O vs Arm U configuration difference; dataset, features, graph construction, splits, training, evaluation (validation-set table); pre-registered thresholds; the threshold inconsistency; limitations |

**On every page:** "Validation only · N diagnostic graphs", "No test-set results", "Arm U: diagnostic deviation", the primary test's "p = … · not significant" (links to Statistics), and a collapsible list of methodological limitations at the foot of the page.

**Memory:** only the selected graph's arrays are fetched; decoded arrays live in a least-recently-used cache bounded at 48 MB (raw size).

## 3. Develop and rebuild (Node.js 18+; internet needed once for `npm ci`)

```bat
cd app
npm ci
npm test            REM unit + UI tests on the synthetic fixture
npm run build       REM type-check + production build into app\dist (committed)
```

- **Synthetic fixture:** `app/test-fixtures/synthetic_bundle.zip` is test data from random synthetic designs, produced by the real HEADPATH and Stage 1 code. Regenerate it with `.venv\Scripts\python tools\make_dashboard_fixture.py` (about 2–4 min, CPU). Its values are deliberately different from the real experiment.
- **Real-bundle tests:** `app/test/real_bundle.test.tsx` runs against the actual export when it is on this machine (`VERIHGN_REAL_ZIP`, or the known file in Downloads) and is skipped otherwise. It reads the ZIP only.
- **Python tests:** `tests/test_dashboard_launcher.py`.
- **End-to-end test** (dashboard running through the launcher): `node app/e2e/e2e_live.mjs --out app/qa/m3` drives headless Edge over the DevTools protocol (Node 22+, no dependencies). It opens every page at 1440 px and 390 px in light and dark theme, checks headings, caveat chips, charts drawn, no integrity block, no horizontal page overflow and no browser errors, saves full-page screenshots, then exercises sidebar navigation, the theme toggle and every map graph / seed / scale, and fails if any request leaves the local server. Results: `app/qa/m3/e2e_results.json`.
- **Screenshots:** `app/qa/m1/`, `app/qa/m2/` (iframe method) and `app/qa/m3/` (`<page>_{desktop,mobile390}_{light,dark}.png`, all twelve pages).

## 4. Layout

```
app/
  index.html, vite.config.ts, tsconfig.json, package.json, package-lock.json   build (exact pins)
  src/theme/        tokens.css (design tokens), app.css, theme.ts (light/dark)
  src/data/         schema.ts, client.ts (verifying loader), arrays.ts (lossless decoder), hash.ts,
                    values.ts (null-safe numbers), derive.ts (all page numbers/sentences), BundleContext.tsx
  src/data/runs.ts  per-run log tables (steps / evals / history), rolling P2 window, log-axis helper
  src/components/   Shell, Chart (one Plotly wrapper), DataTable, ui (Card, Evidence, Status, Value, ArmMark…),
                    charts (CriterionStrip, Dumbbell, SeriesChart, CorrelationHeatmap)
  src/data/maps.ts  map loading (per selection), transpose [x, y] -> image, |error|, summaries, checks
  src/data/yaml.ts  reader for the config.yaml subset PyYAML writes (anything else -> error, never a guess)
  src/pages/        Overview, Provenance (M1); Arms, Seeds, Training, Validation, Metrics, Designs, Statistics (M2);
                    Maps, Collapse, Methods (M3)
  e2e/e2e_live.mjs  end-to-end test of the running dashboard
  test/             setup, fixture loader, data tests, UI tests (incl. the no-hard-coded-numbers scan)
  test-fixtures/    synthetic bundle + golden array vectors (Python encoder)
  dist/             production build (served by the launcher)
  qa/m1/ … qa/m3/   screenshots (+ e2e_results.json, map_timings.json)
```

## 4b. Hosted (static) deployment

The same build can be hosted as a static site (Vercel): `npm run build:vercel` verifies the pinned export in
`app/deploy/` and stages it under `dist/data/`. See `docs/DEPLOY_VERCEL.md`.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "The results bundle could not be verified" | a bundle file differs from its manifest. The launcher window lists the files. Re-download the export. |
| "Integrity failure: results are withheld" | the launcher, the manifest or the integrity report records a failure. Open *Provenance and integrity*. |
| Hosted site: "./data/manifest.json: HTTP 404" | the deployment was built without the data: use `npm run build:vercel` (see `docs/DEPLOY_VERCEL.md`). |
| The page is empty in the browser | open it through the launcher (`http://127.0.0.1:…`), not as a file. Web Crypto and `fetch` need the local server. |
| Port in use | the launcher tries the next 20 ports, or use `--port`. |
| "cached extraction no longer matches the ZIP" | a file in `%LOCALAPPDATA%\VeriHGN-dashboard\bundles\` was changed. The launcher moves that folder aside (`*.modified_<time>`, not deleted) and extracts the ZIP again. |
| "These maps could not be shown" on Prediction maps | an array's shape does not match its graph, or a run has no map for it; nothing is drawn for that pair. Other graphs and pages are unaffected. |
