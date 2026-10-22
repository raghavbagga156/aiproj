/* Checks against the ACTUAL research export (read-only). Skipped when the ZIP is not on this machine.
   Path: $VERIHGN_REAL_ZIP, else the newest known export in the user's Downloads folder. Nothing is written.
   Expected values come from the bundle's own records (report, analysis, exporter checks), never typed in. */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";
import { absError, checkCellLabels, stats } from "../src/data/maps";
import type { MapsIndexEntry, RunIndexEntry } from "../src/data/schema";
import { parseYaml } from "../src/data/yaml";
import { FixtureBundle } from "./fixture";
import { plotlyCalls } from "./setup";

const ZIP = process.env["VERIHGN_REAL_ZIP"] ?? join(homedir(), "Downloads", "headpath_dashboard_20261022_110005.zip");
const HAVE = existsSync(ZIP);
let shared: FixtureBundle | null = null;
const bundle = () => (shared ??= new FixtureBundle(ZIP));
const dec = new TextDecoder();

describe.skipIf(!HAVE)(`actual research bundle (${ZIP})`, () => {
  it("decodes every exported map array losslessly (raw sha256 re-checked) and the maps agree with the exporter's records", async () => {
    const fb = bundle();
    const c = fb.client(16e6);                         // small cache: arrays are decoded, checked and released
    const mi = fb.json<Record<string, MapsIndexEntry>>("headpath/maps/index.json");
    const ri = fb.json<Record<string, RunIndexEntry>>("headpath/runs/index.json");
    const cmp = fb.json("headpath/analysis.json").grid_metrics_npz_vs_evals as Record<string, Record<string, { npz: number | null }>>;
    const cMax = fb.json("headpath/report.json").report.plan.c_max as number;
    let decoded = 0;
    const gts: Record<string, Float32Array> = {};
    for (const [s, e] of Object.entries(mi)) {
      const [gt, H, V, cellGt, tile] = await Promise.all(["grid_gt", "raw_h_overflow", "raw_v_overflow", "cell_gt", "cell_tile"]
        .map((k) => c.array(e.arrays[k]!)));
      decoded += 5;
      expect(gt!.shape).toEqual(e.shape);
      gts[s] = gt!.data as Float32Array;
      expect(checkCellLabels(cellGt!.data as Float32Array, gt!.data as Float32Array, tile!.data as Int32Array)).toBe(0);
      let maxDiff = 0;
      const h = H!.data as Float32Array, v = V!.data as Float32Array, g = gt!.data as Float32Array;
      for (let i = 0; i < g.length; i++) {
        const l = Math.fround(Math.fround(Math.log1p(Math.fround(Math.fround(h[i]! + v[i]!) / 2))) / cMax);
        maxDiff = Math.max(maxDiff, Math.abs(l - g[i]!));
      }
      expect(maxDiff).toBeLessThanOrEqual(1e-6);
      if (e.arrays["cell_index"]) { await c.array(e.arrays["cell_index"]); decoded++; }
    }
    for (const r of Object.values(ri)) {
      let zeros = 0, n = 0;
      for (const [s, m] of Object.entries(r.maps)) {
        const g = await c.array(m["grid_pred"]!), cp = await c.array(m["cell_pred"]!);
        decoded += 2;
        expect(g.shape).toEqual(mi[s]!.shape);
        expect(cp.shape[0]).toBe(mi[s]!.n_cells_subsample);
        const st = stats(g.data as Float32Array);
        expect(st.finite).toBe(st.n);                      // the exporter's "finite" check, re-done
        expect(st.min).toBeGreaterThanOrEqual(0);          // official clamped outputs
        zeros += st.zeroFrac! * st.n; n += st.n;
        expect(absError(g.data as Float32Array, gts[s]!).length).toBe(st.n);
      }
      const rec = cmp[r.name]?.["pooled_zero_frac"]?.npz;
      if (rec !== undefined && rec !== null) expect(Math.abs(zeros / n - rec)).toBeLessThanOrEqual(1e-9);
    }
    const listed = Object.keys(fb.json("manifest.json").files).filter((f) => f.endsWith(".bin.gz")).length;
    expect(decoded).toBe(listed);
    expect(c.cachedArrayBytes().bytes).toBeLessThanOrEqual(16e6);
  }, 600_000);

  it("reads every run's recorded config.yaml, consistent with the plan", () => {
    const fb = bundle();
    for (const r of fb.json("headpath/report.json").report.plan.runs) {
      const y = parseYaml(dec.decode(fb.files.get(`headpath/runs/${r.name}/sources/config.yaml`)!)) as any;
      expect(y.training.seed).toBe(r.seed);
      expect(y.model.cell_head_train_clamp).toBe(r.cell_head_train_clamp);
      expect(y.training.num_epochs).toBe(r.epochs);
      expect(Array.isArray(y.deviations) && y.deviations.length > 0).toBe(true);
    }
  });

  it("renders the M3 pages on the real bundle without errors", async () => {
    const fb = bundle();
    for (const [hash, title] of [["#/maps", "Prediction maps"], ["#/collapse", "Collapse investigation (notebook 07)"], ["#/methods", "Methodology and caveats"]]) {
      window.location.hash = hash!;
      plotlyCalls.length = 0;
      const { unmount } = render(<App client={fb.client()} />);
      await screen.findByRole("heading", { level: 1, name: title }, { timeout: 60_000 });
      if (hash === "#/maps") {
        await screen.findByTestId("check-cells", {}, { timeout: 60_000 });
        expect(screen.getByTestId("check-cells").textContent).toContain("consistent");
        expect(screen.getByTestId("check-label").textContent).toContain("consistent");
      }
      if (hash === "#/collapse") {
        const rows = within(await screen.findByTestId("c07-table")).getAllByRole("row").length - 1;
        expect(rows).toBe(fb.json("context07/report.json").report.results.length);
      }
      if (hash === "#/methods") await screen.findByTestId("arm-diff");
      expect(screen.queryByTestId("maps-error")).toBeNull();
      expect(screen.queryByTestId("config-errors")).toBeNull();
      expect(screen.queryByTestId("integrity-block")).toBeNull();
      expect(screen.queryByTestId("page-error")).toBeNull();
      await waitFor(() => expect(document.body.textContent).toContain("not significant"));
      unmount();
    }
  }, 300_000);
});
