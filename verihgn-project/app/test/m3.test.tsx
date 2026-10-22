/* M3: prediction maps, notebook-07 context, methodology, global caveats and failure handling, on the SYNTHETIC
   bundle (values deliberately different from the real experiment). Every expectation is computed from the
   fixture's own files, never typed in. */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";
import { decodeArray } from "../src/data/arrays";
import { absError, checkCellLabels, density2d, sharedMax, stats, toImage } from "../src/data/maps";
import type { ArrayMeta, MapsIndexEntry, RunIndexEntry } from "../src/data/schema";
import { fmtNum } from "../src/data/values";
import { parseYaml, YamlSubsetError } from "../src/data/yaml";
import { A6_NOTE } from "../src/pages/Collapse";
import { classifyDeviation } from "../src/pages/Methods";
import { FixtureBundle } from "./fixture";
import { plotlyCalls } from "./setup";

const enc = new TextEncoder();
const dec = new TextDecoder();

async function open(hash: string, fb = new FixtureBundle(), budget?: number) {
  window.location.hash = hash;
  plotlyCalls.length = 0;
  render(<App client={fb.client(budget)} />);
  return fb;
}
const report = (fb: FixtureBundle) => fb.json("headpath/report.json").report;
const mapsIndex = (fb: FixtureBundle) => fb.json<Record<string, MapsIndexEntry>>("headpath/maps/index.json");
const runsIndex = (fb: FixtureBundle) => fb.json<Record<string, RunIndexEntry>>("headpath/runs/index.json");
const arr = async (fb: FixtureBundle, m: ArrayMeta) => (await decodeArray(fb.files.get(m.file)!, m)).data;
const heatmaps = () => plotlyCalls.filter((c) => (c.data as any[])[0]?.type === "heatmap").map((c) => (c.data as any[])[0]);
function removeFile(fb: FixtureBundle, rel: string) {
  fb.files.delete(rel);
  const man = fb.json("manifest.json");
  delete man.files[rel];
  fb.files.set("manifest.json", enc.encode(JSON.stringify(man)));
}
const firstPair = (fb: FixtureBundle) => {
  const ri = Object.values(runsIndex(fb)).filter((r) => r.primary_inference);
  const seed = Math.min(...ri.map((r) => r.seed));
  return { seed, O: ri.find((r) => r.arm === "O" && r.seed === seed)!, U: ri.find((r) => r.arm === "U" && r.seed === seed)! };
};
const firstSample = (fb: FixtureBundle) => {
  const plan = report(fb).plan, mi = mapsIndex(fb);
  return [...plan.val_main, ...plan.val.filter((s: string) => !plan.val_main.includes(s))].find((s: string) => mi[s])! as string;
};

describe("map helpers", () => {
  it("transposes [x, y] row-major storage into display rows z[y][x]", () => {
    const a = new Float32Array([0, 1, 2, 10, 11, 12]);            // nx = 2, ny = 3: a[x * 3 + y]
    expect(toImage(a, [2, 3])).toEqual([[0, 10], [1, 11], [2, 12]]);
    expect(() => toImage(a, [3, 3])).toThrow(/does not match shape/);
    expect(toImage(new Float32Array([NaN, 1]), [1, 2])).toEqual([[null], [1]]);   // undefined -> gap, never 0
  });
  it("computes |pred - gt| in float32 and leaves non-finite pairs undefined", () => {
    const p = new Float32Array([0.1, 0, NaN, 3]), g = new Float32Array([0.3, 0, 1, Infinity]);
    const e = absError(p, g);
    expect(e[0]).toBe(Math.fround(Math.abs(Math.fround(0.1) - Math.fround(0.3))));
    expect(e[1]).toBe(0);
    expect(Number.isNaN(e[2]!) && Number.isNaN(e[3]!)).toBe(true);
    expect(() => absError(new Float32Array(2), new Float32Array(3))).toThrow();
  });
  it("summarises maps without treating undefined as zero", () => {
    const s = stats(new Float32Array([0, 0, 2, NaN]));
    expect(s).toMatchObject({ n: 4, finite: 3, min: 0, max: 2, zeroFrac: 2 / 3, allZero: false });
    expect(stats(new Float32Array([0, 0])).allZero).toBe(true);
    expect(stats(new Float32Array([NaN])).allZero).toBe(false);
    expect(sharedMax(new Float32Array([NaN]))).toBeNull();
    expect(sharedMax(new Float32Array([1, 5]), new Float32Array([7]))).toBe(7);
  });
  it("checks cell labels against their tiles and bins densities without dropping points silently", () => {
    const grid = new Float32Array([0.5, 0.25]);
    expect(checkCellLabels(new Float32Array([0.5, 0.25, 0.5]), grid, new Int32Array([0, 1, 0]))).toBe(0);
    expect(checkCellLabels(new Float32Array([0.5, 0.3]), grid, new Int32Array([0, 1]))).toBe(1);
    expect(checkCellLabels(new Float32Array([0.5]), grid, new Int32Array([9]))).toBe(1);       // tile out of range
    const d = density2d(new Float32Array([0, 0.5, 1, 2, NaN]), new Float32Array([0, 0.5, 1, 0, 0]), 2, 1);
    expect(d.z.flat().reduce((a, b) => a + b, 0)).toBe(3);
    expect(d.outside).toBe(1);
    expect(d.nonFinite).toBe(1);
  });
});

describe("config.yaml reader", () => {
  it("reads the PyYAML safe_dump subset: folded quoted scalars, escapes, nested blocks", () => {
    const y = parseYaml([
      "experiment:", "  name: headpath_O", "  n: 3", "  lr: 0.0005", "  flag: true", "  none: null", "  empty: []",
      "deviations:", "- 'A: it''s folded", "  over two lines'", "- plain text", "- \"double \\\"quoted\\\"\"", "runs:", "- name: a", "  seed: 42",
    ].join("\n")) as any;
    expect(y.experiment).toEqual({ name: "headpath_O", n: 3, lr: 0.0005, flag: true, none: null, empty: [] });
    expect(y.deviations).toEqual(["A: it's folded over two lines", "plain text", 'double "quoted"']);
    expect(y.runs).toEqual([{ name: "a", seed: 42 }]);
  });
  it("rejects what it cannot read instead of guessing", () => {
    expect(() => parseYaml("a: [1, 2]")).toThrow(YamlSubsetError);
    expect(() => parseYaml("a: 'unterminated")).toThrow(YamlSubsetError);
    expect(() => parseYaml("just text: : :\n  - x")).toThrow();
  });
  it("parses every run's recorded config.yaml in the fixture, consistent with the plan", () => {
    const fb = new FixtureBundle();
    for (const r of report(fb).plan.runs) {
      const y = parseYaml(dec.decode(fb.files.get(`headpath/runs/${r.name}/sources/config.yaml`)!)) as any;
      expect(y.training.seed).toBe(r.seed);
      expect(y.model.cell_head_train_clamp).toBe(r.cell_head_train_clamp);
      expect(y.training.num_epochs).toBe(r.epochs);
      expect(y.deviations.some((d: string) => d.startsWith("DIAGNOSTIC DEVIATION"))).toBe(r.arm === "U");
    }
  });
  it("classifies recorded deviations by fidelity class", () => {
    expect(classifyDeviation("A: x").kind).toBe("adaptation");
    expect(classifyDeviation("A/D: x")).toMatchObject({ kind: "adaptation", codes: ["A", "D"] });
    expect(classifyDeviation("U: x").kind).toBe("assumption");
    expect(classifyDeviation("P (deviates from official code): x")).toMatchObject({ kind: "fact", codes: ["P"] });
    expect(classifyDeviation("C-ablation: x")).toMatchObject({ kind: "fact", codes: ["C"] });
    expect(classifyDeviation("Not claimed: reproduction").kind).toBe("not-claimed");
    expect(classifyDeviation("DIAGNOSTIC DEVIATION (model)").kind).toBe("diagnostic");
    expect(classifyDeviation("precision fp32 (official ...)").kind).toBe("setting");
  });
});

describe("decoded-array cache is bounded", () => {
  it("evicts least-recently-used arrays beyond the byte budget and re-verifies on reload", async () => {
    const fb = new FixtureBundle();
    const metas = Object.values(mapsIndex(fb)).flatMap((e) => Object.values(e.arrays));
    const budget = metas[0]!.nbytes_raw * 2.5;
    const c = fb.client(budget);
    for (const m of metas) await c.array(m);
    const cached = c.cachedArrayBytes();
    expect(cached.bytes).toBeLessThanOrEqual(budget);
    expect(cached.count).toBeLessThan(metas.length);
    const before = fb.fetches.get(metas[0]!.file);
    await c.array(metas[0]!);                                                   // evicted: fetched and verified again
    expect(fb.fetches.get(metas[0]!.file)).toBe((before ?? 0) + 1);
    const last = metas[metas.length - 1]!;
    const n = fb.fetches.get(last.file);
    await c.array(last);                                                        // still cached
    expect(fb.fetches.get(last.file)).toBe(n);
  });
});

describe("Prediction maps page", () => {
  it("draws the exported label, both arms' predictions and their errors on shared scales", async () => {
    const fb = await open("#/maps");
    await screen.findByRole("heading", { level: 1, name: "Prediction maps" });
    await screen.findByTestId("map-sample-meta");
    const sample = firstSample(fb), { O, U, seed } = firstPair(fb);
    const e = mapsIndex(fb)[sample]!;
    await waitFor(() => expect(heatmaps().length).toBeGreaterThanOrEqual(7));
    const gt = await arr(fb, e.arrays["grid_gt"]!) as Float32Array;
    const pO = await arr(fb, O.maps[sample]!["grid_pred"]!) as Float32Array;
    const pU = await arr(fb, U.maps[sample]!["grid_pred"]!) as Float32Array;
    const h = heatmaps();
    const zs = h.map((t) => JSON.stringify(t.z));
    for (const a of [gt, pO, pU, absError(pO, gt), absError(pU, gt)]) expect(zs).toContain(JSON.stringify(toImage(a, e.shape)));
    const share = sharedMax(gt, pO, pU);
    const labelTrace = h.find((t) => JSON.stringify(t.z) === JSON.stringify(toImage(gt, e.shape)))!;
    expect(labelTrace.zmax).toBe(share);
    expect(h.find((t) => JSON.stringify(t.z) === JSON.stringify(toImage(pU, e.shape)))!.zmax).toBe(share);
    expect(h.find((t) => JSON.stringify(t.z) === JSON.stringify(toImage(absError(pO, gt), e.shape)))!.zmax).toBe(sharedMax(absError(pO, gt), absError(pU, gt)));
    // raw H and V are separate maps on one common scale
    const H = await arr(fb, e.arrays["raw_h_overflow"]!) as Float32Array, V = await arr(fb, e.arrays["raw_v_overflow"]!) as Float32Array;
    const hT = h.find((t) => JSON.stringify(t.z) === JSON.stringify(toImage(H, e.shape)))!, vT = h.find((t) => JSON.stringify(t.z) === JSON.stringify(toImage(V, e.shape)))!;
    expect(hT.zmax).toBe(sharedMax(H, V));
    expect(vT.zmax).toBe(hT.zmax);
    expect(screen.getByTestId("map-sample-meta").textContent).toContain(`HEADPATH_O__seed${seed}`);
    // DEAD tags exactly where a prediction map is all zero
    const dead = [pO, pU].filter((p) => stats(p).allZero).length;
    expect(screen.queryAllByTestId("dead-tag").length).toBeGreaterThanOrEqual(dead);
    // cell subsample: every finite cell is counted once, labels equal their tiles
    const cellGt = await arr(fb, e.arrays["cell_gt"]!) as Float32Array;
    const dens = h.filter((t) => Array.isArray(t.customdata) && Array.isArray(t.x));
    expect(dens.length).toBe(2);
    for (const d of dens) expect((d.customdata as number[][]).flat().reduce((a, b) => a + b, 0)).toBe(cellGt.length);
    expect(screen.getByTestId("check-cells").textContent).toContain("consistent");
    expect(screen.getByTestId("check-label").textContent).toContain("consistent");
    expect(screen.getByTestId("mapstat-Label (grid_gt)").textContent).toContain(fmtNum(stats(gt).max, 4)!);
  });

  it("changes graph, seed, scale and colour mapping and keeps them in the URL", async () => {
    const fb = await open("#/maps");
    await screen.findByTestId("map-sample-meta");
    const samples = Object.keys(mapsIndex(fb));
    const other = samples.find((s) => s !== firstSample(fb))!;
    fireEvent.change(screen.getByLabelText("Validation graph"), { target: { value: other } });
    await waitFor(() => expect(screen.getByTestId("map-sample-meta").textContent).toContain(mapsIndex(fb)[other]!.design));
    await waitFor(() => expect(window.location.hash).toContain(`sample=${encodeURIComponent(other)}`));
    const seeds = Array.from(new Set(Object.values(runsIndex(fb)).filter((r) => r.primary_inference).map((r) => r.seed))).sort();
    fireEvent.click(within(screen.getByRole("group", { name: "Seed" })).getByText(String(seeds[seeds.length - 1])));
    await waitFor(() => expect(screen.getByTestId("map-sample-meta").textContent).toContain(`seed${seeds[seeds.length - 1]}`));
    fireEvent.click(screen.getByText("Per panel (pattern only)"));
    expect(screen.getByTestId("scale-warning")).toBeTruthy();
    plotlyCalls.length = 0;
    fireEvent.click(screen.getByText("Square root"));
    expect(screen.getByTestId("sqrt-note")).toBeTruthy();
    await waitFor(() => expect(heatmaps().some((t) => Array.isArray(t.customdata) && !Array.isArray(t.x))).toBe(true));
    const t = heatmaps().find((x) => Array.isArray(x.customdata) && !Array.isArray(x.x))!;
    const flatZ = (t.z as (number | null)[][]).flat(), flatC = (t.customdata as (number | null)[][]).flat();
    flatC.forEach((v, i) => expect(flatZ[i]).toBe(v === null ? null : Math.sqrt(v)));       // colour only; hover keeps the value
    expect(window.location.hash).toContain("colour=sqrt");
    expect(window.location.hash).toContain("scale=panel");
    expect(screen.queryByTestId("maps-error")).toBeNull();
  });

  it("refuses to compare maps whose shape does not match the sample", async () => {
    const fb = new FixtureBundle();
    const mi = mapsIndex(fb), s = firstSample(fb);
    mi[s]!.shape = [mi[s]!.shape[0] + 1, mi[s]!.shape[1]];
    fb.setJson("headpath/maps/index.json", mi);
    await open("#/maps", fb);
    expect((await screen.findByTestId("maps-error")).textContent).toMatch(/does not match the sample/);
    expect(heatmaps().length).toBe(0);
    expect(screen.queryByTestId("integrity-block")).toBeNull();
  });

  it("reports a prediction missing from the index instead of drawing a partial pair", async () => {
    const fb = new FixtureBundle();
    const ri = runsIndex(fb), { U } = firstPair(fb), s = firstSample(fb);
    delete ri[U.name]!.maps[s];
    fb.setJson("headpath/runs/index.json", ri);
    await open("#/maps", fb);
    expect((await screen.findByTestId("maps-error")).textContent).toContain("grid_pred");
  });

  it("blocks all results when a prediction array was tampered with", async () => {
    const fb = new FixtureBundle();
    const { O } = firstPair(fb), s = firstSample(fb);
    const f = O.maps[s]!["grid_pred"]!.file;
    const b = new Uint8Array(fb.files.get(f)!); b[b.length - 9] = b[b.length - 9]! ^ 0xff;
    fb.setBytes(f, b, false);                                              // not re-signed: manifest hash mismatch
    await open("#/maps", fb);
    expect((await screen.findByTestId("integrity-block")).textContent).toContain(f);
  });

  it("blocks all results when an array decodes to the wrong content, even with a matching file hash", async () => {
    const fb = new FixtureBundle();
    const { U } = firstPair(fb), s = firstSample(fb);
    const meta = U.maps[s]!["grid_pred"]!;
    const { gzipSync } = await import("fflate");
    const raw = new Uint8Array(meta.nbytes_raw); raw[0] = 1;              // valid gzip, wrong bytes
    fb.setBytes(meta.file, gzipSync(raw));                                  // re-signed: only sha256_raw can catch it
    await open("#/maps", fb);
    expect((await screen.findByTestId("integrity-block")).textContent).toMatch(/decoded sha256 mismatch/);
  });

  it("treats a malformed map index as an integrity failure", async () => {
    const fb = new FixtureBundle();
    fb.setBytes("headpath/maps/index.json", enc.encode('{"a": NaN}'));
    await open("#/maps", fb);
    expect((await screen.findByTestId("integrity-block")).textContent).toMatch(/not strict JSON/);
  });

  it("says so when the bundle has no maps", async () => {
    const fb = new FixtureBundle();
    removeFile(fb, "headpath/maps/index.json");
    await open("#/maps", fb);
    expect(await screen.findByText("No prediction maps in this bundle")).toBeTruthy();
  });
});

describe("Collapse investigation (notebook 07) page", () => {
  it("keeps notebook 07 separate and shows every run with the caveats verbatim", async () => {
    const fb = await open("#/collapse");
    await screen.findByRole("heading", { level: 1, name: "Collapse investigation (notebook 07)" });
    const c7 = fb.json("context07/report.json");
    expect(screen.getByText("Data: notebook 07 · separate experiment")).toBeTruthy();
    expect(screen.getByTestId("c07-separation").textContent).toContain("The labels O and U do not apply");
    for (const r of c7.report.results) {
      const row = await screen.findByTestId(`c07-${r.name}`);
      expect(row.textContent).toContain(r.collapsed ? "yes" : "no");
      expect(row.textContent).toContain(String(r.zero_grad_steps));
    }
    const cav = screen.getByTestId("c07-caveats").textContent!;
    for (const [k, v] of Object.entries<string>(c7.field_caveats)) {
      if (k.startsWith("_")) continue;
      expect(cav).toContain(v.replace(/^A\d+:\s*/, ""));
    }
    expect(cav).toContain(A6_NOTE.replace(/^A6:\s*/, ""));
    for (const code of ["A1", "A2", "A3", "A4", "A5", "A6"]) expect(document.getElementById(`caveat-${code}`)).toBeTruthy();
    // never pooled: no HEADPATH arm names on the notebook-07 charts
    await waitFor(() => expect(plotlyCalls.length).toBeGreaterThan(0));
    expect(plotlyCalls.flatMap((c) => c.data as any[]).every((t) => /^seed \d+$/.test(t.name))).toBe(true);
  });

  it("re-derives the forensic counts from the records", async () => {
    const fb = await open("#/collapse");
    await screen.findByTestId("c07-deadcell");
    const res = fb.json("context07/report.json").report.results as any[];
    const dead = res.filter((r) => r.final_val_pred_zero_frac.cell === 1);
    const notC = res.filter((r) => !r.collapsed);
    expect(screen.getByTestId("c07-deadcell").textContent).toContain(`${dead.length} of ${res.length} runs`);
    expect(screen.getByTestId("c07-deadcell").textContent).toContain(`${notC.filter((r) => dead.includes(r)).length} of the ${notC.length} runs`);
    const minTrain = Math.min(...res.map((r) => r.n_train));
    const col = res.filter((r) => r.collapsed);
    expect(screen.getByTestId("c07-early").textContent).toContain(
      `${col.filter((r) => r.collapse_step !== null && r.collapse_step <= minTrain).length} of ${col.length} both-head collapses began by step ${minTrain}`);
  });

  it("marks a non-finite source value as non-finite, never as a number", async () => {
    const fb = new FixtureBundle();
    const c7 = fb.json("context07/report.json");
    c7.report.results[0].max_grad_norm = null;
    c7.undefined_paths = [{ path: "/results/0/max_grad_norm", kind: "inf" }];
    fb.setJson("context07/report.json", c7);
    await open("#/collapse", fb);
    const row = await screen.findByTestId(`c07-${c7.report.results[0].name}`);
    expect(row.textContent).toContain("non-finite (inf)");
    expect(screen.getByTestId("c07-nonfinite").textContent).toContain(c7.report.results[0].name);
  });

  it("says so when the bundle has no notebook-07 records", async () => {
    const fb = new FixtureBundle();
    removeFile(fb, "context07/report.json");
    await open("#/collapse", fb);
    expect(await screen.findByText("Notebook-07 records are not in this bundle")).toBeTruthy();
  });
});

describe("Methodology page", () => {
  it("quotes the recorded deviations by class and compares the arms' configurations", async () => {
    const fb = await open("#/methods");
    await screen.findByTestId("arm-diff");
    const plan = report(fb).plan;
    const cfgO = parseYaml(dec.decode(fb.files.get(`headpath/runs/${plan.runs.find((r: any) => r.arm === "O").name}/sources/config.yaml`)!)) as any;
    for (const d of cfgO.deviations as string[]) {
      const kind = classifyDeviation(d).kind;
      const id = kind === "adaptation" ? "dev-adaptation" : kind === "assumption" ? "dev-assumption" : kind === "not-claimed" ? "not-claimed" : "dev-fact";
      expect(screen.getByTestId(id).textContent).toContain(d);
    }
    const diff = screen.getByTestId("arm-diff");
    expect(diff.textContent).toContain("model.cell_head_train_clamp");
    expect(within(diff).getAllByRole("row").length - 1).toBeGreaterThanOrEqual(1);
    expect(screen.getByTestId("arm-u-deviation").textContent).toMatch(/^DIAGNOSTIC DEVIATION/);
    expect(within(screen.getByTestId("m-valset")).getAllByRole("row").length - 1).toBe(plan.val.length);
    expect(screen.getByTestId("m-thresholds").textContent).toContain(String(plan.thresholds.p3_min_cell_spearman));
    expect(screen.getByTestId("m-limitations")).toBeTruthy();
    expect(document.body.textContent).toContain("not an exact reproduction of the paper");
  });

  it("shows unreadable configurations as errors and still renders the rest", async () => {
    const fb = new FixtureBundle();
    const name = report(fb).plan.runs[0].name;
    fb.setBytes(`headpath/runs/${name}/sources/config.yaml`, enc.encode("experiment: [broken\n"));
    await open("#/methods", fb);
    expect((await screen.findByTestId("config-errors")).textContent).toContain(name);
    expect(screen.getByTestId("m-splits")).toBeTruthy();
  });
});

describe("caveats are visible on every page", () => {
  it("every page shows the scope chips, the test result and the limitations footer", async () => {
    const pages = ["overview", "arms", "seeds", "training", "validation", "metrics", "designs", "maps", "collapse", "statistics", "methods", "provenance"];
    for (const p of pages) {
      const fb = await open(`#/${p}`);
      await screen.findByTestId("chip-p");
      const t = report(fb).analysis.primary_test;
      expect(screen.getByText(`Validation only · ${report(fb).plan.val.length} diagnostic graphs`)).toBeTruthy();
      expect(screen.getByText("No test-set results")).toBeTruthy();
      expect(screen.getByText("Arm U: diagnostic deviation")).toBeTruthy();
      expect(screen.getByTestId("chip-p").textContent).toContain(t.significant ? "significant" : "not significant");
      expect(screen.getByTestId("footer-limitations").textContent).toContain("No test-set results");
      expect(document.querySelectorAll(".nav-item.is-planned").length).toBe(0);
      document.body.innerHTML = "";
    }
  });
});
