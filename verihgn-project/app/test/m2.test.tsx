/* M2 pages on the SYNTHETIC bundle (values deliberately different from the real experiment). Every assertion
   compares what is rendered or plotted with the fixture's own values. */
import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";
import { forLogAxis, num, rollingImprovement, windowImprovement, type Table } from "../src/data/runs";
import { fmtNum, fmtSig } from "../src/data/values";
import { FixtureBundle } from "./fixture";
import { plotlyCalls } from "./setup";

const REAL_P_TEXT = "p = 0.5";

async function open(hash: string, fb = new FixtureBundle()) {
  window.location.hash = hash;
  plotlyCalls.length = 0;
  render(<App client={fb.client()} />);
  return fb;
}
const report = (fb: FixtureBundle) => fb.json("headpath/report.json").report;
const tableOf = (fb: FixtureBundle, run: string, kind: string): Table => fb.json(`headpath/runs/${run}/${kind}.json`);
const traceCalls = () => plotlyCalls.flatMap((c) => c.data as any[]);

describe("number formatting never hides sign or magnitude", () => {
  it("small non-zero values use scientific notation, exact zero stays zero", () => {
    expect(fmtNum(-0.00034, 3)).toBe("-3.4e-4");
    expect(fmtNum(0.0044, 2)).toBe("4.4e-3");
    expect(fmtNum(0, 3)).toBe("0.000");
    expect(fmtNum(0.0123, 3)).toBe("0.012");
    expect(fmtNum(null, 3)).toBeNull();
  });
});

describe("rolling P2 window", () => {
  it("equals a direct window sum and skips undefined pairs", () => {
    const n = [1, 2, null, 4, 5], d = [2, 2, 2, null, 10];
    const r = rollingImprovement(n, d, 2);
    expect(r[0]).toBeNull();
    expect(r[1]).toBeCloseTo(1 - 3 / 4, 12);
    expect(r[2]).toBeCloseTo(1 - 2 / 2, 12);                  // (null,2) pair skipped
    expect(r[4]).toBeCloseTo(1 - 5 / 10, 12);                 // (4,null) pair skipped
    expect(windowImprovement(n, d, 2)).toBeCloseTo(r[4]!, 12);
    expect(forLogAxis([0, 1e-3, null, 2])).toEqual({ y: [null, 1e-3, null, 2], zeros: 1, undefinedCount: 1 });
  });
  it("reproduces every recorded P2 of the fixture from its steps log", () => {
    const fb = new FixtureBundle();
    const rep = report(fb);
    const w = rep.plan.thresholds.p2_last_steps;
    for (const r of rep.results) {
      const t = tableOf(fb, r.name, "steps");
      const v = windowImprovement(num(t, "cell_wmse_clamped"), num(t, "cell_wmse_zero"), w);
      expect(v).toBeCloseTo(r.P2.cell_wmse_clamped_improvement_vs_zero, 9);
    }
  });
});

describe("M2 pages render the bundle's values", () => {
  it("every M2 page renders with its data-scope tag and the global caveats", async () => {
    const routes: [string, string, string][] = [
      ["#/arms", "Arm O vs Arm U", "training + validation"], ["#/seeds", "Seed-wise paired results", "training + validation"],
      ["#/training", "Training dynamics", "training"], ["#/validation", "Validation curves", "validation"],
      ["#/metrics", "Cell- and grid-level metrics", "training + validation"], ["#/designs", "Per-design comparison", "validation"],
      ["#/statistics", "Statistics and limitations", "pre-registered analysis"],
    ];
    for (const [hash, title, scope] of routes) {
      const fb = await open(hash);
      await screen.findByRole("heading", { level: 1, name: title });
      expect(screen.getByText(`Data: ${scope}`)).toBeTruthy();
      expect(screen.getByText(`Validation only · ${report(fb).plan.val.length} diagnostic graphs`)).toBeTruthy();
      expect(screen.getByText("No test-set results")).toBeTruthy();
      expect(document.body.textContent).not.toContain(REAL_P_TEXT);
      document.body.innerHTML = "";
    }
  });

  it("Arm comparison: category counts are the fixture's, retests excluded", async () => {
    const fb = await open("#/arms");
    await screen.findByRole("heading", { level: 1, name: "Arm O vs Arm U" });
    const rep = report(fb);
    const primary = rep.results.filter((r: any) => !r.retest);
    for (const c of rep.plan.categories as string[]) {
      const row = screen.getByTestId(`cat-row-${c}`);
      const cells = within(row).getAllByRole("cell");
      for (const [i, arm] of [[1, "O"], [2, "U"]] as const) {
        const want = primary.filter((r: any) => r.arm === arm && r.category === c).length;
        const tot = primary.filter((r: any) => r.arm === arm).length;
        expect(cells[i]!.textContent).toBe(`${want} / ${tot}`);
      }
    }
  });

  it("Seed pairs: exploratory differences are the fixture's and labelled exploratory", async () => {
    const fb = await open("#/seeds");
    await screen.findByText("Per-seed differences (U − O)");
    const ex = fb.json("headpath/analysis.json").exploratory;
    expect(screen.getAllByText(/Exploratory · not pre-registered · no inference/).length).toBeGreaterThan(0);
    for (const [k, m] of Object.entries<any>(ex.metrics)) {
      const row = screen.getByTestId(`expl-${k}`);
      const cells = within(row).getAllByRole("cell");
      m.per_seed.forEach((p: any, i: number) => {
        const want = fmtNum(p.U_minus_O, 3);
        expect(cells[i + 1]!.textContent).toBe(want ?? "undefined");
      });
    }
    for (const r of report(fb).results.filter((x: any) => x.retest)) expect(screen.getByText(r.name)).toBeTruthy();
  });

  it("Training: rolling P2 ends at the recorded P2; zero-gradient steps are counted, not drawn", async () => {
    const fb = await open("#/training");
    await screen.findByText(/matches the recorded P2/);
    await waitFor(() => expect(traceCalls().length).toBeGreaterThan(0));
    const rep = report(fb);
    const run = rep.results.find((r: any) => !r.retest && r.arm === "U");
    const tr = traceCalls().find((t) => t.name === `Arm U · seed ${run.seed}` && t.y.length > 0 && t.y.some((v: any) => v !== null && v <= 1) && t.x.length === rep.plan.steps_per_run);
    expect(tr).toBeTruthy();
    const steps = tableOf(fb, run.name, "steps");
    let zerosO = 0, zerosU = 0;
    for (const r of rep.results.filter((x: any) => !x.retest)) {
      const z = forLogAxis(num(tableOf(fb, r.name, "steps"), "grad_norm_shared")).zeros;
      if (r.arm === "O") zerosO += z; else zerosU += z;
    }
    expect(screen.getByTestId("zeros-grad_norm_shared").textContent).toContain(`Arm O ${zerosO}, Arm U ${zerosU}`);
    expect(num(steps, "global_step").length).toBe(rep.plan.steps_per_run);
  });

  it("Validation: an undefined correlation stays a gap and is counted", async () => {
    const fb = new FixtureBundle();
    const rep = report(fb);
    const run = rep.results.find((r: any) => !r.retest && r.arm === "O");
    const ev = tableOf(fb, run.name, "evals");
    ev.data["val_cell_spearman_indep"]![2] = null;
    fb.setJson(`headpath/runs/${run.name}/evals.json`, ev);
    await open("#/validation", fb);
    await screen.findByRole("heading", { level: 1, name: "Validation curves" });
    await waitFor(() => expect(traceCalls().some((t) => t.name === `Arm O · seed ${run.seed}`)).toBe(true));
    const cellRho = plotlyCalls.find((c) => (c.layout as any).yaxis?.title?.text === "cell ρ")!;
    const tr = (cellRho.data as any[]).find((t) => t.name === `Arm O · seed ${run.seed}`);
    expect(tr.y[2]).toBeNull();
    expect(tr.connectgaps).toBe(false);
    expect(screen.getByText(/Gaps: ρ undefined because the prediction was constant \(\d+ of \d+ points\)/)).toBeTruthy();
  });

  it("Metrics: final-step values are the fixture's; retests are marked", async () => {
    const fb = await open("#/metrics");
    await screen.findByRole("heading", { level: 1, name: "Cell- and grid-level metrics" });
    const rep = report(fb);
    for (const r of rep.results) {
      const ev = tableOf(fb, r.name, "evals");
      const last = ev.data["val_cell_spearman_indep"]![ev.n_rows - 1] as number | null;
      const row = await screen.findByTestId(`metrics-${r.name}`);
      expect(within(row).getAllByRole("cell")[2]!.textContent).toBe(fmtNum(last, 4) ?? "undefined");
      if (r.retest) expect(row.textContent).toContain("retest · excluded");
    }
  });

  it("Per-design heatmap: values from the evals log, undefined labelled", async () => {
    const fb = new FixtureBundle();
    const rep = report(fb);
    const run = rep.results.find((r: any) => !r.retest && r.arm === "U");
    const ev = tableOf(fb, run.name, "evals");
    const col = ev.columns.find((c: string) => /^design_.*_cell_spearman_indep$/.test(c))!;
    ev.data[col]![ev.n_rows - 1] = null;
    fb.setJson(`headpath/runs/${run.name}/evals.json`, ev);
    await open("#/designs", fb);
    await screen.findByRole("heading", { level: 1, name: "Per-design comparison" });
    await waitFor(() => expect(traceCalls().some((t) => t.type === "heatmap")).toBe(true));
    const hm = traceCalls().find((t) => t.type === "heatmap");
    expect(hm.z.flat()).toContain(null);
    expect(hm.text.flat()).toContain("undef.");
    expect(hm.zmid).toBe(0);
  });

  it("Statistics: test values and pairs are the fixture's", async () => {
    const fb = await open("#/statistics");
    await screen.findByRole("heading", { level: 1, name: "Statistics and limitations" });
    const t = report(fb).analysis.primary_test;
    expect(screen.getByTestId("stat-p").textContent).toBe(`p = ${fmtSig(t.p_two_sided_exact, 4)}`);
    expect(screen.getByTestId("stat-discordant").textContent).toBe(`${t.n_discordant} / ${t.n_valid_pairs}`);
    for (const p of t.pairs) expect(screen.getByTestId(`pair-${p.seed}`).textContent).toContain(p.discordant ? "discordant" : "concordant");
    expect(screen.getByText("matches the recorded test")).toBeTruthy();
    expect(screen.getByTestId("limitations").textContent).toContain(`${report(fb).plan.val.length} graphs`);
  });
});
