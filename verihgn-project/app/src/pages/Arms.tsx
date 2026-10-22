import { useMemo } from "react";
import { CriterionStrip } from "../components/charts";
import { ArmName, Card, Evidence, PageHead, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { CATEGORY_DEFINITION, categoryLabel, primaryRuns } from "../data/derive";
import type { RunResult } from "../data/schema";

const P2 = (r: RunResult) => r.P2.cell_wmse_clamped_improvement_vs_zero;
const P3 = (r: RunResult) => r.P3.final_val_cell_spearman_indep;
const GRID_IMP = (r: RunResult) => r.grid.wmse_improvement_vs_zero;
const GRID_RHO = (r: RunResult) => r.grid.final_val_spearman_indep;
const ACTIVITY = ["HEALTHY", "DEGENERATE", "DEAD"] as const;

export function Arms({ themeKey }: { themeKey: string }) {
  const { report: rf } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const th = plan.thresholds;
  const runs = useMemo(() => primaryRuns(report).filter((r) => r.run_status === "COMPLETED"), [report]);
  const seeds = useMemo(() => Array.from(new Set(runs.map((r) => r.seed))).sort((a, b) => a - b), [runs]);
  const cats = plan.categories;
  const count = (arm: "O" | "U", c: string) => runs.filter((r) => r.arm === arm && r.category === c).length;
  const total = (arm: "O" | "U") => runs.filter((r) => r.arm === arm).length;
  const act = (arm: "O" | "U", head: "cell" | "grid", state: string) =>
    runs.filter((r) => r.arm === arm && (head === "cell" ? r.P1.final_cell_activity : r.grid.final_activity) === state).length;
  const p2Band = useMemo<[number, number]>(() => [th.p2_borderline_low, th.p2_borderline_high], [th]);
  const p2Rules = useMemo(() => [{ x: th.p2_borderline_low, text: `FAIL ≤ ${th.p2_borderline_low}`, anchor: "right" as const },
                                 { x: th.p2_borderline_high, text: `PASS ≥ ${th.p2_borderline_high}`, anchor: "left" as const }], [th]);
  const p3Rules = useMemo(() => [{ x: th.p3_min_cell_spearman, text: `P3 ≥ ${th.p3_min_cell_spearman}` }], [th]);
  const none = useMemo(() => [], []);
  const hsig = report.analysis.hsig;

  return (
    <>
      <PageHead eyebrow="Results" title="Arm O vs Arm U" scope="TRAINING + VALIDATION"
        lede={<>Outcomes of the {total("O") + total("U")} primary runs ({seeds.length} seeds per arm), side by side. Arm O is the
          official model; Arm U removes the cell-head clamp during training only. The retest runs are excluded.</>} />

      <Card title="Run categories by arm" meta="pre-registered categories, mutually exclusive">
        <div className="table-scroll">
          <table className="table" aria-label="Run categories by arm">
            <thead><tr><th>Category</th><th className="num"><ArmName arm="O" short /></th><th className="num"><ArmName arm="U" short /></th><th>Definition</th></tr></thead>
            <tbody>
              {cats.map((c) => (
                <tr key={c} data-testid={`cat-row-${c}`}>
                  <td style={{ fontWeight: c === "LEARNS_AND_GENERALIZES" ? 600 : 400 }}>{categoryLabel(c)}</td>
                  <td className="num">{count("O", c)} <span className="muted">/ {total("O")}</span></td>
                  <td className="num">{count("U", c)} <span className="muted">/ {total("U")}</span></td>
                  <td className="secondary" style={{ fontSize: 12.5 }}>{CATEGORY_DEFINITION[c] ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="section">
        <h2 className="section-title">Head activity at the final evaluation</h2>
        <p className="section-sub">Activity classes from the validation predictions at step {plan.steps_per_run}: DEAD (all predictions exactly zero), DEGENERATE (spread below {th.activity_std_min} or zero fraction ≥ {th.activity_zero_max}), HEALTHY otherwise. Activity says nothing about correlation.</p>
        <div className="grid-2">
          {(["cell", "grid"] as const).map((head) => (
            <Card key={head} title={head === "cell" ? "Cell head" : "Grid head"} meta="number of seeds">
              <table className="table" aria-label={`${head} head activity`}>
                <thead><tr><th>State</th><th className="num"><ArmName arm="O" short /></th><th className="num"><ArmName arm="U" short /></th></tr></thead>
                <tbody>{ACTIVITY.map((s) => (
                  <tr key={s}><td>{s}</td><td className="num">{act("O", head, s)}</td><td className="num">{act("U", head, s)}</td></tr>
                ))}</tbody>
              </table>
              <p className="chart-caption">
                Per seed ({seeds.join(", ")}): O {seeds.map((sd) => { const r = runs.find((x) => x.arm === "O" && x.seed === sd); return r ? (head === "cell" ? r.P1.final_cell_activity : r.grid.final_activity) : "missing"; }).join(" · ")};
                {" "}U {seeds.map((sd) => { const r = runs.find((x) => x.arm === "U" && x.seed === sd); return r ? (head === "cell" ? r.P1.final_cell_activity : r.grid.final_activity) : "missing"; }).join(" · ")}.
              </p>
            </Card>
          ))}
        </div>
      </div>

      <div className="section">
        <h2 className="section-title">Criterion values per run</h2>
        <p className="section-sub">Each mark is one run. Thresholds come from the experiment plan.</p>
        <div className="grid-2">
          <Evidence kind="criterion" title="P2 · cell learning against the zero predictor">
            <CriterionStrip runs={runs} value={P2} label="P2 improvement" xTitle="P2 improvement over the zero predictor"
              band={p2Band} rules={p2Rules} themeKey={themeKey} />
            <p className="chart-caption">Training logs, last {th.p2_last_steps} steps, clamped predictions in both arms.</p>
          </Evidence>
          <Evidence kind="criterion" title="P3 · pooled validation cell Spearman ρ">
            <CriterionStrip runs={runs} value={P3} label="Validation cell ρ" xTitle="pooled validation cell Spearman ρ" rules={p3Rules} themeKey={themeKey} />
            <p className="chart-caption">Validation set, step {plan.steps_per_run}. Runs with an undefined ρ (constant prediction) are not drawn: {runs.filter((r) => P3(r) === null).map((r) => `${r.arm}·${r.seed}`).join(", ") || "none"}.</p>
          </Evidence>
          <Evidence kind="observed" title="Grid head · WMSE improvement over the zero predictor">
            <CriterionStrip runs={runs} value={GRID_IMP} label="Grid WMSE improvement" xTitle="grid WMSE improvement over the zero predictor" rules={none} themeKey={themeKey} />
            <p className="chart-caption">Training logs, last {th.p2_last_steps} steps. Descriptive (no pre-registered threshold).</p>
          </Evidence>
          <Evidence kind="observed" title="Grid head · pooled validation Spearman ρ">
            <CriterionStrip runs={runs} value={GRID_RHO} label="Validation grid ρ" xTitle="pooled validation grid Spearman ρ" rules={none} themeKey={themeKey} />
            <p className="chart-caption">Undefined (constant prediction, not drawn): {runs.filter((r) => GRID_RHO(r) === null).map((r) => `${r.arm}·${r.seed}`).join(", ") || "none"}.</p>
          </Evidence>
        </div>
      </div>

      <div className="section grid-2">
        <Evidence kind="criterion" title="H-sig: active head without useful signal">
          <p>Runs meeting the H-sig criteria (descriptive count, no test applied):</p>
          <ul className="bullets">
            {(["O", "U"] as const).map((a) => (
              <li key={a}><ArmName arm={a} short />: {hsig[a]?.n_seeds_meeting_hsig ?? 0} of {hsig[a]?.n_valid_runs ?? 0} valid runs{(hsig[a]?.seeds.length ?? 0) > 0 ? ` (seeds ${hsig[a]!.seeds.join(", ")})` : ""}.</li>
            ))}
          </ul>
        </Evidence>
        <Evidence kind="limitation" title="Reading the grid-head comparison">
          <p>The grid head's code is official in both arms, but most parameters are shared between the heads. When Arm O's cell head is
            inactive the shared trunk is trained by the grid loss alone; in Arm U both heads train it. Grid differences between the arms
            are therefore expected and are not a clean control.</p>
          <p className="secondary">P1 also requires the cell head to be active on enough of the last {th.p1_last_steps} training steps (dead fraction ≤ {th.p1_max_dead_frac}).
            Dead fraction of the last steps, per seed:</p>
          <ul className="bullets">
            {(["O", "U"] as const).map((a) => (
              <li key={a}><ArmName arm={a} short />: {runs.filter((r) => r.arm === a).sort((x, y) => x.seed - y.seed).map((r, i) => (
                <span key={r.seed}>{i > 0 && " · "}{r.seed}: <Value v={r.P1.dead_frac_last_steps} digits={2} /></span>))}</li>
            ))}
          </ul>
        </Evidence>
      </div>
    </>
  );
}
