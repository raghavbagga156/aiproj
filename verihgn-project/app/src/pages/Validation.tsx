import { useMemo, useState } from "react";
import { SeriesChart, type Series } from "../components/charts";
import { ArmName, Card, Evidence, Loading, PageHead, SeedPicker, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { lastRow, num, str, useRunTables, type Table } from "../data/runs";
import type { Num, RunIndexEntry } from "../data/schema";
import { validationScope } from "../data/derive";

const ACT_ABBR: Record<string, string> = { HEALTHY: "H", DEGENERATE: "Dg", DEAD: "D" };

export function Validation({ themeKey }: { themeKey: string }) {
  const { report: rf } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const th = plan.thresholds;
  const scope = validationScope(report);
  const { index, tables, loading, error } = useRunTables(["evals"]);
  const primary = useMemo(() => (index ? Object.values(index).filter((r) => r.primary_inference) : []), [index]);
  const seeds = useMemo(() => Array.from(new Set(primary.map((r) => r.seed))).sort((a, b) => a - b), [primary]);
  const [seed, setSeed] = useState<number | "all">("all");
  const ev = (r: RunIndexEntry): Table | undefined => tables[r.name]?.evals;

  const build = (col: string): Series[] => primary.filter((r) => seed === "all" || r.seed === seed).map((r) => ({
    name: `Arm ${r.arm} · seed ${r.seed}`, arm: r.arm, seed: r.seed, x: num(ev(r), "global_step") as number[],
    y: num(ev(r), col), emphasis: seed !== "all" ? true : undefined,
  }));
  const p3Rule = useMemo(() => [{ x: th.p3_min_cell_spearman, text: `P3 ≥ ${th.p3_min_cell_spearman}` }], [th]);
  const s = useMemo(() => ({
    cellRho: build("val_cell_spearman_indep"), gridRho: build("val_grid_spearman_indep"),
    cellZero: build("val_cell_pred_zero_frac"), gridZero: build("val_grid_pred_zero_frac"), loss: build("val_loss"),
  }), [primary, seed, tables]);   // eslint-disable-line react-hooks/exhaustive-deps
  const undefCount = (series: Series[]) => series.reduce((a, x) => a + x.y.filter((v) => v === null).length, 0);
  const evalSteps = plan.eval_steps;

  if (loading) return <Loading what="validation logs" />;
  if (error) return <div className="blocking-banner" role="alert">Validation logs could not be verified: {error}</div>;

  return (
    <>
      <PageHead eyebrow="Results" title="Validation curves" scope="VALIDATION"
        lede={<>Evaluation of every primary run at {evalSteps.length} pre-registered steps ({evalSteps.join(", ")}), always with the official
          clamped forward pass, on the diagnostic validation set of {scope.nGraphs} graphs. These are validation results, not test results.</>} />

      <div className="table-tools">
        <SeedPicker seeds={seeds} value={seed} onChange={setSeed} />
        <span className="muted" style={{ fontSize: 12.5 }}>Retest runs excluded.</span>
      </div>

      <div className="grid-2">
        <Evidence kind="criterion" title="Pooled cell Spearman ρ (independent)">
          <SeriesChart series={s.cellRho} label="Validation cell rho" xTitle="global step" yTitle="cell ρ" markers rules={p3Rule} themeKey={themeKey}
            caption={`Gaps: ρ undefined because the prediction was constant (${undefCount(s.cellRho)} of ${s.cellRho.reduce((a, x) => a + x.y.length, 0)} points). At step 0 both arms of a seed have identical weights, so their points coincide.`} />
        </Evidence>
        <Card title="Pooled grid Spearman ρ (independent)">
          <SeriesChart series={s.gridRho} label="Validation grid rho" xTitle="global step" yTitle="grid ρ" markers themeKey={themeKey}
            caption={`Gaps: undefined for a constant prediction (${undefCount(s.gridRho)} of ${s.gridRho.reduce((a, x) => a + x.y.length, 0)} points).`} />
        </Card>
        <Card title="Cell predictions exactly zero">
          <SeriesChart series={s.cellZero} label="Cell zero fraction" xTitle="global step" yTitle="fraction of cells = 0" markers yRange={[-0.02, 1.02]} themeKey={themeKey}
            caption="At 1 the head predicts 0 for every cell (DEAD)." />
        </Card>
        <Card title="Grid predictions exactly zero">
          <SeriesChart series={s.gridZero} label="Grid zero fraction" xTitle="global step" yTitle="fraction of tiles = 0" markers yRange={[-0.02, 1.02]} themeKey={themeKey} />
        </Card>
      </div>

      <div className="section grid-2">
        <Card title="Validation loss (official loss on the validation set)">
          <SeriesChart series={s.loss} label="Validation loss" xTitle="global step" yTitle="validation loss" markers themeKey={themeKey} digits={4}
            caption="Computed on the official clamped predictions in both arms, so the arms are comparable here." />
        </Card>
        <Card title="Activity at each evaluation" meta="H healthy · Dg degenerate · D dead">
          <div className="table-scroll">
            <table className="table activity" aria-label="Activity timeline">
              <thead><tr><th>Run</th><th>Head</th>{evalSteps.map((st) => <th key={st} className="num tabular">{st}</th>)}</tr></thead>
              <tbody>
                {primary.filter((r) => seed === "all" || r.seed === seed).sort((a, b) => a.seed - b.seed || a.arm.localeCompare(b.arm))
                  .flatMap((r) => (["cell", "grid"] as const).map((h) => {
                    const states = str(ev(r), `val_${h}_activity`);
                    return (
                      <tr key={`${r.name}-${h}`}>
                        {h === "cell" ? <td rowSpan={2}><ArmName arm={r.arm} short /> <span className="tabular muted">{r.seed}</span></td> : null}
                        <td className="muted">{h}</td>
                        {states.map((x, i) => <td key={i} className={`num act act-${x ?? "NA"}`}>{x ? ACT_ABBR[x] ?? x : "—"}</td>)}
                      </tr>
                    );
                  }))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      <div className="section">
        <Card title="Official vs independent correlation at the final step" meta={`step ${plan.steps_per_run}`}>
          <div className="table-scroll">
            <table className="table" aria-label="Official vs independent">
              <thead><tr><th>Run</th><th className="num">cell ρ official</th><th className="num">cell ρ independent</th><th className="num">grid ρ official</th><th className="num">grid ρ independent</th></tr></thead>
              <tbody>{[...primary].sort((a, b) => a.seed - b.seed || a.arm.localeCompare(b.arm)).map((r) => {
                const f = lastRow(ev(r));
                const n = (k: string): Num => (typeof f[k] === "number" ? (f[k] as number) : null);
                return (
                  <tr key={r.name}>
                    <td><ArmName arm={r.arm} short /> <span className="tabular muted">{r.seed}</span></td>
                    <td className="num"><Value v={n("val_cell_spearman")} digits={4} /></td><td className="num"><Value v={n("val_cell_spearman_indep")} digits={4} /></td>
                    <td className="num"><Value v={n("val_grid_spearman")} digits={4} /></td><td className="num"><Value v={n("val_grid_spearman_indep")} digits={4} /></td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
          <p className="chart-caption">For a constant prediction the official metric code returns zero, while the independent implementation reports undefined.
            Both are shown; the pre-registered criteria use the independent value.</p>
        </Card>
      </div>
    </>
  );
}
