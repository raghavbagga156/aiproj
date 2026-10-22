import { useMemo, useState } from "react";
import { CorrelationHeatmap } from "../components/charts";
import { Card, Evidence, Loading, PageHead, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { lastRow, useRunTables } from "../data/runs";
import type { Num, RunIndexEntry } from "../data/schema";

const METRICS = [
  { key: "cell_spearman_indep", label: "Cell head · Spearman ρ (independent)" },
  { key: "grid_spearman_indep", label: "Grid head · Spearman ρ (independent)" },
] as const;

export function Designs({ themeKey }: { themeKey: string }) {
  const { report: rf } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const { index, tables, loading, error } = useRunTables(["evals"]);
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("cell_spearman_indep");
  const runs = useMemo<RunIndexEntry[]>(() => (index ? Object.values(index).filter((r) => r.primary_inference)
    .sort((a, b) => a.arm.localeCompare(b.arm) || a.seed - b.seed) : []), [index]);

  // designs of the validation set, with their role and graph count (from the plan)
  const designs = useMemo(() => {
    const valMain = new Set(plan.val_main);
    const by = new Map<string, { n: number; cross: boolean }>();
    for (const s of plan.val) {
      const d = s.split("-")[1]!;
      const e = by.get(d) ?? { n: 0, cross: false };
      e.n += 1;
      e.cross = e.cross || valMain.has(s);
      by.set(d, e);
    }
    return [...by.entries()].sort((a, b) => Number(b[1].cross) - Number(a[1].cross) || a[0].localeCompare(b[0]))
      .map(([d, e]) => ({ design: d, n: e.n, cross: e.cross }));
  }, [plan]);
  const rowLabel = (d: { design: string; n: number; cross: boolean }) =>
    `${d.design.replace("mgc_", "")} · ${d.cross ? "validation design" : "held-out"} · ${d.n} graph${d.n === 1 ? "" : "s"}`;
  const colLabel = (r: RunIndexEntry) => `${r.arm}·${r.seed}`;
  const z = useMemo<Num[][]>(() => designs.map((d) => runs.map((r) => {
    const v = lastRow(tables[r.name]?.evals)[`design_${d.design}_${metric}`];
    return typeof v === "number" ? v : null;
  })), [designs, runs, tables, metric]);
  const undef = z.flat().filter((v) => v === null).length;

  if (loading) return <Loading what="per-design metrics" />;
  if (error) return <div className="blocking-banner" role="alert">Per-design metrics could not be verified: {error}</div>;

  const table = (
    <div className="table-scroll">
      <table className="table" aria-label="Per-design values">
        <thead><tr><th>Design</th>{runs.map((r) => <th key={r.name} className="num tabular">{colLabel(r)}</th>)}</tr></thead>
        <tbody>{designs.map((d, i) => (
          <tr key={d.design}><td style={{ whiteSpace: "nowrap" }}>{rowLabel(d)}</td>{(z[i] ?? []).map((v, j) => <td key={j} className="num"><Value v={v} digits={3} /></td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );

  return (
    <>
      <PageHead eyebrow="Results" title="Per-design comparison" scope="VALIDATION"
        lede={<>Final-step Spearman ρ per validation design and run. The official validation design ({designs.filter((d) => d.cross).map((d) => d.design.replace("mgc_", "")).join(", ")})
          measures generalization across designs; the held-out graphs of training designs measure it within a design only (same netlist, different placement).</>} />

      <Card title="Spearman ρ by design and run" meta={`step ${plan.steps_per_run} · primary runs`}>
        <div className="metric-select">
          <div className="seg" role="group" aria-label="Metric">
            {METRICS.map((m) => <button key={m.key} aria-pressed={metric === m.key} onClick={() => setMetric(m.key)}>{m.label}</button>)}
          </div>
        </div>
        <CorrelationHeatmap rows={designs.map(rowLabel)} cols={runs.map(colLabel)} z={z} label="Per-design correlation heatmap"
          themeKey={`${themeKey}-${metric}`} colTitle="run (arm · seed)" table={table} />
        <p className="chart-caption">Colour runs from magenta (negative) through neutral (zero) to teal (positive); cell labels give the value. "undef." marks a correlation that is
          undefined because the prediction was constant: {undef} of {z.flat().length} cells. Rows with a single graph rest on one placement.</p>
      </Card>

      <div className="section grid-2">
        <Evidence kind="limitation" title="Reading per-design values">
          <ul className="bullets">
            <li>Single-graph rows are one placement of one design: they are noisy and are not design-level estimates.</li>
            <li>The validation-design row pools {designs.find((d) => d.cross)?.n ?? 0} graphs; pooled Spearman across graphs ranks values from differently scaled graphs together.</li>
            <li>These are validation-set values. No test-set design was evaluated.</li>
          </ul>
        </Evidence>
        <Evidence kind="observed" title="Input-feature reference">
          <p>How well the grid input features alone rank cells, per design, is on the <a href="#/metrics">Cell and grid metrics</a> page
            (non-learned baseline). It is descriptive and was computed on the same validation graphs.</p>
        </Evidence>
      </div>
    </>
  );
}
