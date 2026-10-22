import { useMemo } from "react";
import { Dumbbell, type PairPoint } from "../components/charts";
import { ArmName, Card, Evidence, PageHead, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { categoryLabel, primaryRuns, retestRuns } from "../data/derive";
import type { Num, RunResult } from "../data/schema";

interface ExploratoryMetric {
  per_seed: { seed: number; O: Num; U: Num; U_minus_O: Num }[];
  n_defined_pairs: number; n_U_greater: number; n_O_greater: number; n_equal: number;
}
const METRIC_LABEL: Record<string, string> = {
  P2_cell_wmse_clamped_improvement: "P2 cell WMSE improvement (training)",
  final_val_cell_spearman_indep: "Final pooled validation cell ρ",
  final_val_grid_spearman_indep: "Final pooled validation grid ρ",
  grid_wmse_improvement: "Grid WMSE improvement (training)",
};

function pairs(runs: RunResult[], get: (r: RunResult) => Num): PairPoint[] {
  const seeds = Array.from(new Set(runs.map((r) => r.seed))).sort((a, b) => a - b);
  return seeds.map((s) => ({
    seed: s,
    O: (() => { const r = runs.find((x) => x.arm === "O" && x.seed === s); return r ? get(r) : null; })(),
    U: (() => { const r = runs.find((x) => x.arm === "U" && x.seed === s); return r ? get(r) : null; })(),
  }));
}

function PairTable({ points, digits = 3 }: { points: PairPoint[]; digits?: number }) {
  return (
    <table className="table">
      <thead><tr><th>Seed</th><th className="num">Arm O</th><th className="num">Arm U</th></tr></thead>
      <tbody>{points.map((p) => (
        <tr key={p.seed}><td className="tabular">{p.seed}</td><td className="num"><Value v={p.O} digits={digits} /></td><td className="num"><Value v={p.U} digits={digits} /></td></tr>
      ))}</tbody>
    </table>
  );
}

export function Seeds({ themeKey }: { themeKey: string }) {
  const { report: rf, analysis: af } = useCore();
  const report = rf.report;
  const th = report.plan.thresholds;
  const runs = useMemo(() => primaryRuns(report).filter((r) => r.run_status === "COMPLETED"), [report]);
  const p2 = useMemo(() => pairs(runs, (r) => r.P2.cell_wmse_clamped_improvement_vs_zero), [runs]);
  const p3 = useMemo(() => pairs(runs, (r) => r.P3.final_val_cell_spearman_indep), [runs]);
  const grho = useMemo(() => pairs(runs, (r) => r.grid.final_val_spearman_indep), [runs]);
  const gimp = useMemo(() => pairs(runs, (r) => r.grid.wmse_improvement_vs_zero), [runs]);
  const p2Band = useMemo<[number, number]>(() => [th.p2_borderline_low, th.p2_borderline_high], [th]);
  const p2Rules = useMemo(() => [{ x: th.p2_borderline_low, text: `FAIL ≤ ${th.p2_borderline_low}`, anchor: "right" as const },
                                 { x: th.p2_borderline_high, text: `PASS ≥ ${th.p2_borderline_high}`, anchor: "left" as const }], [th]);
  const p3Rules = useMemo(() => [{ x: th.p3_min_cell_spearman, text: `P3 ≥ ${th.p3_min_cell_spearman}` }], [th]);
  const ex = af.exploratory as { label: string; metrics: Record<string, ExploratoryMetric> };
  const retests = retestRuns(report);
  const tr = report.analysis as unknown as { test_retest?: Record<string, Record<string, Num | boolean>> };
  const byName = useMemo(() => new Map(report.results.map((r) => [r.name, r])), [report]);

  return (
    <>
      <PageHead eyebrow="Results" title="Seed-wise paired results" scope="TRAINING + VALIDATION"
        lede={<>Each seed fixes the initial weights, sample order and dropout stream for both arms, so the two runs of a seed form a
          pair. These charts show every pair; the confirmatory test on the paired categories is on the Statistics page.</>} />

      <div className="grid-2">
        <Evidence kind="criterion" title="P2 · cell WMSE improvement over the zero predictor">
          <Dumbbell points={p2} label="P2 improvement" xTitle="P2 improvement (training, last steps)" band={p2Band} rules={p2Rules}
            themeKey={themeKey} table={<PairTable points={p2} />} />
        </Evidence>
        <Evidence kind="criterion" title="P3 · final pooled validation cell Spearman ρ">
          <Dumbbell points={p3} label="Validation cell ρ" xTitle="pooled validation cell ρ (independent)" rules={p3Rules}
            themeKey={themeKey} table={<PairTable points={p3} />} />
        </Evidence>
        <Evidence kind="observed" title="Grid head · WMSE improvement over the zero predictor">
          <Dumbbell points={gimp} label="Grid WMSE improvement" xTitle="grid WMSE improvement (training)" themeKey={themeKey}
            table={<PairTable points={gimp} />} />
        </Evidence>
        <Evidence kind="observed" title="Grid head · final pooled validation Spearman ρ">
          <Dumbbell points={grho} label="Validation grid ρ" xTitle="pooled validation grid ρ (independent)" themeKey={themeKey}
            table={<PairTable points={grho} />} />
        </Evidence>
      </div>
      <p className="chart-caption" style={{ marginTop: 8 }}>"undefined" marks a value that is not defined in the record (a correlation of a constant prediction); it is not drawn and is not zero.</p>

      <div className="section">
        <Evidence kind="exploratory" title="Per-seed differences (U − O)">
          <p className="secondary">{ex.label}</p>
          <div className="table-scroll">
            <table className="table" aria-label="Exploratory per-seed differences">
              <thead>
                <tr><th>Metric</th>{(Object.values(ex.metrics)[0]?.per_seed ?? []).map((p) => <th key={p.seed} className="num tabular">{p.seed}</th>)}
                  <th className="num">U &gt; O</th><th className="num">O &gt; U</th><th className="num">defined pairs</th></tr>
              </thead>
              <tbody>{Object.entries(ex.metrics).map(([k, m]) => (
                <tr key={k} data-testid={`expl-${k}`}>
                  <td>{METRIC_LABEL[k] ?? k}</td>
                  {m.per_seed.map((p) => <td key={p.seed} className="num"><Value v={p.U_minus_O} digits={3} /></td>)}
                  <td className="num">{m.n_U_greater}</td><td className="num">{m.n_O_greater}</td><td className="num">{m.n_defined_pairs}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="chart-caption">Counts are descriptive only. No test is applied to them, and they do not change the pre-registered result.</p>
        </Evidence>
      </div>

      {retests.length > 0 && (
        <div className="section">
          <Card title="Test–retest control (excluded from all inference)" meta="the same seed and arm, run twice">
            <div className="table-scroll">
              <table className="table" aria-label="Test-retest">
                <thead><tr><th>Arm</th><th>Run</th><th>Category</th><th className="num">P2 improvement</th><th className="num">Final cell ρ</th><th className="num">Final grid ρ</th></tr></thead>
                <tbody>{retests.flatMap((re) => {
                  const base = byName.get(re.name.replace("_RETEST", ""));
                  return [base, re].filter((x): x is RunResult => !!x).map((r) => (
                    <tr key={r.name}>
                      <td><ArmName arm={r.arm} short /></td><td className="mono">{r.name}</td><td>{categoryLabel(r.category)}</td>
                      <td className="num"><Value v={r.P2.cell_wmse_clamped_improvement_vs_zero} sig={4} /></td>
                      <td className="num"><Value v={r.P3.final_val_cell_spearman_indep} digits={4} /></td>
                      <td className="num"><Value v={r.grid.final_val_spearman_indep} digits={4} /></td>
                    </tr>
                  ));
                })}</tbody>
              </table>
            </div>
            {tr.test_retest && (
              <p className="chart-caption">
                {Object.entries(tr.test_retest).map(([arm, d]) => (
                  <span key={arm} style={{ marginRight: 16 }}>
                    Arm {arm}: same category {d["same_category"] ? "yes" : "no"}; |Δ P2| <Value v={d["abs_diff_cell_wmse_improvement"] as Num} sig={3} />;
                    |Δ cell ρ| <Value v={d["abs_diff_final_cell_spearman"] as Num} sig={3} />; |Δ grid ρ| <Value v={d["abs_diff_final_grid_spearman"] as Num} sig={3} />.
                  </span>
                ))}
              </p>
            )}
          </Card>
        </div>
      )}
    </>
  );
}
