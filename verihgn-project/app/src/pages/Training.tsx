import { useMemo, useState } from "react";
import { SeriesChart, type Series } from "../components/charts";
import { Card, Evidence, Loading, PageHead, SeedPicker } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { forLogAxis, num, rollingImprovement, useRunTables, windowImprovement, type Table } from "../data/runs";
import type { Num, RunIndexEntry } from "../data/schema";
import { fmtSig } from "../data/values";

const PARTS = [
  { col: "grad_norm_cell_only", title: "Cell-only parameters" },
  { col: "grad_norm_grid_only", title: "Grid-only parameters" },
  { col: "grad_norm_shared", title: "Shared parameters" },
] as const;

export function Training({ themeKey }: { themeKey: string }) {
  const { report: rf } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const th = plan.thresholds;
  const { index, tables, loading, error } = useRunTables(["steps"]);
  const primary = useMemo(() => (index ? Object.values(index).filter((r) => r.primary_inference) : []), [index]);
  const seeds = useMemo(() => Array.from(new Set(primary.map((r) => r.seed))).sort((a, b) => a - b), [primary]);
  const [seed, setSeed] = useState<number | "all">("all");
  const stepsPerEpoch = plan.epochs > 0 ? plan.steps_per_run / plan.epochs : null;
  const vlines = useMemo(() => (stepsPerEpoch ? Array.from({ length: plan.epochs - 1 }, (_, i) => ({ x: (i + 1) * stepsPerEpoch, text: `epoch ${i + 1} ends` })) : []), [stepsPerEpoch, plan.epochs]);
  const p2Rules = useMemo(() => [{ x: th.p2_borderline_low, text: `FAIL ≤ ${th.p2_borderline_low}`, side: "below" as const },
                                 { x: th.p2_borderline_high, text: `PASS ≥ ${th.p2_borderline_high}` }], [th]);

  const steps = (r: RunIndexEntry): Table | undefined => tables[r.name]?.steps;
  const selected = (r: RunIndexEntry) => seed === "all" || r.seed === seed;
  const shown = useMemo(() => primary.filter(selected), [primary, seed]);   // eslint-disable-line react-hooks/exhaustive-deps

  const build = (f: (t: Table) => Num[]): Series[] => primary.filter((r) => seed === "all" || r.seed === seed).map((r) => {
    const t = steps(r);
    return { name: `Arm ${r.arm} · seed ${r.seed}`, arm: r.arm, seed: r.seed, x: num(t, "global_step") as number[],
             y: t ? f(t) : [], emphasis: seed !== "all" ? true : undefined };
  });

  const p2Series = useMemo(() => build((t) => rollingImprovement(num(t, "cell_wmse_clamped"), num(t, "cell_wmse_zero"), th.p2_last_steps)),
    [primary, seed, tables, th.p2_last_steps]);   // eslint-disable-line react-hooks/exhaustive-deps
  const cellLe0 = useMemo(() => build((t) => num(t, "cell_raw_le0")), [primary, seed, tables]);   // eslint-disable-line react-hooks/exhaustive-deps
  const gridLe0 = useMemo(() => build((t) => num(t, "grid_raw_le0")), [primary, seed, tables]);   // eslint-disable-line react-hooks/exhaustive-deps
  const grads = useMemo(() => PARTS.map((p) => {
    let zerosO = 0, zerosU = 0;
    const s = build((t) => { const v = forLogAxis(num(t, p.col)); return v.y; });
    for (const r of shown) {
      const z = forLogAxis(num(steps(r), p.col)).zeros;
      if (r.arm === "O") zerosO += z; else zerosU += z;
    }
    return { ...p, series: s, zerosO, zerosU };
  }), [primary, seed, tables]);   // eslint-disable-line react-hooks/exhaustive-deps

  // consistency: the rolling window ending at the last step must reproduce the recorded P2 of each run
  const p2Check = useMemo(() => primary.map((r) => {
    const t = steps(r);
    const res = report.results.find((x) => x.name === r.name);
    const mine = t ? windowImprovement(num(t, "cell_wmse_clamped"), num(t, "cell_wmse_zero"), th.p2_last_steps) : null;
    const rec = res?.P2.cell_wmse_clamped_improvement_vs_zero ?? null;
    return { name: r.name, mine, rec, ok: mine !== null && rec !== null ? Math.abs(mine - rec) <= 1e-9 * Math.max(1, Math.abs(rec)) : mine === rec };
  }), [primary, tables, report, th.p2_last_steps]);   // eslint-disable-line react-hooks/exhaustive-deps
  const p2Mismatch = p2Check.filter((c) => !c.ok);

  if (loading) return <Loading what="training logs" />;
  if (error) return <div className="blocking-banner" role="alert">Training logs could not be verified: {error}</div>;

  return (
    <>
      <PageHead eyebrow="Results" title="Training dynamics" scope="TRAINING"
        lede={<>Per-step training logs of the {primary.length} primary runs ({plan.steps_per_run} optimizer steps each, {plan.epochs} epochs).
          These are training-graph quantities, not validation results. Arm U's raw-output loss is never compared with Arm O's loss.</>} />

      <div className="table-tools">
        <SeedPicker seeds={seeds} value={seed} onChange={setSeed} />
        <span className="muted" style={{ fontSize: 12.5 }}>{seed === "all" ? "All primary runs (retests excluded)" : `Seed ${seed}: Arm O and Arm U from identical initial weights and sample order`}</span>
      </div>

      <Evidence kind="criterion" title="Rolling P2: cell WMSE improvement over the zero predictor">
        <SeriesChart series={p2Series} label="Rolling P2 improvement" xTitle="global step" yTitle={`improvement (${th.p2_last_steps}-step window)`}
          rules={p2Rules} vlines={vlines} themeKey={themeKey} height={280}
          caption={<>Each point is one minus the ratio of summed clamped cell WMSE to summed zero-predictor WMSE over the preceding {th.p2_last_steps} steps; the last point is the
            pre-registered P2 value. {p2Mismatch.length === 0
              ? `Recomputed in the browser for all ${p2Check.length} runs; matches the recorded P2.`
              : `MISMATCH with the recorded P2 for ${p2Mismatch.map((c) => c.name).join(", ")}.`}</>} />
      </Evidence>

      <div className="section grid-2">
        <Card title="Cell head: fraction of raw outputs at or below zero">
          <SeriesChart series={cellLe0} label="Cell raw outputs at or below zero" xTitle="global step" yTitle="fraction ≤ 0" yRange={[-0.02, 1.02]}
            vlines={vlines} themeKey={themeKey}
            caption="Pre-clamp cell-head outputs on the training graph of each step. At 1, every output is clamped to 0 by the official head (in Arm U only at evaluation)." />
        </Card>
        <Card title="Grid head: fraction of raw outputs at or below zero">
          <SeriesChart series={gridLe0} label="Grid raw outputs at or below zero" xTitle="global step" yTitle="fraction ≤ 0" yRange={[-0.02, 1.02]}
            vlines={vlines} themeKey={themeKey} caption="The grid head is official (clamped) in both arms." />
        </Card>
      </div>

      <div className="section">
        <h2 className="section-title">Gradient norms by parameter partition</h2>
        <p className="section-sub">Pre-clip Euclidean norm of the gradient of the parameters that only the cell head, only the grid head, or both heads can reach. Logarithmic axis.</p>
        <div className="grid-3">
          {grads.map((g) => (
            <Card key={g.col} title={g.title}>
              <SeriesChart series={g.series} label={`${g.title} gradient norm`} xTitle="global step" yTitle="gradient norm" logY
                vlines={vlines} themeKey={themeKey} height={230} digits={2}
                caption={<span data-testid={`zeros-${g.col}`}>Steps with an exactly-zero gradient (not drawable on a log axis, shown as gaps): Arm O {g.zerosO}, Arm U {g.zerosU}.</span>} />
            </Card>
          ))}
        </div>
      </div>

      <div className="section">
        <Evidence kind="limitation" title="Reading these curves">
          <ul className="bullets">
            <li>One training graph per step: per-step values vary with the graph, so single steps are noisy.</li>
            <li>Training-set behaviour only; generalization is judged on the validation set (Validation curves page).</li>
            <li>GPU arithmetic is not bit-deterministic; each run's recorded self-repeat difference is in the validity table on the Cell and grid metrics page.</li>
          </ul>
          {seed !== "all" && <p className="chart-caption">Recorded P2 for seed {seed}: Arm O {fmtSig(report.results.find((r) => r.arm === "O" && r.seed === seed && !r.retest)?.P2.cell_wmse_clamped_improvement_vs_zero ?? null, 4) ?? "undefined"}, Arm U {fmtSig(report.results.find((r) => r.arm === "U" && r.seed === seed && !r.retest)?.P2.cell_wmse_clamped_improvement_vs_zero ?? null, 4) ?? "undefined"}.</p>}
        </Evidence>
      </div>
    </>
  );
}
