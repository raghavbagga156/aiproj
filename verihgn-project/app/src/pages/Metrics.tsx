import { useMemo, useState } from "react";
import { ArmName, Card, Evidence, Loading, PageHead, Status, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { lastRow, useRunTables } from "../data/runs";
import type { Num, RunIndexEntry } from "../data/schema";

const n = (row: Record<string, unknown>, k: string): Num => (typeof row[k] === "number" ? (row[k] as number) : null);

const VAL_COLS = [
  { k: "spearman_indep", h: "ρ Spearman (indep.)" }, { k: "spearman", h: "ρ Spearman (official)" },
  { k: "pearson_indep", h: "r Pearson (indep.)" }, { k: "kendall", h: "τ Kendall (official)" },
  { k: "mae", h: "MAE" }, { k: "rmse", h: "RMSE" }, { k: "pred_std", h: "pred. std" }, { k: "pred_zero_frac", h: "zero fraction" },
] as const;
const FEATURES = ["cell_density", "pin_density", "net_density", "mean_cell_area_um2", "mean_net_hpwl_um"] as const;
const FEATURE_LABEL: Record<string, string> = {
  cell_density: "cell density", pin_density: "pin density", net_density: "net density",
  mean_cell_area_um2: "mean cell area", mean_net_hpwl_um: "mean net HPWL",
};

interface FreshCheck { max_abs_diff_vs_fresh_official?: Num; max_abs_diff_self_repeat?: Num; equivalent?: boolean; bit_identical?: boolean; atol?: number; rtol?: number }

export function Metrics() {
  const { report: rf } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const { index, tables, loading, error } = useRunTables(["evals", "history"]);
  const [head, setHead] = useState<"cell" | "grid">("cell");
  const runs = useMemo(() => (index ? Object.values(index).sort((a, b) => Number(a.retest) - Number(b.retest) || a.seed - b.seed || a.arm.localeCompare(b.arm)) : []), [index]);
  const baseline = (report as unknown as { cell_feature_baseline?: Record<string, number | string | null> }).cell_feature_baseline ?? {};
  const designs = useMemo(() => Array.from(new Set(plan.val.map((s) => s.split("-")[1]!))).sort(), [plan.val]);
  const resultOf = (r: RunIndexEntry) => report.results.find((x) => x.name === r.name);

  if (loading) return <Loading what="run metrics" />;
  if (error) return <div className="blocking-banner" role="alert">Run tables could not be verified: {error}</div>;

  const label = (r: RunIndexEntry) => (
    <><ArmName arm={r.arm} short /> <span className="tabular muted">{r.seed}</span>{r.retest && <span className="chip" style={{ marginLeft: 6, height: 20 }}>retest · excluded</span>}</>
  );

  return (
    <>
      <PageHead eyebrow="Results" title="Cell- and grid-level metrics" scope="TRAINING + VALIDATION"
        lede={<>Final-step metrics of every run on the diagnostic validation set ({plan.val.length} graphs), as computed in the experiment, with
          the training-set metrics and the run validity checks. Retest runs are listed and marked; they are not part of any comparison.</>} />

      <Card title={`Validation metrics at step ${plan.steps_per_run}`} meta="pooled over all validation graphs">
        <div className="table-tools">
          <div className="seg" role="group" aria-label="Head">
            <button aria-pressed={head === "cell"} onClick={() => setHead("cell")}>Cell head</button>
            <button aria-pressed={head === "grid"} onClick={() => setHead("grid")}>Grid head</button>
          </div>
        </div>
        <div className="table-scroll">
          <table className="table" aria-label={`${head} validation metrics`}>
            <thead><tr><th>Run</th><th>Activity</th>{VAL_COLS.map((c) => <th key={c.k} className="num">{c.h}</th>)}</tr></thead>
            <tbody>{runs.map((r) => {
              const f = lastRow(tables[r.name]?.evals);
              return (
                <tr key={r.name} data-testid={`metrics-${r.name}`}>
                  <td style={{ whiteSpace: "nowrap" }}>{label(r)}</td>
                  <td>{String(f[`val_${head}_activity`] ?? "—")}</td>
                  {VAL_COLS.map((c) => <td key={c.k} className="num"><Value v={n(f, `val_${head}_${c.k}`)} digits={4} /></td>)}
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <p className="chart-caption">"indep." is the experiment's independent implementation (undefined for a constant prediction); "official" is the official
          metric code, which returns zero in that case. Cell metrics are over all validation cells, not the cell subsample stored for the prediction maps.</p>
      </Card>

      <div className="section grid-2">
        <Card title="Tile-projection reference (cell head)" meta="descriptive">
          <p className="secondary" style={{ fontSize: 13 }}>Spearman ρ of the grid prediction at each cell's tile against the cell label: how well the
            grid head alone ranks cells. Reference only.</p>
          <table className="table" aria-label="Tile projection">
            <thead><tr><th>Run</th><th className="num">pooled ρ (indep.)</th><th className="num">official</th></tr></thead>
            <tbody>{runs.filter((r) => !r.retest).map((r) => {
              const f = lastRow(tables[r.name]?.evals);
              return <tr key={r.name}><td>{label(r)}</td><td className="num"><Value v={n(f, "tileproj_cell_spearman_indep")} digits={4} /></td><td className="num"><Value v={n(f, "tileproj_cell_spearman")} digits={4} /></td></tr>;
            })}</tbody>
          </table>
        </Card>
        <Card title="Training-set metrics, final epoch" meta="training graphs · not validation">
          <table className="table" aria-label="Training metrics">
            <thead><tr><th>Run</th><th className="num">cell ρ (clamped)</th><th className="num">cell ρ (raw)</th><th className="num">grid ρ</th></tr></thead>
            <tbody>{runs.filter((r) => !r.retest).map((r) => {
              const h = lastRow(tables[r.name]?.history);
              return <tr key={r.name}><td>{label(r)}</td><td className="num"><Value v={n(h, "train_cell_spearman")} digits={4} /></td>
                <td className="num">{r.arm === "U" ? <Value v={n(h, "train_cell_raw_spearman")} digits={4} /> : <span className="muted">n/a</span>}</td>
                <td className="num"><Value v={n(h, "train_grid_spearman")} digits={4} /></td></tr>;
            })}</tbody>
          </table>
          <p className="chart-caption">Official metric code on the training graphs of the epoch. Arm U's raw-output metric is listed separately and is not comparable with Arm O.</p>
        </Card>
      </div>

      <div className="section">
        <Card title="Run validity" meta="checked by the experiment and re-derived by the export">
          <div className="table-scroll">
            <table className="table" aria-label="Run validity">
              <thead><tr><th>Run</th><th>Valid</th><th className="num">optimizer steps</th><th className="num">max |Δ| vs fresh official model</th><th className="num">self-repeat |Δ|</th><th>Within tolerance</th></tr></thead>
              <tbody>{runs.map((r) => {
                const res = resultOf(r) as unknown as { valid?: boolean; optimizer_step_calls?: number; fresh_official_check?: FreshCheck } | undefined;
                const fc = res?.fresh_official_check ?? {};
                return (
                  <tr key={r.name}>
                    <td>{label(r)}</td>
                    <td>{res?.valid ? <Status tone="good">valid</Status> : <Status tone="crit">invalid</Status>}</td>
                    <td className="num">{res?.optimizer_step_calls ?? "—"}</td>
                    <td className="num"><Value v={fc.max_abs_diff_vs_fresh_official ?? null} sig={3} /></td>
                    <td className="num"><Value v={fc.max_abs_diff_self_repeat ?? null} sig={3} /></td>
                    <td>{fc.equivalent ? "yes" : "no"}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
          <p className="chart-caption">The final weights of every run were loaded into a freshly built official model; its predictions must match within
            atol = {plan.thresholds.fresh_official_atol}, rtol = {plan.thresholds.fresh_official_rtol}. The self-repeat difference is GPU non-determinism.</p>
        </Card>
      </div>

      <div className="section">
        <Evidence kind="observed" title="Non-learned baseline: what the input features alone rank" label="Observed · descriptive baseline">
          <p className="secondary" style={{ fontSize: 13 }}>Spearman ρ between each grid input feature at a cell's tile and the cell label, on the validation set.
            Per design: the mean of the per-graph values (unaffected by per-graph standardization). Pooled: all graphs together.</p>
          <div className="table-scroll">
            <table className="table" aria-label="Feature baseline">
              <thead><tr><th>Feature</th><th className="num">pooled</th>{designs.map((d) => <th key={d} className="num">{d.replace("mgc_", "")}</th>)}</tr></thead>
              <tbody>{FEATURES.map((f) => (
                <tr key={f}>
                  <td>{FEATURE_LABEL[f]}</td>
                  <td className="num"><Value v={asNum(baseline[`pooled_${f}`])} digits={3} /></td>
                  {designs.map((d) => <td key={d} className="num"><Value v={asNum(baseline[`design_${d}_${f}_mean_of_samples`])} digits={3} /></td>)}
                </tr>
              ))}</tbody>
            </table>
          </div>
          {typeof baseline["_note"] === "string" && <details style={{ marginTop: 10 }}><summary className="secondary" style={{ cursor: "pointer", fontSize: 13 }}>Standardization note (verbatim)</summary><p className="secondary" style={{ fontSize: 13, marginTop: 8 }}>{baseline["_note"] as string}</p></details>}
        </Evidence>
      </div>
    </>
  );
}

function asNum(v: unknown): Num {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
