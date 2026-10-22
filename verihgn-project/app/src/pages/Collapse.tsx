import { useEffect, useMemo, useState } from "react";
import { InkSeriesChart, type InkSeries } from "../components/charts";
import { Card, Evidence, Loading, PageHead, Value } from "../components/ui";
import { useBundle, useCore } from "../data/BundleContext";
import { forLogAxis, lastRow, num, type Table } from "../data/runs";
import type { C07ReportFile, C07Run, C07RunIndexEntry, Num } from "../data/schema";
import { useJson } from "../data/useJson";

/* Notebook 07 is a SEPARATE, earlier experiment (official model in every arm; arms vary the training set and
   precision). Nothing here is pooled with HEADPATH, and the arm colours O / U are never used on this page. */

/** A6 is a defect of the summary itself (a field it does not export), so it has no field to attach to in the
    bundle's caveat list; its wording follows the forensic report named in field_caveats._source. */
export const A6_NOTE = "A6: the summary exports only each head's validation zero fraction, not the spread of its predictions; "
  + "the spread (val_cell_pred_std, val_grid_pred_std) is read here from each run's history.csv.";

/** Column headings that name a field's definition (thresholds from the notebook-07 code, not results). */
const COL = { either: "Steps either head ≥ 99.9 % clamped", minCell: "Min cell raw ≤ 0", maxGrid: "Max grid raw ≤ 0" };

type Metric = "cell_raw_le0" | "grid_raw_le0" | "grad_norm";
const METRICS: { key: Metric; label: string; axis: string }[] = [
  { key: "cell_raw_le0", label: "Cell head: raw outputs ≤ 0", axis: "fraction ≤ 0" },
  { key: "grid_raw_le0", label: "Grid head: raw outputs ≤ 0", axis: "fraction ≤ 0" },
  { key: "grad_norm", label: "Gradient norm (log)", axis: "gradient norm" },
];

const group = (r: { arm: string; precision: string }) => `${r.arm}__${r.precision}`;
const groupLabel = (g: string) => g.replace("__", " · ");

function useC07Tables(index: Record<string, C07RunIndexEntry> | null) {
  const { client, core } = useBundle();
  const [s, setS] = useState<{ tables: Record<string, { steps?: Table; history?: Table }>; error: string | null; loading: boolean }>(
    { tables: {}, error: null, loading: true });
  useEffect(() => {
    if (!index) return;
    let alive = true;
    (async () => {
      const tables: Record<string, { steps?: Table; history?: Table }> = {};
      await Promise.all(Object.values(index).flatMap((r) => (["steps", "history"] as const).map(async (k) => {
        const rel = `context07/runs/${r.name}/${k}.json`;
        if (!r[k] || !core?.manifest.files[rel]) return;
        (tables[r.name] ??= {})[k] = await client.json<Table>(rel);
      })));
      return tables;
    })().then((tables) => { if (alive) setS({ tables, error: null, loading: false }); })
      .catch((e: Error) => { if (alive) setS({ tables: {}, error: e.message, loading: false }); });
    return () => { alive = false; };
  }, [client, core, index]);
  return s;
}

function Ref({ code }: { code: string }) {
  return <a className="caveat-ref" href={`#caveat-${code}`} title={`Caveat ${code}`}>{code}</a>;
}

export function Collapse({ themeKey }: { themeKey: string }) {
  const { report: hp } = useCore();
  const rep = useJson<C07ReportFile>("context07/report.json");
  const idx = useJson<Record<string, C07RunIndexEntry>>("context07/runs/index.json");
  const tabs = useC07Tables(idx.data);
  const [metric, setMetric] = useState<Metric>("cell_raw_le0");

  const d = rep.data;
  const results = useMemo(() => d?.report.results ?? [], [d]);
  const groups = useMemo(() => Array.from(new Set(results.map(group))), [results]);
  const undefKind = useMemo(() => {
    const m = new Map<string, string>();
    for (const u of d?.undefined_paths ?? []) {
      const mm = /^\/results\/(\d+)\/(.+)$/.exec(u.path);
      if (mm) m.set(`${results[Number(mm[1])]?.name}/${mm[2]}`, u.kind);
    }
    return m;
  }, [d, results]);
  const caveats = useMemo(() => {
    const out: { field: string; code: string; text: string }[] = [];
    for (const [field, text] of Object.entries(d?.field_caveats ?? {})) {
      if (field.startsWith("_")) continue;
      const m = /^(A\d+):\s*/.exec(text);
      out.push({ field, code: m ? m[1]! : "note", text: m ? text.slice(m[0].length) : text });
    }
    return out;
  }, [d]);
  const codeOf = (field: string) => caveats.find((c) => c.field === field)?.code;

  const derived = useMemo(() => {
    if (!results.length) return null;
    const deadCell = results.filter((r) => r.final_val_pred_zero_frac.cell === 1);
    const notCollapsed = results.filter((r) => !r.collapsed);
    const collapsed = results.filter((r) => r.collapsed);
    const minTrain = Math.min(...results.map((r) => r.n_train));
    const early = collapsed.filter((r) => r.collapse_step !== null && r.collapse_step <= minTrain);
    const replica = (p: string) => results.filter((r) => r.arm === "REPLICA" && r.precision === p);
    const precisionSeeds = Array.from(new Set(replica("fp16").map((r) => r.seed))).filter((s) => replica("fp32").some((r) => r.seed === s)).sort();
    const nonFiniteGrad = results.filter((r) => r.max_grad_norm === null && undefKind.has(`${r.name}/max_grad_norm`));
    return { deadCell, notCollapsed, deadCellNotCollapsed: notCollapsed.filter((r) => deadCell.includes(r)), collapsed, minTrain, early,
             replica, precisionSeeds, nonFiniteGrad };
  }, [results, undefKind]);

  const head = (
    <PageHead eyebrow="Context" title="Collapse investigation (notebook 07)" scope="NOTEBOOK 07 · SEPARATE EXPERIMENT"
      lede={<>The earlier experiment that found the training collapse which HEADPATH was designed to diagnose. It is shown for context only:
        it is a different experiment, and none of its values enter HEADPATH's results or statistics.</>} />
  );
  if (rep.missing || idx.missing) return <>{head}<Card title="Notebook-07 records are not in this bundle"><p>This export was made without the notebook-07 context, so this page has nothing to show.</p></Card></>;
  if (rep.loading || idx.loading) return <Loading what="the notebook-07 records" />;
  if (rep.error || idx.error || !d || !derived) return <div className="blocking-banner" role="alert">Notebook-07 records could not be verified: {rep.error ?? idx.error ?? "empty report"}</div>;

  const plan = d.report.plan;
  const seeds = Array.from(new Set(results.map((r) => r.seed))).sort((a, b) => a - b);
  const byArmNote = (d.report.by_arm as { _note?: string })["_note"];
  const hist = (r: C07Run) => lastRow(tabs.tables[r.name]?.history);
  const maxSteps = Math.max(...results.map((r) => r.steps_logged));
  const m = METRICS.find((x) => x.key === metric)!;

  return (
    <>
      {head}
      <Evidence kind="context" title="How notebook 07 differs from HEADPATH (notebook 08)">
        <dl className="kv" data-testid="c07-separation">
          <dt>Model</dt><dd>the official model, both head clamps active, in every run (no Arm U deviation)</dd>
          <dt>Arms</dt><dd>{groups.map(groupLabel).join(", ")}: they vary the training set and the precision, not the cell-head clamp. The labels O and U do not apply.</dd>
          <dt>Runs</dt><dd>{results.length} runs; seeds {seeds.join(", ")}; up to {maxSteps} optimizer steps (comparison at the common budget of {plan.common_steps} steps)</dd>
          <dt>Validation</dt><dd>{byArmNote ?? "not recorded"}</dd>
          <dt>Report</dt><dd className="mono">{d.report_file}</dd>
        </dl>
        <p className="secondary" style={{ marginTop: 10 }}>The badges in the top bar describe HEADPATH. Values on this page are never combined with HEADPATH's,
          and HEADPATH's statistical test does not use them.</p>
      </Evidence>

      <div className="section">
        <Card title="Per-run summary" meta={<>as recorded in the notebook-07 report; superscripts link to the caveats below</>}>
          <div className="table-scroll">
            <table className="table" aria-label="Notebook-07 runs" data-testid="c07-table">
              <thead><tr>
                <th>Group</th><th className="num">Seed</th><th className="num">Training graphs</th><th className="num">Steps</th>
                <th>Collapsed<Ref code={codeOf("collapsed") ?? "A1"} /></th><th className="num">Collapse step</th><th className="num">Zero-gradient steps</th>
                <th className="num">Cell val. zero fraction</th><th className="num">Grid val. zero fraction</th>
                <th className="num">Cell val. pred. std<Ref code="A6" /></th><th className="num">Grid val. pred. std<Ref code="A6" /></th>
                <th className="num">{COL.either}<Ref code={codeOf("steps_both_heads_ge_0999") ?? "A2"} /></th>
                <th className="num">{COL.minCell}<Ref code={codeOf("min_cell_raw_le0") ?? "A3"} /></th>
                <th className="num">{COL.maxGrid}<Ref code={codeOf("max_grid_raw_le0") ?? "A3"} /></th>
                <th className="num">Max grad norm<Ref code={codeOf("max_grad_norm") ?? "A4"} /></th>
                <th className="num">Final val. avg ρ<Ref code={codeOf("final_val_avg_spearman") ?? "A5"} /></th>
              </tr></thead>
              <tbody>{groups.map((g) => results.filter((r) => group(r) === g).map((r, i) => {
                const k = undefKind.get(`${r.name}/max_grad_norm`);
                return (
                  <tr key={r.name} data-testid={`c07-${r.name}`} className={i === 0 ? "group-start" : undefined}>
                    <td style={{ whiteSpace: "nowrap" }}>{i === 0 ? groupLabel(g) : ""}</td><td className="num">{r.seed}</td><td className="num">{r.n_train}</td><td className="num">{r.steps_logged}</td>
                    <td>{r.collapsed ? "yes" : "no"}</td><td className="num">{r.collapse_step ?? "—"}</td><td className="num">{r.zero_grad_steps}</td>
                    <td className="num"><Value v={r.final_val_pred_zero_frac.cell} digits={3} /></td><td className="num"><Value v={r.final_val_pred_zero_frac.grid} digits={3} /></td>
                    <td className="num"><Value v={hist(r)["val_cell_pred_std"] as Num ?? null} sig={3} why="not in history.csv" /></td>
                    <td className="num"><Value v={hist(r)["val_grid_pred_std"] as Num ?? null} sig={3} why="not in history.csv" /></td>
                    <td className="num"><Value v={r.steps_both_heads_ge_0999} digits={0} /></td>
                    <td className="num"><Value v={r.min_cell_raw_le0} digits={3} /></td><td className="num"><Value v={r.max_grid_raw_le0} digits={3} /></td>
                    <td className="num">{k ? <span className="undef" data-undefined="true" title="non-finite in the source report (normalised to null by the export)">non-finite ({k})</span>
                      : <Value v={r.max_grad_norm} sig={4} />}</td>
                    <td className="num"><Value v={r.final_val_avg_spearman} digits={3} /></td>
                  </tr>
                );
              }))}</tbody>
            </table>
          </div>
          <div className="section" style={{ marginTop: 16 }}>
            <h4 className="sm-title">Caveats on these fields</h4>
            <ul className="caveat-list" data-testid="c07-caveats">
              {caveats.map((c) => (
                <li key={c.field} id={`caveat-${c.code}`}><span className="class-badge">{c.code}</span><span><code className="mono">{c.field}</code>: {c.text}</span></li>
              ))}
              <li id="caveat-A6"><span className="class-badge">A6</span><span>{A6_NOTE.replace(/^A6:\s*/, "")}</span></li>
            </ul>
            <p className="chart-caption">Source of the caveats: {d.field_caveats["_source"] ?? "not recorded"}. The report's values are shown unchanged; the caveats say how to read them.</p>
          </div>
        </Card>
      </div>

      <div className="section grid-2">
        <Evidence kind="observed" title="Cell-head state at the end of each run" label="Observed · re-derived from the notebook-07 records">
          <p data-testid="c07-deadcell">{derived.deadCell.length} of {results.length} runs end with every cell-head validation prediction exactly zero.
            {" "}{derived.deadCellNotCollapsed.length} of the {derived.notCollapsed.length} runs that the detector labels "not collapsed" are among them, because the
            detector only registers the death of both heads (A1).</p>
        </Evidence>
        <Evidence kind="observed" title="When the both-head collapses began" label="Observed · re-derived from the notebook-07 records">
          <p data-testid="c07-early">{derived.early.length} of {derived.collapsed.length} both-head collapses began by step {derived.minTrain}. Up to step k a run has used only
            its first k training graphs, and the smallest training set has {derived.minTrain} graphs, so these collapses happened before the arms' training sets could differ in size.</p>
        </Evidence>
        <Evidence kind="observed" title="Precision (REPLICA, FP16 vs FP32)" label="Observed · re-derived from the notebook-07 records">
          <div className="table-scroll"><table className="table" aria-label="Precision comparison">
            <thead><tr><th>Seed</th><th>FP16</th><th>FP32</th><th className="num">FP16 scaler-skipped steps</th></tr></thead>
            <tbody>{derived.precisionSeeds.map((s) => {
              const a = derived.replica("fp16").find((r) => r.seed === s)!, b = derived.replica("fp32").find((r) => r.seed === s)!;
              const f = (r: C07Run) => (r.collapsed ? `collapsed at step ${r.collapse_step ?? "?"}` : "not collapsed");
              return <tr key={s}><td>{s}</td><td>{f(a)}</td><td>{f(b)}</td><td className="num"><Value v={a.scaler_skipped_steps} digits={0} /></td></tr>;
            })}</tbody>
          </table></div>
        </Evidence>
        <Evidence kind="observed" title="The non-finite gradient norm" label="Observed · re-derived from the notebook-07 records">
          {derived.nonFiniteGrad.length === 0 ? <p>No run records a non-finite maximum gradient norm.</p> : (
            <ul className="bullets" data-testid="c07-nonfinite">
              {derived.nonFiniteGrad.map((r) => (
                <li key={r.name}><span className="mono">{r.name}</span>: max_grad_norm is {undefKind.get(`${r.name}/max_grad_norm`)} at step {r.max_grad_step ?? "?"};
                  the run records {r.scaler_skipped_steps ?? "an undefined number of"} GradScaler-skipped step(s). The summary's maximum includes skipped steps (A4).</li>
              ))}
            </ul>
          )}
        </Evidence>
      </div>

      <div className="section">
        <Card title="Collapse timelines" meta="training steps · one panel per group · ink only (these groups are not arms O / U)">
          <div className="metric-select">
            <div className="seg" role="group" aria-label="Timeline metric">
              {METRICS.map((x) => <button key={x.key} aria-pressed={metric === x.key} onClick={() => setMetric(x.key)}>{x.label}</button>)}
            </div>
          </div>
          {tabs.loading ? <Loading what="the notebook-07 step logs" /> : tabs.error ? <div className="blocking-banner" role="alert">{tabs.error}</div> : (
            <div className="small-multiples">
              {groups.map((g) => {
                const runs = results.filter((r) => group(r) === g);
                let zeros = 0, missing = 0;
                const series: InkSeries[] = runs.map((r, i) => {
                  const t = tabs.tables[r.name]?.steps;
                  if (!t) missing++;
                  const raw = num(t, metric);
                  let y: (number | null)[] = raw;
                  if (metric === "grad_norm") { const l = forLogAxis(raw); y = l.y; zeros += l.zeros; }
                  return { name: `seed ${r.seed}`, x: num(t, "global_step").map((v) => v ?? NaN), y, dash: i };
                });
                const vlines = runs.map((r, i) => (r.collapsed && r.collapse_step !== null ? { x: r.collapse_step, dash: i } : null))
                  .filter((v): v is { x: number; dash: number } => v !== null);
                return (
                  <figure key={g} style={{ margin: 0 }} data-testid={`c07-sm-${g}`}>
                    <p className="sm-title">{groupLabel(g)}</p>
                    <p className="sm-sub">{runs[0]?.n_train} training graphs · {runs.filter((r) => r.collapsed).length} of {runs.length} collapsed
                      {metric === "grad_norm" ? ` · ${zeros} zero-gradient steps not drawn (log axis)` : ""}{missing ? ` · ${missing} step logs missing` : ""}</p>
                    <InkSeriesChart series={series} label={`${groupLabel(g)}: ${m.label}`} xTitle="global step" yTitle={m.axis}
                      logY={metric === "grad_norm"} yRange={metric === "grad_norm" ? undefined : [0, 1.02]} xRange={[0, maxSteps]}
                      vlines={vlines} themeKey={`${themeKey}-${metric}`} height={210} />
                  </figure>
                );
              })}
            </div>
          )}
          <p className="chart-caption">Each line is one run, from its steps.csv. Vertical lines mark the recorded collapse step (start of the final zero-gradient run) in the
            matching dash pattern. The raw-output fractions at or below zero are measured on that step's training graph in training mode.</p>
        </Card>
      </div>

      <div className="section grid-2">
        <Card title="How HEADPATH refers to notebook 07">
          <p>HEADPATH's historical replication check compares Arm O's head states at step {hp.report.plan.replication_step} with the notebook-07 states recorded for
            seeds {Object.keys((hp.report.plan as unknown as { replication_reference?: Record<string, unknown> }).replication_reference ?? {}).join(", ") || "none"}.
            Result: <strong>{hp.report.analysis.historical_replication}</strong>. It checks that the earlier failure reappears; it is not part of the
            pre-registered test. Details are on the <a href="#/statistics">Statistics</a> page.</p>
        </Card>
        <Evidence kind="limitation" title="Reading notebook 07">
          <ul className="bullets">
            <li>{seeds.length} seeds per group: the counts above describe these runs, not a rate.</li>
            <li>The validation sets differ between groups, so validation values compare only within the same set (see above).</li>
            <li>Training-set size and diversity cannot be separated from early collapse: see the second finding above.</li>
            {caveats.length === 0 && <li>This bundle carries no field caveats for the report.</li>}
          </ul>
        </Evidence>
      </div>
    </>
  );
}
