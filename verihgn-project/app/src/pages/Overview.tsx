import { useMemo } from "react";
import { CriterionStrip } from "../components/charts";
import { ArmMark, ArmName, Card, EndpointGlyph, Evidence, Status, Value } from "../components/ui";
import { useBundle, useCore } from "../data/BundleContext";
import {
  categoryLabel, CATEGORY_DEFINITION, countCriterion, countEndpoint, experimentStatusTone, integrityState,
  pairRows, primaryRuns, primaryTestSentence, retestRuns, validationScope,
} from "../data/derive";
import type { RunResult } from "../data/schema";
import { fmtSig, isDefined } from "../data/values";

function frac(x: { met: number; total: number }) { return `${x.met} of ${x.total}`; }

// stable accessors (charts re-plot only when their inputs change)
const P2_VALUE = (r: RunResult) => r.P2.cell_wmse_clamped_improvement_vs_zero;
const P3_VALUE = (r: RunResult) => r.P3.final_val_cell_spearman_indep;

export function Overview({ themeKey }: { themeKey: string }) {
  const { report: rf, analysis: af, manifest, integrity } = useCore();
  const { session, verification } = useBundle();
  const report = rf.report;
  const a = report.analysis;
  const plan = report.plan;
  const th = plan.thresholds;
  const t = a.primary_test;
  const scope = validationScope(report);
  const integ = integrityState(manifest, integrity, session, verification);
  const primary = primaryRuns(report);
  const retests = retestRuns(report);
  const rows = pairRows(a);
  const nRuns = plan.runs.length;
  const completed = report.run_status_counts["COMPLETED"] ?? 0;
  const endO = countEndpoint(report, "O"), endU = countEndpoint(report, "U");
  const p2O = countCriterion(report, "O", "P2"), p2U = countCriterion(report, "U", "P2");
  const p3O = countCriterion(report, "O", "P3"), p3U = countCriterion(report, "U", "P3");
  const healthy = (arm: "O" | "U") => ({
    met: primary.filter((r) => r.arm === arm && r.P1.final_cell_activity === "HEALTHY").length,
    total: primary.filter((r) => r.arm === arm).length,
  });
  const minP = af.primary_test_rederived.minimum_p_achievable_with_these_pairs;
  const undefRho = primary.filter((r) => !isDefined(r.P3.final_val_cell_spearman_indep));
  const warns = integrity.checks.filter((c) => c.status === "WARN");
  const seedsUsed = rows.map((r) => r.seed);
  const p2Band = useMemo<[number, number]>(() => [th.p2_borderline_low, th.p2_borderline_high], [th]);
  const p2Rules = useMemo(() => [{ x: th.p2_borderline_low, text: `FAIL ≤ ${th.p2_borderline_low}`, anchor: "right" as const },
                                 { x: th.p2_borderline_high, text: `PASS ≥ ${th.p2_borderline_high}`, anchor: "left" as const }], [th]);
  const p3Rules = useMemo(() => [{ x: th.p3_min_cell_spearman, text: `P3 ≥ ${th.p3_min_cell_spearman}` }], [th]);

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Executive summary</div>
        <h1 className="page-title">Does the train-time clamp keep the cell head from learning?</h1>
        <p className="page-lede">
          HEADPATH compares the official VeriHGN model (Arm O) with a diagnostic variant (Arm U) whose cell-head clamp
          is removed during training only. Both arms are evaluated with the official clamped forward pass, on a
          diagnostic validation set of {scope.nGraphs} graphs from {scope.nDesigns} designs, across {t.n_valid_pairs + t.excluded.length} paired seeds.
        </p>
      </div>

      <div className="strip" role="list" aria-label="Experiment status">
        <div className="strip-item" role="listitem">
          <div className="strip-label">Experiment status</div>
          <div className="strip-value"><Status tone={experimentStatusTone(a.experiment_status)}>{a.experiment_status}</Status></div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Runs completed</div>
          <div className="strip-value tabular">{completed} of {nRuns}<span className="muted" style={{ fontWeight: 400, fontSize: 12.5 }}>{retests.length} retest{retests.length === 1 ? "" : "s"} included</span></div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Notebook-07 replication check</div>
          <div className="strip-value"><Status tone={experimentStatusTone(a.historical_replication)}>{a.historical_replication}</Status></div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Bundle integrity</div>
          <div className="strip-value"><a href="#/provenance" style={{ textDecoration: "none" }}><Status tone={integ.tone}>{integ.label}</Status></a></div>
        </div>
      </div>

      <div className="section">
        <Evidence kind="conclusion" title="Primary result">
          <p className="secondary" style={{ marginTop: 0 }}>
            Endpoint: a run reaches <strong>{categoryLabel("LEARNS_AND_GENERALIZES")}</strong>. {CATEGORY_DEFINITION["LEARNS_AND_GENERALIZES"]}
          </p>
          <div className="matrix-wrap">
            <table className="matrix" aria-label="Primary endpoint by seed and arm">
              <thead>
                <tr><th scope="col" style={{ textAlign: "left", paddingLeft: 0 }}>Seed</th>{rows.map((r) => <th key={r.seed} scope="col" className="tabular">{r.seed}</th>)}</tr>
              </thead>
              <tbody>
                {(["O", "U"] as const).map((arm) => (
                  <tr key={arm}>
                    <th scope="row"><ArmName arm={arm} short /></th>
                    {rows.map((r) => {
                      const cat = arm === "O" ? r.O : r.U;
                      const met = arm === "O" ? r.endpointO : r.endpointU;
                      return (
                        <td key={r.seed} data-testid={`cell-${arm}-${r.seed}`}>
                          <span className="cat"><EndpointGlyph met={met} invalid={cat === "INVALID"} /><span className="cat-text">{categoryLabel(cat)}</span></span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr><td style={{ textAlign: "left", paddingLeft: 0 }}>Pair</td>{rows.map((r) => <td key={r.seed}>{r.discordant ? "discordant" : "concordant"}</td>)}</tr>
              </tfoot>
            </table>
          </div>
          <ul className="matrix-list" aria-label="Primary endpoint by seed (compact)">
            {rows.map((r) => (
              <li key={r.seed}>
                <div className="ml-seed tabular">Seed {r.seed}<span className="muted"> · {r.discordant ? "discordant" : "concordant"}</span></div>
                {(["O", "U"] as const).map((arm) => {
                  const cat = arm === "O" ? r.O : r.U;
                  const met = arm === "O" ? r.endpointO : r.endpointU;
                  return (
                    <div className="ml-arm" key={arm}>
                      <ArmName arm={arm} short />
                      <span className="ml-cat"><EndpointGlyph met={met} invalid={cat === "INVALID"} />{categoryLabel(cat)}</span>
                    </div>
                  );
                })}
              </li>
            ))}
          </ul>
          <div className="legend-row">
            <span><EndpointGlyph met /> meets the primary endpoint</span>
            <span><EndpointGlyph met={false} /> does not</span>
            {t.excluded.length > 0 && <span>Excluded pairs: {t.excluded.map((e) => e.seed).join(", ")}</span>}
          </div>

          <div className="facts">
            <div className="fact">
              <div className="fact-value tabular" data-testid="fact-pairs">{t.n_valid_pairs}</div>
              <div className="fact-label">valid seed pairs</div>
              <div className="fact-note">seeds {seedsUsed.join(", ")}; retests excluded</div>
            </div>
            <div className="fact">
              <div className="fact-value tabular" data-testid="fact-discordant">{t.n_discordant}</div>
              <div className="fact-label">discordant pairs</div>
              <div className="fact-note">U only {t.U_only} · O only {t.O_only}</div>
            </div>
            <div className="fact">
              <div className="fact-value tabular" data-testid="fact-p">p = {fmtSig(t.p_two_sided_exact, 4) ?? "undefined"}</div>
              <div className="fact-label">exact two-sided McNemar test</div>
              <div className="fact-note">
                {t.significant ? "significant" : "not significant"} at α = {t.alpha}
                {isDefined(minP) && <> · smallest attainable p with {t.n_valid_pairs} pairs: {fmtSig(minP, 4)}</>}
              </div>
            </div>
          </div>
          <p style={{ marginTop: 16 }} data-testid="primary-sentence">{primaryTestSentence(a)}</p>
          <p className="secondary" style={{ marginBottom: 0 }}>
            Recorded conclusion, verbatim: <q data-testid="conclusion">{a.conclusion}</q>
          </p>
        </Evidence>
      </div>

      <div className="section">
        <h2 className="section-title">Pre-registered criteria at a glance</h2>
        <p className="section-sub">Primary runs only (seeds {seedsUsed.join(", ")}); the retest runs are excluded. Thresholds are read from the experiment plan.</p>
        <div className="grid-2">
          <Evidence kind="criterion" title="P2 · cell learning against the zero predictor">
            <CriterionStrip runs={primary} themeKey={themeKey} label="P2 improvement"
              value={P2_VALUE}
              xTitle="P2 improvement over the zero predictor"
              band={p2Band} rules={p2Rules} />
            <p className="chart-caption">x: one minus the ratio of summed clamped cell WMSE to summed zero-predictor WMSE over the last {th.p2_last_steps} training steps (clamped predictions in both arms). PASS in {frac(p2U)} Arm-U and {frac(p2O)} Arm-O seeds. Shaded: borderline (inconclusive) interval.</p>
          </Evidence>
          <Evidence kind="criterion" title="P3 · generalization on the validation set">
            <CriterionStrip runs={primary} themeKey={themeKey} label="Pooled validation cell Spearman ρ"
              value={P3_VALUE}
              xTitle="pooled validation cell Spearman ρ"
              rules={p3Rules} />
            <p className="chart-caption">
              x: independent Spearman ρ over all cells of the validation set, final step ({plan.steps_per_run}). Met in {frac(p3U)} Arm-U and {frac(p3O)} Arm-O seeds.{" "}
              {undefRho.length > 0
                ? <>Undefined (constant prediction; not plotted, not zero) for {undefRho.map((r) => `${r.arm}·${r.seed}`).join(", ")}.</>
                : "Defined for every primary run."}
            </p>
          </Evidence>
        </div>
      </div>

      <div className="section grid-2">
        <Evidence kind="observed" title="What the data show">
          <ul className="bullets">
            <li><ArmMark arm="U" /> Arm U reached the primary endpoint in {frac(endU)} seeds; <ArmMark arm="O" /> Arm O in {frac(endO)}.</li>
            <li>Final cell head active (HEALTHY): Arm U {frac(healthy("U"))}, Arm O {frac(healthy("O"))} seeds.</li>
            <li>P2 passed: Arm U {frac(p2U)}, Arm O {frac(p2O)}. P3 met: Arm U {frac(p3U)}, Arm O {frac(p3O)}.</li>
            <li>{t.significant
              ? `The pre-registered test is significant (p = ${fmtSig(t.p_two_sided_exact, 4)}).`
              : `The pre-registered test does not establish a difference between the arms (p = ${fmtSig(t.p_two_sided_exact, 4)}).`}</li>
          </ul>
        </Evidence>
        <Evidence kind="limitation" title="What the data do not show">
          <ul className="bullets">
            <li><strong>No test-set results.</strong> The test split was never evaluated; every performance value here is from the {scope.nGraphs}-graph diagnostic validation set ({scope.nCrossDesign} graphs of the official validation design, {scope.nGraphs - scope.nCrossDesign} held-out graphs of training designs).</li>
            <li><strong>Not a reproduction</strong> of the paper's results: {plan.epochs}-epoch diagnostic runs ({plan.steps_per_run} optimizer steps), adapted labels.</li>
            <li><strong>Arm U is a diagnostic deviation</strong>, not an improved model. Removing the clamp in training changes the loss surface as well as the gradient path.</li>
            <li>With {t.n_valid_pairs} pairs the exact test can only reach p ≤ α if every pair is discordant in the same direction.</li>
          </ul>
        </Evidence>
      </div>

      <div className="section">
        <Card title="Deviation notice (verbatim from the experiment record)">
          <p className="quote" data-testid="deviation">{a.deviation_notice}</p>
        </Card>
        {retests.length > 0 && (
          <Card title="Test–retest runs (excluded from the primary test)" meta="descriptive control only">
            <ul className="list-plain">
              {retests.map((r) => (
                <li key={r.name} style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <ArmName arm={r.arm} short /><span className="mono">{r.name}</span>
                  <span className="secondary">{categoryLabel(r.category)}</span>
                  <span className="muted">P2 improvement <Value v={r.P2.cell_wmse_clamped_improvement_vs_zero} sig={3} /></span>
                </li>
              ))}
            </ul>
          </Card>
        )}
        {warns.length > 0 && (
          <Card title="Integrity warnings" meta={<a href="#/provenance">All checks →</a>}>
            <ul className="list-plain">
              {warns.map((w) => <li key={w.id}><Status tone="warn">WARN</Status> <span className="secondary" style={{ marginLeft: 8 }}>{w.message}</span></li>)}
            </ul>
          </Card>
        )}
      </div>
    </>
  );
}
