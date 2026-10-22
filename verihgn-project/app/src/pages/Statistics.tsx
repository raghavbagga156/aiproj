import { LimitationsList } from "../components/Limitations";
import { ArmName, Card, Evidence, PageHead, Status } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { categoryLabel, experimentStatusTone, primaryTestSentence } from "../data/derive";
import { fmtSig, isDefined } from "../data/values";

interface Replication { expected: Record<string, string>; observed: Record<string, string> | null; match: boolean }

export function Statistics() {
  const { report: rf, analysis: af } = useCore();
  const report = rf.report;
  const a = report.analysis;
  const t = a.primary_test;
  const st = report.plan.statistical_test;
  const re = af.primary_test_rederived;
  const minP = re.minimum_p_achievable_with_these_pairs;
  const rederivedMatches = re.n_valid_pairs === t.n_valid_pairs && re.U_only === t.U_only && re.O_only === t.O_only
    && Math.abs(re.p_two_sided_exact - t.p_two_sided_exact) < 1e-12 && re.significant === t.significant;
  const repl = (a as unknown as { replication?: Record<string, Replication> }).replication ?? {};
  const note = af.threshold_note;
  const plan = report.plan;

  return (
    <>
      <PageHead eyebrow="Evidence" title="Statistics and limitations" scope="PRE-REGISTERED ANALYSIS"
        lede="The single confirmatory test of the experiment, how it was computed, what it can and cannot establish, and every limitation recorded with the data." />

      <Evidence kind="conclusion" title="Pre-registered test">
        <dl className="kv">
          <dt>Test</dt><dd>{st["name"]}</dd>
          <dt>Endpoint</dt><dd>{st["endpoint"]}</dd>
          <dt>Pairs</dt><dd>{st["pairs"]}</dd>
          <dt>Ties and exclusions</dt><dd>{st["ties"]}</dd>
          <dt>Significance rule</dt><dd>{st["significant_if"]}</dd>
        </dl>
        <div className="facts">
          <div className="fact"><div className="fact-value tabular" data-testid="stat-discordant">{t.n_discordant} / {t.n_valid_pairs}</div><div className="fact-label">discordant / valid pairs</div><div className="fact-note">U only {t.U_only} · O only {t.O_only} · direction {t.direction}</div></div>
          <div className="fact"><div className="fact-value tabular" data-testid="stat-p">p = {fmtSig(t.p_two_sided_exact, 4) ?? "undefined"}</div><div className="fact-label">exact two-sided</div><div className="fact-note">α = {t.alpha}</div></div>
          <div className="fact"><div className="fact-value">{t.significant ? "Significant" : "Not significant"}</div><div className="fact-label">at the pre-registered α</div>
            <div className="fact-note">{isDefined(minP) ? `smallest attainable p with ${t.n_valid_pairs} pairs: ${fmtSig(minP, 4)}` : ""}</div></div>
        </div>
        <p style={{ marginTop: 16 }}>{primaryTestSentence(a)}</p>
        <p className="secondary">The p-value is twice the upper tail of a binomial distribution with success probability one half, over the discordant pairs only;
          concordant pairs carry no information about the difference between the arms.</p>
        <p className="secondary" style={{ marginBottom: 0 }}>
          Re-derived by the export from the stored run categories: {rederivedMatches ? <Status tone="good">matches the recorded test</Status> : <Status tone="crit">differs from the recorded test</Status>}
        </p>
      </Evidence>

      <div className="section">
        <Card title="Pairs entering the test">
          <div className="table-scroll">
            <table className="table" aria-label="Test pairs">
              <thead><tr><th>Seed</th><th><ArmName arm="O" short /></th><th><ArmName arm="U" short /></th><th>Pair</th></tr></thead>
              <tbody>{t.pairs.map((p) => (
                <tr key={p.seed} data-testid={`pair-${p.seed}`}>
                  <td className="tabular">{p.seed}</td><td>{categoryLabel(p.O)}</td><td>{categoryLabel(p.U)}</td>
                  <td>{p.discordant ? <strong>discordant</strong> : <span className="muted">concordant</span>}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="chart-caption">{t.excluded.length === 0 ? "No pair was excluded." : `Excluded: ${t.excluded.map((e) => `seed ${e.seed} (${e.reason})`).join("; ")}.`} Retest runs never enter the test.</p>
        </Card>
      </div>

      <div className="section grid-2">
        <Card title="H-sig: active head without useful signal" meta="secondary, descriptive, no test">
          <table className="table" aria-label="H-sig">
            <thead><tr><th>Arm</th><th className="num">runs meeting H-sig</th><th className="num">valid runs</th></tr></thead>
            <tbody>{(["O", "U"] as const).map((arm) => (
              <tr key={arm}><td><ArmName arm={arm} short /></td><td className="num">{a.hsig[arm]?.n_seeds_meeting_hsig ?? 0}</td><td className="num">{a.hsig[arm]?.n_valid_runs ?? 0}</td></tr>
            ))}</tbody>
          </table>
        </Card>
        <Card title="Historical replication check" meta={<Status tone={experimentStatusTone(a.historical_replication)}>{a.historical_replication}</Status>}>
          <p className="secondary" style={{ fontSize: 13 }}>Arm O's head states at step {plan.replication_step} compared with the earlier notebook-07 runs
            (dead vs not dead only). A failure would be flagged for investigation; it would not invalidate the experiment.</p>
          <table className="table" aria-label="Replication">
            <thead><tr><th>Seed</th><th>Expected (cell / grid)</th><th>Observed (cell / grid)</th><th>Match</th></tr></thead>
            <tbody>{Object.entries(repl).map(([seed, r]) => (
              <tr key={seed}><td className="tabular">{seed}</td><td>{r.expected["cell"]} / {r.expected["grid"]}</td>
                <td>{r.observed ? `${r.observed["cell"]} / ${r.observed["grid"]}` : "not evaluated"}</td><td>{r.match ? "yes" : "no"}</td></tr>
            ))}</tbody>
          </table>
        </Card>
      </div>

      <div className="section">
        <Evidence kind="limitation" title="Limitations">
          <LimitationsList report={report} minP={minP} />
        </Evidence>
      </div>

      <div className="section">
        <Card title="Threshold record (documentation inconsistency)" meta="flagged by the export; the record is not modified">
          <dl className="kv">
            <dt>Recorded field</dt><dd className="mono">{note.recorded_field} = {isDefined(note.recorded_value) ? note.recorded_value : "undefined"}</dd>
            <dt>Used by the code</dt><dd>{note.recorded_value_used_by_code ? "yes" : "no"}</dd>
            <dt>Implemented rule</dt><dd>{note.implemented_rule}</dd>
            <dt>Runs affected</dt><dd>{note.runs_in_disputed_interval.length === 0 ? "none: no run's improvement lies between the two readings" : note.runs_in_disputed_interval.map((r) => r.name).join(", ")}</dd>
          </dl>
        </Card>
      </div>
    </>
  );
}
