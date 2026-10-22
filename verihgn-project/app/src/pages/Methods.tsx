import { useMemo } from "react";
import { LimitationsList } from "../components/Limitations";
import { ArmName, Card, Evidence, Loading, Mono, PageHead, Status } from "../components/ui";
import { useCore } from "../data/BundleContext";
import type { MapsIndexEntry } from "../data/schema";
import { useJson, useTexts } from "../data/useJson";
import { LABEL_FORMULA } from "../data/maps";
import { fmtSig, isDefined } from "../data/values";
import { parseYaml, type Yaml } from "../data/yaml";

/* Every value on this page is read from the bundle: the plan and report (headpath/report.json), each run's recorded
   config.yaml (headpath/runs/<run>/sources/), and the map index. The static text only defines terms. */

export type Fidelity = "P" | "C" | "D" | "A" | "U";
const FIDELITY: Record<Fidelity, string> = {
  P: "directly supported by the paper",
  C: "verified in the official implementation",
  D: "derived from the dataset documentation or from inspecting the dataset",
  A: "engineering adaptation, needed because the authors' preprocessing code or details are unavailable",
  U: "unresolved assumption, with no supporting source",
};

export interface Deviation { text: string; codes: Fidelity[]; kind: "fact" | "adaptation" | "assumption" | "not-claimed" | "setting" | "diagnostic" }

/** Classifies one recorded deviation line by its fidelity prefix (docs/RESEARCH_TO_CODE_MAPPING.md). */
export function classifyDeviation(text: string): Deviation {
  if (/^Not claimed:/i.test(text)) return { text, codes: [], kind: "not-claimed" };
  if (/^DIAGNOSTIC DEVIATION/.test(text)) return { text, codes: [], kind: "diagnostic" };
  const m = /^([PCDAU](?:\/[PCDAU])*)(?:-[\w-]+)?(?:\s*\([^)]*\))?:\s/.exec(text);
  if (!m) return { text, codes: [], kind: "setting" };
  const codes = m[1]!.split("/") as Fidelity[];
  const kind = codes.includes("U") ? "assumption" : codes.includes("A") ? "adaptation" : "fact";
  return { text, codes, kind };
}

type Flat = Record<string, string>;
function flatten(y: Yaml, prefix = "", out: Flat = {}): Flat {
  if (y && typeof y === "object" && !Array.isArray(y)) {
    for (const [k, v] of Object.entries(y)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix] = JSON.stringify(y);
  }
  return out;
}
const show = (v: string | undefined) => (v === undefined ? "absent" : v.replace(/^"(.*)"$/, "$1"));

const CRITERIA: { key: string; meaning: string }[] = [
  { key: "activity_std_min", meaning: "a head counts as active if its validation prediction spread exceeds this" },
  { key: "activity_zero_max", meaning: "… and its fraction of exactly-zero predictions stays below this" },
  { key: "p1_last_steps", meaning: "P1 window: last training steps examined" },
  { key: "p1_max_dead_frac", meaning: "P1: largest allowed fraction of dead cell-head steps in that window" },
  { key: "p2_last_steps", meaning: "P2 window: last training steps of the WMSE comparison with the zero predictor" },
  { key: "p2_borderline_low", meaning: "P2 implemented rule: FAIL at or below this improvement" },
  { key: "p2_borderline_high", meaning: "P2 implemented rule: PASS at or above this improvement" },
  { key: "p2_min_improvement", meaning: "recorded but not used by the code (see the flag below)" },
  { key: "p3_min_cell_spearman", meaning: "P3: final pooled validation cell Spearman ρ must reach this" },
  { key: "hsig_max_abs_wmse_change", meaning: "H-sig: WMSE change below this counts as no useful signal" },
  { key: "hsig_max_cell_spearman", meaning: "H-sig: cell ρ below this counts as no useful signal" },
  { key: "sign_test_alpha", meaning: "significance level of the exact McNemar test" },
];

export function Methods() {
  const { report: rf, analysis: af, manifest } = useCore();
  const report = rf.report;
  const plan = report.plan;
  const mi = useJson<Record<string, MapsIndexEntry>>("headpath/maps/index.json");
  const cfgPaths = useMemo(() => plan.runs.map((r) => `headpath/runs/${r.name}/sources/config.yaml`), [plan]);
  const cfgTexts = useTexts(cfgPaths);

  const cfg = useMemo(() => {
    if (!cfgTexts.data) return null;
    const parsed: Record<string, { y: Record<string, Yaml>; flat: Flat; dev: string[] } | null> = {};
    const errors: string[] = [];
    for (const r of plan.runs) {
      const t = cfgTexts.data[`headpath/runs/${r.name}/sources/config.yaml`];
      if (t == null) { parsed[r.name] = null; errors.push(`${r.name}: config.yaml not in the bundle`); continue; }
      try {
        const y = parseYaml(t) as Record<string, Yaml>;
        const dev = Array.isArray(y["deviations"]) ? (y["deviations"] as Yaml[]).map(String) : [];
        const { deviations: _d, ...rest } = y;
        parsed[r.name] = { y, flat: flatten(rest), dev };
      } catch (e) {
        parsed[r.name] = null; errors.push(`${r.name}: ${(e as Error).message}`);
      }
    }
    const O = plan.runs.filter((r) => r.arm === "O" && parsed[r.name]), U = plan.runs.filter((r) => r.arm === "U" && parsed[r.name]);
    const ref = O[0] ? parsed[O[0].name]! : null;
    const refU = U[0] ? parsed[U[0].name]! : null;
    const same = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
    const devSameO = !!ref && O.every((r) => same(parsed[r.name]!.dev, ref.dev));
    const extraU = refU && ref ? refU.dev.filter((x) => !ref.dev.includes(x)) : [];
    const devSameU = !!refU && U.every((r) => same(parsed[r.name]!.dev, refU.dev)) && !!ref && ref.dev.every((x) => refU.dev.includes(x));
    // configuration differences inside each seed pair (Arm O vs Arm U)
    const pairDiffs = new Map<string, number>();
    let nPairs = 0;
    for (const o of O) {
      const u = U.find((x) => x.seed === o.seed && x.retest === o.retest);
      if (!u) continue;
      nPairs++;
      const a = parsed[o.name]!.flat, b = parsed[u.name]!.flat;
      for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) if (a[k] !== b[k]) pairDiffs.set(k, (pairDiffs.get(k) ?? 0) + 1);
    }
    return { parsed, errors, ref, refU, devSameO, devSameU, extraU, nO: O.length, nU: U.length, pairDiffs, nPairs, refO: O[0], refUrun: U[0] };
  }, [cfgTexts.data, plan.runs]);

  const designsOf = (ids: string[]) => Array.from(new Set(ids.map((s) => s.split("-")[1]!))).sort();
  const holdoutN = Object.values(plan.holdout).reduce((n, l) => n + l.length, 0);
  const seeds = Array.from(new Set(plan.runs.filter((r) => !r.retest).map((r) => r.seed))).sort((a, b) => a - b);
  const retests = plan.runs.filter((r) => r.retest);
  const th = plan.thresholds as unknown as Record<string, number | undefined>;
  const note = af.threshold_note;

  if (cfgTexts.error) return <div className="blocking-banner" role="alert">Run configurations could not be verified: {cfgTexts.error}</div>;
  if (!cfg) return <Loading what="the recorded run configurations" />;

  const ref = cfg.ref;
  const v = (k: string) => show(ref?.flat[k]);
  const devs = (ref?.dev ?? []).map(classifyDeviation);
  const byKind = (k: Deviation["kind"]) => devs.filter((d) => d.kind === k);
  const DevList = ({ items, testId }: { items: Deviation[]; testId: string }) => (
    <ul className="caveat-list" data-testid={testId}>
      {items.map((d) => <li key={d.text}><span className="class-badge">{d.codes.join("/") || "—"}</span><span>{d.text}</span></li>)}
    </ul>
  );

  return (
    <>
      <PageHead eyebrow="Evidence" title="Methodology and caveats" scope="METHODOLOGY"
        lede={<>How HEADPATH was set up, read from the configuration and plan recorded with each run. Recorded facts, engineering adaptations and unresolved
          assumptions are marked separately. This is an adapted reproduction of VeriHGN, used for a diagnostic experiment; it is not an exact reproduction of the paper.</>} />

      {cfg.errors.length > 0 && (
        <div className="blocking-banner" role="alert" data-testid="config-errors">Some run configurations could not be read, so they are not used on this page:
          <ul className="bullets mono" style={{ fontSize: 12 }}>{cfg.errors.map((e) => <li key={e}>{e}</li>)}</ul></div>
      )}

      <Evidence kind="fact" title="What this experiment is, and is not">
        <dl className="kv">
          <dt>Experiment</dt><dd>{manifest.experiment}</dd>
          <dt>Official code</dt><dd>VeriHGN reference implementation, commit <Mono>{manifest.official_commit ?? "not recorded"}</Mono>, kept unmodified</dd>
          <dt>Arm U</dt><dd>{report.analysis.deviation_notice}</dd>
          <dt>Not claimed</dt><dd data-testid="not-claimed">{byKind("not-claimed").map((d) => d.text).join(" · ") || "not recorded in the configuration"}</dd>
        </dl>
      </Evidence>

      <div className="section">
        <Card title="Fidelity classes" meta="used in every run's recorded deviation list">
          <ul className="caveat-list">
            {(Object.keys(FIDELITY) as Fidelity[]).map((k) => <li key={k}><span className="class-badge">{k}</span><span>{FIDELITY[k]}</span></li>)}
          </ul>
          <p className="chart-caption">Each run writes its deviations into its config.yaml. Below they are quoted verbatim and grouped by class.
            {ref && <> {cfg.devSameO ? <Status tone="good">identical</Status> : <Status tone="warn">differs</Status>} across the {cfg.nO} Arm O configurations;
              {" "}{cfg.devSameU ? <Status tone="good">identical</Status> : <Status tone="warn">differs</Status>} across the {cfg.nU} Arm U configurations, which add
              {" "}{cfg.extraU.length} line{cfg.extraU.length === 1 ? "" : "s"}.</>}</p>
        </Card>
      </div>

      <div className="section grid-2">
        <Evidence kind="fact" title="Choices with a recorded source (P, C, D)">
          <DevList items={[...byKind("fact"), ...byKind("setting")]} testId="dev-fact" />
          <p className="chart-caption">Lines marked "deviates from" follow one source where the paper and the official code disagree. A line without a class is a run setting.</p>
        </Evidence>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Evidence kind="adaptation" title="Engineering adaptations (A)">
            <DevList items={byKind("adaptation")} testId="dev-adaptation" />
          </Evidence>
          <Evidence kind="assumption" title="Unresolved assumptions (U)">
            <DevList items={byKind("assumption")} testId="dev-assumption" />
          </Evidence>
        </div>
      </div>

      <div className="section">
        <Card title="The two arms" meta="recorded configurations compared key by key">
          <div className="grid-2">
            <div>
              <p><ArmName arm="O" /> runs the official model unchanged.</p>
              <p style={{ marginTop: 8 }}><ArmName arm="U" /> adds one recorded deviation:</p>
              {cfg.extraU.map((x) => <p key={x} className="verbatim" data-testid="arm-u-deviation">{x}</p>)}
            </div>
            <div>
              <div className="table-scroll">
                <table className="table" aria-label="Configuration differences between the arms" data-testid="arm-diff">
                  <thead><tr><th>Setting that differs</th><th>Arm O</th><th>Arm U</th><th className="num">pairs</th></tr></thead>
                  <tbody>{[...cfg.pairDiffs.entries()].map(([k, n]) => (
                    <tr key={k}><td className="mono">{k}</td><td>{show(ref?.flat[k])}</td><td>{show(cfg.refU?.flat[k])}</td><td className="num">{n} / {cfg.nPairs}</td></tr>
                  ))}</tbody>
                </table>
              </div>
              <p className="chart-caption">Every other recorded setting is identical within each of the {cfg.nPairs} seed pairs (deviation lists compared separately above).
                Values shown are from {cfg.refO?.name ?? "—"} and {cfg.refUrun?.name ?? "—"}.</p>
            </div>
          </div>
        </Card>
      </div>

      <div className="section grid-2">
        <Evidence kind="fact" title="Dataset and labels">
          <dl className="kv" data-testid="m-dataset">
            <dt>Dataset scope</dt><dd className="mono">{v("experiment.dataset_scope")}</dd>
            <dt>Training designs</dt><dd>{designsOf(plan.all_train).map((d) => d.replace("mgc_", "")).join(", ")}</dd>
            <dt>Validation design</dt><dd>{designsOf(plan.val_main).map((d) => d.replace("mgc_", "")).join(", ")}</dd>
            <dt>Label</dt><dd>CircuitNet global-routing overflow maps H and V, combined as <Mono>{LABEL_FORMULA}</Mono> (map type: adaptation, see A/D above)</dd>
            <dt>Normalisation</dt><dd><Mono>{v("experiment.label_norm")}</Mono>, c_max = {fmtSig(plan.c_max, 6)} from the training split</dd>
          </dl>
        </Evidence>
        <Evidence kind="fact" title="Features and graph construction">
          <dl className="kv" data-testid="m-features">
            <dt>Grid features</dt><dd className="mono">{v("experiment.grid_features")}</dd>
            <dt>Input sizes</dt><dd>cell {v("model.cell_feat_dim")}, net {v("model.net_feat_dim")}, pin {v("model.pin_feat_dim")}, geometric edge {v("model.geom_feat_dim")}</dd>
            <dt>Model</dt><dd>official VeriHGN, hidden size {v("model.hidden_dim")}, {v("model.num_layers")} layers, dropout {v("model.dropout")}</dd>
            <dt>Graphs</dt><dd>built by the official graph builder from inputs computed by this project's preprocessing; how those inputs were obtained is listed under
              adaptations (A) and assumptions (U)</dd>
          </dl>
        </Evidence>
        <Evidence kind="fact" title="Splits">
          <dl className="kv" data-testid="m-splits">
            <dt>Split</dt><dd className="mono">{v("experiment.split")}</dd>
            <dt>Training graphs</dt><dd>{plan.train.length} used for training, of {plan.all_train.length} in the training split; {holdoutN} held out for validation</dd>
            <dt>Validation graphs</dt><dd>{plan.val.length}: {plan.val_main.length} of the official validation design and {plan.val.length - plan.val_main.length} held-out graphs of training designs</dd>
            <dt>Test split</dt><dd>never evaluated in HEADPATH: there are no test-set results</dd>
          </dl>
        </Evidence>
        <Evidence kind="fact" title="Training">
          <dl className="kv" data-testid="m-training">
            <dt>Runs</dt><dd>{plan.runs.length}: seeds {seeds.join(", ")} for both arms, plus {retests.length} retest run{retests.length === 1 ? "" : "s"} ({retests.map((r) => r.name).join(", ")})</dd>
            <dt>Length</dt><dd>{plan.epochs} epochs, {plan.steps_per_run} optimizer steps, one graph per step</dd>
            <dt>Optimizer</dt><dd>learning rate {v("training.learning_rate")}, weight decay {v("training.weight_decay")}, gradient clip {v("training.grad_clip")}; learning-rate step schedule with step_size {v("training.step_size")} and gamma {v("training.gamma")}</dd>
            <dt>Loss</dt><dd>official weighted MSE (alpha {v("training.wmse_alpha")}) with variance term (lambda {v("training.lambda_var")})</dd>
            <dt>Precision</dt><dd className="mono">{v("training.precision")} (TF32 {v("training.allow_tf32")})</dd>
          </dl>
        </Evidence>
      </div>

      <div className="section">
        <Card title="Evaluation" meta="diagnostic validation set · no test set">
          <p>Validation at steps {plan.eval_steps.join(", ")}, always with the official clamped forward pass. Spearman ρ is pooled over the validation graphs;
            "official" uses the official metric code, which reports the correlation of a constant prediction as zero, while "independent" leaves it undefined.</p>
          {mi.data && (
            <div className="table-scroll" style={{ marginTop: 12 }}>
              <table className="table" aria-label="Validation graphs" data-testid="m-valset">
                <thead><tr><th>Graph</th><th>Design</th><th>Role</th><th className="num">Grid (gcells)</th><th className="num">Cells saved for maps</th></tr></thead>
                <tbody>{plan.val.filter((s) => mi.data![s]).map((s) => {
                  const e = mi.data![s]!;
                  return <tr key={s}><td className="mono">{s}</td><td>{e.design}</td><td>{e.role}</td><td className="num">{e.shape[0]} × {e.shape[1]}</td><td className="num">{e.n_cells_subsample.toLocaleString("en-US")}</td></tr>;
                })}</tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <div className="section grid-2">
        <Evidence kind="criterion" title="Pre-registered thresholds">
          <div className="table-scroll">
            <table className="table" aria-label="Thresholds" data-testid="m-thresholds">
              <thead><tr><th>Field</th><th className="num">Value</th><th>Meaning</th></tr></thead>
              <tbody>{CRITERIA.filter((c) => c.key in th).map((c) => (
                <tr key={c.key}><td className="mono">{c.key}</td><td className="num">{isDefined(th[c.key] ?? null) ? th[c.key] : "undefined"}</td><td>{c.meaning}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </Evidence>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card title="Threshold record (documentation inconsistency)" meta="flagged by the export; the record is not modified">
            <p><Mono>{note.recorded_field}</Mono> = {isDefined(note.recorded_value) ? note.recorded_value : "undefined"} is recorded but
              {note.recorded_value_used_by_code ? " used" : " not used"} by the code. Implemented: {note.implemented_rule}. Runs affected: {note.runs_in_disputed_interval.length}.</p>
          </Card>
          <Card title="Statistical test and retests">
            <dl className="kv">
              <dt>Test</dt><dd>{plan.statistical_test["name"]}</dd>
              <dt>Pairs</dt><dd>{plan.statistical_test["pairs"]}</dd>
              <dt>Endpoint</dt><dd>{plan.statistical_test["endpoint"]}</dd>
            </dl>
          </Card>
        </div>
      </div>

      <div className="section" id="limitations">
        <Evidence kind="limitation" title="Limitations">
          <LimitationsList report={report} minP={af.primary_test_rederived.minimum_p_achievable_with_these_pairs} testId="m-limitations" />
        </Evidence>
      </div>
    </>
  );
}
