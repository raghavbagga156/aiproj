import type { ReactNode } from "react";
import type { Num } from "../data/schema";
import { fmtNum, fmtSig } from "../data/values";

export function Card({ title, meta, children, className = "", id }: {
  title?: ReactNode; meta?: ReactNode; children: ReactNode; className?: string; id?: string;
}) {
  return (
    <section className={`card ${className}`} id={id}>
      {(title || meta) && (
        <div className="card-head">
          {title && <h3 className="card-title">{title}</h3>}
          {meta && <div className="card-meta">{meta}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export type EvidenceKind = "observed" | "criterion" | "conclusion" | "exploratory" | "limitation"
  | "fact" | "adaptation" | "assumption" | "context";
const EVIDENCE_LABEL: Record<EvidenceKind, string> = {
  observed: "Observed",
  criterion: "Pre-registered criterion",
  conclusion: "Pre-registered statistical test",
  exploratory: "Exploratory · not pre-registered · no inference",
  limitation: "Limitation",
  fact: "Fact · from a recorded source",
  adaptation: "Adaptation · engineering choice",
  assumption: "Assumption · no supporting source",
  context: "Separate earlier experiment · notebook 07",
};

/** One visual grammar for the evidence types (docs/DASHBOARD_FRONTEND_PLAN.md section 3.6). */
export function Evidence({ kind, title, children, label }: {
  kind: EvidenceKind; title?: ReactNode; children: ReactNode; label?: string;
}) {
  return (
    <section className={`card evidence evidence--${kind}`} data-evidence={kind}>
      <div className="evidence-label">{label ?? EVIDENCE_LABEL[kind]}</div>
      {title && <h3 className="card-title" style={{ marginBottom: 8 }}>{title}</h3>}
      {children}
    </section>
  );
}

export type Tone = "good" | "warn" | "crit" | "info";

function StatusIcon({ tone }: { tone: Tone }) {
  const common = { width: 12, height: 12, viewBox: "0 0 12 12", "aria-hidden": true } as const;
  if (tone === "good") return <svg {...common}><path d="M2.5 6.2 5 8.6l4.5-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>;
  if (tone === "warn") return <svg {...common}><path d="M6 1.6 11 10.4H1z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" /><path d="M6 5v2.4M6 8.9v.1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>;
  if (tone === "crit") return <svg {...common}><circle cx="6" cy="6" r="4.8" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M4.2 4.2l3.6 3.6M7.8 4.2 4.2 7.8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>;
  return <svg {...common}><circle cx="6" cy="6" r="4.8" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M6 5.4v3M6 3.6v.1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>;
}

/** Status is always icon + text (never colour alone). Reserved for integrity / validity / experiment status. */
export function Status({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return <span className={`status status--${tone}`} title={title}><StatusIcon tone={tone} />{children}</span>;
}

/** A number, or an explicit "undefined" marker. Never renders a missing value as 0. */
export function Value({ v, digits = 3, sig, why = "undefined in the source record (e.g. a correlation of a constant prediction)" }: {
  v: Num | undefined; digits?: number; sig?: number; why?: string;
}) {
  const s = sig ? fmtSig(v ?? null, sig) : fmtNum(v ?? null, digits);
  if (s === null) return <span className="undef" title={why} data-undefined="true">undefined</span>;
  return <span className="tabular">{s}</span>;
}

/** Arm identity mark: colour AND shape (O = circle, U = diamond), so colour is never the only cue. */
export function ArmMark({ arm, size = 10, hollow = false }: { arm: "O" | "U"; size?: number; hollow?: boolean }) {
  const c = `var(--arm-${arm})`;
  const s = size;
  return (
    <svg className="arm-swatch" width={s} height={s} viewBox="0 0 10 10" aria-hidden="true">
      {arm === "O"
        ? <circle cx="5" cy="5" r="4" fill={hollow ? "none" : c} stroke={c} strokeWidth={hollow ? 1.6 : 0} />
        : <path d="M5 0.8 9.2 5 5 9.2 0.8 5z" fill={hollow ? "none" : c} stroke={c} strokeWidth={hollow ? 1.6 : 0} />}
    </svg>
  );
}

export function ArmName({ arm, short = false }: { arm: "O" | "U"; short?: boolean }) {
  return (
    <span className="arm-key">
      <ArmMark arm={arm} />
      <span>{short ? `Arm ${arm}` : arm === "O" ? "Arm O · official" : "Arm U · diagnostic deviation"}</span>
    </span>
  );
}

/** Category glyph: filled = meets the primary endpoint; ring = does not; struck ring = invalid. Ink only. */
export function EndpointGlyph({ met, invalid = false }: { met: boolean; invalid?: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className="cat-glyph">
      {met ? <circle cx="8" cy="8" r="5.5" fill="var(--ink)" />
        : <circle cx="8" cy="8" r="5" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" />}
      {invalid && <path d="M3.5 12.5 12.5 3.5" stroke="var(--status-crit)" strokeWidth="1.5" />}
    </svg>
  );
}

export function Mono({ children, title }: { children: ReactNode; title?: string }) {
  return <code className="mono" title={title}>{children}</code>;
}

export function short(hash: string | undefined | null, n = 12): string {
  return hash ? `${hash.slice(0, n)}…` : "";
}

export type Scope = "TRAINING" | "VALIDATION" | "TRAINING + VALIDATION" | "PRE-REGISTERED ANALYSIS"
  | "NOTEBOOK 07 · SEPARATE EXPERIMENT" | "METHODOLOGY";
const SCOPE_TITLE: Record<Scope, string> = {
  TRAINING: "Values from the training logs (training graphs; clamped predictions in both arms unless stated)",
  VALIDATION: "Values from the diagnostic validation set (not a test set)",
  "TRAINING + VALIDATION": "Training-log and validation-set values; each chart says which",
  "PRE-REGISTERED ANALYSIS": "The pre-registered statistical analysis and its limits",
  "NOTEBOOK 07 · SEPARATE EXPERIMENT": "Records of the earlier notebook-07 collapse experiment; never pooled with HEADPATH (notebook 08)",
  METHODOLOGY: "How the experiment was set up: recorded configuration, plan, deviations and limitations",
};

/** Page header with an explicit data-scope tag (training vs validation), shown on every analysis page. */
export function PageHead({ eyebrow, title, lede, scope }: { eyebrow: string; title: string; lede: ReactNode; scope: Scope }) {
  return (
    <div className="page-head">
      <div className="eyebrow">{eyebrow}</div>
      <h1 className="page-title">{title}</h1>
      <p className="page-lede">{lede}</p>
      <div style={{ marginTop: 12 }}><span className="chip scope-tag" title={SCOPE_TITLE[scope]} data-scope={scope}>Data: {scope.toLowerCase()}</span></div>
    </div>
  );
}

export function SeedPicker({ seeds, value, onChange, allowAll = true }: {
  seeds: number[]; value: number | "all"; onChange: (v: number | "all") => void; allowAll?: boolean;
}) {
  return (
    <div className="seg" role="group" aria-label="Seed">
      {allowAll && <button aria-pressed={value === "all"} onClick={() => onChange("all")}>All seeds</button>}
      {seeds.map((s) => <button key={s} aria-pressed={value === s} onClick={() => onChange(s)} className="tabular">{s}</button>)}
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <div className="loading">Loading and verifying {what}…</div>;
}
