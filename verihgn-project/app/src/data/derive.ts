/* Pure derivations for the pages. Every number and sentence shown on a page is built here from bundle data;
   nothing experimental is written into the UI code. Retest runs are excluded from every primary computation
   (they are flagged in the bundle: retest / primary_inference). */
import type { Analysis, AnalysisFile, Check, IntegrityReport, Manifest, PrimaryPair, Report, RunResult, Session } from "./schema";
import type { VerificationState } from "./client";
import { fmtSig, isDefined } from "./values";

export const ENDPOINT = "LEARNS_AND_GENERALIZES";

/** Human labels for the pre-registered run categories (definitions: docs/HEADPATH_PLAN.md section 5). */
export const CATEGORY_LABEL: Record<string, string> = {
  LEARNS_AND_GENERALIZES: "Learns and generalizes",
  LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION: "Learns; generalization not shown",
  ACTIVE_WITHOUT_USEFUL_SIGNAL: "Active, no useful signal",
  CELL_HEAD_FAILURE_PERSISTS: "Cell-head failure persists",
  INCONCLUSIVE: "Inconclusive",
  INVALID: "Invalid run",
};
export const CATEGORY_DEFINITION: Record<string, string> = {
  LEARNS_AND_GENERALIZES: "P1 active cell head, P2 PASS (clamped cell WMSE improvement vs the zero predictor), and P3 (pooled validation cell Spearman at the final step at or above the pre-registered threshold).",
  LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION: "P1 and P2 PASS, but P3 not met.",
  ACTIVE_WITHOUT_USEFUL_SIGNAL: "H-sig: active cell head, clamped WMSE within the pre-registered band of the zero predictor and low pooled cell Spearman.",
  CELL_HEAD_FAILURE_PERSISTS: "P1 not met: the cell head is not active at the end (dead or degenerate), or predicts all-zero on too many of the last steps.",
  INCONCLUSIVE: "Active cell head, but P2 is borderline or failed without meeting the H-sig criteria.",
  INVALID: "A validity check failed; reported, never dropped.",
};

export function categoryLabel(c: string): string {
  return CATEGORY_LABEL[c] ?? c;
}

export const ARM_LABEL: Record<"O" | "U", string> = {
  O: "Arm O · official model",
  U: "Arm U · diagnostic deviation",
};

export function primaryRuns(report: Report): RunResult[] {
  return report.results.filter((r) => !r.retest);
}

export function retestRuns(report: Report): RunResult[] {
  return report.results.filter((r) => r.retest);
}

export interface PairRow extends PrimaryPair { endpointO: boolean; endpointU: boolean }

export function pairRows(analysis: Analysis): PairRow[] {
  return analysis.primary_test.pairs.map((p) => ({ ...p, endpointO: p.O === ENDPOINT, endpointU: p.U === ENDPOINT }));
}

export function countEndpoint(report: Report, arm: "O" | "U"): { met: number; total: number } {
  const runs = primaryRuns(report).filter((r) => r.arm === arm && r.run_status === "COMPLETED");
  return { met: runs.filter((r) => r.category === ENDPOINT).length, total: runs.length };
}

export function countCriterion(report: Report, arm: "O" | "U", key: "P1" | "P2" | "P3"): { met: number; total: number } {
  const runs = primaryRuns(report).filter((r) => r.arm === arm && r.run_status === "COMPLETED");
  const met = runs.filter((r) => (key === "P2" ? r.P2.outcome === "PASS" : r[key].pass)).length;
  return { met, total: runs.length };
}

/** The sentence that states the pre-registered test result, built only from the bundle. */
export function primaryTestSentence(analysis: Analysis): string {
  const t = analysis.primary_test;
  const p = fmtSig(t.p_two_sided_exact, 4) ?? "undefined";
  const sig = t.significant ? "statistically significant" : "not statistically significant";
  return `${t.n_discordant} of ${t.n_valid_pairs} valid seed pairs are discordant on the primary endpoint ` +
    `(U only: ${t.U_only}; O only: ${t.O_only}). Exact two-sided McNemar p = ${p}, ${sig} at α = ${t.alpha}.`;
}

/** Smallest p the exact test can reach with this many valid pairs (all discordant in one direction). */
export function minimumAttainableP(nPairs: number): number | null {
  if (nPairs <= 0) return null;
  return Math.min(1, 2 * Math.pow(0.5, nPairs));
}

export interface StatusItem { label: string; value: string; tone: "good" | "warn" | "crit" | "info" }

export function experimentStatusTone(s: string): StatusItem["tone"] {
  return s === "VALID" || s === "PASSED" ? "good" : s === "INCOMPLETE" || s === "NOT_EVALUATED" ? "warn" : "crit";
}

export interface IntegrityState {
  ok: boolean;
  tone: StatusItem["tone"];
  label: string;
  reasons: string[];
  warnCount: number;
  failCount: number;
}

/** Integrity of what the viewer is looking at: launcher verification + manifest + integrity report + every
    file the browser has re-verified so far. Any failure is blocking. */
export function integrityState(manifest: Manifest | null, integ: IntegrityReport | null, session: Session | null,
                               ver: VerificationState): IntegrityState {
  const reasons: string[] = [];
  if (session && !session.verified) reasons.push(`launcher verification ${session.verification.status}: ${session.verification.errors.slice(0, 3).join("; ") || "see the launcher output"}`);
  if (manifest && manifest.status !== "PASSED") reasons.push(`bundle manifest status ${manifest.status}`);
  if (integ && integ.status !== "PASSED") reasons.push(`integrity report status ${integ.status}`);
  if (integ && integ.fingerprint && integ.fingerprint.unchanged !== true) reasons.push("source files changed during the export");
  for (const f of ver.failures) reasons.push(`${f.file}: ${f.reason}`);
  const failCount = integ?.summary?.FAIL ?? 0;
  if (failCount > 0) reasons.push(`${failCount} integrity checks failed`);
  const warnCount = integ?.summary?.WARN ?? 0;
  const ok = reasons.length === 0;
  return {
    ok, reasons, warnCount, failCount,
    tone: ok ? (warnCount ? "warn" : "good") : "crit",
    label: ok ? (warnCount ? `Verified · ${warnCount} warning${warnCount === 1 ? "" : "s"}` : "Verified") : "Integrity failure",
  };
}

export function checksByStatus(checks: Check[]): Record<string, Check[]> {
  const out: Record<string, Check[]> = { FAIL: [], WARN: [], INFO: [], PASS: [] };
  for (const c of checks) (out[c.status] ??= []).push(c);
  return out;
}

/** Validation scope facts for the global notice, read from the bundle. */
export function validationScope(report: Report): { nGraphs: number; nDesigns: number; nCrossDesign: number } {
  const designs = new Set(report.plan.val.map((s) => s.split("-")[1]));
  return { nGraphs: report.plan.val.length, nDesigns: designs.size, nCrossDesign: report.plan.val_main.length };
}

export function thresholdRule(a: AnalysisFile): { recorded: string | null; implemented: string; affected: number } {
  const n = a.threshold_note;
  return { recorded: isDefined(n.recorded_value) ? String(n.recorded_value) : null, implemented: n.implemented_rule,
           affected: n.runs_in_disputed_interval.length };
}
