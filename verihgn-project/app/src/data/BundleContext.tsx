import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { BundleClient, type VerificationState } from "./client";
import type { AnalysisFile, IntegrityReport, Manifest, NormalizationIndex, ReportFile, Session } from "./schema";

/** Files every page needs; loaded (and verified) once at start-up. */
export interface CoreBundle {
  manifest: Manifest;
  integrity: IntegrityReport;
  normalization: NormalizationIndex | null;
  report: ReportFile;
  analysis: AnalysisFile;
}

export interface BundleState {
  client: BundleClient;
  session: Session | null;
  core: CoreBundle | null;
  error: string | null;
  verification: VerificationState;
  loading: boolean;
}

const Ctx = createContext<BundleState | null>(null);

export function BundleProvider({ client, children }: { client: BundleClient; children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [core, setCore] = useState<CoreBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [verification, setVerification] = useState<VerificationState>(client.snapshot());

  useEffect(() => client.subscribe(setVerification), [client]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const s = await client.session();
        if (alive) setSession(s);
        const manifest = await client.manifest();
        const [integrity, report, analysis] = await Promise.all([
          client.json<IntegrityReport>("integrity_report.json"),
          client.json<ReportFile>("headpath/report.json"),
          client.json<AnalysisFile>("headpath/analysis.json"),
        ]);
        const normalization = manifest.files["normalization_index.json"]
          ? await client.json<NormalizationIndex>("normalization_index.json") : null;
        if (alive) setCore({ manifest, integrity, normalization, report, analysis });
      } catch (e) {
        if (alive) setError((e as Error).message);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [client]);

  const value = useMemo(() => ({ client, session, core, error, verification, loading }),
    [client, session, core, error, verification, loading]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBundle(): BundleState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useBundle outside BundleProvider");
  return v;
}

/** Core bundle for pages rendered inside the integrity gate (core is guaranteed there). */
export function useCore(): CoreBundle {
  const { core } = useBundle();
  if (!core) throw new Error("core bundle not loaded");
  return core;
}
