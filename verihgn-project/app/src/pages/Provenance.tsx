import { useMemo, useState } from "react";
import { DataTable, type Column } from "../components/DataTable";
import { Card, Evidence, Mono, short, Status, type Tone } from "../components/ui";
import { useBundle, useCore } from "../data/BundleContext";
import { checksByStatus, integrityState } from "../data/derive";
import type { Check, CheckStatus, NormalizationRecord } from "../data/schema";
import { fmtBytes, fmtInt } from "../data/values";

const TONE: Record<CheckStatus, Tone> = { PASS: "good", WARN: "warn", INFO: "info", FAIL: "crit" };
const ORDER: CheckStatus[] = ["FAIL", "WARN", "INFO", "PASS"];

export function Provenance() {
  const { manifest, integrity, normalization } = useCore();
  const { session, verification } = useBundle();
  const integ = integrityState(manifest, integrity, session, verification);
  const [filter, setFilter] = useState<CheckStatus | "ALL">("ALL");
  const by = checksByStatus(integrity.checks);
  const pinned = [...(by["FAIL"] ?? []), ...(by["WARN"] ?? [])];
  const checks = useMemo(() => {
    const all = [...integrity.checks].sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status));
    return filter === "ALL" ? all : all.filter((c) => c.status === filter);
  }, [integrity.checks, filter]);

  const checkCols: Column<Check>[] = [
    { key: "s", header: "Status", width: "92px", render: (c) => <Status tone={TONE[c.status]}>{c.status}</Status>, text: (c) => c.status },
    { key: "id", header: "Check", width: "30%", render: (c) => <Mono>{c.id}</Mono>, text: (c) => c.id },
    { key: "m", header: "Result", render: (c) => c.message, text: (c) => c.message },
  ];
  const sources = useMemo(() => Object.entries(manifest.sources).map(([label, s]) => ({ label, ...s })), [manifest.sources]);
  const srcCols: Column<(typeof sources)[number]>[] = [
    { key: "l", header: "Source file (Drive, read-only)", render: (s) => <Mono>{s.label}</Mono>, text: (s) => s.label },
    { key: "n", header: "Size", numeric: true, render: (s) => fmtBytes(s.size), text: () => "" },
    { key: "h", header: "sha256", render: (s) => <Mono title={s.sha256}>{short(s.sha256, 16)}</Mono>, text: (s) => s.sha256 },
  ];
  const files = useMemo(() => Object.entries(manifest.files).map(([path, f]) => ({ path, ...f })), [manifest.files]);
  const fileCols: Column<(typeof files)[number]>[] = [
    { key: "p", header: "Bundle file", render: (f) => <Mono>{f.path}</Mono>, text: (f) => f.path },
    { key: "n", header: "Size", numeric: true, render: (f) => fmtBytes(f.size), text: () => "" },
    { key: "h", header: "sha256", render: (f) => <Mono title={f.sha256}>{short(f.sha256, 16)}</Mono>, text: (f) => f.sha256 },
  ];
  const norm = useMemo(() => normalization
    ? Object.entries(normalization.files).map(([path, r]) => ({ path, ...r }))
        .sort((a, b) => b.n_replaced - a.n_replaced || a.path.localeCompare(b.path)) : [], [normalization]);
  const normCols: Column<{ path: string } & NormalizationRecord>[] = [
    { key: "p", header: "Normalized copy", render: (r) => <Mono>{r.path}</Mono>, text: (r) => r.path },
    { key: "nan", header: "NaN", numeric: true, render: (r) => r.by_kind.nan },
    { key: "inf", header: "+Inf", numeric: true, render: (r) => r.by_kind.inf },
    { key: "ninf", header: "−Inf", numeric: true, render: (r) => r.by_kind["-inf"] },
    { key: "s", header: "Original sha256 (unchanged)", render: (r) => <Mono title={r.source_sha256}>{short(r.source_sha256, 16)}</Mono>, text: (r) => r.source_sha256 },
  ];
  const fp = integrity.fingerprint;
  const enc = manifest.array_encoding;

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Evidence</div>
        <h1 className="page-title">Provenance and integrity</h1>
        <p className="page-lede">
          Where every value on this dashboard comes from, and how it was checked. The bundle was exported read-only from
          the experiment folders, verified before it was served, and every file is verified again in the browser before use.
        </p>
      </div>

      {!integ.ok && (
        <div className="blocking-banner" role="alert">Integrity failure. Results are withheld until the bundle verifies. {integ.reasons.join(" · ")}</div>
      )}

      <div className="strip" role="list" aria-label="Verification summary">
        <div className="strip-item" role="listitem">
          <div className="strip-label">Launcher verification</div>
          <div className="strip-value">
            {session ? <Status tone={session.verified ? "good" : "crit"}>{session.verification.status}</Status>
              : <Status tone="warn">Not run</Status>}
            {session && <span className="muted tabular" style={{ fontWeight: 400, fontSize: 12.5 }}>{fmtInt(session.verification.files_checked)} files · {fmtInt(session.verification.arrays_checked)} arrays</span>}
          </div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Verified in this browser</div>
          <div className="strip-value tabular">
            <Status tone={verification.failures.length ? "crit" : "good"}>{verification.failures.length ? `${verification.failures.length} failed` : "No mismatches"}</Status>
            <span className="muted" style={{ fontWeight: 400, fontSize: 12.5 }}>{fmtInt(verification.verifiedFiles)} files used so far, all re-hashed</span>
          </div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Export checks</div>
          <div className="strip-value tabular" style={{ gap: 6 }}>
            {ORDER.map((s) => (integrity.summary[s] ?? 0) > 0 && <Status key={s} tone={TONE[s]}>{integrity.summary[s]} {s}</Status>)}
          </div>
        </div>
        <div className="strip-item" role="listitem">
          <div className="strip-label">Experiment sources</div>
          <div className="strip-value">
            <Status tone={fp.unchanged ? "good" : "crit"}>{fp.unchanged ? "Unchanged" : "Changed"}</Status>
            <span className="muted tabular" style={{ fontWeight: 400, fontSize: 12.5 }}>{fmtInt(fp.n_files)} files hashed before and after the export</span>
          </div>
        </div>
      </div>

      <div className="section grid-2">
        <Card title="Bundle identity">
          <dl className="kv">
            <dt>Export file</dt><dd>{session ? <Mono>{session.zip_name}</Mono> : <span className="muted">served without the launcher</span>}</dd>
            <dt>Export sha256</dt><dd>{session ? <Mono>{session.zip_sha256}</Mono> : <span className="muted">unknown</span>}</dd>
            <dt>Export size</dt><dd className="tabular">{session ? fmtBytes(session.zip_bytes) : "—"}</dd>
            <dt>Exported (UTC)</dt><dd className="tabular">{manifest.created_utc}</dd>
            <dt>Manifest status</dt><dd><Status tone={manifest.status === "PASSED" ? "good" : "crit"}>{manifest.status}</Status></dd>
            <dt>Exporter / schema</dt><dd className="tabular">{manifest.exporter_version} · schema {manifest.schema_version}</dd>
            <dt>Official code commit</dt><dd>{manifest.official_commit ? <Mono>{manifest.official_commit}</Mono> : <span className="muted">not recorded</span>}</dd>
            <dt>Files in bundle</dt><dd className="tabular">{fmtInt(Object.keys(manifest.files).length)} + manifest</dd>
            {session && <><dt>Opened from</dt><dd><Mono>{session.extract_dir}</Mono></dd></>}
          </dl>
        </Card>
        <Evidence kind="limitation" title="Scope of this record">
          <p>{manifest.integrity_notice}</p>
          <p className="secondary">Checkpoints were re-checked during the export but are not part of the bundle. The bundle never modifies the experiment folders.</p>
        </Evidence>
      </div>

      {pinned.length > 0 && (
        <div className="section">
          <Card title="Warnings and failures" meta={`${pinned.length} pinned`}>
            <ul className="list-plain">
              {pinned.map((c) => (
                <li key={c.id}>
                  <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
                    <Status tone={TONE[c.status]}>{c.status}</Status><Mono>{c.id}</Mono>
                  </div>
                  <p className="secondary" style={{ margin: "6px 0 0" }}>{c.message}</p>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      )}

      <div className="section">
        <Card title="All export checks" meta={`${integrity.checks.length} checks`}>
          <DataTable rows={checks} columns={checkCols} searchable searchLabel="Filter checks" pageSize={20}
            caption="Export integrity checks"
            toolbar={
              <div className="seg" role="group" aria-label="Filter by status">
                {(["ALL", ...ORDER] as const).map((s) => (
                  <button key={s} aria-pressed={filter === s} onClick={() => setFilter(s)}>
                    {s === "ALL" ? "All" : s} <span className="muted tabular">{s === "ALL" ? integrity.checks.length : (by[s]?.length ?? 0)}</span>
                  </button>
                ))}
              </div>
            } />
        </Card>
      </div>

      <div className="section grid-2">
        <Evidence kind="observed" title="Undefined values in the source records" label="JSON normalization">
          {normalization ? (
            <>
              <p>
                <strong className="tabular">{fmtInt(normalization.total_replaced)}</strong> non-finite numbers (NaN, ±Infinity) in{" "}
                {norm.filter((r) => r.n_replaced > 0).length} source file{norm.filter((r) => r.n_replaced > 0).length === 1 ? "" : "s"} were written as <code>null</code> in the bundle.
                They mean <strong>undefined</strong> (for example, the correlation of a constant prediction), never zero, and this dashboard shows them as such.
              </p>
              <p className="secondary">The original files were only read; their sha256 is unchanged.</p>
            </>
          ) : <p className="muted">This bundle has no normalization index.</p>}
        </Evidence>
        <Card title="Array encoding">
          <dl className="kv">
            <dt>Encoding</dt><dd><Mono>{enc.name}</Mono></dd>
            <dt>Lossless</dt><dd>{enc.lossless ? "yes: every array is decoded and its sha256 re-checked" : "no"}</dd>
            <dt>Steps</dt><dd><ol className="bullets" style={{ paddingLeft: 16 }}>{enc.steps.map((s) => <li key={s}>{s}</li>)}</ol></dd>
            <dt>Derived, not stored</dt><dd>{enc.derived_not_stored}</dd>
          </dl>
        </Card>
      </div>

      {norm.length > 0 && (
        <div className="section">
          <Card title="Normalized JSON copies" meta={`${norm.length} files`}>
            <DataTable rows={norm} columns={normCols} pageSize={10} searchable searchLabel="Filter files" caption="Normalized JSON files" />
          </Card>
        </div>
      )}

      <div className="section">
        <Card title="Source fingerprints" meta={`${sources.length} files the export read (each hashed); ${fp.n_files} experiment files were fingerprinted before and after`}>
          <DataTable rows={sources} columns={srcCols} pageSize={15} searchable searchLabel="Filter sources" caption="Source fingerprints" />
        </Card>
      </div>
      <div className="section">
        <Card title="Bundle contents" meta={`${files.length} files, each checked against its sha256 before use`}>
          <DataTable rows={files} columns={fileCols} pageSize={15} searchable searchLabel="Filter bundle files" caption="Bundle files" />
        </Card>
      </div>
    </>
  );
}
