import { Component, useMemo, type ReactNode } from "react";
import { Shell } from "./components/Shell";
import { Card, Status } from "./components/ui";
import { BundleProvider, useBundle } from "./data/BundleContext";
import { BundleClient } from "./data/client";
import { integrityState, validationScope } from "./data/derive";
import { LimitationsList } from "./components/Limitations";
import { fmtSig } from "./data/values";
import { Arms } from "./pages/Arms";
import { Collapse } from "./pages/Collapse";
import { Designs } from "./pages/Designs";
import { Maps } from "./pages/Maps";
import { Methods } from "./pages/Methods";
import { Metrics } from "./pages/Metrics";
import { Overview } from "./pages/Overview";
import { Provenance } from "./pages/Provenance";
import { Seeds } from "./pages/Seeds";
import { Statistics } from "./pages/Statistics";
import { Training } from "./pages/Training";
import { Validation } from "./pages/Validation";
import { IMPLEMENTED, NAV, useRoute } from "./router";
import { useTheme } from "./theme/theme";

export function App({ client }: { client?: BundleClient }) {
  const c = useMemo(() => client ?? new BundleClient("./data/"), [client]);
  return <BundleProvider client={c}><Root /></BundleProvider>;
}

function Gate({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="gate">
      <div className="card gate-card blocking" role="alert">
        <Status tone="crit">Integrity</Status>
        <h1 style={{ marginTop: 12 }}>{title}</h1>
        {children}
      </div>
    </div>
  );
}

/** A render error on one page shows a message on that page instead of blanking the whole application. */
class PageBoundary extends Component<{ path: string; children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };
  static getDerivedStateFromError(e: Error) { return { error: e.message }; }
  componentDidUpdate(prev: { path: string }) { if (prev.path !== this.props.path && this.state.error) this.setState({ error: null }); }
  render() {
    return this.state.error
      ? <div className="blocking-banner" role="alert" data-testid="page-error">This page could not be drawn: {this.state.error}. Other pages are unaffected.</div>
      : this.props.children;
  }
}

function Root() {
  const route = useRoute();
  const [theme, toggleTheme] = useTheme();
  const { core, error, loading, session, verification } = useBundle();

  if (loading) return <div className="gate"><div className="loading">Verifying the results bundle…</div></div>;
  if (error || !core) {
    return (
      <Gate title="The results bundle could not be verified">
        <p>No results are shown. The dashboard only displays data that matches its manifest exactly.</p>
        <p className="mono" style={{ whiteSpace: "pre-wrap" }} data-testid="gate-error">{error ?? "bundle not loaded"}</p>
        {session ? (
          <div className="secondary" data-testid="gate-launcher">
            <p style={{ marginBottom: 6 }}>
              Launcher verification of <code>{session.zip_name}</code>: <strong>{session.verification.status}</strong>
            </p>
            {session.verification.errors.length > 0 && (
              <ul className="bullets mono" style={{ fontSize: 12 }}>
                {session.verification.errors.slice(0, 8).map((e) => <li key={e}>{e}</li>)}
              </ul>
            )}
            <p className="muted" style={{ marginTop: 8 }}>Export sha256 <code>{session.zip_sha256}</code></p>
          </div>
        ) : (
          <p className="secondary">Start the dashboard with <code>run_dashboard.bat &lt;export.zip&gt;</code>: the launcher verifies the
            export, extracts it outside the project and serves it locally.</p>
        )}
      </Gate>
    );
  }
  const integ = integrityState(core.manifest, core.integrity, session, verification);
  const scope = validationScope(core.report.report);
  const pt = core.report.report.analysis.primary_test;
  const themeKey = `${theme}`;
  const notice = (
    <>
      <span className="chip" title="All performance values are on the diagnostic validation set">Validation only · {scope.nGraphs} diagnostic graphs</span>
      <span className="chip" title="The test split was never evaluated">No test-set results</span>
      <span className="chip" title="Arm U removes the cell-head clamp during training only">Arm U: diagnostic deviation</span>
      <a className="chip chip--link" href="#/statistics" title="The pre-registered exact McNemar test (Statistics page)" data-testid="chip-p">
        p = {fmtSig(pt.p_two_sided_exact, 4) ?? "undefined"} · {pt.significant ? "significant" : "not significant"}</a>
    </>
  );
  const path = IMPLEMENTED.has(route.path) ? route.path : "overview";
  const planned = !IMPLEMENTED.has(route.path) && NAV.some((g) => g.items.some((i) => i.path === route.path));

  let page: React.ReactNode;
  if (!integ.ok && path !== "provenance") {
    page = (
      <>
        <div className="blocking-banner" role="alert" data-testid="integrity-block">
          Integrity failure: results are withheld. {integ.reasons.join(" · ")}
        </div>
        <Card title="Why no results are shown">
          <p>The dashboard shows experiment results only when the export bundle verifies completely. Open <a href="#/provenance">Provenance and integrity</a> for the failed checks.</p>
        </Card>
      </>
    );
  } else if (path === "provenance") {
    page = <Provenance />;
  } else if (path === "arms") {
    page = <Arms themeKey={themeKey} />;
  } else if (path === "seeds") {
    page = <Seeds themeKey={themeKey} />;
  } else if (path === "training") {
    page = <Training themeKey={themeKey} />;
  } else if (path === "validation") {
    page = <Validation themeKey={themeKey} />;
  } else if (path === "metrics") {
    page = <Metrics />;
  } else if (path === "designs") {
    page = <Designs themeKey={themeKey} />;
  } else if (path === "statistics") {
    page = <Statistics />;
  } else if (path === "maps") {
    page = <Maps themeKey={themeKey} params={route.params} />;
  } else if (path === "collapse") {
    page = <Collapse themeKey={themeKey} />;
  } else if (path === "methods") {
    page = <Methods />;
  } else {
    page = (
      <>
        {planned && <div className="card" style={{ marginBottom: 20 }}>This section is planned for a later milestone and is not built yet. Showing the executive summary.</div>}
        <Overview themeKey={themeKey} />
      </>
    );
  }
  return (
    <Shell route={{ ...route, path }} integrity={integ} scopeNotice={notice} theme={theme} onToggleTheme={toggleTheme}
      footer={integ.ok ? (
        <details className="footer-limits" data-testid="footer-limits">
          <summary>Methodological limitations of HEADPATH</summary>
          <LimitationsList report={core.report.report} minP={core.analysis.primary_test_rederived.minimum_p_achievable_with_these_pairs} testId="footer-limitations" />
          <p className="chart-caption">Full methodology, adaptations and assumptions: <a href="#/methods">Methodology and caveats</a>.</p>
        </details>
      ) : null}>
      <PageBoundary path={path}>{page}</PageBoundary>
    </Shell>
  );
}
