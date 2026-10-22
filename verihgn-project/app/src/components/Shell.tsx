import { useEffect, useState, type ReactNode } from "react";
import { IMPLEMENTED, NAV, type Route } from "../router";
import type { IntegrityState } from "../data/derive";
import { Status } from "./ui";

export function Shell({ route, integrity, scopeNotice, theme, onToggleTheme, children, footer }: {
  route: Route; integrity: IntegrityState | null; scopeNotice: ReactNode; theme: "light" | "dark";
  onToggleTheme: () => void; children: ReactNode; footer?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  useEffect(() => { setOpen(false); }, [route.path]);
  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "is-open" : ""}`} aria-label="Sections">
        <div className="brand">
          <div className="brand-name">VeriHGN · HEADPATH</div>
          <div className="brand-sub">Diagnostic results · offline</div>
        </div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {NAV.map((g) => (
            <div className="nav-group" key={g.label}>
              <div className="nav-label">{g.label}</div>
              {g.items.map((it) => IMPLEMENTED.has(it.path) ? (
                <a key={it.path} className="nav-item" href={`#/${it.path}`}
                   aria-current={route.path === it.path ? "page" : undefined}>
                  <span>{it.label}</span>
                </a>
              ) : (
                <span key={it.path} className="nav-item is-planned" aria-disabled="true"
                      title={`Planned for milestone ${it.milestone}; not built yet`}>
                  <span>{it.label}</span><span className="nav-tag">{it.milestone}</span>
                </span>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          Reads a verified export bundle locally. No backend, no network access.
        </div>
      </aside>
      {open && <div className="scrim" onClick={() => setOpen(false)} aria-hidden="true" />}
      <div className="main">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setOpen(true)} aria-label="Open sections">
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M1.5 3.5h11M1.5 7h11M1.5 10.5h11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
          </button>
          <div className="topbar-title">HEADPATH <span>· train-time cell-head clamp diagnostic</span></div>
          <div className="scope-notice">{scopeNotice}</div>
          {integrity && (
            <a href="#/provenance" style={{ textDecoration: "none" }} title={integrity.reasons.join("\n") || "All bundle checks passed"}>
              <Status tone={integrity.tone}>{integrity.label}</Status>
            </a>
          )}
          <button className="icon-button" onClick={onToggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            {theme === "dark"
              ? <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="2.8" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M7 1v1.4M7 11.6V13M1 7h1.4M11.6 7H13M2.8 2.8l1 1M10.2 10.2l1 1M2.8 11.2l1-1M10.2 3.8l1-1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" /></svg>
              : <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M11.5 8.6A5 5 0 0 1 5.4 2.5a5 5 0 1 0 6.1 6.1z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" /></svg>}
          </button>
        </header>
        <main className="content">{children}{footer}</main>
      </div>
    </div>
  );
}
