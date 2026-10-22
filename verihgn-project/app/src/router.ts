import { useEffect, useState } from "react";

/* Hash routing: works from any static server and keeps the view shareable (#/provenance?status=WARN). */
export interface Route { path: string; params: URLSearchParams }

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [p, q] = h.split("?");
  return { path: p || "overview", params: new URLSearchParams(q ?? "") };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export interface NavItem { path: string; label: string; milestone: "M1" | "M2" | "M3" }
export interface NavGroup { label: string; items: NavItem[] }

/** Sections of the dashboard (docs/DASHBOARD_FRONTEND_PLAN.md section 4); IMPLEMENTED lists the built ones. */
export const NAV: NavGroup[] = [
  { label: "Overview", items: [{ path: "overview", label: "Executive summary", milestone: "M1" }] },
  {
    label: "Results",
    items: [
      { path: "arms", label: "Arm O vs Arm U", milestone: "M2" },
      { path: "seeds", label: "Seed-wise pairs", milestone: "M2" },
      { path: "training", label: "Training dynamics", milestone: "M2" },
      { path: "validation", label: "Validation curves", milestone: "M2" },
      { path: "metrics", label: "Cell and grid metrics", milestone: "M2" },
      { path: "designs", label: "Per-design comparison", milestone: "M2" },
      { path: "maps", label: "Prediction maps", milestone: "M3" },
    ],
  },
  { label: "Context", items: [{ path: "collapse", label: "Collapse investigation", milestone: "M3" }] },
  {
    label: "Evidence",
    items: [
      { path: "statistics", label: "Statistics and limitations", milestone: "M2" },
      { path: "methods", label: "Methodology and caveats", milestone: "M3" },
      { path: "provenance", label: "Provenance and integrity", milestone: "M1" },
    ],
  },
];

export const IMPLEMENTED = new Set(["overview", "provenance", "arms", "seeds", "training", "validation", "metrics", "designs", "statistics",
  "maps", "collapse", "methods"]);
