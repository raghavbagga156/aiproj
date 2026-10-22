import { useEffect, useRef, useState, type ReactNode } from "react";
import { cssVar } from "../theme/theme";

/* One Plotly wrapper carries every chart convention (docs/DASHBOARD_FRONTEND_PLAN.md section 3.5):
   theme-token colours, system font, hairline y-grid only, no dual axes, minimal toolbar (PNG + reset),
   no Plotly logo, gaps for undefined values (callers pass null; connectgaps is forced off). */

type PlotlyModule = {
  react: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
  Plots: { resize: (el: HTMLElement) => void };
};
let plotlyPromise: Promise<PlotlyModule> | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) plotlyPromise = import("plotly.js-dist-min").then((m) => ((m as { default?: unknown }).default ?? m) as PlotlyModule);
  return plotlyPromise;
}

export type Trace = Record<string, unknown>;

export function baseLayout(): Record<string, unknown> {
  const ink = cssVar("--ink"), ink2 = cssVar("--ink-2"), ink3 = cssVar("--ink-3");
  const font = cssVar("--font-ui", "Segoe UI, system-ui, sans-serif");
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: font, size: 12, color: ink2 },
    margin: { l: 56, r: 16, t: 8, b: 44 },
    hoverlabel: { bgcolor: cssVar("--surface"), bordercolor: cssVar("--hairline-strong"), font: { family: font, color: ink, size: 12 } },
    xaxis: { showgrid: false, zeroline: false, linecolor: cssVar("--axis"), tickcolor: cssVar("--axis"), ticks: "outside",
             ticklen: 4, tickfont: { color: ink3 }, title: { font: { color: ink2, size: 12 }, standoff: 10 } },
    yaxis: { gridcolor: cssVar("--grid"), gridwidth: 1, zeroline: false, linecolor: "rgba(0,0,0,0)", tickfont: { color: ink3 },
             title: { font: { color: ink2, size: 12 }, standoff: 10 }, automargin: true },
    showlegend: false,
    dragmode: false,
  };
}

function merge(a: Record<string, unknown>, b: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...a };
  for (const [k, v] of Object.entries(b)) {
    const av = out[k];
    out[k] = av && v && typeof av === "object" && typeof v === "object" && !Array.isArray(v)
      ? merge(av as Record<string, unknown>, v as Record<string, unknown>) : v;
  }
  return out;
}

export function Chart({ traces, layout = {}, height = 260, label, caption, themeKey, table, preferTableBelow }: {
  traces: (theme: { ink: string; ink2: string; ink3: string; armO: string; armU: string; surface: string; hairline: string }) => Trace[];
  layout?: Record<string, unknown> | ((t: Record<string, unknown>) => Record<string, unknown>);
  height?: number; label: string; caption?: ReactNode; themeKey: string; table?: ReactNode;
  /** below this viewport width the table view is shown first (charts too dense for a phone) */
  preferTableBelow?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(() => !!table && preferTableBelow !== undefined
    && typeof window !== "undefined" && window.innerWidth < preferTableBelow);

  useEffect(() => {
    const el = ref.current;
    if (!el || showTable) return;
    let cancelled = false;
    const t = { ink: cssVar("--ink"), ink2: cssVar("--ink-2"), ink3: cssVar("--ink-3"), armO: cssVar("--arm-O"),
                armU: cssVar("--arm-U"), surface: cssVar("--surface"), hairline: cssVar("--hairline") };
    const data = traces(t).map((tr) => ({ ...tr, connectgaps: false }));   // undefined values stay gaps
    const base = merge(baseLayout(), { height });
    const lay = typeof layout === "function" ? merge(base, layout(base)) : merge(base, layout);
    const config = { displaylogo: false, responsive: true, modeBarButtons: [["toImage", "resetScale2d"]],
                     toImageButtonOptions: { format: "png", scale: 2 } };
    loadPlotly().then((P) => { if (!cancelled && ref.current) return P.react(ref.current, data, lay, config); })
      .catch((e: Error) => setFailed(e.message));
    const ro = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => { loadPlotly().then((P) => { if (ref.current) P.Plots.resize(ref.current); }).catch(() => undefined); })
      : null;
    ro?.observe(el);
    return () => { cancelled = true; ro?.disconnect(); };
  }, [traces, layout, height, themeKey, showTable]);

  useEffect(() => () => { const el = ref.current; if (el) loadPlotly().then((P) => P.purge(el)).catch(() => undefined); }, []);

  return (
    <figure style={{ margin: 0 }}>
      {table && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4 }}>
          <div className="seg" role="group" aria-label={`${label}: view`}>
            <button aria-pressed={!showTable} onClick={() => setShowTable(false)}>Chart</button>
            <button aria-pressed={showTable} onClick={() => setShowTable(true)}>Table</button>
          </div>
        </div>
      )}
      {showTable ? table : (
        failed ? <div className="chart-fallback">Chart could not be drawn ({failed}). The table view shows the same values.</div>
          : <div ref={ref} className="chart" role="img" aria-label={label} style={{ height }} />
      )}
      {caption && <figcaption className="chart-caption">{caption}</figcaption>}
    </figure>
  );
}
