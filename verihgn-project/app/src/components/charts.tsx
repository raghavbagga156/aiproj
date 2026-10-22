/* Reusable research charts on top of <Chart> (one Plotly wrapper, one set of conventions).
   Arm identity is always colour AND marker shape (O circle, U diamond). Undefined values are gaps or an explicit
   "undefined" mark; they are never drawn at 0. */
import { useCallback, useMemo, type ReactNode } from "react";
import type { Num, RunResult } from "../data/schema";
import { fmtNum, isDefined, toPlotSeries } from "../data/values";
import { Chart, type Trace } from "./Chart";

type Theme = Record<string, string>;
export const armColor = (t: Theme, arm: "O" | "U") => (arm === "O" ? t["armO"]! : t["armU"]!);
export const armSymbol = (arm: "O" | "U") => (arm === "O" ? "circle" : "diamond");

export interface Rule { x: number; text: string; anchor?: "left" | "right" | "center"; side?: "above" | "below" }

function ruleShapes(rules: Rule[], ink3: string, axis: "x" | "y" = "x") {
  return rules.map((r) => axis === "x"
    ? { type: "line", xref: "x", yref: "paper", x0: r.x, x1: r.x, y0: 0, y1: 1, line: { width: 1, color: ink3 }, layer: "below" }
    : { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: r.x, y1: r.x, line: { width: 1, color: ink3 }, layer: "below" });
}
function ruleNotes(rules: Rule[], ink3: string, axis: "x" | "y" = "x") {
  return rules.map((r) => axis === "x"
    ? { x: r.x, y: 1.0, xref: "x", yref: "paper", yanchor: "bottom", text: r.text, xanchor: r.anchor ?? "center",
        xshift: r.anchor === "right" ? -4 : r.anchor === "left" ? 4 : 0, showarrow: false, font: { size: 11, color: ink3 } }
    : { x: 1, y: r.x, xref: "paper", yref: "y", xanchor: "right", yanchor: r.side === "below" ? "top" : "bottom", text: r.text,
        showarrow: false, font: { size: 11, color: ink3 } });
}
const ink3Of = (base: Record<string, unknown>) => (base["font"] as { color: string }).color;

/** Per-run criterion values on two arm rows, with pre-registered thresholds drawn from the plan. */
export function CriterionStrip({ runs, value, label, xTitle, rules, band, themeKey, digits = 3, height = 210 }: {
  runs: RunResult[]; value: (r: RunResult) => Num; label: string; xTitle: string;
  rules: Rule[]; band?: [number, number]; themeKey: string; digits?: number; height?: number;
}) {
  const seeds = useMemo(() => Array.from(new Set(runs.map((r) => r.seed))).sort((a, b) => a - b), [runs]);
  const traces = useCallback((t: Theme): Trace[] => (["O", "U"] as const).map((arm) => {
    const rs = runs.filter((r) => r.arm === arm);
    const off = (s: number) => (seeds.indexOf(s) - (seeds.length - 1) / 2) * 0.07;
    return {
      type: "scatter", mode: "markers", name: arm === "O" ? "Arm O" : "Arm U",
      x: toPlotSeries(rs.map(value)), y: rs.map((r) => (arm === "O" ? 1 : 0) + off(r.seed)),
      customdata: rs.map((r) => [r.seed, r.category]),
      marker: { size: 11, symbol: armSymbol(arm), color: armColor(t, arm), line: { width: 1.5, color: t["surface"] } },
      hovertemplate: `Arm ${arm} · seed %{customdata[0]}<br>${label}: %{x:.${digits}f}<extra></extra>`,
    };
  }), [runs, value, seeds, label, digits]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const ink3 = ink3Of(base);
    return {
      margin: { l: 64, r: 16, t: 28, b: 46 },
      yaxis: { tickvals: [1, 0], ticktext: ["Arm O", "Arm U"], range: [-0.55, 1.55], showgrid: false, fixedrange: true },
      xaxis: { title: { text: xTitle }, zeroline: true, zerolinecolor: ink3, zerolinewidth: 1, fixedrange: true, rangemode: "tozero" },
      shapes: [
        ...(band ? [{ type: "rect", xref: "x", yref: "paper", x0: band[0], x1: band[1], y0: 0, y1: 1, line: { width: 0 },
                     fillcolor: "rgba(137,135,129,0.14)", layer: "below" }] : []),
        ...ruleShapes(rules, ink3),
      ],
      annotations: ruleNotes(rules, ink3),
    };
  }, [rules, band, xTitle]);
  return <Chart label={label} traces={traces} layout={layout} height={height} themeKey={themeKey} />;
}

export interface PairPoint { seed: number; O: Num; U: Num }

/** Dumbbell per seed: O and U on one row joined by a hairline. Undefined ends are not drawn; the row is labelled. */
export function Dumbbell({ points, label, xTitle, rules = [], band, themeKey, digits = 3, table }: {
  points: PairPoint[]; label: string; xTitle: string; rules?: Rule[]; band?: [number, number]; themeKey: string;
  digits?: number; table?: ReactNode;
}) {
  const traces = useCallback((t: Theme): Trace[] => {
    const lines: Trace[] = points.map((p, i) => ({ p, i })).filter(({ p }) => isDefined(p.O) && isDefined(p.U)).map(({ p, i }) => ({
      type: "scatter", mode: "lines", x: [p.O, p.U], y: [i, i], hoverinfo: "skip",
      line: { color: t["hairline"], width: 3 }, showlegend: false,
    }));
    const marks = (["O", "U"] as const).map((arm) => ({
      type: "scatter", mode: "markers", name: arm === "O" ? "Arm O" : "Arm U",
      x: toPlotSeries(points.map((p) => p[arm])), y: points.map((_, i) => i), customdata: points.map((p) => p.seed),
      marker: { size: 11, symbol: armSymbol(arm), color: armColor(t, arm), line: { width: 1.5, color: t["surface"] } },
      hovertemplate: `Arm ${arm} · seed %{customdata}<br>${label}: %{x:.${digits}f}<extra></extra>`,
    }));
    return [...lines, ...marks];
  }, [points, label, digits]);
  const undefinedNotes = useMemo(() => points.map((p, i) => ({ i, arms: (["O", "U"] as const).filter((a) => !isDefined(p[a])) }))
    .filter((u) => u.arms.length > 0), [points]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const ink3 = ink3Of(base);
    return {
      margin: { l: 56, r: 92, t: 28, b: 46 },
      yaxis: { title: { text: "seed" }, showgrid: false, fixedrange: true, tickvals: points.map((_, i) => i),
               ticktext: points.map((p) => String(p.seed)), range: [points.length - 0.5, -0.5], zeroline: false },
      xaxis: { title: { text: xTitle }, zeroline: true, zerolinecolor: ink3, fixedrange: true, rangemode: "tozero" },
      shapes: [
        ...(band ? [{ type: "rect", xref: "x", yref: "paper", x0: band[0], x1: band[1], y0: 0, y1: 1, line: { width: 0 },
                     fillcolor: "rgba(137,135,129,0.14)", layer: "below" }] : []),
        ...ruleShapes(rules, ink3),
      ],
      annotations: [
        ...ruleNotes(rules, ink3),
        ...undefinedNotes.map((u) => ({ x: 1, xref: "paper", xanchor: "left", xshift: 8, y: u.i, yref: "y",
          text: `${u.arms.join(", ")}: undefined`, showarrow: false, font: { size: 11, color: ink3 } })),
      ],
    };
  }, [rules, band, xTitle, undefinedNotes, points]);
  return <Chart label={label} traces={traces} layout={layout} height={64 + points.length * 34} themeKey={themeKey} table={table} />;
}

export interface Series { name: string; arm: "O" | "U"; x: (number | string)[]; y: Num[]; emphasis?: boolean; seed?: number }

/** Lines over steps (one per run). Emphasised runs are solid; others faint. Gaps for undefined values. */
export function SeriesChart({ series, label, xTitle, yTitle, themeKey, height = 240, logY = false, rules = [],
                              vlines = [], yRange, markers = false, digits = 3, table, caption }: {
  series: Series[]; label: string; xTitle: string; yTitle: string; themeKey: string; height?: number; logY?: boolean;
  rules?: Rule[]; vlines?: { x: number; text: string }[]; yRange?: [number, number]; markers?: boolean; digits?: number;
  table?: ReactNode; caption?: ReactNode;
}) {
  const traces = useCallback((t: Theme): Trace[] => series.map((s) => ({
    type: "scatter", mode: markers ? "lines+markers" : "lines", name: s.name, x: s.x, y: toPlotSeries(s.y),
    line: { color: armColor(t, s.arm), width: s.emphasis === undefined ? 1.25 : s.emphasis ? 2 : 1, dash: "solid" },
    opacity: s.emphasis === undefined ? 0.7 : s.emphasis ? 1 : 0.28,
    marker: { size: 7, symbol: armSymbol(s.arm), color: armColor(t, s.arm) },
    hovertemplate: `${s.name}<br>${xTitle} %{x}<br>${yTitle}: %{y:.${digits}${logY ? "e" : "f"}}<extra></extra>`,
  })), [series, xTitle, yTitle, digits, logY, markers]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const ink3 = ink3Of(base);
    return {
      margin: { l: 60, r: 16, t: 24, b: 44 },
      xaxis: { title: { text: xTitle }, fixedrange: true },
      yaxis: { title: { text: yTitle }, type: logY ? "log" : "linear", fixedrange: true, ...(yRange ? { range: yRange } : {}),
               exponentformat: "power" },
      shapes: [...ruleShapes(rules, ink3, "y"),
               ...vlines.map((v) => ({ type: "line", xref: "x", yref: "paper", x0: v.x, x1: v.x, y0: 0, y1: 1,
                                       line: { width: 1, color: ink3 }, layer: "below" }))],
      annotations: [...ruleNotes(rules, ink3, "y"),
                    ...vlines.map((v) => ({ x: v.x, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: v.text,
                                            showarrow: false, font: { size: 11, color: ink3 } }))],
    };
  }, [xTitle, yTitle, logY, yRange, rules, vlines]);
  return <Chart label={label} traces={traces} layout={layout} height={height} themeKey={themeKey} table={table} caption={caption} />;
}

/** Diverging heatmap for correlations (centred on 0). Poles teal / magenta: never the arm hues. Undefined cells
    are left empty and labelled "undef.". */
export function CorrelationHeatmap({ rows, cols, z, label, themeKey, height, colTitle, table }: {
  rows: string[]; cols: string[]; z: Num[][]; label: string; themeKey: string; height?: number; colTitle?: string; table?: ReactNode;
}) {
  const traces = useCallback((t: Theme): Trace[] => {
    const dark = document.documentElement.dataset.theme === "dark";
    const mid = dark ? "#383835" : "#f0efec";
    const neg = dark ? "#8e3d68" : "#d47ba7", pos = dark ? "#226f69" : "#4aa79f";
    return [{
      type: "heatmap", x: cols, y: rows, z: z.map((r) => toPlotSeries(r)), zmin: -1, zmax: 1, zmid: 0,
      colorscale: [[0, neg], [0.5, mid], [1, pos]], xgap: 2, ygap: 2, hoverongaps: true,
      text: z.map((r) => r.map((v) => fmtNum(v, 2) ?? "undef.")), texttemplate: "%{text}",
      textfont: { size: 11, color: t["ink"] },
      colorbar: { thickness: 10, len: 0.8, outlinewidth: 0, tickfont: { size: 11 }, title: { text: "ρ", side: "right" } },
      hovertemplate: "%{y} · %{x}<br>ρ = %{text}<extra></extra>",
    }];
  }, [rows, cols, z]);
  const layout = useCallback(() => ({
    margin: { l: 150, r: 20, t: 10, b: 70 },
    xaxis: { type: "category", tickangle: -40, fixedrange: true, title: { text: colTitle ?? "" } },
    yaxis: { type: "category", autorange: "reversed", showgrid: false, fixedrange: true },
  }), [colTitle]);
  return <Chart label={label} traces={traces} layout={layout} height={height ?? 80 + rows.length * 34} themeKey={themeKey} table={table}
                preferTableBelow={720} />;
}

/** Sequential single-hue ramp (surface -> ink) for maps: never an arm hue, so a map colour never means "Arm O/U".
    In dark mode the same tokens give a dark-to-light ramp. */
export const seqScale = (t: Theme) => [[0, t["surface"]!], [1, t["ink"]!]];

/** One spatial map at native resolution (no resampling). z is [y][x]; zmax is shared by the caller when panels
    are compared. Tiles are square (scaleanchor); undefined tiles would be gaps (none in a verified export). */
export type MapTransform = "linear" | "sqrt";

/** Colour-bar ticks at round VALUES; with the square-root transform they sit at sqrt(value), so the bar is read in
    the data's own units. */
function mapTicks(zmax: number, transform: MapTransform) {
  const step = Math.pow(10, Math.floor(Math.log10(zmax / 4 || 1)));
  const nice = [1, 2, 2.5, 5, 10].map((m) => m * step).find((s) => zmax / s <= 5) ?? zmax / 4;
  const vals: number[] = [];
  for (let v = 0; v <= zmax * 1.0001; v += nice) vals.push(Number(v.toPrecision(6)));
  return { tickvals: vals.map((v) => (transform === "sqrt" ? Math.sqrt(v) : v)), ticktext: vals.map((v) => String(v)) };
}

export function MapPanel({ z, zmax, label, colorTitle, themeKey, height = 320, digits = 4, transform = "linear" }: {
  z: (number | null)[][]; zmax: number | null; label: string; colorTitle: string; themeKey: string; height?: number; digits?: number;
  transform?: MapTransform;
}) {
  const traces = useCallback((t: Theme): Trace[] => {
    const top = zmax && zmax > 0 ? zmax : 1;
    const shown = transform === "sqrt" ? z.map((r) => r.map((v) => (v === null ? null : Math.sqrt(Math.max(v, 0))))) : z;
    return [{
      type: "heatmap", z: shown, zmin: 0, zmax: transform === "sqrt" ? Math.sqrt(top) : top, zauto: false, colorscale: seqScale(t),
      hoverongaps: false, ...(transform === "sqrt" ? { customdata: z } : {}),
      colorbar: { thickness: 8, len: 0.9, outlinewidth: 0, tickfont: { size: 10 }, tickmode: "array", ...mapTicks(top, transform),
                  title: { text: transform === "sqrt" ? `${colorTitle} (√ scale)` : colorTitle, side: "right", font: { size: 11 } } },
      hovertemplate: `x %{x}, y %{y}<br>${colorTitle} %{${transform === "sqrt" ? "customdata" : "z"}:.${digits}f}<extra></extra>`,
    }];
  }, [z, zmax, colorTitle, digits, transform]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const axisLine = (base["xaxis"] as Record<string, unknown>)["linecolor"];
    return {
      margin: { l: 44, r: 8, t: 6, b: 38 },
      xaxis: { title: { text: "x (gcell)" }, showline: true, mirror: true, linecolor: axisLine, constrain: "domain", fixedrange: true },
      yaxis: { title: { text: "y (gcell)" }, showline: true, mirror: true, linecolor: axisLine, scaleanchor: "x", constrain: "domain",
               showgrid: false, fixedrange: true },
    };
  }, []);
  return <Chart label={label} traces={traces} layout={layout} height={height} themeKey={themeKey} />;
}

/** Pred-vs-label density of the cell subsample (counts per bin; empty bins blank). Shared square axes; the dotted
    diagonal is prediction = label. */
export function DensityPanel({ z, edges, label, themeKey, height = 300, xTitle, yTitle }: {
  z: number[][]; edges: number[]; label: string; themeKey: string; height?: number; xTitle: string; yTitle: string;
}) {
  const centers = useMemo(() => edges.slice(0, -1).map((e, i) => (e + edges[i + 1]!) / 2), [edges]);
  const traces = useCallback((t: Theme): Trace[] => [{
    type: "heatmap", x: centers, y: centers, z: z.map((r) => r.map((c) => (c > 0 ? Math.log10(c) : null))),
    customdata: z, colorscale: seqScale(t), hoverongaps: false,
    colorbar: { thickness: 8, len: 0.9, outlinewidth: 0, tickfont: { size: 10 }, title: { text: "log₁₀ cells", side: "right", font: { size: 11 } } },
    hovertemplate: `${xTitle} ≈ %{x:.3f}<br>${yTitle} ≈ %{y:.3f}<br>%{customdata} cells<extra></extra>`,
  }], [z, centers, xTitle, yTitle]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const ink3 = ink3Of(base);
    const hi = edges[edges.length - 1] ?? 1;
    return {
      margin: { l: 52, r: 8, t: 6, b: 42 },
      xaxis: { title: { text: xTitle }, range: [0, hi], fixedrange: true, constrain: "domain" },
      yaxis: { title: { text: yTitle }, range: [0, hi], scaleanchor: "x", fixedrange: true, constrain: "domain", showgrid: false },
      shapes: [{ type: "line", x0: 0, y0: 0, x1: hi, y1: hi, line: { width: 1, color: ink3, dash: "dot" } }],
    };
  }, [edges, xTitle, yTitle]);
  return <Chart label={label} traces={traces} layout={layout} height={height} themeKey={themeKey} />;
}

/** Lines in ink only (no arm colours): used for the separate notebook-07 experiment, whose groups are not arms
    O / U. Series are told apart by dash pattern and shade, and named in the legend. */
export interface InkSeries { name: string; x: number[]; y: (number | null)[]; dash: number }
const DASHES = ["solid", "dash", "dot", "dashdot"];
export function InkSeriesChart({ series, label, xTitle, yTitle, themeKey, height = 200, logY = false, yRange, xRange, vlines = [] }: {
  series: InkSeries[]; label: string; xTitle: string; yTitle: string; themeKey: string; height?: number; logY?: boolean;
  yRange?: [number, number]; xRange?: [number, number]; vlines?: { x: number; dash: number }[];
}) {
  const traces = useCallback((t: Theme): Trace[] => series.map((s) => ({
    type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.y,
    line: { color: s.dash === 0 ? t["ink"] : t["ink2"], width: 1.5, dash: DASHES[s.dash % DASHES.length] },
    hovertemplate: `${s.name}<br>${xTitle} %{x}<br>${yTitle}: %{y:.3${logY ? "e" : "f"}}<extra></extra>`,
  })), [series, xTitle, yTitle, logY]);
  const layout = useCallback((base: Record<string, unknown>) => {
    const ink3 = ink3Of(base);
    return {
      margin: { l: 52, r: 8, t: 8, b: 40 },
      showlegend: true, legend: { orientation: "h", y: -0.32, x: 0, font: { size: 11 } },
      xaxis: { title: { text: xTitle }, fixedrange: true, ...(xRange ? { range: xRange } : {}) },
      yaxis: { title: { text: yTitle }, type: logY ? "log" : "linear", fixedrange: true, exponentformat: "power",
               ...(yRange ? { range: yRange } : {}) },
      shapes: vlines.map((v) => ({ type: "line", xref: "x", yref: "paper", x0: v.x, x1: v.x, y0: 0, y1: 1,
                                   line: { width: 1, color: ink3, dash: DASHES[v.dash % DASHES.length] }, layer: "below" })),
    };
  }, [xTitle, yTitle, logY, yRange, xRange, vlines]);
  return <Chart label={label} traces={traces} layout={layout} height={height} themeKey={themeKey} />;
}
