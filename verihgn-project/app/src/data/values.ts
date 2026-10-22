/* Null-safe number handling. `null` is UNDEFINED (a correlation of a constant prediction, a NaN in the source
   record, ...). It is never converted to 0: formatting returns undefined markers, and chart adapters turn it
   into a gap. */
import type { Num } from "./schema";

export const isDefined = (v: Num | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/** Fixed decimals; null/undefined -> null so the caller must render an explicit "undefined" marker. */
export function fmtNum(v: Num | undefined, digits = 3): string | null {
  if (!isDefined(v)) return null;
  // a non-zero value too small for `digits` decimals is shown in scientific notation, never as "0.000" or "-0.000"
  if (v !== 0 && Math.abs(v) < 0.5 * Math.pow(10, -digits)) return v.toExponential(1);
  return v.toFixed(digits);
}

/** Significant-figure formatting for p-values and small quantities (no trailing-zero noise). */
export function fmtSig(v: Num | undefined, sig = 4): string | null {
  if (!isDefined(v)) return null;
  if (v === 0) return "0";
  const s = Number(v.toPrecision(sig)).toString();
  return s;
}

export function fmtInt(v: Num | undefined): string | null {
  return isDefined(v) ? Math.round(v).toLocaleString("en-US") : null;
}

export function fmtBytes(n: Num | undefined): string | null {
  if (!isDefined(n)) return null;
  const u = ["B", "kB", "MB", "GB"];
  let i = 0;
  let x = n;
  while (x >= 1000 && i < u.length - 1) { x /= 1000; i++; }
  return `${i === 0 ? x.toFixed(0) : x.toFixed(1)} ${u[i]}`;
}

/** Plotly-ready series: undefined values become null (a gap with connectgaps:false), never 0. */
export function toPlotSeries(values: Num[]): (number | null)[] {
  return values.map((v) => (isDefined(v) ? v : null));
}

/** Count defined / undefined members without coercion. */
export function definedCount(values: Num[]): { defined: number; undefined: number } {
  let d = 0;
  for (const v of values) if (isDefined(v)) d++;
  return { defined: d, undefined: values.length - d };
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}
