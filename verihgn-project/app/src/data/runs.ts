/* Per-run tables (steps / evals / history), loaded on demand from the verified bundle and cached by the client.
   Column-oriented, as exported: { columns, n_rows, data: {col: values}, undefined: {col: {row: kind}} }.
   Undefined source values are null and stay null. */
import { useEffect, useState } from "react";
import { useBundle } from "./BundleContext";
import type { Num, RunIndexEntry } from "./schema";

export interface Table {
  columns: string[];
  n_rows: number;
  data: Record<string, (number | string | boolean | null)[]>;
  undefined: Record<string, Record<string, string>>;
}
export type TableKind = "steps" | "evals" | "history";

export function num(t: Table | undefined, col: string): Num[] {
  if (!t) return [];
  const c = t.data[col];
  if (!c) return Array.from({ length: t.n_rows }, () => null);
  return c.map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
}

export function str(t: Table | undefined, col: string): (string | null)[] {
  if (!t) return [];
  const c = t.data[col];
  if (!c) return Array.from({ length: t.n_rows }, () => null);
  return c.map((v) => (v === null || v === undefined ? null : String(v)));
}

/** Last row of a table as a column -> value map (e.g. the final evaluation). */
export function lastRow(t: Table | undefined): Record<string, number | string | boolean | null> {
  if (!t || t.n_rows === 0) return {};
  const out: Record<string, number | string | boolean | null> = {};
  for (const c of t.columns) out[c] = t.data[c]?.[t.n_rows - 1] ?? null;
  return out;
}

export function rowAt(t: Table | undefined, col: string, value: number): Record<string, number | string | boolean | null> | null {
  if (!t) return null;
  const i = (t.data[col] ?? []).findIndex((v) => v === value);
  if (i < 0) return null;
  const out: Record<string, number | string | boolean | null> = {};
  for (const c of t.columns) out[c] = t.data[c]?.[i] ?? null;
  return out;
}

export interface RunsState {
  index: Record<string, RunIndexEntry> | null;
  tables: Record<string, Partial<Record<TableKind, Table>>>;
  loading: boolean;
  error: string | null;
}

/** Loads the run index and the requested table kinds for every run (cached by the BundleClient). */
export function useRunTables(kinds: TableKind[]): RunsState {
  const { client } = useBundle();
  const key = kinds.join(",");
  const [state, setState] = useState<RunsState>({ index: null, tables: {}, loading: true, error: null });
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const index = await client.json<Record<string, RunIndexEntry>>("headpath/runs/index.json");
        const tables: RunsState["tables"] = {};
        await Promise.all(Object.keys(index).flatMap((name) => kinds.map(async (k) => {
          const t = await client.json<Table>(`headpath/runs/${name}/${k}.json`);
          (tables[name] ??= {})[k] = t;
        })));
        if (alive) setState({ index, tables, loading: false, error: null });
      } catch (e) {
        if (alive) setState({ index: null, tables: {}, loading: false, error: (e as Error).message });
      }
    })();
    return () => { alive = false; };
  }, [client, key]);   // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

/** Rolling 1 - sum(numer)/sum(denom) over the trailing `window` steps; the same rule the pre-registered P2 uses
    over its last `window` steps (diagnostics/headpath.py _improvement). Pairs with an undefined value are
    skipped, as in the experiment code. null until the window is full. */
export function rollingImprovement(numer: Num[], denom: Num[], window: number): Num[] {
  const out: Num[] = [];
  let sn = 0, sd = 0;
  const ok = numer.map((v, i) => typeof v === "number" && typeof denom[i] === "number");
  for (let i = 0; i < numer.length; i++) {
    if (ok[i]) { sn += numer[i] as number; sd += denom[i] as number; }
    const j = i - window;
    if (j >= 0 && ok[j]) { sn -= numer[j] as number; sd -= denom[j] as number; }
    out.push(i >= window - 1 && sd > 0 ? 1 - sn / sd : null);
  }
  return out;
}

/** Exact recomputation of the window ending at the last step (no running-sum rounding), for checks. */
export function windowImprovement(numer: Num[], denom: Num[], window: number): Num {
  let sn = 0, sd = 0;
  for (let i = Math.max(0, numer.length - window); i < numer.length; i++) {
    const a = numer[i], b = denom[i];
    if (typeof a === "number" && typeof b === "number") { sn += a; sd += b; }
  }
  return sd > 0 ? 1 - sn / sd : null;
}

/** Values for a log axis: exactly-zero and undefined become gaps; the number of zeros is reported. */
export function forLogAxis(values: Num[]): { y: (number | null)[]; zeros: number; undefinedCount: number } {
  let zeros = 0, undef = 0;
  const y = values.map((v) => {
    if (v === null) { undef++; return null; }
    if (v <= 0) { zeros++; return null; }
    return v;
  });
  return { y, zeros, undefinedCount: undef };
}
