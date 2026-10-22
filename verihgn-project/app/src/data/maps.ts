/* Prediction-map helpers. Everything here works on the exported arrays exactly as decoded (float32, lossless):
   no resampling, no smoothing, no invented values. Grids are stored row-major [x, y] (shape [M, N] = [nx, ny],
   tile index = x * N + y; src/verihgn_repro/data/features.py), so images are transposed for display with x
   horizontal and y vertical. Cell arrays are the experiment's 20,000-cell subsample, never a full-cell map. */
import { useEffect, useState } from "react";
import { useBundle } from "./BundleContext";
import type { DecodedArray } from "./arrays";
import type { ArrayMeta, MapsIndexEntry, RunIndexEntry } from "./schema";

/** The experiment's grid label (src/verihgn_repro/data/graph.py grid_label, divided by c_max). */
export const LABEL_FORMULA = "log1p((H + V) / 2) / c_max";

export interface MapStats { n: number; finite: number; min: number | null; max: number | null; mean: number | null;
                            zeroFrac: number | null; allZero: boolean }

export function stats(a: Float32Array): MapStats {
  let finite = 0, zeros = 0, sum = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i]!;
    if (!Number.isFinite(v)) continue;
    finite++;
    if (v === 0) zeros++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { n: a.length, finite, min: finite ? min : null, max: finite ? max : null, mean: finite ? sum / finite : null,
           zeroFrac: finite ? zeros / finite : null, allZero: finite > 0 && zeros === finite };
}

/** |pred - gt| in float32 (the exporter checked this to be bit-exact against the NPZ's grid_abs_err).
    Defined only where both values are finite; elsewhere NaN, which the map draws as a gap. */
export function absError(pred: Float32Array, gt: Float32Array): Float32Array {
  if (pred.length !== gt.length) throw new Error(`length ${pred.length} != ${gt.length}`);
  const out = new Float32Array(pred.length);
  for (let i = 0; i < out.length; i++) {
    const p = pred[i]!, g = gt[i]!;
    out[i] = Number.isFinite(p) && Number.isFinite(g) ? Math.abs(Math.fround(p - g)) : NaN;
  }
  return out;
}

/** Display rows: z[y][x] = a[x * ny + y]. Non-finite values become null (a gap), never 0. */
export function toImage(a: Float32Array, shape: readonly number[]): (number | null)[][] {
  const [nx, ny] = shape as [number, number];
  if (a.length !== nx * ny) throw new Error(`array of ${a.length} values does not match shape ${nx} x ${ny}`);
  const z: (number | null)[][] = new Array(ny);
  for (let y = 0; y < ny; y++) {
    const row: (number | null)[] = new Array(nx);
    for (let x = 0; x < nx; x++) { const v = a[x * ny + y]!; row[x] = Number.isFinite(v) ? v : null; }
    z[y] = row;
  }
  return z;
}

/** Common [0, max] range of several maps (for a shared colour scale); null if nothing is finite. */
export function sharedMax(...arrs: Float32Array[]): number | null {
  let m = -Infinity;
  for (const a of arrs) for (let i = 0; i < a.length; i++) { const v = a[i]!; if (Number.isFinite(v) && v > m) m = v; }
  return Number.isFinite(m) ? m : null;
}

/** cell_gt must equal grid_gt[cell_tile] bit for bit (the experiment's cell label definition). Returns mismatches. */
export function checkCellLabels(cellGt: Float32Array, gridGt: Float32Array, cellTile: Int32Array): number {
  let bad = 0;
  for (let i = 0; i < cellGt.length; i++) {
    const t = cellTile[i]!;
    if (t < 0 || t >= gridGt.length || !Object.is(cellGt[i], gridGt[t])) bad++;
  }
  return bad;
}

/** 2-D count histogram on a shared [0, hi] x [0, hi] square (pred vs label); counts outside are reported, not dropped silently. */
export function density2d(x: Float32Array, y: Float32Array, bins: number, hi: number):
  { z: number[][]; edges: number[]; outside: number; nonFinite: number } {
  const z = Array.from({ length: bins }, () => new Array<number>(bins).fill(0));
  let outside = 0, nonFinite = 0;
  const w = hi / bins;
  for (let i = 0; i < x.length; i++) {
    const a = x[i]!, b = y[i]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) { nonFinite++; continue; }
    if (a < 0 || b < 0 || a > hi || b > hi || w <= 0) { outside++; continue; }
    const ix = Math.min(bins - 1, Math.floor(a / w)), iy = Math.min(bins - 1, Math.floor(b / w));
    z[iy]![ix]! += 1;
  }
  return { z, edges: Array.from({ length: bins + 1 }, (_, i) => i * w), outside, nonFinite };
}

export interface SampleMaps {
  gridGt: Float32Array; rawH: Float32Array; rawV: Float32Array;
  cellGt: Float32Array; cellTile: Int32Array;
  O: { grid: Float32Array; cell: Float32Array } | null;
  U: { grid: Float32Array; cell: Float32Array } | null;
}

function need(meta: Record<string, ArrayMeta> | undefined, key: string, where: string): ArrayMeta {
  const m = meta?.[key];
  if (!m) throw new Error(`${where}: array "${key}" is not in the bundle index`);
  return m;
}

function checkShape(a: DecodedArray, shape: readonly number[], what: string) {
  if (a.shape.length !== shape.length || a.shape.some((v, i) => v !== shape[i])) {
    throw new Error(`${what}: shape ${a.shape.join(" x ")} does not match the sample's ${shape.join(" x ")}; maps are not compared`);
  }
}

/** Loads one sample's labels and the two arms' predictions for one seed (only these arrays are fetched). */
export function useSampleMaps(sample: string | null, entry: MapsIndexEntry | undefined,
                              runO: RunIndexEntry | undefined, runU: RunIndexEntry | undefined) {
  const { client } = useBundle();
  const key = `${sample}|${runO?.name}|${runU?.name}`;
  const [state, setState] = useState<{ key: string; data: SampleMaps | null; error: string | null; loading: boolean }>(
    { key: "", data: null, error: null, loading: true });
  useEffect(() => {
    if (!sample || !entry) { setState({ key, data: null, error: null, loading: false }); return; }
    let alive = true;
    setState({ key, data: null, loading: true, error: null });
    (async () => {
      const a = entry.arrays;
      const [gridGt, rawH, rawV, cellGt, cellTile] = await Promise.all(
        ["grid_gt", "raw_h_overflow", "raw_v_overflow", "cell_gt", "cell_tile"].map((k) => client.array(need(a, k, sample))));
      for (const [arr, k] of [[gridGt, "grid_gt"], [rawH, "raw_h_overflow"], [rawV, "raw_v_overflow"]] as const) checkShape(arr!, entry.shape, k);
      const nCells = cellGt!.shape[0]!;
      if (cellTile!.shape[0] !== nCells) throw new Error("cell_tile and cell_gt have different lengths");
      const arm = async (r: RunIndexEntry | undefined) => {
        if (!r) return null;
        const m = r.maps[sample];
        const [g, c] = await Promise.all([client.array(need(m, "grid_pred", `${r.name}/${sample}`)),
                                          client.array(need(m, "cell_pred", `${r.name}/${sample}`))]);
        checkShape(g, entry.shape, `${r.name} grid_pred`);
        if (c.shape[0] !== nCells) throw new Error(`${r.name} cell_pred: ${c.shape[0]} cells, labels have ${nCells}`);
        return { grid: g.data as Float32Array, cell: c.data as Float32Array };
      };
      const [O, U] = await Promise.all([arm(runO), arm(runU)]);
      return { gridGt: gridGt!.data as Float32Array, rawH: rawH!.data as Float32Array, rawV: rawV!.data as Float32Array,
               cellGt: cellGt!.data as Float32Array, cellTile: cellTile!.data as Int32Array, O, U };
    })().then((data) => { if (alive) setState({ key, data, error: null, loading: false }); })
      .catch((e: Error) => { if (alive) setState({ key, data: null, error: e.message, loading: false }); });
    return () => { alive = false; };
  }, [client, key, sample, entry, runO, runU]);
  // arrays loaded for an earlier selection are never paired with the current one
  return state.key === key ? state : { key, data: null, error: null, loading: true };
}
