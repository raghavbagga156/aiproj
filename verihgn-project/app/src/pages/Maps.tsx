import { useEffect, useMemo, useState } from "react";
import { DensityPanel, MapPanel, type MapTransform } from "../components/charts";
import { ArmName, Card, Evidence, Loading, PageHead, SeedPicker, Status, Value } from "../components/ui";
import { useCore } from "../data/BundleContext";
import { LABEL_FORMULA, absError, checkCellLabels, density2d, sharedMax, stats, toImage, useSampleMaps, type MapStats, type SampleMaps } from "../data/maps";
import type { MapsIndexEntry, Num, RunIndexEntry } from "../data/schema";
import { useJson } from "../data/useJson";
import { fmtNum, fmtSig, isDefined } from "../data/values";

type Scale = "shared" | "panel";
const BINS = 48;
/** Tolerance of the in-browser label check (float32 rounding), a definition, not a result. */
const LABEL_TOL = 1e-6;

interface NpzCheck { npz: Num; evals_csv: Num; abs_diff: Num }

function setHashParams(update: Record<string, string>) {
  const [path, q] = window.location.hash.replace(/^#\/?/, "").split("?");
  const p = new URLSearchParams(q ?? "");
  for (const [k, v] of Object.entries(update)) p.set(k, v);
  window.history.replaceState(null, "", `#/${path || "maps"}?${p.toString()}`);
}

function sampleLabel(id: string, e: MapsIndexEntry) {
  const util = id.split("-util_")[1];
  return `${e.design.replace("mgc_", "")}${util ? ` · util ${util}` : ""} · ${e.shape[0]} × ${e.shape[1]} · #${id.split("-")[0]}`;
}

function DeadTag({ s }: { s: MapStats | null }) {
  if (!s?.allZero) return null;
  return <span className="chip chip--dead" data-testid="dead-tag" title="Every value of this prediction map is exactly zero (the clamped head output is constant)">DEAD · all predictions zero</span>;
}

/** Everything the page draws for one graph and seed: images, shared ranges, summaries and in-browser checks. */
function buildView(d: SampleMaps | null, entry: MapsIndexEntry | undefined, cMax: number) {
  if (!d || !entry || !d.O || !d.U) return null;
  const errO = absError(d.O.grid, d.gridGt), errU = absError(d.U.grid, d.gridGt);
  const st = { gt: stats(d.gridGt), O: stats(d.O.grid), U: stats(d.U.grid), errO: stats(errO), errU: stats(errU),
               H: stats(d.rawH), V: stats(d.rawV) };
  const valueMax = sharedMax(d.gridGt, d.O.grid, d.U.grid), errMax = sharedMax(errO, errU), rawMax = sharedMax(d.rawH, d.rawV);
  // label recomputed from the raw maps in float32: log1p((H + V) / 2) / c_max
  let maxDiff = 0;
  for (let i = 0; i < d.gridGt.length; i++) {
    const s = Math.fround(Math.fround(d.rawH[i]! + d.rawV[i]!) / 2);
    const l = Math.fround(Math.fround(Math.log1p(s)) / cMax);
    maxDiff = Math.max(maxDiff, Math.abs(l - d.gridGt[i]!));
  }
  const cellHi = sharedMax(d.cellGt, d.O.cell, d.U.cell) ?? 1;
  const cellErr = (p: Float32Array) => { const e = absError(p, d.cellGt); return stats(e).mean; };
  return {
    img: { gt: toImage(d.gridGt, entry.shape), O: toImage(d.O.grid, entry.shape), U: toImage(d.U.grid, entry.shape),
           errO: toImage(errO, entry.shape), errU: toImage(errU, entry.shape),
           H: toImage(d.rawH, entry.shape), V: toImage(d.rawV, entry.shape) },
    st, valueMax, errMax, rawMax, maxDiff,
    cellMismatch: checkCellLabels(d.cellGt, d.gridGt, d.cellTile),
    nCells: d.cellGt.length,
    cell: { O: stats(d.O.cell), U: stats(d.U.cell), errO: cellErr(d.O.cell), errU: cellErr(d.U.cell),
            dO: density2d(d.cellGt, d.O.cell, BINS, cellHi), dU: density2d(d.cellGt, d.U.cell, BINS, cellHi) },
  };
}

export function Maps({ themeKey, params }: { themeKey: string; params: URLSearchParams }) {
  const { report: rf, analysis: af } = useCore();
  const plan = rf.report.plan;
  const mi = useJson<Record<string, MapsIndexEntry>>("headpath/maps/index.json");
  const ri = useJson<Record<string, RunIndexEntry>>("headpath/runs/index.json");

  const samples = useMemo(() => {
    if (!mi.data) return [];
    const order = [...plan.val_main, ...plan.val.filter((s) => !plan.val_main.includes(s))];
    return [...order.filter((s) => mi.data![s]), ...Object.keys(mi.data).filter((s) => !order.includes(s)).sort()];
  }, [mi.data, plan]);
  const primary = useMemo(() => (ri.data ? Object.values(ri.data).filter((r) => r.primary_inference) : []), [ri.data]);
  const seeds = useMemo(() => Array.from(new Set(primary.map((r) => r.seed))).sort((a, b) => a - b), [primary]);

  const [sample, setSample] = useState<string | null>(params.get("sample"));
  const [seed, setSeed] = useState<number | null>(params.get("seed") ? Number(params.get("seed")) : null);
  const [scale, setScale] = useState<Scale>(params.get("scale") === "panel" ? "panel" : "shared");
  const [transform, setTransform] = useState<MapTransform>(params.get("colour") === "sqrt" ? "sqrt" : "linear");
  const sel = sample && samples.includes(sample) ? sample : samples[0] ?? null;
  const selSeed = seed !== null && seeds.includes(seed) ? seed : seeds[0] ?? null;
  useEffect(() => { if (sel && selSeed !== null) setHashParams({ sample: sel, seed: String(selSeed), scale, colour: transform }); },
    [sel, selSeed, scale, transform]);

  const entry = sel ? mi.data?.[sel] : undefined;
  const runO = primary.find((r) => r.arm === "O" && r.seed === selSeed);
  const runU = primary.find((r) => r.arm === "U" && r.seed === selSeed);
  const maps = useSampleMaps(sel, entry, runO, runU);

  const computed = useMemo(() => {
    try { return { view: buildView(maps.data, entry, plan.c_max), error: null }; }
    catch (e) { return { view: null, error: (e as Error).message }; }
  }, [maps.data, entry, plan.c_max]);
  const view = computed.view;
  const mapError = maps.error ?? computed.error;

  if (mi.missing || ri.missing) {
    return (
      <>
        <PageHead eyebrow="Results" title="Prediction maps" scope="VALIDATION" lede="Spatial predictions of both arms on each validation graph." />
        <Card title="No prediction maps in this bundle"><p>This export does not list the map index, so no maps are shown.</p></Card>
      </>
    );
  }
  if (mi.loading || ri.loading) return <Loading what="the map index" />;
  if (mi.error || ri.error) return <div className="blocking-banner" role="alert">Map index could not be verified: {mi.error ?? ri.error}</div>;

  const zShared = (own: MapStats) => (scale === "shared" ? view?.valueMax ?? null : own.max);
  const zErr = (own: MapStats) => (scale === "shared" ? view?.errMax ?? null : own.max);
  const zRaw = (own: MapStats) => (scale === "shared" ? view?.rawMax ?? null : own.max);
  const cmp = (run: RunIndexEntry | undefined) =>
    run ? ((af as unknown as { grid_metrics_npz_vs_evals?: Record<string, Record<string, NpzCheck>> }).grid_metrics_npz_vs_evals?.[run.name] ?? null) : null;

  const statRows: [string, MapStats | undefined][] = view ? [
    ["Label (grid_gt)", view.st.gt], ["Arm O prediction", view.st.O], ["Arm U prediction", view.st.U],
    ["|error| Arm O", view.st.errO], ["|error| Arm U", view.st.errU], ["Raw H overflow", view.st.H], ["Raw V overflow", view.st.V],
  ] : [];

  return (
    <>
      <PageHead eyebrow="Results" title="Prediction maps" scope="VALIDATION"
        lede={<>Grid-level predictions of both arms on one validation graph, at the end of training (step {plan.steps_per_run}), next to the label and the
          absolute error. Every map is the exported float32 array at its native gcell resolution: nothing is resampled or smoothed.</>} />

      <Card>
        <div className="map-controls">
          <label className="field">
            <span className="field-label">Validation graph</span>
            <select className="select" value={sel ?? ""} onChange={(e) => setSample(e.target.value)} aria-label="Validation graph">
              {["official validation design", "held-out"].map((g) => {
                const opts = samples.filter((s) => (g === "held-out") !== (mi.data![s]!.role === "official validation design"));
                return opts.length ? (
                  <optgroup key={g} label={g === "held-out" ? "Held-out graphs of training designs" : "Official validation design"}>
                    {opts.map((s) => <option key={s} value={s}>{sampleLabel(s, mi.data![s]!)}</option>)}
                  </optgroup>
                ) : null;
              })}
            </select>
          </label>
          <div className="field">
            <span className="field-label">Seed (paired runs)</span>
            {selSeed !== null && <SeedPicker seeds={seeds} value={selSeed} onChange={(v) => v !== "all" && setSeed(v)} allowAll={false} />}
          </div>
          <div className="field">
            <span className="field-label">Colour scale</span>
            <div className="seg" role="group" aria-label="Colour scale">
              <button aria-pressed={scale === "shared"} onClick={() => setScale("shared")}>Shared (comparable)</button>
              <button aria-pressed={scale === "panel"} onClick={() => setScale("panel")}>Per panel (pattern only)</button>
            </div>
          </div>
          <div className="field">
            <span className="field-label">Colour mapping</span>
            <div className="seg" role="group" aria-label="Colour mapping">
              <button aria-pressed={transform === "linear"} onClick={() => setTransform("linear")}>Linear</button>
              <button aria-pressed={transform === "sqrt"} onClick={() => setTransform("sqrt")}>Square root</button>
            </div>
          </div>
        </div>
        {entry && <p className="chart-caption" data-testid="map-sample-meta">
          {entry.design} · {entry.role} · grid {entry.shape[0]} × {entry.shape[1]} gcells ({(entry.shape[0] * entry.shape[1]).toLocaleString("en-US")} tiles) ·
          runs {runO?.name ?? "missing"} and {runU?.name ?? "missing"}
        </p>}
        {scale === "panel" && <p className="chart-caption" data-testid="scale-warning"><strong>Per-panel scales:</strong> each map is stretched to its own maximum, so
          brightness cannot be compared between panels. Switch back to the shared scale to compare values.</p>}
        {transform === "sqrt" && <p className="chart-caption" data-testid="sqrt-note"><strong>Square-root colour mapping:</strong> small values are made visible;
          the colour bar is labelled in the data's own units and hovering shows the exact value. The data are unchanged.</p>}
      </Card>

      {maps.loading && <div className="section"><Loading what="the maps of this graph" /></div>}
      {mapError && <div className="section blocking-banner" role="alert" data-testid="maps-error">These maps could not be shown: {mapError}</div>}
      {!maps.loading && !mapError && maps.data && (!maps.data.O || !maps.data.U) && (
        <div className="section blocking-banner" role="alert" data-testid="maps-error">Seed {selSeed} has no complete Arm O / Arm U pair in this bundle, so no maps are compared.</div>
      )}

      {view && !maps.loading && (
        <>
          <div className="section map-grid map-grid--3">
            <Card title="Label" meta="grid_gt · normalised">
              <MapPanel transform={transform} z={view.img.gt} zmax={zShared(view.st.gt)} label={`Label map of ${sel}`} colorTitle="label" themeKey={`${themeKey}-${sel}-${scale}-${transform}`} />
            </Card>
            <Card title={<><ArmName arm="O" short /> prediction</>} meta={<DeadTag s={view.st.O} />}>
              <MapPanel transform={transform} z={view.img.O} zmax={zShared(view.st.O)} label={`Arm O prediction, seed ${selSeed}`} colorTitle="prediction" themeKey={`${themeKey}-${sel}-${selSeed}-${scale}-${transform}`} />
            </Card>
            <Card title={<><ArmName arm="U" short /> prediction</>} meta={<DeadTag s={view.st.U} />}>
              <MapPanel transform={transform} z={view.img.U} zmax={zShared(view.st.U)} label={`Arm U prediction, seed ${selSeed}`} colorTitle="prediction" themeKey={`${themeKey}-${sel}-${selSeed}-${scale}-${transform}`} />
            </Card>
          </div>
          <p className="chart-caption">{scale === "shared"
            ? <>Label and both predictions share one scale, from zero to {fmtNum(view.valueMax, 3) ?? "undefined"} (the largest of the three maps).</>
            : <>Each panel uses its own maximum.</>} Dark = high in light mode, bright = high in dark mode.</p>

          <div className="section map-grid map-grid--2">
            <Card title={<>|error| · <ArmName arm="O" short /></>} meta="|prediction − label|">
              <MapPanel transform={transform} z={view.img.errO} zmax={zErr(view.st.errO)} label={`Absolute error, Arm O, seed ${selSeed}`} colorTitle="|error|" themeKey={`${themeKey}-${sel}-${selSeed}-${scale}-${transform}`} />
            </Card>
            <Card title={<>|error| · <ArmName arm="U" short /></>} meta="|prediction − label|">
              <MapPanel transform={transform} z={view.img.errU} zmax={zErr(view.st.errU)} label={`Absolute error, Arm U, seed ${selSeed}`} colorTitle="|error|" themeKey={`${themeKey}-${sel}-${selSeed}-${scale}-${transform}`} />
            </Card>
          </div>
          <p className="chart-caption">The error maps are computed in this browser as |prediction − label| in float32 (the exporter checked this
            to be bit-exact against the experiment's stored error map). {scale === "shared" ? <>Both share one scale, from zero to {fmtNum(view.errMax, 3) ?? "undefined"}.</> : null} Where
            a prediction map is all zero, its error map equals the label.</p>

          <div className="section map-grid map-grid--2">
            <Card title="Raw horizontal overflow" meta="raw_h_overflow">
              <MapPanel transform={transform} z={view.img.H} zmax={zRaw(view.st.H)} label={`Raw horizontal overflow of ${sel}`} colorTitle="H" themeKey={`${themeKey}-${sel}-${scale}-${transform}`} />
            </Card>
            <Card title="Raw vertical overflow" meta="raw_v_overflow">
              <MapPanel transform={transform} z={view.img.V} zmax={zRaw(view.st.V)} label={`Raw vertical overflow of ${sel}`} colorTitle="V" themeKey={`${themeKey}-${sel}-${scale}-${transform}`} />
            </Card>
          </div>
          <p className="chart-caption">Horizontal and vertical routing overflow are kept separate, as exported{scale === "shared" ? <>; they share one scale, from zero to {fmtNum(view.rawMax, 3) ?? "undefined"}</> : null}.
            The label combines them as <code className="mono">{LABEL_FORMULA}</code> with c_max = {fmtNum(plan.c_max, 4)} (training split).</p>

          <div className="section grid-2">
            <Card title="Map summary" meta="computed in this browser from the exported arrays">
              <div className="table-scroll">
                <table className="table" aria-label="Map summary">
                  <thead><tr><th>Map</th><th className="num">min</th><th className="num">max</th><th className="num">mean</th><th className="num">zero fraction</th></tr></thead>
                  <tbody>{statRows.map(([name, s]) => (
                    <tr key={name} data-testid={`mapstat-${name}`}>
                      <td>{name}</td><td className="num"><Value v={s?.min ?? null} digits={4} /></td><td className="num"><Value v={s?.max ?? null} digits={4} /></td>
                      <td className="num"><Value v={s?.mean ?? null} digits={4} /></td><td className="num"><Value v={s?.zeroFrac ?? null} digits={4} /></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </Card>
            <Card title="Consistency checks" meta="this graph and seed">
              <ul className="checklist">
                <li data-testid="check-label">
                  <Status tone={view.maxDiff <= LABEL_TOL ? "good" : "warn"}>{view.maxDiff <= LABEL_TOL ? "consistent" : "differs"}</Status>
                  <span>Label recomputed from raw H and V in float32: largest difference {fmtSig(view.maxDiff, 2) ?? "undefined"} (tolerance {LABEL_TOL.toExponential(0)}, float32 rounding).</span>
                </li>
                <li data-testid="check-cells">
                  <Status tone={view.cellMismatch === 0 ? "good" : "crit"}>{view.cellMismatch === 0 ? "consistent" : "mismatch"}</Status>
                  <span>Cell labels equal the label of the cell's tile for {(view.nCells - view.cellMismatch).toLocaleString("en-US")} of {view.nCells.toLocaleString("en-US")} subsampled cells.</span>
                </li>
                {[runO, runU].map((r) => {
                  const c = cmp(r);
                  const z = c?.["pooled_zero_frac"];
                  return r && z ? (
                    <li key={r.name}>
                      <Status tone={isDefined(z.abs_diff) && z.abs_diff <= 1e-4 ? "good" : "info"}>exporter</Status>
                      <span>{r.name}: grid zero fraction over all {plan.val.length} validation graphs, maps <Value v={z.npz} digits={4} /> vs
                        validation log <Value v={z.evals_csv} digits={4} /> (recomputed by the exporter).</span>
                    </li>
                  ) : null;
                })}
              </ul>
            </Card>
          </div>

          <div className="section">
            <Card title="Cell head · subsample of this graph" meta={`${view.nCells.toLocaleString("en-US")}-cell subsample · same cells in both arms`}>
              <div className="map-grid map-grid--2">
                {(["O", "U"] as const).map((arm) => (
                  <figure key={arm} style={{ margin: 0 }}>
                    <div className="panel-head"><ArmName arm={arm} /> <DeadTag s={view.cell[arm]} /></div>
                    <DensityPanel z={arm === "O" ? view.cell.dO.z : view.cell.dU.z} edges={view.cell.dO.edges}
                      label={`Cell prediction vs label, Arm ${arm}, seed ${selSeed}`} xTitle="cell label (its tile's label)" yTitle="cell prediction"
                      themeKey={`${themeKey}-${sel}-${selSeed}`} />
                    <p className="chart-caption" data-testid={`cell-${arm}`}>Zero fraction <Value v={view.cell[arm].zeroFrac} digits={4} /> · mean |error| <Value v={arm === "O" ? view.cell.errO : view.cell.errU} digits={4} /></p>
                  </figure>
                ))}
              </div>
              <p className="chart-caption">Counts per bin on shared axes; the dotted line is prediction = label. Only the {view.nCells.toLocaleString("en-US")} cells saved by the
                experiment are available, so there is no full-cell map: the cell head is shown as this distribution, never as an image.</p>
            </Card>
          </div>

          <div className="section grid-2">
            <Evidence kind="observed" title="What these maps show">
              <ul className="bullets">
                <li>One validation graph and one seed at a time, from the final evaluation pass of each run.</li>
                <li>The two arms of a seed share initial weights and data order, so the maps form a pair.</li>
                <li>"DEAD" marks a prediction map whose values are all exactly zero.</li>
              </ul>
            </Evidence>
            <Evidence kind="limitation" title="What they do not show">
              <ul className="bullets">
                <li>No test-set graph: these are diagnostic validation graphs.</li>
                <li>A single graph is one placement; pooled metrics are on the Validation and Per-design pages.</li>
                <li>Retest runs ({rf.report.results.filter((r) => r.retest).map((r) => r.name).join(", ") || "none"}) are not shown here; they are excluded from all comparisons.</li>
              </ul>
            </Evidence>
          </div>
        </>
      )}
    </>
  );
}
