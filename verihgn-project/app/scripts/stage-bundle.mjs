// Static deployment (e.g. Vercel): stage the ACTUAL export bundle under dist/data/, the path the dashboard reads
// at runtime (./data/manifest.json, ./data/<file> ...), after verifying it exactly like tools/run_dashboard.py does.
//
//   node scripts/stage-bundle.mjs [--out dist/data] [--lock deploy/bundle.lock.json]
//
// Fail-closed: any check that fails aborts with exit code 1 and leaves no data in the output, so a deployment
// built from an unverified bundle shows no results (the app's own gate reports the missing manifest).
// Checks: the ZIP's sha256 and size equal the pinned values in deploy/bundle.lock.json (which covers every byte,
// so zip CRCs add nothing); one top-level folder; every member listed in manifest.json with its size and sha256 and nothing unlisted;
// every .json is strict JSON (no NaN / Infinity); every array decodes (gzip -> byte un-shuffle) to its recorded
// sha256_raw; manifest and integrity report status PASSED; not the synthetic test fixture.
// The ZIP is only read. The browser re-checks every file and array again on load (src/data/client.ts).
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync, unzipSync } from "fflate";

export const STAGER_VERSION = "1.0.0";
const APP = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const sha256 = (b) => createHash("sha256").update(b).digest("hex");
const dec = new TextDecoder();

/** Strict JSON: JSON.parse already rejects NaN / Infinity literals. */
const strictJson = (b) => JSON.parse(dec.decode(b));

function unshuffle(planes, itemsize) {
  if (planes.length % itemsize !== 0) throw new Error("byte length is not a multiple of itemsize");
  const n = planes.length / itemsize, out = new Uint8Array(planes.length);
  for (let b = 0; b < itemsize; b++) for (let i = 0; i < n; i++) out[i * itemsize + b] = planes[b * n + i];
  return out;
}

/** Same checks as tools/run_dashboard.py verify_zip, on the in-memory ZIP. Returns { status, errors, files, top, ... }. */
export function verifyBundle(zipBytes) {
  const errors = [];
  let members;
  try { members = unzipSync(zipBytes); } catch (e) { return { status: "FAILED", errors: [`not a readable zip (${e.message})`] }; }
  const names = Object.keys(members).filter((n) => !n.endsWith("/"));
  const tops = new Set(names.map((n) => n.split("/")[0]));
  if (tops.size !== 1) return { status: "FAILED", errors: [`expected one top-level folder, found ${[...tops].slice(0, 5)}`] };
  const top = [...tops][0];
  const read = (rel) => members[`${top}/${rel}`];
  let man;
  try { man = strictJson(read("manifest.json")); } catch (e) { return { status: "FAILED", errors: [`manifest.json: ${e.message}`] }; }
  const present = new Set(names.map((n) => n.slice(top.length + 1)));
  const listed = new Set([...Object.keys(man.files), "manifest.json"]);
  const diff = [...present].filter((p) => !listed.has(p)).concat([...listed].filter((p) => !present.has(p)));
  if (diff.length) errors.push(`unlisted or missing files: ${diff.slice(0, 10)}`);
  const metas = {};
  try {
    for (const r of Object.values(strictJson(read("headpath/runs/index.json")))) for (const arrs of Object.values(r.maps)) for (const m of Object.values(arrs)) metas[m.file] = m;
    for (const e of Object.values(strictJson(read("headpath/maps/index.json")))) for (const m of Object.values(e.arrays)) metas[m.file] = m;
  } catch (e) { errors.push(`array index: ${e.message}`); }
  let nFiles = 0, nArr = 0, nJson = 0;
  for (const [rel, info] of Object.entries(man.files)) {
    const data = read(rel);
    if (!data) continue;
    nFiles++;
    if (data.length !== info.size || sha256(data) !== info.sha256) { errors.push(`${rel}: size/sha256 differs from manifest.json`); continue; }
    if (rel.endsWith(".json")) { try { strictJson(data); nJson++; } catch (e) { errors.push(`${rel}: not strict JSON (${e.message})`); } }
    if (rel.endsWith(".bin.gz")) {
      const m = metas[rel];
      if (!m) { errors.push(`${rel}: array not described in the run/map index`); continue; }
      try {
        const raw = unshuffle(gunzipSync(data), m.itemsize);
        if (raw.length !== m.nbytes_raw || sha256(raw) !== m.sha256_raw) errors.push(`${rel}: decoded array differs from its sha256_raw`);
        else nArr++;
      } catch (e) { errors.push(`${rel}: cannot decode (${e.message})`); }
    }
  }
  try {
    const integ = strictJson(read("integrity_report.json"));
    if (man.status !== "PASSED") errors.push(`manifest status ${man.status}`);
    if (integ.status !== "PASSED") errors.push(`integrity report status ${integ.status}`);
  } catch (e) { errors.push(`integrity_report.json: ${e.message}`); }
  return { status: errors.length ? "FAILED" : "PASSED", errors, files_checked: nFiles, arrays_checked: nArr, json_strict: nJson,
           top, members, manifest: man };
}

function fail(msg, out) {
  console.error(`\nSTAGING FAILED: ${msg}\nNo bundle data was written; the deployment would show no results.`);
  if (out && existsSync(out)) rmSync(out, { recursive: true, force: true });       // our own output folder only
  process.exit(1);
}

function main() {
  const lockPath = resolve(APP, arg("--lock", "deploy/bundle.lock.json"));
  const out = resolve(APP, arg("--out", "dist/data"));
  if (!existsSync(resolve(out, "..", "index.html"))) fail(`the production build is missing (${resolve(out, "..", "index.html")}); run "npm run build" first`);
  if (existsSync(out)) fail(`${out} already exists; refusing to overwrite (the build empties dist/ first)`);
  const lock = JSON.parse(readFileSync(lockPath, "utf8"));
  const zipPath = resolve(dirname(lockPath), lock.zip);
  if (!existsSync(zipPath)) fail(`export ZIP not found: ${zipPath}`);
  const zip = new Uint8Array(readFileSync(zipPath));
  const zsha = sha256(zip);
  console.log(`export: ${lock.zip} (${zip.length} bytes)\nsha256: ${zsha}`);
  if (zip.length !== lock.bytes || zsha !== lock.sha256) fail(`the ZIP does not match deploy/bundle.lock.json (pinned ${lock.sha256}, ${lock.bytes} bytes)`);
  // never the synthetic test fixture
  const fixtureMeta = join(APP, "test-fixtures", "fixture_meta.json");
  if (existsSync(fixtureMeta) && JSON.parse(readFileSync(fixtureMeta, "utf8")).zip_sha256 === zsha) fail("this is the synthetic test fixture, not the experiment export");
  const t0 = Date.now();
  const v = verifyBundle(zip);
  console.log(`verification ${v.status} in ${((Date.now() - t0) / 1000).toFixed(1)}s: ${v.files_checked ?? 0} files, ${v.arrays_checked ?? 0} arrays, ${v.json_strict ?? 0} JSON files`);
  if (v.status !== "PASSED") fail(v.errors.slice(0, 20).join("\n  "));
  if (/synthetic/i.test(v.top) || /synthetic/i.test(v.manifest.experiment ?? "")) fail(`bundle "${v.top}" looks synthetic`);

  // write to a temporary folder, then rename: a half-written data folder is never left behind
  const tmp = `${out}.tmp_${process.pid}`;
  rmSync(tmp, { recursive: true, force: true });
  for (const [name, data] of Object.entries(v.members)) {
    if (name.endsWith("/")) continue;
    const rel = name.slice(v.top.length + 1);
    const p = resolve(tmp, rel);
    if (!p.startsWith(resolve(tmp))) fail(`unsafe path in zip: ${name}`, tmp);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, data);
  }
  // re-read what was written: every file byte-identical to the verified member
  for (const [rel, info] of Object.entries(v.manifest.files)) {
    const b = readFileSync(resolve(tmp, rel));
    if (b.length !== info.size || sha256(b) !== info.sha256) fail(`${rel}: written file differs from the verified member`, tmp);
  }
  // the launcher's session record, written by this build-time verifier instead (read by the Provenance page)
  writeFileSync(join(tmp, "session.json"), JSON.stringify({
    app: "verihgn-headpath-dashboard", zip_name: lock.zip, zip_path: `app/deploy/${lock.zip} (repository copy)`,
    zip_sha256: zsha, zip_bytes: zip.length, verified: true,
    verification: { status: v.status, errors: [], files_checked: v.files_checked, arrays_checked: v.arrays_checked, json_strict: v.json_strict },
    extract_dir: "static deployment: /data/ (verified and extracted at build time by app/scripts/stage-bundle.mjs)",
    launched_utc: new Date().toISOString().replace(/\.\d+Z$/, "Z"), launcher_version: `static build stager ${STAGER_VERSION}`,
  }));
  renameSync(tmp, out);
  const bytes = Object.values(v.manifest.files).reduce((a, f) => a + f.size, 0);
  console.log(`staged ${Object.keys(v.manifest.files).length + 1} bundle files (${(bytes / 1e6).toFixed(1)} MB) + session.json into ${out}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
