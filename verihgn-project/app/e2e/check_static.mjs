// Checks a STATIC deployment of the dashboard over HTTP (a local static server on the production build, or the
// live Vercel URL) against the actual export ZIP:
//
//   node app/e2e/check_static.mjs --url http://127.0.0.1:8790 --zip <export.zip>
//
// Every file listed in the ZIP's manifest must return HTTP 200 with bytes identical to the ZIP member (after the
// browser-style transfer decoding fetch applies); every array must decode to its sha256_raw; the app shell, the
// build-time verification record and the assets must load; no synthetic fixture may be reachable. Read-only.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { gunzipSync, unzipSync } from "fflate";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg("--url", "http://127.0.0.1:8790").replace(/\/$/, "");
const ZIP = arg("--zip");
if (!ZIP) { console.error("--zip <export.zip> is required"); process.exit(2); }
const sha256 = (b) => createHash("sha256").update(b).digest("hex");
const results = [];
const record = (name, ok, detail = "") => { results.push({ name, ok }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  ${detail}` : ""}`); };

const members = unzipSync(new Uint8Array(readFileSync(ZIP)));
const top = Object.keys(members)[0].split("/")[0];
const man = JSON.parse(new TextDecoder().decode(members[`${top}/manifest.json`]));
const metas = {};
for (const r of Object.values(JSON.parse(new TextDecoder().decode(members[`${top}/headpath/runs/index.json`])))) for (const a of Object.values(r.maps)) for (const m of Object.values(a)) metas[m.file] = m;
for (const e of Object.values(JSON.parse(new TextDecoder().decode(members[`${top}/headpath/maps/index.json`])))) for (const m of Object.values(e.arrays)) metas[m.file] = m;

async function get(path, tries = 4) {
  let r;
  try {
    r = await fetch(`${URL_}/${path}`, { headers: { "Accept-Encoding": "gzip, deflate, br" }, cache: "no-store" });
  } catch (e) {          // a small local test server can refuse bursts of connections; retry only connection errors
    if (tries <= 1) throw e;
    await new Promise((res) => setTimeout(res, 300));
    return get(path, tries - 1);
  }
  return { status: r.status, enc: r.headers.get("content-encoding"), type: r.headers.get("content-type"), body: new Uint8Array(await r.arrayBuffer()) };
}

const shell = await get("");
const html = new TextDecoder().decode(shell.body);
record("app shell (/) returns 200", shell.status === 200, `${shell.status}`);
const assets = [...html.matchAll(/(?:src|href)="\.\/(assets\/[^"]+)"/g)].map((m) => m[1]);
for (const a of assets) { const r = await get(a); record(`asset ${a}`, r.status === 200 && r.body.length > 0, `${r.status}, ${r.body.length} bytes`); }

const sess = await get("data/session.json");
let s = null;
try { s = JSON.parse(new TextDecoder().decode(sess.body)); } catch { /* reported below */ }
record("data/session.json (build-time verification record) returns 200", sess.status === 200 && !!s, `${sess.status}`);
record("verification record names this ZIP and PASSED", !!s && s.zip_sha256 === sha256(readFileSync(ZIP)) && s.verified === true && s.verification.status === "PASSED",
  s ? `${s.zip_name} · ${s.verification.files_checked} files · ${s.verification.arrays_checked} arrays` : "");

let ok200 = 0, same = 0, arrays = 0, encodings = new Map();
const bad = [];
const rels = ["manifest.json", ...Object.keys(man.files)];
for (let i = 0; i < rels.length; i += 6) {
  await Promise.all(rels.slice(i, i + 6).map(async (rel) => {
    const r = await get(`data/${rel.split("/").map(encodeURIComponent).join("/")}`);
    const k = `${rel.split(".").pop()}:${r.enc ?? "identity"}`;
    encodings.set(k, (encodings.get(k) ?? 0) + 1);
    if (r.status !== 200) { bad.push(`${rel}: HTTP ${r.status}`); return; }
    ok200++;
    if (sha256(r.body) !== sha256(members[`${top}/${rel}`])) { bad.push(`${rel}: bytes differ from the ZIP`); return; }
    same++;
    const m = metas[rel];
    if (rel.endsWith(".bin.gz") && m) {
      const planes = gunzipSync(r.body), n = planes.length / m.itemsize, raw = new Uint8Array(planes.length);
      for (let b = 0; b < m.itemsize; b++) for (let j = 0; j < n; j++) raw[j * m.itemsize + b] = planes[b * n + j];
      if (raw.length === m.nbytes_raw && sha256(raw) === m.sha256_raw) arrays++; else bad.push(`${rel}: decoded sha256 mismatch`);
    }
  }));
}
record(`every manifest file returns HTTP 200 (${rels.length})`, ok200 === rels.length, `${ok200}/${rels.length}`);
record("every served file is byte-identical to the ZIP member", same === rels.length, `${same}/${rels.length}`);
const nArr = Object.keys(man.files).filter((f) => f.endsWith(".bin.gz")).length;
record(`every array decodes to its sha256_raw (${nArr})`, arrays === nArr, `${arrays}/${nArr}`);
if (bad.length) console.log(bad.slice(0, 10).join("\n"));
console.log("transfer encodings seen (extension:content-encoding -> files):", Object.fromEntries(encodings));

for (const p of ["test-fixtures/synthetic_bundle.zip", "test-fixtures/fixture_meta.json", "synthetic_bundle.zip", "data/headpath_dashboard_synthetic/manifest.json"]) {
  const r = await get(p);
  const leak = r.status === 200 && !/^\s*<!doctype html/i.test(new TextDecoder().decode(r.body.slice(0, 64)));
  record(`no synthetic fixture at /${p}`, !leak, `${r.status}`);
}
record("manifest names the real experiment", !/synthetic/i.test(man.experiment ?? "") && !/synthetic/i.test(top), man.experiment);

const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
