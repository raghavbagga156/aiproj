/* scripts/stage-bundle.mjs (static deployments): stages the ACTUAL export into <build>/data/ only after full
   verification, and fails closed otherwise. Uses the real export committed at app/deploy/ (pinned by
   deploy/bundle.lock.json); the synthetic fixture appears only to prove that it is refused. */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { unzipSync, zipSync } from "fflate";
import { afterAll, describe, expect, it } from "vitest";

const APP = resolve(__dirname, "..");
const LOCK = JSON.parse(readFileSync(join(APP, "deploy", "bundle.lock.json"), "utf8")) as { zip: string; sha256: string; bytes: number };
const REAL = join(APP, "deploy", LOCK.zip);
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const tmps: string[] = [];
afterAll(() => tmps.forEach((d) => rmSync(d, { recursive: true, force: true })));

/** A scratch site (index.html present) and a lock pointing at `zipPath`. */
function scratch(zipPath: string, pin: { sha256: string; bytes: number }) {
  const d = mkdtempSync(join(tmpdir(), "stage-test-"));
  tmps.push(d);
  mkdirSync(join(d, "site"));
  writeFileSync(join(d, "site", "index.html"), "<!doctype html>");
  writeFileSync(join(d, "lock.json"), JSON.stringify({ ...pin, zip: zipPath }));
  return d;
}
const stage = (d: string) => spawnSync(process.execPath, [join(APP, "scripts", "stage-bundle.mjs"), "--lock", join(d, "lock.json"), "--out", join(d, "site", "data")],
  { cwd: APP, encoding: "utf8" });
const walk = (d: string, base = d): string[] => readdirSync(d).flatMap((f) => {
  const p = join(d, f);
  return statSync(p).isDirectory() ? walk(p, base) : [p.slice(base.length + 1).replace(/\\/g, "/")];
});

describe.skipIf(!existsSync(REAL))("build-time staging of the export for static hosting", () => {
  it("the committed export matches its pin", () => {
    const b = new Uint8Array(readFileSync(REAL));
    expect(b.length).toBe(LOCK.bytes);
    expect(sha(b)).toBe(LOCK.sha256);
  });

  it("stages every manifest file byte-identically, plus the verification record", () => {
    const d = scratch(REAL, LOCK);
    const r = stage(d);
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toMatch(/verification PASSED/);
    const out = join(d, "site", "data");
    const members = unzipSync(new Uint8Array(readFileSync(REAL)));
    const top = Object.keys(members)[0]!.split("/")[0]!;
    const man = JSON.parse(new TextDecoder().decode(members[`${top}/manifest.json`]));
    const files = walk(out).sort();
    expect(files).toEqual(["manifest.json", "session.json", ...Object.keys(man.files)].sort());
    for (const rel of ["manifest.json", ...Object.keys(man.files)]) expect(sha(readFileSync(join(out, rel)))).toBe(sha(members[`${top}/${rel}`]!));
    const s = JSON.parse(readFileSync(join(out, "session.json"), "utf8"));
    expect(s).toMatchObject({ zip_sha256: LOCK.sha256, verified: true, verification: { status: "PASSED", errors: [] } });
    expect(s.verification.arrays_checked).toBe(Object.keys(man.files).filter((f) => f.endsWith(".bin.gz")).length);
  }, 120_000);

  it("refuses a ZIP that does not match the pin, and writes nothing", () => {
    const d = scratch(REAL, { sha256: "0".repeat(64), bytes: LOCK.bytes });
    const r = stage(d);
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/does not match deploy\/bundle.lock.json/);
    expect(existsSync(join(d, "site", "data"))).toBe(false);
  });

  it("refuses the synthetic test fixture even when it is pinned", () => {
    const fx = join(APP, "test-fixtures", "synthetic_bundle.zip");
    const b = new Uint8Array(readFileSync(fx));
    const d = scratch(fx, { sha256: sha(b), bytes: b.length });
    const r = stage(d);
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/synthetic/);
    expect(existsSync(join(d, "site", "data"))).toBe(false);
  });

  it("fails verification when a member was altered, even if the altered ZIP is re-pinned", () => {
    const members = unzipSync(new Uint8Array(readFileSync(REAL)));
    const name = Object.keys(members).find((n) => n.endsWith("/headpath/report.json"))!;
    const b = new Uint8Array(members[name]!); b[b.length - 2] = b[b.length - 2] === 32 ? 10 : 32;
    members[name] = b;
    const d0 = mkdtempSync(join(tmpdir(), "stage-zip-")); tmps.push(d0);
    const zp = join(d0, "altered.zip");
    const z = zipSync(members);
    writeFileSync(zp, z);
    const d = scratch(zp, { sha256: sha(z), bytes: z.length });
    const r = stage(d);
    expect(r.status).toBe(1);
    expect(r.stderr).toMatch(/headpath\/report.json: size\/sha256 differs from manifest.json/);
    expect(existsSync(join(d, "site", "data"))).toBe(false);
  }, 120_000);

  it("never overwrites an existing data folder and needs the built app", () => {
    const d = scratch(REAL, LOCK);
    mkdirSync(join(d, "site", "data"));
    expect(stage(d).stderr).toMatch(/refusing to overwrite/);
    rmSync(join(d, "site", "index.html"));
    expect(stage(d).stderr).toMatch(/production build is missing/);
  });
});
