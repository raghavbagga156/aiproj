import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { decodeArray, rawBytes } from "../src/data/arrays";
import { BundleIntegrityError } from "../src/data/client";
import { integrityState, minimumAttainableP, primaryTestSentence } from "../src/data/derive";
import type { ArrayMeta } from "../src/data/schema";
import { definedCount, fmtNum, fmtSig, toPlotSeries } from "../src/data/values";
import { FIXTURE_DIR, FixtureBundle, fixtureMeta } from "./fixture";

const hex = (b: Uint8Array) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");

describe("array decoder (golden vectors from the Python encoder)", () => {
  const golden = JSON.parse(readFileSync(resolve(FIXTURE_DIR, "golden_arrays.json"), "utf8")) as
    Record<string, { blob_b64: string; meta: ArrayMeta; raw_hex: string }>;
  for (const [name, g] of Object.entries(golden)) {
    it(`decodes ${name} bit-exactly`, async () => {
      const blob = Uint8Array.from(Buffer.from(g.blob_b64, "base64"));
      const a = await decodeArray(blob, g.meta);
      expect(hex(rawBytes(a.data))).toBe(g.raw_hex);            // NaN, -0.0, subnormal, int32 extremes preserved
      expect(a.shape).toEqual(g.meta.shape);
    });
  }
  it("rejects a corrupted array", async () => {
    const g = golden["float_ramp"]!;
    const blob = Uint8Array.from(Buffer.from(g.blob_b64, "base64"));
    await expect(decodeArray(blob, { ...g.meta, sha256_raw: "0".repeat(64) })).rejects.toThrow(/sha256/);
  });
});

describe("undefined values are never zero", () => {
  it("formats and plots null as undefined / gap", () => {
    expect(fmtNum(null)).toBeNull();
    expect(fmtNum(0)).toBe("0.000");
    expect(fmtSig(null)).toBeNull();
    expect(fmtSig(0.03125, 4)).toBe("0.03125");
    expect(toPlotSeries([0.2, null, 0, Number.NaN])).toEqual([0.2, null, 0, null]);
    expect(definedCount([0, null, 1])).toEqual({ defined: 2, undefined: 1 });
  });
});

describe("bundle client verification", () => {
  it("loads and verifies the synthetic bundle", async () => {
    const fb = new FixtureBundle();
    const c = fb.client();
    const rep = await c.json<any>("headpath/report.json");
    expect(rep.report.analysis.primary_test.p_two_sided_exact).toBe(fixtureMeta.primary_p);
    expect(c.snapshot().failures).toEqual([]);
  });
  it("rejects a file whose bytes differ from the manifest (tampering)", async () => {
    const fb = new FixtureBundle();
    const b = new Uint8Array(fb.files.get("headpath/analysis.json")!);
    b[10] = b[10]! ^ 1;
    fb.setBytes("headpath/analysis.json", b, false);
    const c = fb.client();
    await expect(c.json("headpath/analysis.json")).rejects.toThrow(BundleIntegrityError);
    expect(c.snapshot().failures[0]!.reason).toMatch(/sha256/);
  });
  it("rejects non-strict JSON even when the hash matches", async () => {
    const fb = new FixtureBundle();
    fb.setBytes("headpath/analysis.json", new TextEncoder().encode('{"x": NaN}'), true);
    await expect(fb.client().json("headpath/analysis.json")).rejects.toThrow(/strict JSON/);
  });
  it("rejects files not listed in the manifest", async () => {
    const fb = new FixtureBundle();
    fb.files.set("extra.json", new TextEncoder().encode("{}"));
    await expect(fb.client().json("extra.json")).rejects.toThrow(/not listed/);
  });
});

describe("derivations come from the data", () => {
  it("builds the test sentence from the fixture", async () => {
    const fb = new FixtureBundle();
    const a = fb.json("headpath/report.json").report.analysis;
    const s = primaryTestSentence(a);
    expect(s).toContain(`${a.primary_test.n_discordant} of ${a.primary_test.n_valid_pairs}`);
    expect(s).toContain(`p = ${fmtSig(a.primary_test.p_two_sided_exact, 4)}`);
  });
  it("minimum attainable p of the exact sign test", () => {
    expect(minimumAttainableP(6)).toBe(0.03125);
    expect(minimumAttainableP(1)).toBe(1);
    expect(minimumAttainableP(0)).toBeNull();
  });
  it("integrity state: any failure is blocking", () => {
    const fb = new FixtureBundle();
    const man = fb.json("manifest.json"), integ = fb.json("integrity_report.json");
    const ok = integrityState(man, integ, fb.session as any, { verifiedFiles: 1, failures: [] });
    expect(ok.ok).toBe(true);
    expect(integrityState({ ...man, status: "FAILED" }, integ, null, { verifiedFiles: 0, failures: [] }).ok).toBe(false);
    expect(integrityState(man, { ...integ, status: "FAILED" }, null, { verifiedFiles: 0, failures: [] }).ok).toBe(false);
    expect(integrityState(man, integ, { ...(fb.session as any), verified: false, verification: { status: "FAILED", errors: ["x"] } },
      { verifiedFiles: 0, failures: [] }).ok).toBe(false);
    expect(integrityState(man, integ, null, { verifiedFiles: 1, failures: [{ file: "a", reason: "b" }] }).ok).toBe(false);
  });
});
