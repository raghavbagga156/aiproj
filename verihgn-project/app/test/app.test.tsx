import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { render, screen, waitFor, within } from "@testing-library/react";
import ts from "typescript";
import { describe, expect, it } from "vitest";
import { App } from "../src/App";
import { fmtSig } from "../src/data/values";
import { FixtureBundle, fixtureMeta } from "./fixture";
import { plotlyCalls } from "./setup";

// Values of the REAL experiment that must never appear when the synthetic bundle is loaded: if one did, it would
// have been hard-coded. (Report sha256 and test outcome of the real HEADPATH report; documented in
// docs/DASHBOARD_FRONTEND_PLAN.md section 1.3.)
const REAL_REPORT_SHA_PREFIX = "ce485b2569b5d355";
const REAL_P_TEXT = "p = 0.5";

async function renderAt(hash: string, fb = new FixtureBundle()) {
  window.location.hash = hash;
  const utils = render(<App client={fb.client()} />);
  return { ...utils, fb };
}

describe("synthetic fixture is independent of the real experiment", () => {
  it("has different headline values", () => {
    const fb = new FixtureBundle();
    const src = fb.json("manifest.json").sources["reports/08_headpath_experiment.json"];
    expect(src.sha256.startsWith(REAL_REPORT_SHA_PREFIX)).toBe(false);
    expect(fixtureMeta.primary_p).not.toBe(0.5);
  });
});

describe("Overview renders only bundle values", () => {
  it("shows the fixture's test result, scope and verbatim conclusion", async () => {
    const { fb } = await renderAt("#/overview");
    const a = fb.json("headpath/report.json").report.analysis;
    const plan = fb.json("headpath/report.json").report.plan;
    await screen.findByText("Primary result");
    expect(screen.getByTestId("fact-p").textContent).toBe(`p = ${fmtSig(a.primary_test.p_two_sided_exact, 4)}`);
    expect(screen.getByTestId("fact-discordant").textContent).toBe(String(a.primary_test.n_discordant));
    expect(screen.getByTestId("fact-pairs").textContent).toBe(String(a.primary_test.n_valid_pairs));
    expect(screen.getByTestId("conclusion").textContent).toBe(a.conclusion);
    expect(screen.getByTestId("deviation").textContent).toBe(a.deviation_notice);
    expect(screen.getByText(`Validation only · ${plan.val.length} diagnostic graphs`)).toBeTruthy();
    expect(screen.getByText("No test-set results")).toBeTruthy();
    expect(document.body.textContent).not.toContain(REAL_P_TEXT);
    expect(document.body.textContent).not.toContain(REAL_REPORT_SHA_PREFIX);
  });

  it("pairs come from the primary test; retest runs never enter the matrix", async () => {
    const fb = new FixtureBundle();
    const rep = fb.json("headpath/report.json");
    const retest = rep.report.results.find((r: any) => r.retest);
    retest.category = "INVALID";                                   // a retest outcome must not reach the matrix
    fb.setJson("headpath/report.json", rep);
    await renderAt("#/overview", fb);
    await screen.findByText("Primary result");
    const seeds = rep.report.analysis.primary_test.pairs.map((p: any) => p.seed);
    for (const s of seeds) {
      for (const arm of ["O", "U"]) {
        const pair = rep.report.analysis.primary_test.pairs.find((p: any) => p.seed === s);
        expect(within(screen.getByTestId(`cell-${arm}-${s}`)).getByText(/./).textContent).toBeTruthy();
        expect(pair[arm]).not.toBe(undefined);
      }
    }
    const matrix = screen.getByRole("table", { name: "Primary endpoint by seed and arm" });
    expect(matrix.textContent).not.toContain("Invalid run");
  });

  it("undefined values stay undefined (never plotted or printed as 0)", async () => {
    const fb = new FixtureBundle();
    const rep = fb.json("headpath/report.json");
    const target = rep.report.results.find((r: any) => !r.retest && r.arm === "U");
    target.P3.final_val_cell_spearman_indep = null;
    fb.setJson("headpath/report.json", rep);
    plotlyCalls.length = 0;
    await renderAt("#/overview", fb);
    await screen.findByText(new RegExp(`Undefined .* for .*U·${target.seed}`));
    await waitFor(() => expect(plotlyCalls.length).toBeGreaterThan(0));
    const p3 = plotlyCalls.find((c) => (c.layout as any).xaxis?.title?.text === "pooled validation cell Spearman ρ");
    expect(p3).toBeTruthy();
    const uTrace = (p3!.data as any[]).find((t) => t.name === "Arm U");
    const i = uTrace.customdata.findIndex((d: any[]) => d[0] === target.seed);
    expect(uTrace.x[i]).toBeNull();                                // a gap, not 0
    expect(uTrace.connectgaps).toBe(false);
  });
});

describe("Provenance", () => {
  it("shows the launcher session, manifest and check counts from the bundle", async () => {
    const { fb } = await renderAt("#/provenance");
    await screen.findByText("Bundle identity");
    const integ = fb.json("integrity_report.json");
    expect(screen.getByText(fixtureMeta.zip_sha256)).toBeTruthy();
    expect(screen.getAllByText(`${integ.summary.PASS} PASS`).length).toBeGreaterThan(0);
    expect(screen.getByText(`${integ.checks.length} checks`)).toBeTruthy();
  });
});

describe("integrity failures are blocking", () => {
  it("a tampered core file stops the app before any result is shown", async () => {
    const fb = new FixtureBundle();
    const b = new Uint8Array(fb.files.get("headpath/report.json")!);
    b[20] = b[20]! ^ 1;
    fb.setBytes("headpath/report.json", b, false);
    await renderAt("#/overview", fb);
    expect((await screen.findByTestId("gate-error")).textContent).toMatch(/sha256/);
    expect(screen.queryByText("Primary result")).toBeNull();
  });

  it("a FAILED manifest withholds results; provenance stays available", async () => {
    const fb = new FixtureBundle();
    const man = fb.json("manifest.json");
    man.status = "FAILED";
    fb.files.set("manifest.json", new TextEncoder().encode(JSON.stringify(man)));
    await renderAt("#/overview", fb);
    expect((await screen.findByTestId("integrity-block")).textContent).toMatch(/manifest status FAILED/);
    expect(screen.queryByText("Primary result")).toBeNull();
    window.location.hash = "#/provenance";
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    await screen.findByText("Bundle identity");
    expect(screen.getByRole("alert").textContent).toMatch(/Integrity failure/);
  });

  it("a failed launcher verification withholds results", async () => {
    const fb = new FixtureBundle();
    fb.session = { ...(fb.session as object), verified: false,
                   verification: { status: "FAILED", errors: ["x.json: size/sha256 differs"], files_checked: 1, arrays_checked: 0, json_strict: 0 } };
    await renderAt("#/overview", fb);
    expect((await screen.findByTestId("integrity-block")).textContent).toMatch(/launcher verification FAILED/);
  });
});

describe("no hard-coded results in the UI source", () => {
  // JSX text may name criteria and versions but must not contain numbers: every number on a page comes
  // from the bundle through an expression.
  const ALLOWED = /\b(P[1-3]|M[1-3]|A[1-6]|sha256|int32|float32|Notebook-07|notebook-07|Arm-[OU]|FP16|FP32|TF32)\b|\bnotebook 0[78]\b/g;
  const files: string[] = [];
  const walk = (d: string) => readdirSync(d).forEach((f) => {
    const p = join(d, f);
    if (statSync(p).isDirectory()) walk(p); else if (p.endsWith(".tsx")) files.push(p);
  });
  walk(resolve(__dirname, "../src"));
  const jsxTexts = (f: string): string[] => {
    const sf = ts.createSourceFile(f, readFileSync(f, "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    const out: string[] = [];
    const visit = (n: ts.Node) => {
      if (ts.isJsxText(n) && n.getText().trim()) out.push(n.getText());
      if (ts.isJsxAttribute(n) && n.initializer && ts.isStringLiteral(n.initializer)
          && ["title", "aria-label", "label", "placeholder", "meta"].includes(n.name.getText())) out.push(n.initializer.text);
      ts.forEachChild(n, visit);
    };
    visit(sf);
    return out;
  };
  it("the scanner sees JSX text", () => { expect(files.length).toBeGreaterThan(3); });
  it.each(files.map((f) => [f.slice(f.indexOf("src")), f]))("%s has no numeric literal text", (_name, f) => {
    const offending = jsxTexts(f).map((t) => t.replace(ALLOWED, "")).filter((t) => /\d/.test(t));
    expect(offending).toEqual([]);
  });
});
