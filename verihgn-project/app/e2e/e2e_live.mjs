// End-to-end check of the RUNNING dashboard (started by run_dashboard.bat / tools/run_dashboard.py) in headless
// Microsoft Edge or Chrome, driven over the DevTools protocol (Node 22+, no dependencies).
//
//   node app/e2e/e2e_live.mjs [--url http://127.0.0.1:8765] [--out <dir>] [--no-shots] [--browser <exe>]
//
// For every page, at desktop (1440 px) and phone (390 px) width, in light and dark theme, it checks that the page
// renders its heading, that the global research caveats are visible, that no integrity block or page error is shown,
// that every chart was drawn, and that the browser logged no exceptions; it saves a full-page screenshot. It then
// exercises navigation (sidebar links), the theme toggle and the prediction-map controls. Exit code 0 = all passed.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const URL_ = arg("--url", "http://127.0.0.1:8765").replace(/\/$/, "");
const OUT = resolve(arg("--out", "app/qa/m3"));
const SHOTS = !process.argv.includes("--no-shots");
const PORT = 9333;
const BROWSERS = [arg("--browser", ""), "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Google/Chrome/Application/chrome.exe"].filter(Boolean);

const PAGES = [
  ["overview", "Does the train-time clamp keep the cell head from learning?"], ["arms", "Arm O vs Arm U"], ["seeds", "Seed-wise paired results"],
  ["training", "Training dynamics"], ["validation", "Validation curves"], ["metrics", "Cell- and grid-level metrics"],
  ["designs", "Per-design comparison"], ["maps", "Prediction maps"], ["collapse", "Collapse investigation (notebook 07)"],
  ["statistics", "Statistics and limitations"], ["methods", "Methodology and caveats"], ["provenance", "Provenance and integrity"],
];
const CAVEATS = ["Validation only", "diagnostic graphs", "No test-set results", "Arm U: diagnostic deviation", "not significant"];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const record = (name, ok, detail = "") => { results.push({ name, ok, detail }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  ${detail}` : ""}`); };

async function main() {
  const exe = BROWSERS.find((b) => existsSync(b));
  if (!exe) throw new Error("no Edge or Chrome found (pass --browser <exe>)");
  const prof = mkdtempSync(join(tmpdir(), "verihgn-e2e-"));
  const br = spawn(exe, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`, "about:blank"], { stdio: "ignore" });
  try {
    let ver = null;
    for (let i = 0; i < 50 && !ver; i++) { try { ver = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); } catch { await sleep(200); } }
    if (!ver) throw new Error("browser did not expose the DevTools endpoint");
    const target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page");
    const cdp = await connect(target.webSocketDebuggerUrl);
    await run(cdp);
    cdp.close();
  } finally {
    br.kill();
    await sleep(500);
    try { rmSync(prof, { recursive: true, force: true }); } catch { /* profile still locked: left in temp */ }
  }
  const failed = results.filter((r) => !r.ok);
  writeFileSync(join(OUT, "e2e_results.json"), JSON.stringify({ url: URL_, when: new Date().toISOString(), passed: results.length - failed.length,
    failed: failed.length, results }, null, 1));
  console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
  process.exit(failed.length ? 1 : 0);
}

function connect(wsUrl) {
  return new Promise((res, rej) => {
    const ws = new WebSocket(wsUrl);
    let id = 0;
    const pending = new Map(), events = [];
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && pending.has(msg.id)) { const p = pending.get(msg.id); pending.delete(msg.id); msg.error ? p.rej(new Error(msg.error.message)) : p.res(msg.result); }
      else if (msg.method) events.push(msg);
    };
    ws.onerror = rej;
    ws.onopen = () => res({
      send: (method, params = {}) => new Promise((r, j) => { const i = ++id; pending.set(i, { res: r, rej: j }); ws.send(JSON.stringify({ id: i, method, params })); }),
      events, close: () => ws.close(),
    });
  });
}

async function run(cdp) {
  mkdirSync(OUT, { recursive: true });
  await cdp.send("Page.enable"); await cdp.send("Runtime.enable"); await cdp.send("Network.enable");
  const evaluate = async (expr) => (await cdp.send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result.value;
  const errorsSince = (n) => cdp.events.slice(n).filter((e) => e.method === "Runtime.exceptionThrown"
    || (e.method === "Runtime.consoleAPICalled" && e.params.type === "error"))
    .map((e) => e.params.exceptionDetails?.exception?.description ?? e.params.exceptionDetails?.text ?? e.params.args?.map((a) => a.value ?? a.description).join(" "));
  // ready = heading present, nothing loading, every chart container drawn (or replaced by its table)
  const waitReady = async (title, ms = 45000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const s = await evaluate(`(() => { const h = document.querySelector("h1"); const charts = [...document.querySelectorAll(".chart")];
        return { h: h ? h.textContent : null, loading: document.querySelectorAll(".loading").length,
                 drawn: charts.filter((c) => c.querySelector(".main-svg")).length, charts: charts.length }; })()`);
      if (s && s.h === title && s.loading === 0 && s.drawn === s.charts) return { ...s, ms: Date.now() - t0 };
      await sleep(250);
    }
    return null;
  };
  let reload = 0;
  const open = async (route, theme) => {
    await cdp.send("Page.navigate", { url: `${URL_}/?r=${++reload}#/${route}?theme=${theme}` });
    await sleep(300);
  };

  for (const [vpName, w, h, mobile] of [["desktop", 1440, 900, false], ["mobile390", 390, 844, true]]) {
    await cdp.send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile });
    for (const theme of ["light", "dark"]) {
      for (const [route, title] of PAGES) {
        const mark = cdp.events.length;
        await open(route, theme);
        const ready = await waitReady(title);
        const name = `${route} · ${vpName} · ${theme}`;
        if (!ready) { record(`${name}: renders`, false, "heading/charts not ready within the timeout"); continue; }
        const st = await evaluate(`(() => { const t = document.body.innerText; return {
          caveats: ${JSON.stringify(CAVEATS)}.filter((c) => !t.includes(c)),
          blocked: !!document.querySelector("[data-testid=integrity-block]"),
          alerts: [...document.querySelectorAll(".blocking-banner")].map((b) => b.textContent.slice(0, 160)),
          themeOk: document.documentElement.dataset.theme === ${JSON.stringify(theme)},
          overflowX: document.documentElement.scrollWidth > window.innerWidth + 1,
          footer: !!document.querySelector("[data-testid=footer-limits]") }; })()`);
        const errs = errorsSince(mark);
        const ok = st.caveats.length === 0 && !st.blocked && st.alerts.length === 0 && st.themeOk && !st.overflowX && st.footer && errs.length === 0;
        record(`${name}: renders`, ok, ok ? `${ready.charts} charts, ${ready.ms} ms` :
          JSON.stringify({ missingCaveats: st.caveats, blocked: st.blocked, alerts: st.alerts, themeOk: st.themeOk, overflowX: st.overflowX, footer: st.footer, errs }));
        if (SHOTS) {
          const m = await cdp.send("Page.getLayoutMetrics");
          const fullH = Math.ceil(m.cssContentSize?.height ?? m.contentSize.height);
          await cdp.send("Emulation.setDeviceMetricsOverride", { width: w, height: Math.min(fullH, 16000), deviceScaleFactor: 1, mobile });
          await sleep(600);
          const shot = await cdp.send("Page.captureScreenshot", { format: "png" });
          writeFileSync(join(OUT, `${route}_${vpName}_${theme}.png`), Buffer.from(shot.data, "base64"));
          await cdp.send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile });
        }
      }
    }
  }

  // ---- interactions (desktop, light)
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  let mark = cdp.events.length;
  await open("overview", "light");
  await waitReady(PAGES[0][1]);
  for (const [route, title] of PAGES) {
    await evaluate(`document.querySelector('a.nav-item[href="#/${route}"]')?.click()`);
    const r = await waitReady(title);
    record(`sidebar link → ${route}`, !!r, r ? `${r.ms} ms` : "page did not render");
  }
  const before = await evaluate("document.documentElement.dataset.theme");
  await evaluate(`document.querySelector('button[aria-label^="Switch to"]').click()`);
  await sleep(400);
  const after = await evaluate("document.documentElement.dataset.theme");
  record("theme toggle", before !== after, `${before} → ${after}`);

  await open("maps", "light");
  await waitReady("Prediction maps");
  const opts = await evaluate(`[...document.querySelectorAll("select.select option")].map((o) => o.value)`);
  record("maps: one option per validation graph", Array.isArray(opts) && opts.length > 0, `${opts?.length} graphs`);
  const timings = [];
  for (const v of opts ?? []) {
    const t0 = Date.now();
    await evaluate(`(() => { const s = document.querySelector("select.select");
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(s, ${JSON.stringify(v)});
      s.dispatchEvent(new Event("change", { bubbles: true })); })()`);
    await sleep(150);
    const r = await waitReady("Prediction maps");
    const meta = await evaluate(`document.querySelector("[data-testid=map-sample-meta]")?.textContent ?? ""`);
    const hash = await evaluate("location.hash");
    const err = await evaluate(`document.querySelector("[data-testid=maps-error]")?.textContent ?? null`);
    const ok = !!r && !err && hash.includes(`sample=${encodeURIComponent(v)}`);
    timings.push(Date.now() - t0);
    record(`maps: ${v}`, ok, ok ? `${Date.now() - t0} ms · ${meta.split("·").slice(0, 3).join("·").trim()}` : `error=${err} hash=${hash}`);
  }
  const seedBtns = await evaluate(`[...document.querySelectorAll('[aria-label="Seed"] button')].map((b) => b.textContent)`);
  for (const s of seedBtns ?? []) {
    await evaluate(`[...document.querySelectorAll('[aria-label="Seed"] button')].find((b) => b.textContent === ${JSON.stringify(s)}).click()`);
    await sleep(150);
    const r = await waitReady("Prediction maps");
    const hash = await evaluate("location.hash");
    record(`maps: seed ${s}`, !!r && hash.includes(`seed=${s}`), r ? `${r.ms} ms` : "not ready");
  }
  await evaluate(`[...document.querySelectorAll('[aria-label="Colour scale"] button')][1].click()`);
  await sleep(300);
  const warn = await evaluate(`!!document.querySelector("[data-testid=scale-warning]")`);
  record("maps: per-panel scale shows its warning", warn);
  const heap = await evaluate("performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null");
  record("maps: JS heap after visiting every graph and seed", heap === null || heap < 600, heap === null ? "not reported" : `${heap} MB`);
  const errs = errorsSince(mark);
  record("no browser errors during interactions", errs.length === 0, errs.slice(0, 3).join(" | "));
  writeFileSync(join(OUT, "map_timings.json"), JSON.stringify({ ms_per_graph: timings, heap_mb: heap }, null, 1));

  // offline: every request of the whole run went to the local server (or was an inline data:/blob: URL)
  const urls = cdp.events.filter((e) => e.method === "Network.requestWillBeSent").map((e) => e.params.request.url);
  const external = [...new Set(urls.filter((u) => !u.startsWith(`${URL_}/`) && !/^(data|blob|about):/.test(u)))];
  record("offline: no request left the local server", external.length === 0, external.length ? external.slice(0, 5).join(" ") : `${urls.length} requests, all local`);
}

main().catch((e) => { console.error(e); process.exit(2); });
