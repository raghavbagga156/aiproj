# Static deployment (Vercel)

## Why the first deployment showed "/data/manifest.json: HTTP 404"

The dashboard reads its data at runtime from `./data/` next to `index.html` (`app/src/data/client.ts`):
`./data/manifest.json`, then every other file listed in it, each checked against the manifest's size and
sha256; arrays are decoded and checked against their `sha256_raw`. Locally, `tools/run_dashboard.py` provides
`/data/` from the verified, extracted export. A plain `vite build` contains no data (by design: data is never
bundled into the app), so the Vercel deployment served the app without `/data/`, and the app's integrity gate
correctly refused to show results.

## How the deployment works now

```
app/deploy/headpath_dashboard_20261022_110005.zip   the actual export (byte-identical copy, 16,675,322 bytes)
app/deploy/bundle.lock.json                         its pinned sha256 and size
npm run build:vercel  =  npm run build  +  node scripts/stage-bundle.mjs
        1. type-check and build the app into dist/ (unchanged)
        2. stage-bundle: refuse unless the ZIP matches the pin; verify it exactly like the launcher
           (every member listed in the manifest with its size and sha256, nothing unlisted, strict JSON,
           every array decoded to its sha256_raw, manifest and integrity report PASSED, not the synthetic
           fixture); extract the individual files to dist/data/; re-read them; write dist/data/session.json,
           the build-time verification record shown on the Provenance page
        any failure -> the build fails, nothing is deployed
```

In the browser nothing changes: every file and array is verified again on load, and any mismatch withholds all
results. The deployed site is static and read-only; it makes no request outside its own origin (CSP `'self'`).

**Vercel limits checked (vercel.com/docs/limits, 2026-09):** build step 45 min (this build takes about 30 s);
no upper limit on build output files from Git deployments (this one has 642 files, 32 MB); the 100 MB / 15,000
file limits apply to CLI source uploads only. Vercel compresses only text-like MIME types
(vercel.com/docs/how-vercel-cdn-works/compression); `.bin.gz` arrays are served as stored, and JSON that is
compressed in transit is decoded by the browser before the hash check.

## Vercel project settings

| Setting | Value |
|---|---|
| Root Directory | `verihgn-project/app` (the Git repository root is the folder above `verihgn-project`) |
| Framework Preset | Other |
| Install / Build / Output | taken from `app/vercel.json`: `npm ci` / `npm run build:vercel` / `dist` (leave the dashboard overrides off) |
| Node.js | 20.x or 22.x |
| Environment variables | none |

## Checking a deployment

```bat
cd "C:\Users\ragha\Desktop\ai for electronics project\verihgn-project\app"
node e2e\check_static.mjs --url https://aiproj-theta.vercel.app --zip deploy\headpath_dashboard_20261022_110005.zip
node e2e\e2e_live.mjs --url https://aiproj-theta.vercel.app --out qa\vercel --no-shots
```

`check_static.mjs`: every manifest file returns 200 and is byte-identical to the ZIP, all arrays decode to their
sha256_raw, the verification record names this ZIP, no synthetic fixture is reachable. `e2e_live.mjs`: all 12
pages at two widths and both themes, navigation, every map graph and seed, caveats visible, no request leaves the
site.

## Replacing the export later

Copy the new ZIP into `app/deploy/`, update `zip`, `sha256` and `bytes` in `app/deploy/bundle.lock.json`, run
`npm run build:vercel` locally (it must print `verification PASSED`), commit, push. Do not edit files inside a ZIP.

## Limits of this setup

- The deployment is public unless Vercel Deployment Protection is enabled for the project.
- The export ZIP lives in the Git history (16.7 MB per export version).
- The browser checks files against the served manifest; the manifest itself is anchored by the build-time check
  against the pinned sha256 and by Vercel's immutable deployments. A host that served a consistently altered
  manifest and files would not be detected by the browser (the same trust model as any static site).
- Data is fetched per page (`cache: "no-store"`), so the hosted site needs a connection to its own origin while
  browsing. Fully offline use is the local launcher (`run_dashboard.bat`).
