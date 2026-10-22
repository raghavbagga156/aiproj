/* Loads the SYNTHETIC test bundle (app/test-fixtures/synthetic_bundle.zip, produced by
   tools/make_dashboard_fixture.py from random synthetic designs; never experimental results) and serves it
   to BundleClient through an in-memory fetcher. Tests can mutate files; `resign` updates the manifest hash
   so a change looks like a legitimately different bundle, while leaving it unsigned simulates tampering. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { unzipSync } from "fflate";
import { BundleClient } from "../src/data/client";

export const FIXTURE_DIR = resolve(__dirname, "../test-fixtures");
export const fixtureMeta = JSON.parse(readFileSync(resolve(FIXTURE_DIR, "fixture_meta.json"), "utf8")) as {
  zip: string; zip_sha256: string; primary_p: number; n_discordant: number;
};

export function loadFixtureFiles(zipPath = resolve(FIXTURE_DIR, "synthetic_bundle.zip")): Map<string, Uint8Array> {
  const zip = unzipSync(new Uint8Array(readFileSync(zipPath)));
  const files = new Map<string, Uint8Array>();
  for (const [name, data] of Object.entries(zip)) {
    if (name.endsWith("/")) continue;
    files.set(name.slice(name.indexOf("/") + 1), data);
  }
  return files;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

export class FixtureBundle {
  files: Map<string, Uint8Array>;
  /** fetches per bundle path (tests of caching / memory) */
  fetches = new Map<string, number>();
  constructor(zipPath?: string) { this.files = loadFixtureFiles(zipPath); }
  session: Record<string, unknown> | null = {
    app: "verihgn-headpath-dashboard", zip_name: fixtureMeta.zip, zip_path: "C:/fixtures/synthetic_bundle.zip",
    zip_sha256: fixtureMeta.zip_sha256, zip_bytes: 1, verified: true,
    verification: { status: "PASSED", errors: [], files_checked: 1, arrays_checked: 1, json_strict: 1 },
    extract_dir: "C:/cache/synthetic", launched_utc: "test", launcher_version: "test",
  };

  json<T = any>(rel: string): T { return JSON.parse(dec.decode(this.files.get(rel)!)) as T; }

  setJson(rel: string, obj: unknown, resign = true) { this.setBytes(rel, enc.encode(JSON.stringify(obj)), resign); }

  setBytes(rel: string, bytes: Uint8Array, resign = true) {
    this.files.set(rel, bytes);
    if (resign) {
      const man = this.json("manifest.json");
      man.files[rel] = { size: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
      this.files.set("manifest.json", enc.encode(JSON.stringify(man)));
    }
  }

  client(arrayBudget?: number): BundleClient {
    return new BundleClient("./data/", async (url) => {
      const rel = url.replace(/^\.\/data\//, "");
      this.fetches.set(rel, (this.fetches.get(rel) ?? 0) + 1);
      if (rel === "session.json") {
        if (!this.session) throw new Error("no session");
        return enc.encode(JSON.stringify(this.session));
      }
      const b = this.files.get(rel);
      if (!b) throw new Error(`404 ${rel}`);
      return b;
    }, arrayBudget);
  }
}
