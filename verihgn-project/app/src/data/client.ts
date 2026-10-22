/* Reads the verified export bundle served by tools/run_dashboard.py under ./data/.
   Every file is checked against manifest.json (size + sha256) before it is used; JSON is parsed strictly
   (JSON.parse rejects NaN/Infinity); arrays are decoded losslessly and their raw sha256 re-checked. Any
   mismatch is recorded and puts the application into the blocking integrity-failure state. Nothing is ever
   written; there are no requests outside ./data/. */
import { decodeArray, type DecodedArray } from "./arrays";
import { sha256Hex } from "./hash";
import type { ArrayMeta, Manifest, Session } from "./schema";

export type Fetcher = (url: string) => Promise<Uint8Array>;

export const defaultFetcher: Fetcher = async (url) => {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
};

export class BundleIntegrityError extends Error {}

export interface VerificationState {
  verifiedFiles: number;
  failures: { file: string; reason: string }[];
}

type Listener = (s: VerificationState) => void;

export class BundleClient {
  readonly base: string;
  private readonly fetcher: Fetcher;
  private manifestP: Promise<Manifest> | null = null;
  private readonly cache = new Map<string, Promise<unknown>>();
  private readonly arrays = new Map<string, { p: Promise<DecodedArray>; bytes: number }>();
  private arrayBytes = 0;
  readonly arrayBudget: number;
  private readonly state: VerificationState = { verifiedFiles: 0, failures: [] };
  private readonly listeners = new Set<Listener>();

  constructor(base = "./data/", fetcher: Fetcher = defaultFetcher, arrayBudget = 48e6) {
    this.base = base.endsWith("/") ? base : `${base}/`;
    this.fetcher = fetcher;
    this.arrayBudget = arrayBudget;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.snapshot());
    return () => this.listeners.delete(fn);
  }

  snapshot(): VerificationState {
    return { verifiedFiles: this.state.verifiedFiles, failures: [...this.state.failures] };
  }

  private emit() {
    const s = this.snapshot();
    this.listeners.forEach((l) => l(s));
  }

  private fail(file: string, reason: string): never {
    this.state.failures.push({ file, reason });
    this.emit();
    throw new BundleIntegrityError(`${file}: ${reason}`);
  }

  /** Launcher session record (not part of the bundle). Missing -> null (e.g. a plain static server). */
  async session(): Promise<Session | null> {
    try {
      const b = await this.fetcher(`${this.base}session.json`);
      return JSON.parse(new TextDecoder().decode(b)) as Session;
    } catch {
      return null;
    }
  }

  manifest(): Promise<Manifest> {
    if (!this.manifestP) {
      this.manifestP = (async () => {
        const b = await this.fetcher(`${this.base}manifest.json`);
        let m: Manifest;
        try {
          m = JSON.parse(new TextDecoder().decode(b)) as Manifest;
        } catch (e) {
          this.fail("manifest.json", `not strict JSON (${(e as Error).message})`);
        }
        if (m.schema_version !== 1) this.fail("manifest.json", `unsupported schema_version ${String(m.schema_version)}`);
        return m;
      })();
    }
    return this.manifestP;
  }

  /** Bytes of a bundle file, verified against the manifest (size + sha256). */
  async bytes(rel: string): Promise<Uint8Array> {
    const m = await this.manifest();
    const entry = m.files[rel];
    if (!entry) this.fail(rel, "not listed in manifest.json");
    let b: Uint8Array;
    try {
      b = await this.fetcher(`${this.base}${rel}`);
    } catch (e) {
      this.fail(rel, `could not be read (${(e as Error).message})`);
    }
    if (b.length !== entry.size) this.fail(rel, `size ${b.length} != manifest ${entry.size}`);
    const h = await sha256Hex(b);
    if (h !== entry.sha256) this.fail(rel, "sha256 does not match manifest.json");
    this.state.verifiedFiles += 1;
    this.emit();
    return b;
  }

  json<T>(rel: string): Promise<T> {
    if (!this.cache.has(rel)) {
      this.cache.set(rel, (async () => {
        const b = await this.bytes(rel);
        try {
          return JSON.parse(new TextDecoder().decode(b)) as T;   // strict: NaN / Infinity are rejected
        } catch (e) {
          this.fail(rel, `not strict JSON (${(e as Error).message})`);
        }
      })());
    }
    return this.cache.get(rel) as Promise<T>;
  }

  text(rel: string): Promise<string> {
    const key = `text:${rel}`;
    if (!this.cache.has(key)) this.cache.set(key, this.bytes(rel).then((b) => new TextDecoder().decode(b)));
    return this.cache.get(key) as Promise<string>;
  }

  /** Decoded arrays are kept in a least-recently-used cache bounded by their raw size, so browsing many maps
      never holds the whole bundle in memory (148 MB raw for the real export). */
  array(meta: ArrayMeta): Promise<DecodedArray> {
    const hit = this.arrays.get(meta.file);
    if (hit) {
      this.arrays.delete(meta.file);
      this.arrays.set(meta.file, hit);
      return hit.p;
    }
    const p = (async () => {
      const b = await this.bytes(meta.file);
      try {
        return await decodeArray(b, meta);
      } catch (e) {
        this.fail(meta.file, (e as Error).message);
      }
    })();
    this.arrays.set(meta.file, { p, bytes: meta.nbytes_raw });
    this.arrayBytes += meta.nbytes_raw;
    for (const [k, v] of this.arrays) {
      if (this.arrayBytes <= this.arrayBudget || k === meta.file) break;
      this.arrays.delete(k);
      this.arrayBytes -= v.bytes;
    }
    return p;
  }

  /** Raw bytes of the decoded arrays currently cached (tests and diagnostics). */
  cachedArrayBytes(): { bytes: number; count: number; budget: number } {
    return { bytes: this.arrayBytes, count: this.arrays.size, budget: this.arrayBudget };
  }
}
