import { useEffect, useState } from "react";
import { useBundle } from "./BundleContext";

export interface Loaded<T> { data: T | null; error: string | null; loading: boolean; missing: boolean }

/** A verified bundle JSON file. `missing` = not listed in the manifest (an older or partial export), which is
    reported as absent rather than as an integrity failure. */
export function useJson<T>(rel: string | null): Loaded<T> {
  const { client, core } = useBundle();
  const listed = !!rel && !!core?.manifest.files[rel];
  const [s, setS] = useState<Loaded<T>>({ data: null, error: null, loading: listed, missing: !listed });
  useEffect(() => {
    if (!rel || !listed) { setS({ data: null, error: null, loading: false, missing: true }); return; }
    let alive = true;
    setS({ data: null, error: null, loading: true, missing: false });
    client.json<T>(rel).then((data) => { if (alive) setS({ data, error: null, loading: false, missing: false }); })
      .catch((e: Error) => { if (alive) setS({ data: null, error: e.message, loading: false, missing: false }); });
    return () => { alive = false; };
  }, [client, rel, listed]);
  return s;
}

/** Several verified text files (e.g. every run's config.yaml); a file absent from the manifest maps to null. */
export function useTexts(rels: string[]): { data: Record<string, string | null> | null; error: string | null } {
  const { client, core } = useBundle();
  const key = rels.join("\n");
  const [s, setS] = useState<{ data: Record<string, string | null> | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let alive = true;
    const list = key ? key.split("\n") : [];
    Promise.all(list.map((r) => (core?.manifest.files[r] ? client.text(r) : Promise.resolve(null))))
      .then((texts) => { if (alive) setS({ data: Object.fromEntries(list.map((r, i) => [r, texts[i] ?? null])), error: null }); })
      .catch((e: Error) => { if (alive) setS({ data: null, error: e.message }); });
    return () => { alive = false; };
  }, [client, core, key]);
  return s;
}
