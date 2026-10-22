/* Lossless decoder for the export's array encoding "le+byteshuffle+gzip" (src/verihgn_repro/export/dashboard_bundle.py):
   gzip -> itemsize byte planes -> interleave back -> little-endian float32 / int32. The decoded raw bytes are
   re-hashed and must equal the manifest's sha256_raw, otherwise the array is rejected. */
import { gunzipSync } from "fflate";
import type { ArrayMeta } from "./schema";
import { sha256Hex } from "./hash";

export const ENCODING = "le+byteshuffle+gzip";

export class ArrayIntegrityError extends Error {}

export function unshuffle(planes: Uint8Array, itemsize: number): Uint8Array {
  if (planes.length % itemsize !== 0) throw new ArrayIntegrityError("byte length is not a multiple of itemsize");
  const n = planes.length / itemsize;
  const out = new Uint8Array(planes.length);
  for (let b = 0; b < itemsize; b++) {
    const off = b * n;
    for (let i = 0; i < n; i++) out[i * itemsize + b] = planes[off + i]!;
  }
  return out;
}

export interface DecodedArray { data: Float32Array | Int32Array; shape: number[]; meta: ArrayMeta }

export async function decodeArray(blob: Uint8Array, meta: ArrayMeta): Promise<DecodedArray> {
  if (meta.encoding !== ENCODING) throw new ArrayIntegrityError(`unsupported encoding ${meta.encoding}`);
  const planes = gunzipSync(blob);
  const raw = unshuffle(planes, meta.itemsize);
  if (raw.length !== meta.nbytes_raw) throw new ArrayIntegrityError(`${meta.file}: ${raw.length} bytes, expected ${meta.nbytes_raw}`);
  const h = await sha256Hex(raw);
  if (h !== meta.sha256_raw) throw new ArrayIntegrityError(`${meta.file}: decoded sha256 mismatch`);
  const n = meta.shape.reduce((a, b) => a * b, 1);
  // copy into an aligned buffer, read as little-endian explicitly
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  let data: Float32Array | Int32Array;
  if (meta.dtype === "float32") {
    data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = dv.getFloat32(i * 4, true);
  } else if (meta.dtype === "int32") {
    data = new Int32Array(n);
    for (let i = 0; i < n; i++) data[i] = dv.getInt32(i * 4, true);
  } else {
    throw new ArrayIntegrityError(`unsupported dtype ${String(meta.dtype)}`);
  }
  return { data, shape: meta.shape, meta };
}

/** Raw little-endian bytes of a decoded array (bit-exact comparison in tests). */
export function rawBytes(a: Float32Array | Int32Array): Uint8Array {
  const out = new Uint8Array(a.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < a.length; i++) {
    if (a instanceof Float32Array) dv.setFloat32(i * 4, a[i]!, true);
    else dv.setInt32(i * 4, a[i]!, true);
  }
  return out;
}
