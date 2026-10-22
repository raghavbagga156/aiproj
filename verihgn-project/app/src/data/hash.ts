/* SHA-256 via Web Crypto (available on http://127.0.0.1, a secure context, and in Node for tests). */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("Web Crypto (crypto.subtle) is not available: open the dashboard through the launcher (http://127.0.0.1)");
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const d = await subtle.digest("SHA-256", copy.buffer);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}
