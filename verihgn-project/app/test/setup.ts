import { webcrypto } from "node:crypto";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Web Crypto (sha256 checks) in the jsdom environment
if (!globalThis.crypto?.subtle) {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
}
// jsdom has no matchMedia
if (!window.matchMedia) {
  window.matchMedia = ((q: string) => ({ matches: false, media: q, onchange: null, addListener: () => {}, removeListener: () => {},
    addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => false })) as unknown as typeof window.matchMedia;
}

// Plotly cannot render in jsdom: record what the app asks it to draw instead.
export const plotlyCalls: { data: unknown[]; layout: unknown }[] = [];
vi.mock("plotly.js-dist-min", () => ({
  default: {
    react: (_el: HTMLElement, data: unknown[], layout: unknown) => { plotlyCalls.push({ data, layout }); return Promise.resolve(); },
    purge: () => {},
    Plots: { resize: () => {} },
  },
}));

afterEach(() => { cleanup(); window.location.hash = ""; });
