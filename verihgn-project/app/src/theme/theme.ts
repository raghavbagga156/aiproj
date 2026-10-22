import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";

function systemTheme(): Theme {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function storedTheme(): Theme | null {
  try {
    const v = window.localStorage.getItem("verihgn.theme");
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function urlTheme(): Theme | null {
  const q = window.location.hash.split("?")[1] ?? "";
  const v = new URLSearchParams(q).get("theme") ?? new URLSearchParams(window.location.search).get("theme");
  return v === "light" || v === "dark" ? v : null;
}

/** Light/dark: URL (?theme=) > the viewer's saved choice > OS preference. Applied as <html data-theme>. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => urlTheme() ?? storedTheme() ?? systemTheme());
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      try { window.localStorage.setItem("verihgn.theme", next); } catch { /* storage unavailable: session only */ }
      return next;
    });
  }, []);
  return [theme, toggle];
}

/** Resolved CSS token values (charts read the live tokens, so they follow the theme). */
export function cssVar(name: string, fallback = "#000"): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
