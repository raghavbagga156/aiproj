/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static, offline build. `base: "./"` keeps every asset path relative, so the build can be served from any
// directory by tools/run_dashboard.py. Data is never bundled: it is read at runtime from /data/ (the launcher
// serves the verified, extracted export there).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    include: ["src/**/*.test.ts", "src/**/*.test.tsx", "test/**/*.test.ts", "test/**/*.test.tsx"],
    testTimeout: 30000,
  },
});
