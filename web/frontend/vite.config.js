import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The production bundle is emitted into the FastAPI static tree so it can be
// served from the same origin as the legacy dashboard. `base` controls the
// public URL prefix Vite bakes into asset references — FastAPI mounts
// `web/static/` at `/static/`, so we want `/static/dashboard/dist/`.
//
// During `vite dev`, Vite serves on its own port and ignores `base`/`outDir`;
// it just uses `index.html` from this folder as a host shell.
export default defineConfig({
  plugins: [react()],
  base: "/static/dashboard/dist/",
  build: {
    outDir: resolve(__dirname, "../static/dashboard/dist"),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        // Single bundle keeps things easy to drop into the existing template
        // when a future PR wires it up. Filenames are hashed for cache-busting.
        entryFileNames: "assets/dashboard-[hash].js",
        chunkFileNames: "assets/dashboard-[hash].js",
        assetFileNames: "assets/dashboard-[hash][extname]",
        manualChunks: undefined,
      },
    },
  },
  server: {
    // Pick a port that doesn't collide with FastAPI (4941).
    port: 5173,
    strictPort: false,
  },
});
