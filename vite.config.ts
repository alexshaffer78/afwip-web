import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static build for GitHub Pages. `base` must match the repo name for project
// pages (https://<user>.github.io/afwip-web/); override with VITE_BASE for a
// custom domain or local root serving. No dev proxy — the engine runs in the
// browser (Pyodide), so there is no backend to proxy to.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/afwip-web/",
  plugins: [react()],
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
