import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Built bundle lands in dist/ and is served by the stdlib Console Agent
// (titan_console.agent._serve_static). Relative base so it works behind any
// path / reverse proxy. During dev, proxy /console → the agent on :7799.
export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5174,
    proxy: { "/console": "http://127.0.0.1:7799" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
