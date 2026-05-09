import path from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/ws": { target: "ws://localhost:8000", ws: true, changeOrigin: true },
      "/transcribe": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/models": { target: "http://localhost:8000", changeOrigin: true },
      // YouTube replay: HTTP for /extract, WS for /stream. Both share the
      // /youtube prefix; Vite picks the matching ws/http handler per request.
      "/youtube": { target: "ws://localhost:8000", ws: true, changeOrigin: true },
    },
  },
})
