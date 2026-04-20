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
      "/ws": { target: "ws://96.38.133.243:22961", ws: true, changeOrigin: true },
      "/transcribe": { target: "http://96.38.133.243:22961", changeOrigin: true },
      "/health": { target: "http://96.38.133.243:22961", changeOrigin: true },
      "/models": { target: "http://96.38.133.243:22961", changeOrigin: true },
    },
  },
})
