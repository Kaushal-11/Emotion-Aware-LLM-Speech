import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // loadEnv is the correct way to read .env files inside vite.config.js
  // process.env.VITE_* does NOT work here — only inside React code via import.meta.env
  const env = loadEnv(mode, process.cwd(), "");
  const serverUrl = env.VITE_SERVER_URL || "http://localhost:8000";

  return {
    plugins: [react({ jsxRuntime: "automatic" })],
    server: {
      port: 3000,
      proxy: {
        "/api": {
          target: serverUrl,
          changeOrigin: true,
          // needed for Cloudflare tunnels (https target)
          secure: false,
        },
        "/ws": {
          target: serverUrl,
          changeOrigin: true,
          secure: false,
          ws: true,
        },
      },
    },
  };
});