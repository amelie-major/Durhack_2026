import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// Yahoo Finance blocks browser calls (CORS), so the dev/preview server forwards /yahoo/* to it.
const yahooProxy = {
  '/yahoo': {
    target: 'https://query1.finance.yahoo.com',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/yahoo/, ''),
    headers: { 'User-Agent': 'Mozilla/5.0 (fx-forecast-frontend)' },
  },
  // Forecast server (XGBoost or PRAGMA): `python -m live.server` from the repo root
  '/api': { target: 'http://localhost:8000', changeOrigin: true },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: yahooProxy },
  preview: { proxy: yahooProxy },
})
