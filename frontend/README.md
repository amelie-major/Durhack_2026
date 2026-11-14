# FX Forecast — frontend (DurHack 2026)

Vite + React + TypeScript + Tailwind v4 + react-globe.gl.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production build
```

- Shared data contract: `src/types.ts` (agree changes with the model team)
- All data goes through `src/data/api.ts` (mock now, real later)
- Globe textures + countries GeoJSON live in `public/assets/` so the demo works offline.
  Country code rule: use `ISO_A3`, but fall back to `ADM0_A3` when it is `"-99"`
  (France, Norway, Kosovo, N. Cyprus, Somaliland).
