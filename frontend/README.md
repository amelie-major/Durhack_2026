# FX Forecast — frontend (DurHack 2026)

Vite + React + TypeScript + Tailwind v4 + react-globe.gl.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production build
```

- Shared data contract: `src/types.ts` (agree changes with the model team)
- All data goes through `src/data/api.ts`. `VITE_DATA_SOURCE` in `.env.local` picks the source:
  `model` (default: `python -m live.server` from the repo root, falls back to `yahoo` if it is down),
  `yahoo` (in-browser estimate) or `mock`.
- Replay: `ReplayBar` + `replay` in the store play back the last 24 h; in replay every data call
  passes `asOf` and gets only what was known at that moment.
- Globe textures + countries GeoJSON live in `public/assets/` so the demo works offline.
  Country code rule: use `ISO_A3`, but fall back to `ADM0_A3` when it is `"-99"`
  (France, Norway, Kosovo, N. Cyprus, Somaliland).
