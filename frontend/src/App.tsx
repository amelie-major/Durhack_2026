/**
 * Layout: the globe fills the screen; small panels sit at the edges so it stays visible.
 * Steps 5–7 add the forecast panel, execution strip and replay controls.
 */
import { useEffect } from 'react'
import FxGlobe from './components/globe/FxGlobe'
import ConvertCard from './components/convert/ConvertCard'
import RatesTicker from './components/layout/RatesTicker'
import StormLegend from './components/layout/StormLegend'
import { getSnapshot } from './data/api'
import { useAppStore } from './store/useAppStore'

/** Load data for the current simulated "now" whenever it (or the From currency) changes */
function useSnapshotLoader() {
  const asOf = useAppStore((s) => s.asOf)
  const base = useAppStore((s) => s.base)
  const setSnapshot = useAppStore((s) => s.setSnapshot)
  useEffect(() => {
    let cancelled = false
    getSnapshot(asOf, base)
      .then((snap) => {
        if (!cancelled) setSnapshot(snap)
      })
      .catch((err: unknown) => console.error('[data] could not load snapshot', err))
    return () => {
      cancelled = true
    }
  }, [asOf, base, setSnapshot])
}

export default function App() {
  useSnapshotLoader()

  return (
    <div className="relative h-full w-full">
      {/* Globe layer (full screen, behind everything) */}
      <FxGlobe />

      {/* Top: thin rates "conveyor belt" with a small brand */}
      <header className="absolute top-3 left-3 right-3">
        <RatesTicker />
      </header>

      {/* Bottom-left: convert card */}
      <div className="absolute bottom-3 left-3">
        <ConvertCard />
      </div>

      {/* Bottom-right: volatility key */}
      <div className="absolute bottom-3 right-3 hidden sm:block">
        <StormLegend />
      </div>
    </div>
  )
}
