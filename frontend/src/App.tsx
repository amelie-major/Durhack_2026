/**
 * Layout: the globe fills the screen; small panels sit in the corners so it stays visible.
 * Steps 5–7 add the forecast panel, execution strip and replay controls.
 */
import FxGlobe from './components/globe/FxGlobe'
import ConvertCard from './components/convert/ConvertCard'
import StormLegend from './components/layout/StormLegend'

export default function App() {
  return (
    <div className="relative h-full w-full">
      {/* Globe layer (full screen, behind everything) */}
      <FxGlobe />

      {/* Top-left: title. Top-right: last updated. Small pills, not a full-width bar */}
      <header className="pointer-events-none absolute top-4 left-4 right-4 flex items-start justify-between gap-4">
        <h1 className="glass pointer-events-auto px-4 py-2 font-display text-lg font-semibold tracking-tight">
          FX Forecast <span className="text-ink-muted font-normal">· next 60 min outlook</span>
        </h1>
        <p className="glass pointer-events-auto px-4 py-2 text-sm text-ink-muted">
          Last updated <span className="num text-ink">14:32</span>
        </p>
      </header>

      {/* Bottom-left: convert card */}
      <div className="absolute bottom-4 left-4">
        <ConvertCard />
      </div>

      {/* Bottom-right: storm scale key */}
      <div className="absolute bottom-4 right-4 hidden sm:block">
        <StormLegend />
      </div>
    </div>
  )
}
