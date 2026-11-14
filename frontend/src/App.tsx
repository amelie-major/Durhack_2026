/**
 * Layout: the globe fills the screen; small panels sit at the edges so it stays visible.
 * Steps 5–7 add the forecast panel, execution strip and replay controls.
 */
import FxGlobe from './components/globe/FxGlobe'
import ConvertCard from './components/convert/ConvertCard'
import RatesTicker from './components/layout/RatesTicker'
import StormLegend from './components/layout/StormLegend'

export default function App() {
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
