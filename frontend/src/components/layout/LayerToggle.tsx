/**
 * What the globe shows: Volatility (how much rates may move) or the model's Buy / Sell signal.
 * Sits in the top bar next to Simple | Pro, with a "Map" label and an icon on each option so it reads
 * as its own control (not a second view switch).
 */
import { CloudLightning, TrendingUpDown, type LucideIcon } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'
import type { MapLayer } from '../../types'

const LAYERS: { layer: MapLayer; label: string; hint: string; Icon: LucideIcon }[] = [
  { layer: 'volatility', label: 'Volatility', hint: 'Colour the map by how much each rate is likely to move (next 30 min)', Icon: CloudLightning },
  {
    layer: 'signal',
    label: 'Buy / Sell',
    hint: "Colour the map by the model's signal from your currency: green buy, red sell, grey hold (next 10 min). Not financial advice.",
    Icon: TrendingUpDown,
  },
]

export default function LayerToggle() {
  const mapLayer = useAppStore((s) => s.mapLayer)
  const setMapLayer = useAppStore((s) => s.setMapLayer)
  return (
    <div className="flex items-center gap-1.5">
      <span id="map-layer-label" className="text-ink-muted text-[11px] uppercase tracking-wider max-[1400px]:sr-only">
        Map
      </span>
      <div role="radiogroup" aria-labelledby="map-layer-label" className="seg-track">
        {LAYERS.map(({ layer, label, hint, Icon }) => {
          const on = mapLayer === layer
          return (
            <button
              key={layer}
              type="button"
              role="radio"
              aria-checked={on}
              title={hint}
              onClick={() => setMapLayer(layer)}
              className="seg-btn seg-ink"
            >
              <Icon className="size-4" aria-hidden />
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
