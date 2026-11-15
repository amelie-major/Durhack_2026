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
      <span id="map-layer-label" className="text-ink-muted text-[10px] uppercase tracking-wider">
        Map
      </span>
      <div role="radiogroup" aria-labelledby="map-layer-label" className="flex rounded-full bg-white/[0.06] p-0.5">
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
              className={`flex cursor-pointer items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-accent ${
                on ? 'bg-ink text-space shadow-sm' : 'text-ink-muted hover:text-ink'
              }`}
            >
              <Icon className="size-3.5" aria-hidden />
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
