/** What the globe shows: Volatility (how much rates may move) or the model's Buy / Sell signal. */
import { useAppStore } from '../../store/useAppStore'
import type { MapLayer } from '../../types'

const LAYERS: { layer: MapLayer; label: string; hint: string }[] = [
  { layer: 'volatility', label: 'Volatility', hint: 'Colour the map by how much each rate is likely to move' },
  { layer: 'signal', label: 'Buy / Sell', hint: "Colour the map by the model's signal: green buy, red sell, grey hold" },
]

export default function LayerToggle() {
  const mapLayer = useAppStore((s) => s.mapLayer)
  const setMapLayer = useAppStore((s) => s.setMapLayer)
  return (
    <div role="radiogroup" aria-label="Map layer" className="flex rounded-full border border-panel-border p-0.5">
      {LAYERS.map((l) => (
        <button
          key={l.layer}
          type="button"
          role="radio"
          aria-checked={mapLayer === l.layer}
          title={l.hint}
          onClick={() => setMapLayer(l.layer)}
          className={`cursor-pointer whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-accent ${
            mapLayer === l.layer ? 'bg-accent text-space' : 'text-ink-muted hover:text-ink'
          }`}
        >
          {l.label}
        </button>
      ))}
    </div>
  )
}
