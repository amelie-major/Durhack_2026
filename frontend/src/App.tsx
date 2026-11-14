/**
 * Step 2 shell: real globe behind placeholder panels.
 * Steps 3–7 replace the placeholders one by one.
 */
import { useCallback, useState } from 'react'
import { CloudLightning, CloudSun, Sun } from 'lucide-react'
import FxGlobe, { type GlobeSelection } from './components/globe/FxGlobe'

const levels = [
  { name: 'Calm', Icon: Sun, cls: 'text-calm' },
  { name: 'Unsettled', Icon: CloudSun, cls: 'text-unsettled' },
  { name: 'Storm', Icon: CloudLightning, cls: 'text-storm' },
]

export default function App() {
  const [selection, setSelection] = useState<GlobeSelection | null>(null)
  const handleSelect = useCallback((s: GlobeSelection | null) => setSelection(s), [])

  return (
    <div className="relative h-full w-full">
      {/* Globe layer (full screen, behind everything) */}
      <FxGlobe selectedCode={selection?.code ?? null} onSelect={handleSelect} />

      {/* Top bar */}
      <header className="glass absolute top-4 left-4 right-4 flex items-center justify-between px-5 py-3">
        <h1 className="font-display text-xl font-semibold tracking-tight">
          FX Forecast <span className="text-ink-muted font-normal">· next 60 min outlook</span>
        </h1>
        <p className="text-ink-muted text-sm">
          Last updated <span className="num text-ink">14:32</span>
        </p>
      </header>

      {/* Left: convert card placeholder */}
      <section className="glass absolute top-24 left-4 w-80 p-5">
        <h2 className="font-display text-lg mb-3">Convert</h2>
        <p className="num text-3xl">£10,000.00</p>
        <p className="text-ink-muted text-sm mt-1">GBP → JPY · step 4</p>
      </section>

      {/* Right: forecast panel placeholder — shows the globe selection + storm scale tokens */}
      <section className="glass absolute top-24 right-4 w-80 p-5 space-y-3" aria-live="polite">
        <h2 className="font-display text-lg">Selected</h2>
        {selection ? (
          <p>
            <span className="font-medium">{selection.name}</span>{' '}
            <span className="text-ink-muted">
              · {selection.zone ? selection.zone.code : 'No forecast'}
            </span>
          </p>
        ) : (
          <p className="text-ink-muted text-sm">Click a country on the globe (Esc to clear)</p>
        )}

        <h2 className="font-display text-lg pt-2">Storm scale</h2>
        {levels.map(({ name, Icon, cls }) => (
          <div key={name} className={`flex items-center gap-3 ${cls}`}>
            <Icon className="size-5" aria-hidden />
            <span className="font-medium">{name}</span>
          </div>
        ))}
      </section>

      {/* Bottom: execution strip placeholder */}
      <footer className="glass absolute bottom-4 left-4 right-4 px-5 py-3 text-ink-muted text-sm">
        Execution window strip · step 6 — Replay controls · step 7
      </footer>
    </div>
  )
}