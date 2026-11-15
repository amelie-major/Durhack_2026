/**
 * BETA card: 30-minute vol and spread forecasts from the experimental hybrid model
 * (XGBoost + PRAGMA embeddings). Collapsed by default so it never crowds the globe.
 * The numbers come from historical data (data/processed), not the live feed, and the card says so.
 * Hidden entirely when the model server isn't running.
 */
import { useEffect, useState } from 'react'
import { ChevronDown, FlaskConical } from 'lucide-react'
import { getHybridPredictions } from '../../data/api'
import type { HybridPredictions } from '../../types'

const REFRESH_MS = 60_000

const fmt = (x: number | null) => (x == null ? '–' : x.toFixed(2))
const utc = (iso?: string) => (iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : 'unknown')

export default function HybridBetaCard() {
  const [data, setData] = useState<HybridPredictions | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    const load = () => getHybridPredictions(controller.signal).then((d) => !controller.signal.aborted && setData(d))
    load()
    const id = window.setInterval(load, REFRESH_MS)
    return () => {
      controller.abort()
      window.clearInterval(id)
    }
  }, [])

  if (!data) return null

  return (
    <section aria-label="Hybrid model forecasts (beta)" className="glass w-[min(300px,calc(100vw-2rem))] p-3.5 text-xs">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 text-left"
      >
        <FlaskConical className="size-3.5 text-accent" aria-hidden />
        <span className="font-medium uppercase tracking-wider text-ink">Hybrid model</span>
        <span className="rounded-full border border-unsettled/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-unsettled">
          Beta
        </span>
        <ChevronDown className={`ml-auto size-3.5 text-ink-muted transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>

      {open && (
        <div className="mt-3 space-y-2">
          <p className="text-ink-muted">
            Experimental: {data.model}. Next {data.horizonMin} min, 1M forward. Built on historical data, not live prices.
          </p>

          {data.available ? (
            <>
              <table className="w-full">
                <thead className="text-ink-faint">
                  <tr>
                    <th className="py-1 text-left font-normal">Pair</th>
                    <th className="py-1 text-right font-normal">Vol (bp)</th>
                    <th className="py-1 text-right font-normal">Spread (bp)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.pairs.map((p) => (
                    <tr key={p.pair} className="border-t border-panel-border">
                      <td className="py-1 text-ink">{p.pair}</td>
                      <td className="num py-1 text-right text-ink">{fmt(p.volBp)}</td>
                      <td className="num py-1 text-right text-ink">{fmt(p.spreadBps)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-ink-faint">
                Data up to {utc(data.dataAsOf)} · run {utc(data.generatedAt)}
              </p>
            </>
          ) : (
            <p className="text-ink-muted">
              No predictions yet. From the repo root run <code className="num text-ink">scripts/run_hybrid_beta.sh</code>.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
