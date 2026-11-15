/**
 * Top bar: small brand on the left, a slowly scrolling "conveyor belt" of rates in the middle,
 * last-updated + "Sample data" on the right.
 * Each rate shows its volatility level (icon + word); in the Buy / Sell map layer it shows the model's signal instead.
 * Hovering pauses the belt; "reduce motion" turns the scrolling off.
 */
import { useMemo } from 'react'
import { USING_SAMPLE_DATA } from '../../data/api'
import LayerToggle from './LayerToggle'
import ModeToggle from './ModeToggle'
import NextUpdate from './NextUpdate'
import { decimalsFor } from '../../data/mock'
import { actionFor, SIGNAL_INFO } from '../../lib/signal'
import { LEVEL_INFO, levelFor } from '../../lib/storm'
import { useAppStore } from '../../store/useAppStore'
import type { SignalAction, Snapshot, StormLevel, Tolerance } from '../../types'

interface TickerItem {
  pair: string // "EUR/USD"
  mid: number | null
  level: StormLevel
  signal: SignalAction | null
}

/** Every pair the data has (USD pairs first, then crosses), each with its own level and signal */
function tickerItems(snapshot: Snapshot | null, tolerance: Tolerance): TickerItem[] {
  const pairs = [...(snapshot?.pairs ?? [])].sort(
    (a, b) => Number(!a.pair.includes('USD')) - Number(!b.pair.includes('USD')) || a.pair.localeCompare(b.pair),
  )
  return pairs.map((p) => ({
    pair: `${p.base}/${p.quote}`,
    mid: p.mid,
    level: levelFor(p, tolerance),
    signal: actionFor(p),
  }))
}

function TickerEntry({ item, copy = false, showSignal }: { item: TickerItem; copy?: boolean; showSignal: boolean }) {
  // Volatility layer: the weather word. Buy / Sell layer: the model's signal.
  const info = showSignal ? (item.signal ? SIGNAL_INFO[item.signal] : null) : item.level === 'nodata' ? null : LEVEL_INFO[item.level]
  return (
    <li aria-hidden={copy || undefined} className="flex shrink-0 items-center gap-2 px-5">
      <span className="text-ink-muted font-medium">{item.pair}</span>
      <span className="num text-ink">{item.mid == null ? '—' : item.mid.toFixed(decimalsFor(item.pair))}</span>
      {info ? (
        <span className={`flex items-center gap-1 ${info.textClass}`}>
          <info.Icon className="size-3.5" aria-hidden />
          <span>{info.name}</span>
        </span>
      ) : (
        <span className="text-ink-muted">{showSignal ? 'No signal' : 'No data'}</span>
      )}
      <span className="text-ink-faint pl-3" aria-hidden>
        •
      </span>
    </li>
  )
}


export default function RatesTicker() {
  const snapshot = useAppStore((s) => s.snapshot)
  const tolerance = useAppStore((s) => s.tolerance)
  const showSignal = useAppStore((s) => s.mapLayer === 'signal')
  const items = useMemo(() => tickerItems(snapshot, tolerance), [snapshot, tolerance])

  return (
    <div className="glass flex h-9 items-center overflow-hidden rounded-full text-xs">
      <span className="font-display shrink-0 border-r border-panel-border px-4 font-semibold tracking-tight">
        FX Forecast
      </span>

      {/* The moving belt. Edges fade out so items glide in and out */}
      <div
        className="group relative min-w-0 flex-1 overflow-hidden"
        style={{ maskImage: 'linear-gradient(90deg, transparent, black 6%, black 94%, transparent)' }}
      >
        <ul aria-label="Rates" className="flex w-max motion-safe:animate-ticker group-hover:[animation-play-state:paused]">
          {items.map((item) => (
            <TickerEntry key={item.pair} item={item} showSignal={showSignal} />
          ))}
          {/* Second copy for a seamless loop; hidden from screen readers */}
          {items.map((item) => (
            <TickerEntry key={`${item.pair}-copy`} item={item} copy showSignal={showSignal} />
          ))}
        </ul>
      </div>

      <span className="text-ink-muted flex shrink-0 items-center gap-2 border-l border-panel-border px-4">
        {USING_SAMPLE_DATA && (
          <span className="rounded-full border border-panel-border px-2 py-0.5 text-[10px] uppercase tracking-wider">
            Sample data
          </span>
        )}
        <NextUpdate />
      </span>
      <span className="flex shrink-0 items-center gap-1.5 border-l border-panel-border px-2">
        <LayerToggle />
        <ModeToggle />
      </span>
    </div>
  )
}
