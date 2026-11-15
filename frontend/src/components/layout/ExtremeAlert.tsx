/**
 * Extreme volatility alerts: a compact list in the left column (above the convert card).
 * - One row per pair whose forecast 30-min move is ≥ 1.5× its normal size, worst first.
 * - Your own route (From → To) is marked. Clicking a row lights that pair up bright red on the globe
 *   (both countries, the line between them, rings) and flies there; clicking it again clears it.
 * - If the rows don't fit, the list scrolls slowly upwards like the rates ticker (pauses on hover/focus);
 *   with "reduce motion", or while a row is picked, it's a normal scrollable list instead.
 * Colour + icon + words (colour-blind safe). It never says which way the rate will go.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { TriangleAlert } from 'lucide-react'
import { EXTREME_RATIO, levelFor, volRatio } from '../../lib/storm'
import { slashPair } from '../../lib/pro'
import { usePrefersReducedMotion } from '../../lib/hooks'
import { routeLegs } from '../../lib/zones'
import { useAppStore } from '../../store/useAppStore'
import type { PairSnapshot } from '../../types'

/** Seconds each row stays in view while the list loops */
const SECONDS_PER_ROW = 3
/** The bar is full at this many × normal */
const BAR_MAX_RATIO = 4

function AlertRow({
  pair,
  ratio,
  onRoute,
  picked,
  onShow,
  copy = false,
}: {
  pair: PairSnapshot
  ratio: number
  onRoute: boolean
  picked: boolean
  onShow: () => void
  copy?: boolean
}) {
  const fill = Math.min(1, (ratio - 1) / (BAR_MAX_RATIO - 1))
  return (
    <li aria-hidden={copy || undefined}>
      <button
        type="button"
        tabIndex={copy ? -1 : undefined}
        onClick={onShow}
        aria-pressed={copy ? undefined : picked}
        title={picked ? 'Shown on the globe · click again to clear' : `Show ${slashPair(pair.pair)} on the globe`}
        className={`grid w-full cursor-pointer grid-cols-[4.5rem_1fr_2.75rem] items-center gap-2 rounded-lg px-2 py-1.5 text-left outline-none transition focus-visible:ring-2 focus-visible:ring-extreme ${
          picked ? 'bg-extreme/20 ring-1 ring-extreme/70' : 'hover:bg-white/[0.06]'
        }`}
      >
        <span className="flex flex-col leading-tight">
          <span className="num text-ink text-[12px] font-medium">{slashPair(pair.pair)}</span>
          {onRoute && <span className="text-accent text-[10px]">your route</span>}
        </span>
        <span className="h-1.5 overflow-hidden rounded-full bg-white/[0.08]" aria-hidden>
          <span className="block h-full rounded-full bg-extreme" style={{ width: `${Math.max(8, fill * 100)}%` }} />
        </span>
        <span className="num text-extreme text-right text-[12px] font-semibold">{ratio.toFixed(1)}×</span>
      </button>
    </li>
  )
}

export default function ExtremeAlert() {
  const snapshot = useAppStore((s) => s.snapshot)
  const tolerance = useAppStore((s) => s.tolerance)
  const base = useAppStore((s) => s.base)
  const quote = useAppStore((s) => s.quote)
  const alertPair = useAppStore((s) => s.alertPair)
  const showAlertPair = useAppStore((s) => s.showAlertPair)
  const reducedMotion = usePrefersReducedMotion()

  const rows = useMemo(() => {
    const route = new Set(routeLegs(snapshot, base, quote).map((l) => l.pair.pair))
    return (snapshot?.pairs ?? [])
      .filter((p) => levelFor(p, tolerance) === 'extreme')
      .map((p) => ({ pair: p, ratio: volRatio(p) ?? EXTREME_RATIO, onRoute: route.has(p.pair) }))
      .sort((a, b) => Number(b.onRoute) - Number(a.onRoute) || b.ratio - a.ratio)
  }, [snapshot, tolerance, base, quote])

  // Does the list fit? Measure one copy of the rows against the space we have.
  const viewportRef = useRef<HTMLDivElement>(null)
  const firstCopyRef = useRef<HTMLUListElement>(null)
  const [overflows, setOverflows] = useState(false)
  useLayoutEffect(() => {
    const viewport = viewportRef.current
    const list = firstCopyRef.current
    if (!viewport || !list) return
    const measure = () => setOverflows(list.offsetHeight > viewport.clientHeight + 1)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(viewport)
    ro.observe(list)
    return () => ro.disconnect()
  }, [rows.length])

  // While a row is picked the list stops looping (a normal scrollable list) and keeps that row in view
  const pickedInList = rows.some((r) => r.pair.pair === alertPair)
  useEffect(() => {
    if (!pickedInList) return
    firstCopyRef.current?.querySelector('[aria-pressed="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [pickedInList, alertPair])

  if (!rows.length) return null
  const loop = overflows && !reducedMotion && !pickedInList

  const renderRows = (copy: boolean) =>
    rows.map(({ pair, ratio, onRoute }) => (
      <AlertRow
        key={`${pair.pair}${copy ? '-copy' : ''}`}
        pair={pair}
        ratio={ratio}
        onRoute={onRoute}
        picked={alertPair === pair.pair}
        copy={copy}
        onShow={() => showAlertPair(pair)}
      />
    ))

  return (
    <section
      role="status"
      aria-label={`Extreme volatility: ${rows.length} ${rows.length === 1 ? 'pair' : 'pairs'}`}
      className="glass flex max-h-full min-h-0 w-[min(280px,calc(100vw-2rem))] flex-col overflow-hidden border-extreme/45! p-2"
    >
      <header className="flex items-center gap-2 px-2 pt-1">
        <TriangleAlert className="text-extreme size-4 shrink-0" aria-hidden />
        <span className="font-display text-extreme text-[12px] font-semibold uppercase tracking-wider">Extreme volatility</span>
        <span className="motion-safe:animate-alarm num ml-auto rounded-full bg-extreme/20 px-2 py-0.5 text-[11px] font-semibold text-ink">
          {rows.length}
        </span>
      </header>
      <p className="text-ink-muted px-2 pb-1.5 pt-0.5 text-[11px] leading-snug">
        Next 30 min likely to move this many × its normal size, either direction
      </p>

      <div
        ref={viewportRef}
        className={`group relative min-h-0 flex-1 ${loop ? 'overflow-hidden' : 'overflow-y-auto'}`}
        style={loop ? { maskImage: 'linear-gradient(180deg, transparent, black 10%, black 90%, transparent)' } : undefined}
      >
        <div
          className={
            loop ? 'motion-safe:animate-ticker-y group-hover:[animation-play-state:paused] group-focus-within:[animation-play-state:paused]' : ''
          }
          style={loop ? { animationDuration: `${rows.length * SECONDS_PER_ROW}s` } : undefined}
        >
          <ul ref={firstCopyRef} aria-label="Pairs at Extreme, worst first" className="flex flex-col">
            {renderRows(false)}
          </ul>
          {/* Second copy for a seamless loop; hidden from screen readers and the keyboard */}
          {loop && <ul className="flex flex-col">{renderRows(true)}</ul>}
        </div>
      </div>
    </section>
  )
}
