/**
 * Top bar: small brand on the left, a slowly scrolling "conveyor belt" of rates in the middle,
 * last-updated + "Sample data" on the right.
 * Each rate shows its volatility level (icon + word), never an up/down arrow — we don't forecast direction.
 * Hovering pauses the belt; "reduce motion" turns the scrolling off.
 */
import { SAMPLE_TICKER, type TickerItem } from '../../data/sampleRates'
import { STORM_LEVELS } from '../../lib/storm'

const LEVEL = Object.fromEntries(STORM_LEVELS.map((l) => [l.level, l]))

function TickerEntry({ item, copy = false }: { item: TickerItem; copy?: boolean }) {
  const { Icon, name, textClass } = LEVEL[item.level]
  return (
    <li aria-hidden={copy || undefined} className="flex shrink-0 items-center gap-2 px-5">
      <span className="text-ink-muted font-medium">{item.pair}</span>
      <span className="num text-ink">{item.mid.toFixed(item.decimals)}</span>
      <span className={`flex items-center gap-1 ${textClass}`}>
        <Icon className="size-3.5" aria-hidden />
        <span>{name}</span>
      </span>
      <span className="text-ink-faint pl-3" aria-hidden>
        •
      </span>
    </li>
  )
}

export default function RatesTicker() {
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
        <ul aria-label="Sample rates" className="flex w-max motion-safe:animate-ticker group-hover:[animation-play-state:paused]">
          {SAMPLE_TICKER.map((item) => (
            <TickerEntry key={item.pair} item={item} />
          ))}
          {/* Second copy for a seamless loop; hidden from screen readers */}
          {SAMPLE_TICKER.map((item) => (
            <TickerEntry key={`${item.pair}-copy`} item={item} copy />
          ))}
        </ul>
      </div>

      <span className="text-ink-muted flex shrink-0 items-center gap-2 border-l border-panel-border px-4">
        <span className="rounded-full border border-panel-border px-2 py-0.5 text-[10px] uppercase tracking-wider">
          Sample data
        </span>
        Updated <span className="num text-ink">14:32</span>
      </span>
    </div>
  )
}
