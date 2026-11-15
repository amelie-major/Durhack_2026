/** Storm-scale key: weather word + volatility level + icon + colour */
import { STORM_LEVELS } from '../../lib/storm'

export default function StormLegend() {
  return (
    <ul aria-label="Volatility scale" className="glass flex flex-col gap-1.5 px-4 py-3 text-xs">
      {STORM_LEVELS.map(({ name, volatility, Icon, textClass }) => (
        <li key={name} className="flex items-center gap-2">
          <Icon className={`size-4 ${textClass}`} aria-hidden />
          <span className={`w-16 font-medium ${textClass}`}>{name}</span>
          <span className="text-ink-muted">{volatility}</span>
        </li>
      ))}
      <li className="text-ink-muted flex items-center gap-2">
        <span className="mx-0.5 size-3 rounded-sm bg-accent" aria-hidden />
        <span className="w-16">From</span>
        <span>Your own currency</span>
      </li>
      <li className="text-ink-muted flex items-center gap-2">
        <span className="mx-0.5 size-3 rounded-sm bg-nodata" aria-hidden />
        <span className="w-16">No data</span>
        <span>No forecast here</span>
      </li>
    </ul>
  )
}
