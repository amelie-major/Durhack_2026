/**
 * "Updated 18:40 UTC · next in 0:42" with a small countdown ring and a pause/resume button.
 * Re-renders once a second on its own, so the rest of the app doesn't.
 */
import { useEffect, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import { hhmm } from '../../lib/pro'
import { UPDATE_EVERY_MS, useAppStore } from '../../store/useAppStore'

/** Safe for empty / malformed timestamps (shows "--:--" instead of crashing) */
const formatUtc = (iso: string | undefined) => `${hhmm(iso)} UTC`

export default function NextUpdate() {
  const lastUpdated = useAppStore((s) => s.snapshot?.lastUpdated)
  const live = useAppStore((s) => s.live)
  const nextUpdateAt = useAppStore((s) => s.nextUpdateAt)
  const setLive = useAppStore((s) => s.setLive)
  const replaying = useAppStore((s) => s.replay !== null)

  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  const leftMs = Math.max(0, nextUpdateAt - now)
  const secs = Math.ceil(leftMs / 1000)
  const progress = 1 - leftMs / UPDATE_EVERY_MS // 0 → 1 as the next update approaches
  const R = 6
  const C = 2 * Math.PI * R

  // Replay has its own clock and controls (ReplayBar); here we only say that this isn't live
  if (replaying) {
    return (
      <span className="flex items-center gap-2" title="Playing back the last day of data, not live prices">
        <span className="rounded-full border border-unsettled/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-unsettled">
          Replay
        </span>
        <span>
          Data at <span className="num text-ink">{formatUtc(lastUpdated)}</span>
        </span>
      </span>
    )
  }

  return (
    <span
      className="flex items-center gap-2"
      title={`Forecasts refresh every ${Math.round(UPDATE_EVERY_MS / 1000)} s, built on 1-minute data`}
    >
      <span>
        Updated <span className="num text-ink">{formatUtc(lastUpdated)}</span>
      </span>
      <span className="text-ink-faint" aria-hidden>
        ·
      </span>
      {live ? (
        <span className="flex items-center gap-1.5" aria-live="off">
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden className="-rotate-90">
            <circle cx="8" cy="8" r={R} fill="none" stroke="var(--color-panel-border)" strokeWidth="2" />
            <circle
              cx="8"
              cy="8"
              r={R}
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={C}
              strokeDashoffset={C * (1 - progress)}
            />
          </svg>
          <span>
            next in{' '}
            <span className="num text-ink">
              {Math.floor(secs / 60)}:{String(secs % 60).padStart(2, '0')}
            </span>
          </span>
        </span>
      ) : (
        <span className="text-unsettled">Paused</span>
      )}
      <button
        type="button"
        onClick={() => setLive(!live)}
        aria-label={live ? 'Pause live updates' : 'Resume live updates'}
        title={live ? 'Pause live updates' : 'Resume live updates'}
        className="icon-btn"
      >
        {live ? <Pause className="size-3.5" aria-hidden /> : <Play className="size-3.5" aria-hidden />}
      </button>
    </span>
  )
}
