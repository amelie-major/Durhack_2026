/**
 * Replay: play back through the last day of data as if it were happening now.
 * Live: one "Replay last day" button. Replaying: play/pause, speed, a scrubber and "Back to live".
 * Every number on screen is what the model said AT the replayed moment, never anything later.
 */
import { useState } from 'react'
import { History, Pause, Play, Radio } from 'lucide-react'
import { getReplayRange } from '../../data/api'
import { REPLAY_SPEEDS, useAppStore } from '../../store/useAppStore'

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** "2026-11-13T14:30:00Z" → "Fri 13 Nov 14:30 UTC" (never throws on a bad timestamp) */
function dayTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '--:-- UTC'
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.toISOString().slice(11, 16)} UTC`
}

const pill =
  'flex cursor-pointer items-center gap-1.5 rounded-full border border-panel-border px-3 py-1 text-xs font-medium text-ink outline-none transition hover:bg-white/10 focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-wait disabled:opacity-60'

export default function ReplayBar() {
  const replay = useAppStore((s) => s.replay)
  const asOf = useAppStore((s) => s.asOf)
  const startReplay = useAppStore((s) => s.startReplay)
  const exitReplay = useAppStore((s) => s.exitReplay)
  const setPlaying = useAppStore((s) => s.setReplayPlaying)
  const setSpeed = useAppStore((s) => s.setReplaySpeed)
  const seek = useAppStore((s) => s.seekReplay)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!replay) {
    const begin = () => {
      setLoading(true)
      setError(null)
      getReplayRange(asOf, useAppStore.getState().model)
        .then((range) => startReplay(range))
        .catch((err: unknown) => {
          console.error('[data] could not start replay', err)
          setError('Replay is not available right now')
        })
        .finally(() => setLoading(false))
    }
    return (
      <div className="glass flex items-center gap-2 rounded-full p-1">
        <button type="button" onClick={begin} disabled={loading} className={pill} title="Play back the last 24 hours of data">
          <History className="size-3.5 text-accent" aria-hidden />
          {loading ? 'Loading…' : 'Replay last day'}
        </button>
        {error && (
          <span role="alert" className="pr-3 text-[11px] text-unsettled">
            {error}
          </span>
        )}
      </div>
    )
  }

  const startMs = new Date(replay.start).getTime()
  const endMs = new Date(replay.end).getTime()
  const totalMin = Math.max(1, Math.round((endMs - startMs) / 60_000))
  const atMin = Math.min(totalMin, Math.max(0, Math.round((new Date(asOf).getTime() - startMs) / 60_000)))
  const atEnd = atMin >= totalMin

  return (
    <section aria-label="Replay" className="glass flex w-[min(620px,100%)] flex-col gap-1.5 rounded-2xl px-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setPlaying(!replay.playing)}
          aria-label={replay.playing ? 'Pause replay' : atEnd ? 'Play again from the start' : 'Play replay'}
          title={replay.playing ? 'Pause' : atEnd ? 'Play again from the start' : 'Play'}
          className="grid size-7 shrink-0 cursor-pointer place-items-center rounded-full bg-accent text-space outline-none transition hover:brightness-110 focus-visible:ring-2 focus-visible:ring-accent"
        >
          {replay.playing ? <Pause className="size-3.5" aria-hidden /> : <Play className="size-3.5" aria-hidden />}
        </button>

        <span className="rounded-full border border-unsettled/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-unsettled">
          Replay
        </span>
        <span className="num text-ink min-w-0 flex-1 truncate" aria-live="off">
          {dayTime(asOf)}
        </span>

        <div role="radiogroup" aria-label="Replay speed" className="flex shrink-0 rounded-full border border-panel-border p-0.5">
          {REPLAY_SPEEDS.map((sp) => (
            <button
              key={sp}
              type="button"
              role="radio"
              aria-checked={replay.speed === sp}
              title={`${sp} minute${sp === 1 ? '' : 's'} of data per second`}
              onClick={() => setSpeed(sp)}
              className={`num cursor-pointer rounded-full px-2 py-0.5 text-[10px] font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-accent ${
                replay.speed === sp ? 'bg-white/12 text-ink' : 'text-ink-muted hover:text-ink'
              }`}
            >
              {sp}×
            </button>
          ))}
        </div>

        <button type="button" onClick={exitReplay} className={`${pill} shrink-0`} title="Stop the replay and show live data">
          <Radio className="size-3.5 text-calm" aria-hidden />
          Back to live
        </button>
      </div>

      <div className="flex items-center gap-2">
        <span className="num text-ink-faint shrink-0 text-[10px]">{dayTime(replay.start).slice(0, -4)}</span>
        <input
          type="range"
          min={0}
          max={totalMin}
          step={1}
          value={atMin}
          onChange={(e) => seek(new Date(startMs + Number(e.target.value) * 60_000).toISOString())}
          aria-label="Replay position"
          aria-valuetext={dayTime(asOf)}
          className="h-1 min-w-0 flex-1 cursor-pointer accent-[var(--color-accent)]"
        />
        <span className="num text-ink-faint shrink-0 text-[10px]">{dayTime(replay.end).slice(0, -4)}</span>
      </div>
    </section>
  )
}
