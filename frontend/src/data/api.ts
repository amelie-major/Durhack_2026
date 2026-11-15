/**
 * THE ONLY DOOR TO DATA. Every component gets forecasts through these functions.
 *
 * VITE_DATA_SOURCE (set in frontend/.env.local):
 *   model (default) — the team's XGBoost models on live Yahoo bars, from `python -m live.server`
 *                     (proxied at /api). If the server is down, falls back to "yahoo".
 *   yahoo           — live Yahoo bars with a simple in-browser vol estimate (src/data/yahoo.ts)
 *   mock            — deterministic sample data (src/data/mock.ts)
 *
 * REPLAY: pass `replay = true` and the data is what it looked like AT `asOf` (nothing later),
 * so the app can play back through the last day. getReplayRange() says which day that is.
 */
import type { CurrencyCode, HybridPredictions, PairHistory, ReplayRange, Snapshot } from '../types'
import { mockHistory, mockSnapshot } from './mock'
import { liveHistory, liveReplayRange, liveSnapshot } from './yahoo'

type Source = 'model' | 'yahoo' | 'mock'
const SOURCE: Source = (['yahoo', 'mock'] as const).find((s) => s === import.meta.env.VITE_DATA_SOURCE) ?? 'model'

export const USING_LIVE_DATA = SOURCE !== 'mock'
export const USING_SAMPLE_DATA = !USING_LIVE_DATA

async function fromServer<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, { signal })
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`)
  return res.json() as Promise<T>
}

let warned = false
function fallback(err: unknown) {
  if (!warned) console.warn('[data] model server unavailable, using in-browser Yahoo estimate:', err)
  warned = true
}

/** "&asOf=…" in replay, nothing when live (the server then uses the real now) */
const at = (asOf: string, replay: boolean) => (replay ? `&asOf=${encodeURIComponent(asOf)}` : '')

/** `signal` cancels the request when it's no longer wanted (replay moves on every second) */
export async function getSnapshot(asOf: string, base: CurrencyCode, replay = false, signal?: AbortSignal): Promise<Snapshot> {
  if (SOURCE === 'mock') return mockSnapshot(asOf, base)
  if (SOURCE === 'model') {
    try {
      return await fromServer<Snapshot>(`/api/snapshot?base=${encodeURIComponent(base)}${at(asOf, replay)}`, signal)
    } catch (err) {
      if (signal?.aborted) throw err // superseded, not a server failure: don't fall back
      fallback(err)
    }
  }
  return liveSnapshot(base, replay ? asOf : undefined)
}

/** The stretch of time replay plays through: the last 24 h that we have data for */
export async function getReplayRange(nowIso: string): Promise<ReplayRange> {
  if (SOURCE === 'mock') {
    return { start: new Date(new Date(nowIso).getTime() - 24 * 3_600_000).toISOString().replace('.000Z', 'Z'), end: nowIso, stepMin: 1 }
  }
  if (SOURCE === 'model') {
    try {
      return await fromServer<ReplayRange>('/api/replay-range')
    } catch (err) {
      fallback(err)
    }
  }
  return liveReplayRange()
}

/** Past history of one pair up to asOf (Pro view): forecasts, what actually happened, spread, rate. */
export async function getHistory(pair: string, asOf: string, hours: number, replay = false): Promise<PairHistory | null> {
  if (SOURCE === 'mock') return mockHistory(pair, asOf, hours)
  if (SOURCE === 'model') {
    try {
      return await fromServer<PairHistory>(`/api/history?pair=${encodeURIComponent(pair)}&hours=${hours}${at(asOf, replay)}`)
    } catch (err) {
      fallback(err)
    }
  }
  return liveHistory(pair, hours, replay ? asOf : undefined)
}

/**
 * BETA: the hybrid model's latest predictions (XGBoost + PRAGMA embeddings), or null when the
 * model server isn't in use or doesn't answer. Refresh them with scripts/run_hybrid_beta.sh.
 */
export async function getHybridPredictions(signal?: AbortSignal): Promise<HybridPredictions | null> {
  if (SOURCE !== 'model') return null
  try {
    return await fromServer<HybridPredictions>('/api/hybrid', signal)
  } catch (err) {
    if (!signal?.aborted) console.warn('[data] hybrid (beta) predictions unavailable:', err)
    return null
  }
}
