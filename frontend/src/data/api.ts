/**
 * THE ONLY DOOR TO DATA. Every component gets forecasts through these functions.
 *
 * VITE_DATA_SOURCE (set in frontend/.env.local):
 *   model (default) — the team's XGBoost models on live Yahoo bars, from `python -m live.server`
 *                     (proxied at /api). If the server is down, falls back to "yahoo".
 *   yahoo           — live Yahoo bars with a simple in-browser vol estimate (src/data/yahoo.ts)
 *   mock            — deterministic sample data (src/data/mock.ts)
 *
 * MODEL: `model` picks the forecasts the server returns, XGBoost (live) or PRAGMA (saved predictions for
 * its test week, replay only). PRAGMA never falls back to Yahoo: that would show numbers that are not PRAGMA's.
 *
 * REPLAY: pass `replay = true` and the data is what it looked like AT `asOf` (nothing later),
 * so the app can play back through the last day. getReplayRange() says which day that is.
 */
import type { CurrencyCode, ForecastModel, PairHistory, ReplayRange, Snapshot } from '../types'
import { mockHistory, mockSnapshot } from './mock'
import { liveHistory, liveReplayRange, liveSnapshot } from './yahoo'

type Source = 'model' | 'yahoo' | 'mock'
const SOURCE: Source = (['yahoo', 'mock'] as const).find((s) => s === import.meta.env.VITE_DATA_SOURCE) ?? 'model'

export const USING_LIVE_DATA = SOURCE !== 'mock'
export const USING_SAMPLE_DATA = !USING_LIVE_DATA

async function fromServer<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`)
  return res.json() as Promise<T>
}

let warned = false
function fallback(err: unknown, model: ForecastModel) {
  if (model === 'pragma') throw err
  if (!warned) console.warn('[data] model server unavailable, using in-browser Yahoo estimate:', err)
  warned = true
}

/** "&asOf=…" in replay, nothing when live (the server then uses the real now) */
const at = (asOf: string, replay: boolean) => (replay ? `&asOf=${encodeURIComponent(asOf)}` : '')

/** Which models the forecast server can serve, and the one it was started with (`--model`) */
export async function getModelInfo(): Promise<{ models: ForecastModel[]; defaultModel: ForecastModel }> {
  if (SOURCE !== 'model') return { models: ['xgboost'], defaultModel: 'xgboost' }
  const health = await fromServer<{ models?: ForecastModel[]; defaultModel?: ForecastModel }>('/api/health')
  return { models: health.models ?? ['xgboost'], defaultModel: health.defaultModel ?? 'xgboost' }
}

export async function getSnapshot(
  asOf: string,
  base: CurrencyCode,
  replay = false,
  model: ForecastModel = 'xgboost',
): Promise<Snapshot> {
  if (SOURCE === 'mock') return mockSnapshot(asOf, base)
  if (SOURCE === 'model') {
    try {
      return await fromServer<Snapshot>(`/api/snapshot?base=${encodeURIComponent(base)}&model=${model}${at(asOf, replay)}`)
    } catch (err) {
      fallback(err, model)
    }
  }
  return liveSnapshot(base, replay ? asOf : undefined)
}

/** The stretch of time replay plays through: the last 24 h that we have data for */
export async function getReplayRange(nowIso: string, model: ForecastModel = 'xgboost'): Promise<ReplayRange> {
  if (SOURCE === 'mock') {
    return { start: new Date(new Date(nowIso).getTime() - 24 * 3_600_000).toISOString().replace('.000Z', 'Z'), end: nowIso, stepMin: 1 }
  }
  if (SOURCE === 'model') {
    try {
      return await fromServer<ReplayRange>(`/api/replay-range?model=${model}`)
    } catch (err) {
      fallback(err, model)
    }
  }
  return liveReplayRange()
}

/** Past history of one pair up to asOf (Pro view): forecasts, what actually happened, spread, rate. */
export async function getHistory(
  pair: string,
  asOf: string,
  hours: number,
  replay = false,
  model: ForecastModel = 'xgboost',
): Promise<PairHistory | null> {
  if (SOURCE === 'mock') return mockHistory(pair, asOf, hours)
  if (SOURCE === 'model') {
    try {
      return await fromServer<PairHistory>(`/api/history?pair=${encodeURIComponent(pair)}&hours=${hours}&model=${model}${at(asOf, replay)}`)
    } catch (err) {
      fallback(err, model)
    }
  }
  return liveHistory(pair, hours, replay ? asOf : undefined)
}
