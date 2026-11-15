/**
 * THE ONLY DOOR TO DATA. Every component gets forecasts through these functions.
 *
 * VITE_DATA_SOURCE (set in frontend/.env.local):
 *   model (default) — the team's XGBoost models on live Yahoo bars, from `python -m live.server`
 *                     (proxied at /api). If the server is down, falls back to "yahoo".
 *   yahoo           — live Yahoo bars with a simple in-browser vol estimate (src/data/yahoo.ts)
 *   mock            — deterministic sample data (src/data/mock.ts)
 */
import type { CurrencyCode, PairHistory, Snapshot } from '../types'
import { mockHistory, mockSnapshot } from './mock'
import { liveHistory, liveSnapshot } from './yahoo'

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
function fallback(err: unknown) {
  if (!warned) console.warn('[data] model server unavailable, using in-browser Yahoo estimate:', err)
  warned = true
}

export async function getSnapshot(asOf: string, base: CurrencyCode): Promise<Snapshot> {
  if (SOURCE === 'mock') return mockSnapshot(asOf, base)
  if (SOURCE === 'model') {
    try {
      return await fromServer<Snapshot>(`/api/snapshot?base=${encodeURIComponent(base)}`)
    } catch (err) {
      fallback(err)
    }
  }
  return liveSnapshot(base)
}

/** Past history of one pair up to asOf (Pro view): forecasts, what actually happened, spread, rate. */
export async function getHistory(pair: string, asOf: string, hours: number): Promise<PairHistory | null> {
  if (SOURCE === 'mock') return mockHistory(pair, asOf, hours)
  if (SOURCE === 'model') {
    try {
      return await fromServer<PairHistory>(`/api/history?pair=${encodeURIComponent(pair)}&hours=${hours}`)
    } catch (err) {
      fallback(err)
    }
  }
  return liveHistory(pair, hours)
}
