/**
 * LIVE data from Yahoo Finance 1-minute FX bars (via the /yahoo dev-server proxy, see vite.config.ts).
 * Builds the same Snapshot / PairHistory shapes as the mock (src/types.ts), so no component changes.
 *
 * Yahoo gives spot mid prices only — no bid/ask — so spreads are ESTIMATES (typical interbank spread
 * scaled by how volatile the pair is right now). Vol numbers are real, computed from the bars:
 * - realizedVol30m: √(Σ r²) of the last 30 one-minute log returns, in bp
 * - forecast: EWMA (λ = 0.97) per-minute variance × √horizon
 * - percentile: current 30-min vol ranked against the last 5 days of rolling 30-min vols
 */
import type { CurrencyCode, HistoryPoint, PairHistory, PairSnapshot, PairStatus, ReplayRange, Snapshot } from '../types'
import { HORIZONS_MIN } from './mock'

interface PairSpec {
  pair: string
  base: CurrencyCode
  quote: CurrencyCode
  spreadBps: number // typical interbank spread (Yahoo has none)
}

/**
 * Every pair in data/processed/instruments.csv (tests/test_live_pairs.py checks they match).
 * spreadBps: the 1M-forward value for the three pairs with forwards, else the median spot spread in the data.
 */
export const LIVE_PAIRS: PairSpec[] = [
  { pair: 'EURUSD', base: 'EUR', quote: 'USD', spreadBps: 0.3 },
  { pair: 'USDJPY', base: 'USD', quote: 'JPY', spreadBps: 0.45 },
  { pair: 'USDSGD', base: 'USD', quote: 'SGD', spreadBps: 0.7 },
  { pair: 'AUDUSD', base: 'AUD', quote: 'USD', spreadBps: 1.28 },
  { pair: 'EURGBP', base: 'EUR', quote: 'GBP', spreadBps: 0.82 },
  { pair: 'EURJPY', base: 'EUR', quote: 'JPY', spreadBps: 0.44 },
  { pair: 'EURSGD', base: 'EUR', quote: 'SGD', spreadBps: 2.3 },
  { pair: 'GBPJPY', base: 'GBP', quote: 'JPY', spreadBps: 0.74 },
  { pair: 'GBPUSD', base: 'GBP', quote: 'USD', spreadBps: 0.52 },
  { pair: 'NZDUSD', base: 'NZD', quote: 'USD', spreadBps: 1.68 },
  { pair: 'SGDJPY', base: 'SGD', quote: 'JPY', spreadBps: 2.07 },
  { pair: 'USDCAD', base: 'USD', quote: 'CAD', spreadBps: 0.79 },
  { pair: 'USDCHF', base: 'USD', quote: 'CHF', spreadBps: 1.0 },
  { pair: 'USDCNH', base: 'USD', quote: 'CNH', spreadBps: 0.25 },
  { pair: 'USDHKD', base: 'USD', quote: 'HKD', spreadBps: 0.24 },
  { pair: 'USDMXN', base: 'USD', quote: 'MXN', spreadBps: 1.86 },
  { pair: 'USDNOK', base: 'USD', quote: 'NOK', spreadBps: 2.17 },
  { pair: 'USDPLN', base: 'USD', quote: 'PLN', spreadBps: 2.07 },
  { pair: 'USDSEK', base: 'USD', quote: 'SEK', spreadBps: 1.91 },
  { pair: 'USDTRY', base: 'USD', quote: 'TRY', spreadBps: 1.04 },
  { pair: 'USDZAR', base: 'USD', quote: 'ZAR', spreadBps: 2.62 },
]

const WINDOW = 30 // minutes for realised vol
const LAMBDA = 0.97
const CACHE_MS = 15_000

interface Bars {
  t: number[] // ms
  close: number[]
  lastPrice: number // regularMarketPrice — fresher than the last bar
}

const cache = new Map<string, { at: number; bars: Promise<Bars> }>()

async function fetchBars(pair: string): Promise<Bars> {
  const res = await fetch(`/yahoo/v8/finance/chart/${pair}=X?interval=1m&range=5d`)
  if (!res.ok) throw new Error(`Yahoo ${pair}: HTTP ${res.status}`)
  const json = await res.json()
  const r = json?.chart?.result?.[0]
  if (!r) throw new Error(`Yahoo ${pair}: ${json?.chart?.error?.description ?? 'no result'}`)
  const ts: number[] = r.timestamp ?? []
  const cl: (number | null)[] = r.indicators?.quote?.[0]?.close ?? []
  const t: number[] = []
  const close: number[] = []
  ts.forEach((s, i) => {
    const c = cl[i]
    if (c != null && Number.isFinite(c) && c > 0) {
      t.push(s * 1000)
      close.push(c)
    }
  })
  if (!close.length) throw new Error(`Yahoo ${pair}: no bars`)
  return { t, close, lastPrice: r.meta?.regularMarketPrice ?? close[close.length - 1] }
}

/** Bars for a pair, shared between callers and refreshed at most every CACHE_MS */
function getBars(pair: string): Promise<Bars> {
  const hit = cache.get(pair)
  if (hit && Date.now() - hit.at <= CACHE_MS) return hit.bars
  const bars = fetchBars(pair)
  cache.set(pair, { at: Date.now(), bars })
  bars.catch(() => cache.delete(pair)) // retry on the next call after a failure
  return bars
}

/** Replay: only the bars at or before asOf, priced at the last of them (nothing from the future) */
function upTo(b: Bars, asOfMs: number | undefined): Bars {
  if (asOfMs == null) return b
  let n = b.t.length
  while (n > 0 && b.t[n - 1] > asOfMs) n--
  if (!n) throw new Error('no bars at or before the replay time')
  return { t: b.t.slice(0, n), close: b.close.slice(0, n), lastPrice: b.close[n - 1] }
}

const toMs = (asOf: string | undefined) => (asOf ? new Date(asOf).getTime() : undefined)

/** 1-minute log returns in bp; ret[i] is the move from bar i to bar i+1 (gaps > 5 min count as 0) */
function returnsBp(b: Bars): number[] {
  const out: number[] = []
  for (let i = 1; i < b.close.length; i++) {
    out.push(b.t[i] - b.t[i - 1] > 5 * 60_000 ? 0 : Math.log(b.close[i] / b.close[i - 1]) * 1e4)
  }
  return out
}

/** Realised vol (bp) over returns [from, to) */
function realised(ret: number[], from: number, to: number): number {
  let s = 0
  for (let i = Math.max(0, from); i < to; i++) s += ret[i] * ret[i]
  return Math.sqrt(s)
}

/** EWMA per-minute variance after every return (same length as ret) */
function ewmaVar(ret: number[]): number[] {
  const out: number[] = []
  const seed = ret.slice(0, WINDOW)
  let v = seed.reduce((a, r) => a + r * r, 0) / Math.max(1, seed.length)
  for (const r of ret) {
    v = LAMBDA * v + (1 - LAMBDA) * r * r
    out.push(v)
  }
  return out
}

function rankPct(sorted: number[], x: number): number {
  if (!sorted.length) return 0.5
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const m = (lo + hi) >> 1
    if (sorted[m] < x) lo = m + 1
    else hi = m
  }
  return Math.min(0.99, Math.max(0.01, lo / sorted.length))
}

function statusAt(lastBarMs: number, now: number): PairStatus {
  const d = new Date(now)
  const wd = d.getUTCDay()
  const h = d.getUTCHours()
  if (wd === 6 || (wd === 0 && h < 21) || (wd === 5 && h >= 22)) return 'thin' // weekend
  const mins = h * 60 + d.getUTCMinutes()
  if (mins >= 21 * 60 + 6 && mins <= 21 * 60 + 10) return 'rollover'
  if (now - lastBarMs > 5 * 60_000) return 'stale'
  return 'live'
}

const round = (x: number, d = 3) => Number(x.toFixed(d))

function analyse(b: Bars) {
  const ret = returnsBp(b)
  const ev = ewmaVar(ret)
  const sorted: number[] = []
  for (let i = WINDOW; i <= ret.length; i++) sorted.push(realised(ret, i - WINDOW, i))
  sorted.sort((a, z) => a - z)
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0
  return { ret, ev, sorted, median }
}

async function livePair(spec: PairSpec, now: number, asOfMs?: number): Promise<PairSnapshot> {
  const b = upTo(await getBars(spec.pair), asOfMs)
  const { ret, ev, sorted, median } = analyse(b)
  const n = ret.length
  const rv30 = realised(ret, n - WINDOW, n)
  const perMin = Math.sqrt(ev[n - 1] ?? 0)
  const percentile = rankPct(sorted, rv30)
  const vol = HORIZONS_MIN.map((h) => round(perMin * Math.sqrt(h)))
  const mid = b.lastPrice
  const spreadBps = round(spec.spreadBps * (1 + percentile * 0.8))
  const half = (mid * spreadBps) / 20_000
  return {
    pair: spec.pair,
    base: spec.base,
    quote: spec.quote,
    bid: mid - half,
    ask: mid + half,
    mid,
    spreadBps,
    realizedVol30m: round(rv30),
    forecast: {
      horizonsMin: HORIZONS_MIN,
      vol,
      volLow: vol.map((v) => round(v * 0.75)),
      volHigh: vol.map((v) => round(v * 1.3)),
      spreadBps: HORIZONS_MIN.map((_, i) => round(spreadBps * (1 + 0.03 * i))),
    },
    percentile: round(percentile),
    volNormal30m: round(median),
    status: statusAt(b.t[b.t.length - 1], now),
  }
}

export async function liveSnapshot(base: CurrencyCode, asOf?: string): Promise<Snapshot> {
  const asOfMs = toMs(asOf)
  const now = asOfMs ?? Date.now()
  const results = await Promise.allSettled(LIVE_PAIRS.map((s) => livePair(s, now, asOfMs)))
  const pairs = results.flatMap((r) => (r.status === 'fulfilled' ? [r.value] : []))
  results.forEach((r) => r.status === 'rejected' && console.warn('[yahoo]', r.reason))
  if (!pairs.length) throw new Error('Yahoo Finance: no pairs could be loaded')
  const lastBars = await Promise.all(
    LIVE_PAIRS.map((s) => getBars(s.pair).then((b) => upTo(b, asOfMs)).then((b) => b.t[b.t.length - 1], () => 0)),
  )
  return { asOf: new Date(now).toISOString(), base, lastUpdated: new Date(Math.max(...lastBars)).toISOString(), pairs }
}

/** Last 24 h of bars that every pair has */
export async function liveReplayRange(): Promise<ReplayRange> {
  // Skip a lone straggler bar after a gap (Yahoo adds one closing bar hours after the market shuts)
  const lastBusy = (b: Bars) => {
    let i = b.t.length - 1
    while (i > 0 && b.t[i] - b.t[i - 1] > 5 * 60_000) i--
    return b.t[i]
  }
  const lasts = await Promise.all(LIVE_PAIRS.map((s) => getBars(s.pair).then(lastBusy)))
  const end = Math.floor(Math.min(...lasts) / 60_000) * 60_000
  const iso = (ms: number) => new Date(ms).toISOString().replace('.000Z', 'Z')
  return { start: iso(end - 24 * 3_600_000), end: iso(end), stepMin: 1 }
}

export async function liveHistory(pair: string, hours: number, asOf?: string, stepMin = 5): Promise<PairHistory | null> {
  const spec = LIVE_PAIRS.find((s) => s.pair === pair)
  if (!spec) return null
  const asOfMs = toMs(asOf)
  const b = upTo(await getBars(pair), asOfMs)
  const { ret, ev, sorted } = analyse(b)
  const horizon = HORIZONS_MIN[HORIZONS_MIN.length - 1]
  const step = stepMin * 60_000
  const end = Math.floor((asOfMs ?? Date.now()) / step) * step
  const steps = Math.floor((hours * 60) / stepMin)
  const lastBar = b.t[b.t.length - 1]
  const points: HistoryPoint[] = []

  // index of the last bar at or before time x (-1 if none)
  const idxAt = (x: number) => {
    if (b.t[0] > x) return -1
    let lo = 0
    let hi = b.t.length - 1
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1
      if (b.t[m] <= x) lo = m
      else hi = m - 1
    }
    return lo
  }

  for (let i = steps; i >= 0; i--) {
    const tMs = end - i * step
    const t = new Date(tMs).toISOString()
    const k = idxAt(tMs)
    if (k < 1 || tMs - b.t[k] > 5 * 60_000) {
      points.push({ t, mid: null, spreadBps: null, volForecast: null, volActual: null, percentile: null, status: 'no_data' })
      continue
    }
    const r = k - 1 // return ending at bar k
    const fc = Math.sqrt(ev[r]) * Math.sqrt(horizon)
    const pct = rankPct(sorted, realised(ret, r + 1 - WINDOW, r + 1))
    const actual = tMs + horizon * 60_000 <= lastBar ? realised(ret, r + 1, Math.min(ret.length, r + 1 + horizon)) : null
    points.push({
      t,
      mid: b.close[k],
      spreadBps: round(spec.spreadBps * (1 + pct * 0.8)),
      volForecast: round(fc),
      volForecastLow: round(fc * 0.75),
      volForecastHigh: round(fc * 1.3),
      volActual: actual == null ? null : round(actual),
      percentile: round(pct),
      status: 'live',
    })
  }
  return { pair: spec.pair, base: spec.base, quote: spec.quote, stepMin, horizonMin: horizon, points }
}
