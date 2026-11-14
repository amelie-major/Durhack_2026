/**
 * Shared data contract between the model team and the UI.
 * PROPOSAL v1 — agree with the team, then change it in ONE place (here).
 * See project doc "UI plan & build instructions (Emilia)", section 4.
 */

export type CurrencyCode = string // e.g. "GBP", "JPY"

export interface PairForecast {
  /** Minutes ahead for each forecast point, e.g. [5, 10, ..., 60] */
  horizonsMin: number[]
  /** Forecast volatility at each horizon (same length as horizonsMin) */
  vol: number[]
  /** Forecast spread in basis points at each horizon */
  spreadBps: number[]
  /** Optional uncertainty band */
  volLow?: number[]
  volHigh?: number[]
}

export interface PairSnapshot {
  pair: string // "GBPJPY"
  quote: CurrencyCode // "JPY"
  bid: number
  ask: number
  mid: number
  spreadBps: number
  realizedVol30m: number
  forecast: PairForecast
  /** How unusual current vol is vs this pair's own history, 0–1. Drives globe colour. */
  percentile: number
}

export interface Snapshot {
  asOf: string // ISO timestamp — the (simulated) "now"
  base: CurrencyCode
  lastUpdated: string // ISO timestamp of the newest data used
  pairs: PairSnapshot[]
}

/** UI-side derived types */
export type StormLevel = 'calm' | 'unsettled' | 'storm' | 'nodata'
export type Tolerance = 'relaxed' | 'balanced' | 'cautious'
export type ViewMode = 'simple' | 'pro'
