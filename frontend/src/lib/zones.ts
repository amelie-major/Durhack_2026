/**
 * Volatility level for each currency zone on the globe, against the "From" (home) currency.
 * A zone's level comes from its direct pair with the From currency (e.g. EURGBP) when the data has one,
 * otherwise from the two legs through USD.
 */
import type { CurrencyCode, PairSnapshot, Snapshot, StormLevel, Tolerance } from '../types'
import { levelFor, worseLevel } from './storm'

export type ZoneLevel = StormLevel | 'home'

/** Find the pair between two currencies in either direction (e.g. USD+JPY → USDJPY) */
export function findPair(snapshot: Snapshot | null, a: CurrencyCode, b: CurrencyCode): PairSnapshot | undefined {
  return snapshot?.pairs.find((p) => (p.base === a && p.quote === b) || (p.base === b && p.quote === a))
}

/**
 * Level of a currency zone as seen from the "From" currency.
 * - the From currency itself → "home"
 * - a direct pair exists → its level
 * - otherwise go through USD (e.g. EUR → JPY = worse of EURUSD and USDJPY)
 */
export function zoneLevel(snapshot: Snapshot | null, zone: CurrencyCode, from: CurrencyCode, tolerance: Tolerance): ZoneLevel {
  if (zone === from) return 'home'
  const direct = findPair(snapshot, from, zone)
  if (direct) return levelFor(direct, tolerance)
  const legA = findPair(snapshot, from, 'USD')
  const legB = findPair(snapshot, 'USD', zone)
  return worseLevel(levelFor(legA, tolerance), levelFor(legB, tolerance))
}

/**
 * The pairs a From → To conversion goes through, each with the currency you hold on that leg:
 * the direct pair if the data has one, otherwise the two legs through USD. Empty if not covered.
 */
export function routeLegs(
  snapshot: Snapshot | null,
  from: CurrencyCode,
  to: CurrencyCode,
): { pair: PairSnapshot; from: CurrencyCode }[] {
  if (from === to) return []
  const direct = findPair(snapshot, from, to)
  if (direct) return [{ pair: direct, from }]
  const legA = findPair(snapshot, from, 'USD')
  const legB = findPair(snapshot, 'USD', to)
  return legA && legB ? [{ pair: legA, from }, { pair: legB, from: 'USD' }] : []
}
