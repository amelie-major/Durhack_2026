/**
 * Volatility level for each currency zone on the globe, against the "From" (home) currency.
 * All the data's pairs involve USD, so a zone's level comes from the USD/<zone> pair.
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
