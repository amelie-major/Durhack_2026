/**
 * THE ONLY DOOR TO DATA. Every component gets forecasts through getSnapshot().
 * Today it returns mock data; when the team's export is ready, swap the body for a
 * fetch of /data/<day>.json and keep the same return type (src/types.ts).
 */
import type { CurrencyCode, Snapshot } from '../types'
import { mockSnapshot } from './mock'

export const USING_SAMPLE_DATA = true

export async function getSnapshot(asOf: string, base: CurrencyCode): Promise<Snapshot> {
  return mockSnapshot(asOf, base)
}
