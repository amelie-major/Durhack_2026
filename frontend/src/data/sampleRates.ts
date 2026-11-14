/**
 * SAMPLE rates for the top ticker, so the layout can be built before the team's data exists.
 * These are made-up, plausible numbers — NOT live or historical prices. The UI labels them "Sample data".
 * Step 3 replaces this with values from src/data/api.ts.
 */
import type { StormLevel } from '../lib/storm'

export interface TickerItem {
  pair: string // display form, e.g. "EUR/USD"
  mid: number
  decimals: number
  level: StormLevel // forecast volatility level for that pair
}

const EURUSD = 1.165
const USDJPY = 147.2
const USDSGD = 1.285

export const SAMPLE_TICKER: TickerItem[] = [
  { pair: 'EUR/USD', mid: EURUSD, decimals: 4, level: 'calm' },
  { pair: 'USD/JPY', mid: USDJPY, decimals: 2, level: 'storm' },
  { pair: 'USD/SGD', mid: USDSGD, decimals: 4, level: 'unsettled' },
  // Crosses derived from the 3 pairs above
  { pair: 'EUR/JPY', mid: EURUSD * USDJPY, decimals: 2, level: 'unsettled' },
  { pair: 'EUR/SGD', mid: EURUSD * USDSGD, decimals: 4, level: 'calm' },
  { pair: 'SGD/JPY', mid: USDJPY / USDSGD, decimals: 2, level: 'unsettled' },
]
