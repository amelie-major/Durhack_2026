/**
 * The model's Buy / Sell / Hold signal for a pair (the "Buy / Sell" map layer).
 *
 * Rule (models/signal.py, computed by the forecast server, never in the browser):
 *   expected move = (2 × P(up) − 1) × forecast vol × √(2/π)
 *   BUY if it is above the forecast spread, SELL if below minus the spread, otherwise HOLD.
 * BUY EUR/USD means buy EUR and sell USD. It is a model output, not advice.
 */
import { Minus, TrendingDown, TrendingUp, type LucideIcon } from 'lucide-react'
import type { PairSnapshot, SignalAction } from '../types'
import { slashPair } from './pro'

export interface SignalInfo {
  action: SignalAction
  name: string
  meaning: string
  Icon: LucideIcon
  textClass: string // Tailwind colour class from the @theme tokens
  colorVar: string // CSS variable, for the globe (WebGL can't use classes)
}

export const SIGNALS: SignalInfo[] = [
  { action: 'buy', name: 'Buy', meaning: 'Expected rise beats the cost', Icon: TrendingUp, textClass: 'text-buy', colorVar: '--color-buy' },
  { action: 'sell', name: 'Sell', meaning: 'Expected fall beats the cost', Icon: TrendingDown, textClass: 'text-sell', colorVar: '--color-sell' },
  { action: 'hold', name: 'Hold', meaning: 'Move too small to pay the spread', Icon: Minus, textClass: 'text-hold', colorVar: '--color-hold' },
]

export const SIGNAL_INFO = Object.fromEntries(SIGNALS.map((s) => [s.action, s])) as Record<SignalAction, SignalInfo>

/** The pair's signal, or null when there isn't one (no pair, or the model server is offline) */
export function actionFor(pair: PairSnapshot | undefined): SignalAction | null {
  return pair?.signal?.action ?? null
}

const signed = (bp: number) => `${bp >= 0 ? '+' : '−'}${Math.abs(bp).toFixed(1)}`

/** "BUY USD/JPY: buy USD, sell JPY" / "HOLD USD/JPY" */
export function signalHeadline(pair: PairSnapshot): string | null {
  const action = actionFor(pair)
  if (!action) return null
  const name = `${SIGNAL_INFO[action].name.toUpperCase()} ${slashPair(pair.pair)}`
  if (action === 'buy') return `${name}: buy ${pair.base}, sell ${pair.quote}`
  if (action === 'sell') return `${name}: sell ${pair.base}, buy ${pair.quote}`
  return name
}

/** "expected +1.2 bp vs 0.4 bp cost, next 10 min" */
export function signalDetail(pair: PairSnapshot): string | null {
  const s = pair.signal
  if (!s) return null
  return `expected ${signed(s.expectedMoveBps)} bp vs ${s.hurdleBps.toFixed(1)} bp cost, next ${s.horizonMin} min`
}
