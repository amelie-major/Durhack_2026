/**
 * The 3 storm levels. Each has a weather word, a plain volatility label, an icon and a colour,
 * so it never relies on colour alone and never implies which way the rate will move.
 * "High" means high compared with this pair's own normal, not "bad".
 */
import { CloudLightning, CloudSun, Sun, type LucideIcon } from 'lucide-react'

export type StormLevel = 'calm' | 'unsettled' | 'storm'

export interface StormLevelInfo {
  level: StormLevel
  name: string // weather word
  volatility: string // plain-English meaning
  Icon: LucideIcon
  textClass: string // Tailwind colour class from the @theme tokens
}

export const STORM_LEVELS: StormLevelInfo[] = [
  { level: 'calm', name: 'Calm', volatility: 'Low volatility', Icon: Sun, textClass: 'text-calm' },
  { level: 'unsettled', name: 'Unsettled', volatility: 'Medium volatility', Icon: CloudSun, textClass: 'text-unsettled' },
  { level: 'storm', name: 'Storm', volatility: 'High volatility', Icon: CloudLightning, textClass: 'text-storm' },
]
