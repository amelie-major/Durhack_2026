/**
 * App state shared by the globe and the panels.
 * Step 3 adds asOf / tolerance / mode / replay to this same store.
 */
import { create } from 'zustand'
import type { CurrencyCode } from '../types'
import { HOME_CURRENCY, ZONES, type CurrencyZone } from '../data/currencies'

export interface SelectedCountry {
  code: string // ISO alpha-3 of the clicked country ("" when chosen from the dropdown)
  name: string
  zone: CurrencyZone | null // null = "No forecast"
}

export interface CameraFocus {
  lat: number
  lng: number
}

interface AppState {
  amount: number
  base: CurrencyCode
  quote: CurrencyCode
  selected: SelectedCountry | null
  /** Where the globe camera should be; null = free spin */
  focus: CameraFocus | null

  setAmount: (amount: number) => void
  setBase: (code: CurrencyCode) => void
  setQuote: (code: CurrencyCode) => void
  swap: () => void
  /** Called by the globe when a country is clicked */
  selectCountry: (country: SelectedCountry, focus: CameraFocus) => void
  clearSelection: () => void
}

const zoneOf = (code: CurrencyCode) => ZONES.find((z) => z.code === code)

/** Focus the camera on a currency zone chosen from the dropdowns */
function focusZone(code: CurrencyCode): Pick<AppState, 'selected' | 'focus'> {
  const zone = zoneOf(code)
  if (!zone) return { selected: null, focus: null }
  return { selected: { code: '', name: zone.name, zone }, focus: { ...zone.anchor } }
}

export const useAppStore = create<AppState>()((set, get) => ({
  amount: 10_000,
  base: HOME_CURRENCY,
  quote: 'JPY',
  selected: null,
  focus: null,

  setAmount: (amount) => set({ amount }),

  setBase: (code) => {
    const { quote, base } = get()
    if (code === quote) set({ base: code, quote: base }) // picking the same currency swaps them
    else set({ base: code })
  },

  setQuote: (code) => {
    const { base, quote } = get()
    if (code === base) set({ quote: code, base: quote, ...focusZone(code) })
    else set({ quote: code, ...focusZone(code) })
  },

  swap: () => {
    const { base, quote } = get()
    set({ base: quote, quote: base, ...focusZone(base) })
  },

  selectCountry: (country, focus) => {
    const { base } = get()
    const zoneCode = country.zone?.code
    // Clicking a lit zone that isn't the "From" currency makes it the "To" currency
    if (zoneCode && zoneCode !== base) set({ quote: zoneCode, selected: country, focus })
    else set({ selected: country, focus })
  },

  clearSelection: () => set({ selected: null, focus: null }),
}))
