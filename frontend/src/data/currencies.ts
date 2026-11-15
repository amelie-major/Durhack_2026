/**
 * Currency zones the model has data for (EURUSD, USDJPY, USDSGD → 4 currencies).
 * Every other country on the globe shows "No forecast".
 * Step 3 adds names/flags for the convert card's currency search.
 */
import type { CurrencyCode } from '../types'

export interface CurrencyZone {
  code: CurrencyCode
  name: string
  /** ISO 3166-1 alpha-3 codes of the countries that use this currency */
  countries: string[]
  /** Where the camera flies to for single-country zones */
  anchor: { lat: number; lng: number }
}

export const HOME_CURRENCY: CurrencyCode = 'USD'

/**
 * Euro area: 21 members since Bulgaria joined on 1 Jan 2026.
 * Countries that use the euro without being members (Montenegro, Kosovo, Andorra…) are left out on purpose.
 * Malta (MLT) is too small for the 110m map, so it has no polygon.
 */
const EURO_AREA = [
  'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL',
  'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN', 'ESP',
]

export const ZONES: CurrencyZone[] = [
  { code: 'USD', name: 'US dollar', countries: ['USA'], anchor: { lat: 39.5, lng: -98.4 } },
  { code: 'EUR', name: 'Euro', countries: EURO_AREA, anchor: { lat: 50.1, lng: 8.7 } },
  { code: 'JPY', name: 'Japanese yen', countries: ['JPN'], anchor: { lat: 36.2, lng: 138.3 } },
  { code: 'SGD', name: 'Singapore dollar', countries: ['SGP'], anchor: { lat: 1.35, lng: 103.82 } },
]

/** Country code → its currency zone (only for the 4 zones above) */
export const ZONE_BY_COUNTRY = new Map<string, CurrencyZone>(
  ZONES.flatMap((zone) => zone.countries.map((country) => [country, zone] as const)),
)

/**
 * Singapore isn't in the 110m countries file (too small), so the globe
 * draws it as a point marker instead of a polygon.
 */
export const SINGAPORE_MARKER = {
  code: 'SGP',
  name: 'Singapore',
  lat: 1.35,
  lng: 103.82,
}