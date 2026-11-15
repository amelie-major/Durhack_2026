/**
 * Currency zones the model has data for: every currency in a pair in data/processed/instruments.csv
 * (21 pairs → 17 currencies; tests/test_live_pairs.py keeps the two in step).
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
  { code: 'GBP', name: 'British pound', countries: ['GBR'], anchor: { lat: 52.5, lng: -1.5 } },
  { code: 'AUD', name: 'Australian dollar', countries: ['AUS'], anchor: { lat: -25.3, lng: 133.8 } },
  { code: 'NZD', name: 'New Zealand dollar', countries: ['NZL'], anchor: { lat: -41.3, lng: 174.0 } },
  { code: 'CAD', name: 'Canadian dollar', countries: ['CAN'], anchor: { lat: 56.1, lng: -106.3 } },
  { code: 'CHF', name: 'Swiss franc', countries: ['CHE'], anchor: { lat: 46.8, lng: 8.2 } },
  { code: 'CNH', name: 'Chinese yuan (offshore)', countries: ['CHN'], anchor: { lat: 35.9, lng: 104.2 } },
  { code: 'HKD', name: 'Hong Kong dollar', countries: ['HKG'], anchor: { lat: 22.32, lng: 114.17 } },
  { code: 'MXN', name: 'Mexican peso', countries: ['MEX'], anchor: { lat: 23.6, lng: -102.6 } },
  { code: 'NOK', name: 'Norwegian krone', countries: ['NOR'], anchor: { lat: 61.0, lng: 9.0 } },
  { code: 'SEK', name: 'Swedish krona', countries: ['SWE'], anchor: { lat: 62.0, lng: 15.0 } },
  { code: 'PLN', name: 'Polish zloty', countries: ['POL'], anchor: { lat: 52.1, lng: 19.4 } },
  { code: 'TRY', name: 'Turkish lira', countries: ['TUR'], anchor: { lat: 39.0, lng: 35.2 } },
  { code: 'ZAR', name: 'South African rand', countries: ['ZAF'], anchor: { lat: -30.6, lng: 22.9 } },
]

/** Country code → its currency zone (only for the zones above) */
export const ZONE_BY_COUNTRY = new Map<string, CurrencyZone>(
  ZONES.flatMap((zone) => zone.countries.map((country) => [country, zone] as const)),
)

export interface PointMarker {
  code: string
  name: string
  lat: number
  lng: number
}

/**
 * Singapore and Hong Kong aren't in the 110m countries file (too small), so the globe
 * draws them as point markers instead of polygons.
 */
export const POINT_MARKERS: PointMarker[] = [
  { code: 'SGP', name: 'Singapore', lat: 1.35, lng: 103.82 },
  { code: 'HKG', name: 'Hong Kong', lat: 22.32, lng: 114.17 },
]