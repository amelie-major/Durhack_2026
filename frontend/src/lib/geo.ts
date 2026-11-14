/** Helpers for the Natural Earth 110m countries GeoJSON in public/assets/ */

export interface CountryProps {
  ISO_A3: string
  ADM0_A3: string
  NAME: string
}

export interface CountryFeature {
  type: 'Feature'
  properties: CountryProps
  geometry: { type: string; coordinates: number[] }
}

/**
 * France, Norway, Kosovo, N. Cyprus and Somaliland have ISO_A3 = "-99" in this file.
 * Fall back to ADM0_A3 only for those (ADM0_A3 differs from ISO elsewhere, e.g. South Sudan).
 */
export function countryCode(p: CountryProps): string {
  return p.ISO_A3 !== '-99' ? p.ISO_A3 : p.ADM0_A3
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (ch) => `&#${ch.charCodeAt(0)};`)
}