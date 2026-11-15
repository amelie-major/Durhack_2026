/**
 * Read design tokens from CSS (index.css @theme) so WebGL code uses the same colours.
 * Change the palette in index.css only — the globe follows automatically.
 */
export function cssVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

/** "#rrggbb" (or "#rrggbbaa") + alpha → "rgba(r, g, b, a)" */
export function withAlpha(hex: string, alpha: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(hex)
  if (!m) return hex
  const [r, g, b] = m.slice(1, 4).map((h) => parseInt(h, 16))
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}