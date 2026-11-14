/**
 * Full-screen interactive globe.
 * - Auto-spins; pauses while you drag; resumes 3 s after you let go (unless something is in focus)
 * - Hover tooltip on every country
 * - Click a country → it's selected and the camera flies there; a lit zone also becomes the "To" currency
 * - The "To" currency's whole zone is highlighted (e.g. every eurozone country for EUR)
 * - Click empty globe or press Esc to clear the selection and resume spinning
 * - Respects the OS "reduce motion" setting (no spin, instant camera moves)
 *
 * Step 3 replaces the single "forecast available" tint with storm colours from mock data.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Globe, { type GlobeMethods } from 'react-globe.gl'
import { HOME_CURRENCY, SINGAPORE_MARKER, ZONE_BY_COUNTRY, type CurrencyZone } from '../../data/currencies'
import { countryCode, escapeHtml, type CountryFeature } from '../../lib/geo'
import { usePrefersReducedMotion, useWindowSize } from '../../lib/hooks'
import { cssVar, withAlpha } from '../../lib/theme'
import { useAppStore } from '../../store/useAppStore'

const RESUME_SPIN_AFTER_MS = 3000
const FLY_MS = 1200
const SPIN_SPEED = 0.35
const START_VIEW = { lat: 22, lng: 20, altitude: 2.4 }
const FOCUS_ALTITUDE = 1.9

type SingaporeMarker = typeof SINGAPORE_MARKER

function tooltipHtml(name: string, zone: CurrencyZone | null): string {
  const detail = !zone
    ? 'No forecast'
    : zone.code === HOME_CURRENCY
      ? `${zone.code} · home currency`
      : `${zone.code} · forecast available`
  return `<div class="globe-tip"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(detail)}</span></div>`
}

export default function FxGlobe() {
  const globeRef = useRef<GlobeMethods | undefined>(undefined)
  const resumeTimer = useRef<number | undefined>(undefined)

  const selectedCode = useAppStore((s) => s.selected?.code ?? null)
  const quote = useAppStore((s) => s.quote)
  const focus = useAppStore((s) => s.focus)
  const selectCountry = useAppStore((s) => s.selectCountry)
  const clearSelection = useAppStore((s) => s.clearSelection)

  const focusRef = useRef(focus) // read by the spin timer, which outlives renders
  useEffect(() => {
    focusRef.current = focus
  }, [focus])

  const reducedMotion = usePrefersReducedMotion()
  const { width, height } = useWindowSize()

  const [countries, setCountries] = useState<CountryFeature[]>([])
  const [loadError, setLoadError] = useState(false)
  const [hoveredCode, setHoveredCode] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  // Colours come from the @theme tokens in index.css
  const colors = useMemo(
    () => ({
      accent: cssVar('--color-accent', '#7c9cff'),
      ink: cssVar('--color-ink', '#e6ebf5'),
      nodata: cssVar('--color-nodata', '#334155'),
    }),
    [],
  )

  // 1. Load country shapes once (local file, so venue wifi can't break it)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/assets/countries.geojson', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<{ features: CountryFeature[] }>
      })
      .then((geo) => setCountries(geo.features.filter((f) => countryCode(f.properties) !== 'ATA')))
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return
        console.error('[FxGlobe] could not load countries.geojson', err)
        setLoadError(true)
      })
    return () => controller.abort()
  }, [])

  const singaporeData = useMemo(() => [SINGAPORE_MARKER], [])

  // 2. Spin + pause-on-drag, set up once the globe exists
  useEffect(() => {
    const globe = globeRef.current
    if (!ready || !globe) return
    const controls = globe.controls()

    controls.autoRotateSpeed = SPIN_SPEED
    controls.autoRotate = !reducedMotion && !focusRef.current

    const onStart = () => {
      window.clearTimeout(resumeTimer.current)
      controls.autoRotate = false
    }
    const onEnd = () => {
      if (reducedMotion) return
      window.clearTimeout(resumeTimer.current)
      resumeTimer.current = window.setTimeout(() => {
        if (!focusRef.current) controls.autoRotate = true
      }, RESUME_SPIN_AFTER_MS)
    }

    controls.addEventListener('start', onStart)
    controls.addEventListener('end', onEnd)
    return () => {
      controls.removeEventListener('start', onStart)
      controls.removeEventListener('end', onEnd)
      window.clearTimeout(resumeTimer.current)
    }
  }, [ready, reducedMotion])

  // 3. Fly to whatever is in focus (from a globe click OR the convert card); spin again when cleared
  useEffect(() => {
    const globe = globeRef.current
    if (!ready || !globe) return
    if (focus) {
      window.clearTimeout(resumeTimer.current)
      globe.controls().autoRotate = false
      globe.pointOfView({ ...focus, altitude: FOCUS_ALTITUDE }, reducedMotion ? 0 : FLY_MS)
    } else if (!reducedMotion) {
      globe.controls().autoRotate = true
    }
  }, [ready, focus, reducedMotion])

  // Esc clears the selection (but not while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (e.key === 'Escape' && !target?.closest('input, select, textarea')) clearSelection()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clearSelection])

  // 4. Polygon styling (accessors re-run when hover / selection / "To" currency change)
  const isActive = useCallback(
    (code: string) =>
      code === hoveredCode || code === selectedCode || ZONE_BY_COUNTRY.get(code)?.code === quote,
    [hoveredCode, selectedCode, quote],
  )

  const capColor = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      const lit = ZONE_BY_COUNTRY.has(code)
      if (lit) return withAlpha(colors.accent, isActive(code) ? 0.85 : 0.45)
      return withAlpha(colors.nodata, isActive(code) ? 0.75 : 0.35)
    },
    [colors, isActive],
  )

  const sideColor = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      return ZONE_BY_COUNTRY.has(code) ? withAlpha(colors.accent, 0.25) : withAlpha(colors.nodata, 0.15)
    },
    [colors],
  )

  const strokeColor = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      return code === selectedCode ? colors.ink : withAlpha(colors.ink, 0.12)
    },
    [colors, selectedCode],
  )

  const altitude = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      if (code === selectedCode) return 0.03
      if (isActive(code)) return 0.02
      return ZONE_BY_COUNTRY.has(code) ? 0.012 : 0.006
    },
    [selectedCode, isActive],
  )

  const polygonLabel = useCallback((obj: object) => {
    const p = (obj as CountryFeature).properties
    return tooltipHtml(p.NAME, ZONE_BY_COUNTRY.get(countryCode(p)) ?? null)
  }, [])

  // 5. Interaction handlers
  const handlePolygonClick = useCallback(
    (obj: object, _event: MouseEvent, coords: { lat: number; lng: number }) => {
      const p = (obj as CountryFeature).properties
      const code = countryCode(p)
      const zone = ZONE_BY_COUNTRY.get(code) ?? null
      // Single-country zones fly to a fixed anchor (so clicking Alaska still centres the US)
      const target = zone && zone.countries.length === 1 ? zone.anchor : coords
      selectCountry({ code, name: p.NAME, zone }, { lat: target.lat, lng: target.lng })
    },
    [selectCountry],
  )

  const handlePolygonHover = useCallback((obj: object | null) => {
    setHoveredCode(obj ? countryCode((obj as CountryFeature).properties) : null)
  }, [])

  const handlePointClick = useCallback(
    (obj: object) => {
      const m = obj as SingaporeMarker
      selectCountry({ code: m.code, name: m.name, zone: ZONE_BY_COUNTRY.get(m.code) ?? null }, { lat: m.lat, lng: m.lng })
    },
    [selectCountry],
  )

  const handlePointHover = useCallback((obj: object | null) => {
    setHoveredCode(obj ? (obj as SingaporeMarker).code : null)
  }, [])

  const handleGlobeReady = useCallback(() => {
    globeRef.current?.pointOfView(START_VIEW, 0)
    setReady(true)
  }, [])

  return (
    <div className="absolute inset-0" aria-label="Interactive globe of FX forecast zones">
      <Globe
        ref={globeRef}
        width={width}
        height={height}
        onGlobeReady={handleGlobeReady}
        // Textures (local copies in public/assets)
        globeImageUrl="/assets/earth-dark.jpg"
        bumpImageUrl="/assets/earth-topology.png"
        backgroundImageUrl="/assets/night-sky.png"
        atmosphereColor={colors.accent}
        atmosphereAltitude={0.18}
        // Countries
        polygonsData={countries}
        polygonCapColor={capColor}
        polygonSideColor={sideColor}
        polygonStrokeColor={strokeColor}
        polygonAltitude={altitude}
        polygonLabel={polygonLabel}
        polygonsTransitionDuration={reducedMotion ? 0 : 250}
        onPolygonClick={handlePolygonClick}
        onPolygonHover={handlePolygonHover}
        // Singapore (too small to be a polygon at this map scale)
        pointsData={singaporeData}
        pointLat="lat"
        pointLng="lng"
        pointColor={() => colors.accent}
        pointAltitude={(obj: object) => (isActive((obj as SingaporeMarker).code) ? 0.05 : 0.025)}
        pointRadius={0.9}
        pointLabel={(obj: object) => tooltipHtml((obj as SingaporeMarker).name, ZONE_BY_COUNTRY.get('SGP') ?? null)}
        onPointClick={handlePointClick}
        onPointHover={handlePointHover}
        // Clicking empty globe clears the selection
        onGlobeClick={clearSelection}
      />
      {loadError && (
        <p role="alert" className="glass absolute top-20 left-1/2 -translate-x-1/2 px-4 py-2 text-sm text-ink-muted">
          Couldn&rsquo;t load the country map. Check that public/assets/countries.geojson exists.
        </p>
      )}
    </div>
  )
}
