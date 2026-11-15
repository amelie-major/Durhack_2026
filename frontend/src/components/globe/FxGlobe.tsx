/**
 * Full-screen interactive globe.
 * - Auto-spins; pauses while you drag; resumes 3 s after you let go (unless something is in focus)
 * - Hover tooltip on every country
 * - Click a country → it's selected and the camera flies there; a lit zone also becomes the "To" currency
 * - The "To" currency's whole zone is highlighted (e.g. every eurozone country for EUR)
 * - Click empty globe or press Esc to clear the selection and resume spinning
 * - Respects the OS "reduce motion" setting (no spin, instant camera moves)
 *
 * - Lit zones are coloured by their volatility level vs the "From" currency (Calm / Unsettled / Storm);
 *   the "From" currency's own zone is the accent colour; zones with no usable data are slate.
 * - "Extreme" zones (forecast ≥ 1.5× normal) are bright red with pulsing warning rings.
 * - A line (arc) joins the two countries of every pair we have data for.
 * - Map layer toggle: "Volatility" (above) or "Buy / Sell": zones and lines go green (buy), red (sell)
 *   or grey (hold) from the model's signal, with pulsing rings at both ends while a buy/sell is on.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Globe, { type GlobeMethods } from 'react-globe.gl'
import { POINT_MARKERS, ZONES, ZONE_BY_COUNTRY, type CurrencyZone, type PointMarker } from '../../data/currencies'
import { countryCode, escapeHtml, type CountryFeature } from '../../lib/geo'
import { usePrefersReducedMotion, useWindowSize } from '../../lib/hooks'
import { slashPair } from '../../lib/pro'
import { actionFor, signalDetail, signalHeadline } from '../../lib/signal'
import { costBand, costRatioPct, LEVEL_INFO, levelFor, volRatio } from '../../lib/storm'
import { cssVar, withAlpha } from '../../lib/theme'
import { findPair, zoneLevel, type ZoneLevel } from '../../lib/zones'
import type { MapLayer, PairSnapshot, Snapshot } from '../../types'
import { useAppStore } from '../../store/useAppStore'

const RESUME_SPIN_AFTER_MS = 3000
const FLY_MS = 1200
const SPIN_SPEED = 0.35
const START_VIEW = { lat: 22, lng: 20, altitude: 2.4 }
const FOCUS_ALTITUDE = 1.9

/** A line between the two countries of a pair */
interface PairArc {
  pair: PairSnapshot
  startLat: number
  startLng: number
  endLat: number
  endLng: number
  color: string
  /** A buy or sell is on (drawn thicker, with rings at both ends) */
  active: boolean
}

interface Ring {
  lat: number
  lng: number
  color: string
}

const ZONE_BY_CODE = new Map(ZONES.map((z) => [z.code, z]))

function signalLines(pair: PairSnapshot | undefined): string[] {
  if (!pair) return ['No signal: no direct pair with your "From" currency']
  const headline = signalHeadline(pair)
  if (!headline) return ['No signal (forecast server offline)']
  const detail = signalDetail(pair)
  return detail ? [headline, detail] : [headline]
}

function arcTooltip(arc: PairArc, layer: MapLayer, level: ZoneLevel): string {
  const lines: string[] = []
  if (layer === 'signal') lines.push(...signalLines(arc.pair))
  else if (level === 'nodata' || level === 'home') lines.push('No reliable data right now')
  else lines.push(`${LEVEL_INFO[level].name} · ${LEVEL_INFO[level].volatility.toLowerCase()} (next 30 min)`)
  const body = lines.map((l) => `<span>${escapeHtml(l)}</span>`).join('')
  return `<div class="globe-tip"><strong>${escapeHtml(slashPair(arc.pair.pair))}</strong>${body}</div>`
}

function tooltipHtml(
  name: string,
  zone: CurrencyZone | null,
  level: ZoneLevel | undefined,
  snapshot: Snapshot | null,
  from: string,
  layer: MapLayer,
): string {
  const lines: string[] = []
  if (!zone) lines.push('No forecast for this country')
  else if (level === 'home') lines.push(`${zone.code} · your "From" currency`)
  else if (layer === 'signal') lines.push(...signalLines(findPair(snapshot, from, zone.code)))
  else if (!level || level === 'nodata') lines.push(`${zone.code} · no reliable data right now`)
  else {
    const info = LEVEL_INFO[level]
    lines.push(`${zone.code} · ${info.name} · ${info.volatility.toLowerCase()} (next 30 min)`)
    const pair = findPair(snapshot, from, zone.code)
    const ratio = volRatio(pair)
    if (level === 'extreme' && ratio) lines.push(`⚠ Expected move ≈ ${ratio.toFixed(1)}× its normal size`)
    const band = pair ? costBand(costRatioPct(pair)) : null
    if (band) lines.push(band.label)
  }
  const body = lines.map((l) => `<span>${escapeHtml(l)}</span>`).join('')
  return `<div class="globe-tip"><strong>${escapeHtml(name)}</strong>${body}</div>`
}

export default function FxGlobe() {
  const globeRef = useRef<GlobeMethods | undefined>(undefined)
  const resumeTimer = useRef<number | undefined>(undefined)

  const selectedCode = useAppStore((s) => s.selected?.code ?? null)
  const quote = useAppStore((s) => s.quote)
  const base = useAppStore((s) => s.base)
  const snapshot = useAppStore((s) => s.snapshot)
  const tolerance = useAppStore((s) => s.tolerance)
  const mapLayer = useAppStore((s) => s.mapLayer)
  const setQuote = useAppStore((s) => s.setQuote)
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
      calm: cssVar('--color-calm', '#2dd4bf'),
      unsettled: cssVar('--color-unsettled', '#fbbf24'),
      storm: cssVar('--color-storm', '#f43f8e'),
      extreme: cssVar('--color-extreme', '#ff2e3b'),
      buy: cssVar('--color-buy', '#22c55e'),
      sell: cssVar('--color-sell', '#ef4444'),
      hold: cssVar('--color-hold', '#94a3b8'),
    }),
    [],
  )

  // Volatility level of each currency zone, seen from the "From" currency
  const zoneLevels = useMemo(
    () => new Map<string, ZoneLevel>(ZONES.map((z) => [z.code, zoneLevel(snapshot, z.code, base, tolerance)])),
    [snapshot, base, tolerance],
  )

  const levelColor = useCallback(
    (level: ZoneLevel | undefined) => {
      if (level === 'home') return colors.accent
      if (!level || level === 'nodata') return colors.nodata
      return colors[level]
    },
    [colors],
  )

  /** Buy / Sell layer: green, red or grey from the model's signal; slate when there is no signal */
  const signalColor = useCallback(
    (pair: PairSnapshot | undefined) => {
      const action = actionFor(pair)
      return action ? colors[action] : colors.nodata
    },
    [colors],
  )

  /** Colour for a country code, or null if it isn't in a currency zone */
  const zoneColor = useCallback(
    (code: string) => {
      const zone = ZONE_BY_COUNTRY.get(code)
      if (!zone) return null
      if (mapLayer === 'volatility') return levelColor(zoneLevels.get(zone.code))
      // Buy / Sell: the signal of this zone's pair with the "From" currency (same colour as its line)
      return zone.code === base ? colors.accent : signalColor(findPair(snapshot, base, zone.code))
    },
    [levelColor, zoneLevels, mapLayer, base, colors, signalColor, snapshot],
  )

  // One line per pair we have data for, between the two currencies' countries
  const arcs = useMemo<PairArc[]>(
    () =>
      (snapshot?.pairs ?? []).flatMap((pair) => {
        const a = ZONE_BY_CODE.get(pair.base)
        const b = ZONE_BY_CODE.get(pair.quote)
        if (!a || !b) return []
        const action = actionFor(pair)
        const signalLayer = mapLayer === 'signal'
        return [
          {
            pair,
            startLat: a.anchor.lat,
            startLng: a.anchor.lng,
            endLat: b.anchor.lat,
            endLng: b.anchor.lng,
            color: signalLayer ? signalColor(pair) : levelColor(levelFor(pair, tolerance)),
            active: signalLayer && (action === 'buy' || action === 'sell'),
          },
        ]
      }),
    [snapshot, mapLayer, tolerance, signalColor, levelColor],
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

  // Pulsing rings (none when "reduce motion" is on):
  // volatility layer → on every "Extreme" zone; Buy / Sell layer → green/red at both ends of an active line
  const rings = useMemo<Ring[]>(() => {
    if (reducedMotion) return []
    if (mapLayer === 'volatility') {
      return ZONES.filter((z) => zoneLevels.get(z.code) === 'extreme').map((z) => ({ ...z.anchor, color: colors.extreme }))
    }
    return arcs
      .filter((a) => a.active)
      .flatMap((a) => [
        { lat: a.startLat, lng: a.startLng, color: a.color },
        { lat: a.endLat, lng: a.endLng, color: a.color },
      ])
  }, [zoneLevels, reducedMotion, mapLayer, arcs, colors])
  const ringColor = useCallback((obj: object) => (t: number) => withAlpha((obj as Ring).color, Math.max(0, 1 - t)), [])

  const arcLabel = useCallback(
    (obj: object) => {
      const arc = obj as PairArc
      return arcTooltip(arc, mapLayer, levelFor(arc.pair, tolerance))
    },
    [mapLayer, tolerance],
  )

  /** Clicking a line points the app at that pair (its currency that isn't the "From" one) */
  const handleArcClick = useCallback(
    (obj: object) => {
      const { pair } = obj as PairArc
      setQuote(pair.base === base ? pair.quote : pair.quote === base ? pair.base : pair.quote === 'USD' ? pair.base : pair.quote)
    },
    [setQuote, base],
  )

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
      const color = zoneColor(code)
      if (color) return withAlpha(color, isActive(code) ? 1 : 0.78)
      return withAlpha(colors.nodata, isActive(code) ? 0.75 : 0.35)
    },
    [colors, isActive, zoneColor],
  )

  const sideColor = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      const color = zoneColor(code)
      return color ? withAlpha(color, 0.55) : withAlpha(colors.nodata, 0.15)
    },
    [colors, zoneColor],
  )

  const strokeColor = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      if (code === selectedCode) return colors.ink
      return ZONE_BY_COUNTRY.has(code) ? withAlpha(colors.ink, 0.45) : withAlpha(colors.ink, 0.12)
    },
    [colors, selectedCode],
  )

  const altitude = useCallback(
    (obj: object) => {
      const code = countryCode((obj as CountryFeature).properties)
      if (code === selectedCode) return 0.04
      if (isActive(code)) return 0.03
      return ZONE_BY_COUNTRY.has(code) ? 0.018 : 0.006
    },
    [selectedCode, isActive],
  )

  const tooltipFor = useCallback(
    (name: string, code: string) => {
      const zone = ZONE_BY_COUNTRY.get(code) ?? null
      return tooltipHtml(name, zone, zone ? zoneLevels.get(zone.code) : undefined, snapshot, base, mapLayer)
    },
    [zoneLevels, snapshot, base, mapLayer],
  )

  const polygonLabel = useCallback(
    (obj: object) => {
      const p = (obj as CountryFeature).properties
      return tooltipFor(p.NAME, countryCode(p))
    },
    [tooltipFor],
  )

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
      const m = obj as PointMarker
      selectCountry({ code: m.code, name: m.name, zone: ZONE_BY_COUNTRY.get(m.code) ?? null }, { lat: m.lat, lng: m.lng })
    },
    [selectCountry],
  )

  const handlePointHover = useCallback((obj: object | null) => {
    setHoveredCode(obj ? (obj as PointMarker).code : null)
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
        // Singapore and Hong Kong (too small to be polygons at this map scale)
        pointsData={POINT_MARKERS}
        pointLat="lat"
        pointLng="lng"
        pointColor={(obj: object) => zoneColor((obj as PointMarker).code) ?? colors.nodata}
        pointAltitude={(obj: object) => (isActive((obj as PointMarker).code) ? 0.05 : 0.025)}
        pointRadius={1.1}
        pointLabel={(obj: object) => tooltipFor((obj as PointMarker).name, (obj as PointMarker).code)}
        onPointClick={handlePointClick}
        onPointHover={handlePointHover}
        // Lines joining the two countries of each pair
        arcsData={arcs}
        arcColor="color"
        arcStroke={(obj: object) => ((obj as PairArc).active ? 0.9 : 0.5)}
        arcAltitudeAutoScale={0.35}
        arcDashLength={reducedMotion ? 1 : 0.45}
        arcDashGap={reducedMotion ? 0 : 0.12}
        arcDashAnimateTime={(obj: object) => (reducedMotion ? 0 : (obj as PairArc).active ? 1400 : 4200)}
        arcsTransitionDuration={0}
        arcLabel={arcLabel}
        onArcClick={handleArcClick}
        // Warning rings (extreme volatility) or buy/sell pulses
        ringsData={rings}
        ringLat="lat"
        ringLng="lng"
        ringColor={ringColor}
        ringMaxRadius={7}
        ringPropagationSpeed={3}
        ringRepeatPeriod={700}
        ringAltitude={0.02}
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
