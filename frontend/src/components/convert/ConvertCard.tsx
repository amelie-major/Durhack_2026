/**
 * Compact convert card: amount, what it comes to in the To currency right now (mid-market), From ⇄ To, swap.
 * Only currencies we have forecasts for are offered (USD, EUR, JPY, SGD).
 * Step 5 adds the rate, "you'd receive" and the forecast underneath.
 */
import { useState, type ChangeEvent } from 'react'
import { ArrowLeftRight, MapPin } from 'lucide-react'
import { ZONES } from '../../data/currencies'
import { SIGNAL_DISCLAIMER, SIGNAL_INFO, signalFrom, signalFromDetail, signalFromMeaning, signalFromTitle } from '../../lib/signal'
import { hhmm } from '../../lib/pro'
import { findPair, routeLegs, routeRate } from '../../lib/zones'
import { useAppStore } from '../../store/useAppStore'

const MAX_AMOUNT = 1_000_000_000
/** Digits and thousands commas, then optionally a dot and up to 2 decimals. Anything else is ignored. */
const ALLOWED_INPUT = /^[\d,]*(\.\d{0,2})?$/

function formatAmount(n: number): string {
  return n.toLocaleString('en-GB', { maximumFractionDigits: 2 })
}

/** Keep digits and one decimal point; "12,500.5" → 12500.5 (not capped: see the input handler) */
function parseAmount(text: string): number {
  const cleaned = text.replace(/[^\d.]/g, '').replace(/(\..*)\./g, '$1')
  const n = Number(cleaned)
  return Number.isFinite(n) ? n : 0
}

function CurrencySelect({
  id,
  label,
  value,
  onChange,
}: {
  id: string
  label: string
  value: string
  onChange: (code: string) => void
}) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      <label htmlFor={id} className="text-ink-muted text-[10px] uppercase tracking-wider">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value)}
        className="h-9 w-full cursor-pointer rounded-lg border border-panel-border bg-white/5 px-2.5 text-sm font-medium text-ink outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        {ZONES.map((z) => (
          <option key={z.code} value={z.code} title={z.name} className="bg-space">
            {z.code}
          </option>
        ))}
      </select>
    </div>
  )
}

/** Amount in `currency` with that currency's usual decimals and its code: "1,557,240 JPY" (never an ambiguous "$") */
function formatIn(value: number, currency: string): string {
  let digits = 2
  try {
    digits = new Intl.NumberFormat('en-GB', { style: 'currency', currency }).resolvedOptions().maximumFractionDigits ?? 2
  } catch {
    // unknown code: keep 2 decimals
  }
  return `${value.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits })} ${currency}`
}

/** "155.72" / "1.1028" / "0.78432": enough digits to be useful at any size */
function formatRate(rate: number): string {
  const digits = rate >= 100 ? 2 : rate >= 1 ? 4 : 5
  return rate.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

/** What the amount comes to right now, at the mid-market rate (updates with every data refresh / replay step) */
function Converted() {
  const amount = useAppStore((s) => s.amount)
  const base = useAppStore((s) => s.base)
  const quote = useAppStore((s) => s.quote)
  const snapshot = useAppStore((s) => s.snapshot)
  const rate = routeRate(snapshot, base, quote)
  const viaUsd = routeLegs(snapshot, base, quote).length === 2

  if (base === quote) return null
  if (rate == null) {
    return <p className="text-ink-muted text-[11px]">{snapshot ? `No ${base} → ${quote} rate right now` : 'Loading rates…'}</p>
  }
  const value = formatIn(amount * rate, quote)
  const [number, code] = [value.slice(0, value.lastIndexOf(' ')), quote]
  return (
    <div
      aria-live="polite"
      className="group relative space-y-0.5 overflow-hidden rounded-xl border border-accent/25 bg-gradient-to-b from-[color-mix(in_oklab,var(--color-accent)_16%,transparent)] to-white/[0.02] px-3 pt-2.5 pb-2 transition duration-200 ease-out hover:border-accent/60 hover:shadow-[0_10px_28px_-10px_var(--color-accent)] motion-safe:hover:-translate-y-0.5"
    >
      {/* Glowing top edge, like the Pro tiles */}
      <span
        aria-hidden
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, var(--color-accent-bright), transparent)' }}
      />
      <p className="text-ink-muted text-[10px] uppercase tracking-wider">You&rsquo;d get about · mid-market</p>
      {/* Re-keyed on every new value so it gives a short glow each time the number changes */}
      <p
        key={value}
        className="num text-ink motion-safe:animate-value-pop flex items-baseline gap-1.5 truncate text-2xl font-semibold tracking-tight"
        title="Mid-market value, before fees and spread"
      >
        <span className="truncate">{number}</span>
        <span className="text-accent-bright text-sm font-semibold">{code}</span>
      </p>
      <p className="num text-ink-muted truncate text-[11px]">
        1 {base} = {formatRate(rate)} {quote}
        {viaUsd ? ' via USD' : ''} · {hhmm(snapshot?.lastUpdated ?? snapshot?.asOf)} UTC
      </p>
    </div>
  )
}

/** Buy / Sell layer only: the model's call for YOUR route, in one plain sentence (said from the From currency) */
function RouteSignal() {
  const base = useAppStore((s) => s.base)
  const quote = useAppStore((s) => s.quote)
  const snapshot = useAppStore((s) => s.snapshot)
  const pair = findPair(snapshot, base, quote)
  const view = signalFrom(pair, base)

  let body
  if (base === quote) body = <p className="text-ink-muted">Pick two different currencies to see a signal.</p>
  else if (!pair) body = <p className="text-ink-muted">No signal for {base} → {quote}: the model has no direct pair.</p>
  else if (!view) body = <p className="text-ink-muted">No signal right now: the forecast server isn&rsquo;t running.</p>
  else {
    const { Icon, textClass } = SIGNAL_INFO[view.action]
    body = (
      <>
        <p className={`flex items-center gap-1.5 text-[13px] font-semibold ${textClass}`}>
          <Icon className="size-4 shrink-0" aria-hidden />
          {signalFromTitle(view)}
          <span className="text-ink font-normal">· {signalFromMeaning(view)}</span>
        </p>
        <p className="num text-ink-muted">{signalFromDetail(view)}</p>
      </>
    )
  }
  return (
    <div aria-live="polite" className="space-y-0.5 rounded-lg border border-panel-border bg-white/[0.03] px-2.5 py-2 text-[11px] leading-snug">
      <p className="text-ink-muted text-[10px] uppercase tracking-wider">Model signal · {base} → {quote}</p>
      {body}
      <p className="text-ink-faint text-[10px]">{SIGNAL_DISCLAIMER}</p>
    </div>
  )
}

export default function ConvertCard() {
  const amount = useAppStore((s) => s.amount)
  const base = useAppStore((s) => s.base)
  const quote = useAppStore((s) => s.quote)
  const selected = useAppStore((s) => s.selected)
  const setAmount = useAppStore((s) => s.setAmount)
  const setBase = useAppStore((s) => s.setBase)
  const setQuote = useAppStore((s) => s.setQuote)
  const swap = useAppStore((s) => s.swap)
  const showSignal = useAppStore((s) => s.mapLayer === 'signal')

  // The text box keeps what you type; the store keeps the number
  const [draft, setDraft] = useState(() => formatAmount(amount))

  return (
    <section aria-label="Convert" className="glass w-[min(280px,calc(100vw-2rem))] space-y-2.5 p-3">
      <div className="flex flex-col gap-1">
        <label htmlFor="amount" className="text-ink-muted text-[10px] uppercase tracking-wider">
          Amount
        </label>
        <div className="group/amount relative">
          <input
            id="amount"
            inputMode="decimal"
            autoComplete="off"
            value={draft}
            onChange={(e) => {
              const text = e.target.value
              if (!ALLOWED_INPUT.test(text)) return // letters, spaces, a 2nd dot… are refused
              const n = parseAmount(text)
              if (n > MAX_AMOUNT) {
                // Over the cap: show the capped number in the box too, so box and conversion always agree
                setDraft(formatAmount(MAX_AMOUNT))
                setAmount(MAX_AMOUNT)
                return
              }
              setDraft(text)
              setAmount(n)
            }}
            onBlur={() => setDraft(formatAmount(amount))}
            className="num h-12 w-full rounded-xl border border-panel-border bg-gradient-to-b from-white/[0.09] to-white/[0.02] pr-16 pl-3 text-xl font-medium text-ink shadow-[inset_0_1px_0_rgb(255_255_255/0.07)] outline-none transition duration-200 hover:border-accent/40 focus:border-accent/80 focus:shadow-[0_0_0_3px_rgb(124_156_255/0.18),0_0_24px_-6px_var(--color-accent)]"
          />
          {/* The currency you're converting from, as a glowing chip inside the box */}
          <span
            aria-hidden
            className="num pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 rounded-md border border-accent/30 bg-accent/15 px-1.5 py-0.5 text-[11px] font-semibold text-accent-bright transition group-focus-within/amount:shadow-[0_0_12px_-2px_var(--color-accent)]"
          >
            {base}
          </span>
        </div>
      </div>

      <div className="flex items-end gap-2">
        <CurrencySelect id="from" label="From" value={base} onChange={setBase} />
        <button
          type="button"
          onClick={swap}
          aria-label={`Swap: convert ${quote} to ${base} instead`}
          title="Swap currencies"
          className="grid size-9 shrink-0 cursor-pointer place-items-center rounded-full border border-panel-border text-accent transition hover:bg-white/10 focus-visible:ring-2 focus-visible:ring-accent outline-none"
        >
          <ArrowLeftRight className="size-4" aria-hidden />
        </button>
        <CurrencySelect id="to" label="To" value={quote} onChange={setQuote} />
      </div>

      <Converted />

      {showSignal && <RouteSignal />}

      <p className="text-ink-muted flex items-center gap-1.5 text-[11px]" aria-live="polite">
        <MapPin className="size-3.5 shrink-0" aria-hidden />
        {selected
          ? selected.zone
            ? `${selected.name} · ${selected.zone.code}`
            : `${selected.name} · no forecast for this country`
          : 'Tip: click a lit country on the globe to pick it'}
      </p>
    </section>
  )
}
