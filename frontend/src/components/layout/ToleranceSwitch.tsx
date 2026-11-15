/**
 * Relaxed / Balanced / Cautious switch for the Simple view (Pro has its own compact one in the panel
 * header). Same store value (tolerance), same meaning: it moves the Calm / Unsettled / Storm cut-offs
 * (LEVEL_CUTOFFS) and how big a move the numbers plan for (TOLERANCE_PLAN). Extreme never changes.
 *
 * Keyboard: a proper radio group. Tab lands on the chosen option; ←/→ (or ↑/↓) move and select;
 * Home/End jump to the ends.
 */
import { useRef, type KeyboardEvent } from 'react'
import { Feather, Scale, ShieldCheck, type LucideIcon } from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'
import type { Tolerance } from '../../types'

const TOLERANCE_OPTIONS: { value: Tolerance; label: string; Icon: LucideIcon }[] = [
  { value: 'relaxed', label: 'Relaxed', Icon: Feather },
  { value: 'balanced', label: 'Balanced', Icon: Scale },
  { value: 'cautious', label: 'Cautious', Icon: ShieldCheck },
]

export default function ToleranceSwitch({
  onChange,
  size = 'sm',
}: {
  onChange?: (from: Tolerance, to: Tolerance) => void
  /** 'md' = roomier version for the Simple-view tolerance card; 'sm' = compact (Pro panel header) */
  size?: 'sm' | 'md'
}) {
  const tolerance = useAppStore((s) => s.tolerance)
  const setTolerance = useAppStore((s) => s.setTolerance)
  const buttons = useRef<(HTMLButtonElement | null)[]>([])

  const choose = (next: Tolerance) => {
    if (next === tolerance) return
    onChange?.(tolerance, next)
    setTolerance(next)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TOLERANCE_OPTIONS.findIndex((o) => o.value === tolerance)
    const last = TOLERANCE_OPTIONS.length - 1
    const next =
      e.key === 'ArrowRight' || e.key === 'ArrowDown'
        ? Math.min(last, i + 1)
        : e.key === 'ArrowLeft' || e.key === 'ArrowUp'
          ? Math.max(0, i - 1)
          : e.key === 'Home'
            ? 0
            : e.key === 'End'
              ? last
              : null
    if (next == null) return
    e.preventDefault()
    choose(TOLERANCE_OPTIONS[next].value)
    buttons.current[next]?.focus()
  }

  return (
    <div
      role="radiogroup"
      aria-label="Your tolerance for volatility"
      onKeyDown={onKeyDown}
      className="seg-track"
    >
      {TOLERANCE_OPTIONS.map(({ value, label, Icon }, i) => {
        const on = value === tolerance
        return (
          <button
            key={value}
            ref={(el) => {
              buttons.current[i] = el
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            onClick={() => choose(value)}
            className={`seg-btn flex-1 ${size === 'md' ? 'px-2 py-1.5' : 'seg-sm'}`}
          >
            <Icon className={`${size === 'md' ? 'size-4' : 'size-3.5'} shrink-0`} aria-hidden />
            {label}
          </button>
        )
      })}
    </div>
  )
}
