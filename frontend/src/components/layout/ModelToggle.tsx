/**
 * Whose forecasts the site shows: XGBoost (live) or PRAGMA.
 * PRAGMA is served from saved predictions for the challenge data's test week, so choosing it
 * starts a replay of that week; going back to XGBoost returns to live.
 * Hidden unless the forecast server says it has PRAGMA predictions.
 */
import { useEffect, useState } from "react";
import { getModelInfo, getReplayRange } from "../../data/api";
import { useAppStore } from "../../store/useAppStore";
import type { ForecastModel } from "../../types";

const MODELS: { model: ForecastModel; label: string; hint: string }[] = [
  {
    model: "xgboost",
    label: "XGBoost",
    hint: "XGBoost forecasts on live prices",
  },
  {
    model: "pragma",
    label: "PRAGMA",
    hint: "Saved PRAGMA forecasts, replayed over its test week (24–31 Aug). 30-minute forecasts only; no Buy / Sell signal",
  },
];

export default function ModelToggle() {
  const model = useAppStore((s) => s.model);
  const startReplay = useAppStore((s) => s.startReplay);
  const exitReplay = useAppStore((s) => s.exitReplay);
  const [available, setAvailable] = useState<ForecastModel[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const choose = (next: ForecastModel) => {
    if (next === useAppStore.getState().model || busy) return;
    setError(null);
    if (next === "xgboost") return exitReplay();
    setBusy(true);
    getReplayRange(useAppStore.getState().asOf, next)
      .then((range) => startReplay(range, next))
      .catch((err: unknown) => {
        console.error("[data] could not switch to PRAGMA", err);
        setError("PRAGMA is not available: is the forecast server running?");
      })
      .finally(() => setBusy(false));
  };

  // Ask the server once which models it has; start on PRAGMA if that is its default
  useEffect(() => {
    let cancelled = false;
    getModelInfo()
      .then((info) => {
        if (cancelled) return;
        setAvailable(info.models);
        if (info.defaultModel === "pragma") choose("pragma");
      })
      .catch(() => undefined); // server offline: no toggle
    return () => {
      cancelled = true;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!available.includes("pragma")) return null;
  return (
    <>
      {error && (
        <span
          role="alert"
          className="whitespace-nowrap text-[11px] text-unsettled"
        >
          {error}
        </span>
      )}
      <div
        role="radiogroup"
        aria-label="Forecast model"
        className="flex rounded-full border border-panel-border p-0.5"
      >
        {MODELS.map((m) => (
          <button
            key={m.model}
            type="button"
            role="radio"
            aria-checked={model === m.model}
            title={m.hint}
            disabled={busy}
            onClick={() => choose(m.model)}
            className={`cursor-pointer whitespace-nowrap rounded-full px-2.5 py-0.5 text-[11px] font-medium outline-none transition focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-wait ${
              model === m.model
                ? "bg-accent text-space"
                : "text-ink-muted hover:text-ink"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </>
  );
}
