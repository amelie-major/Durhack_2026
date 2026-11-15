"""Live forecast server: the team's XGBoost vol/spread models scored on live
Yahoo Finance 1-minute FX bars, served as JSON in the frontend's shapes
(frontend/src/types.ts: Snapshot, PairHistory).

Run from the repo root::

    python -m live.server            # http://localhost:8000
    python -m live.server --retrain  # rebuild the cached models first

Endpoints
    GET /api/snapshot?base=USD
    GET /api/history?pair=USDJPY&hours=120
    GET /api/health

Yahoo gives spot mid OHLC only (no bid/ask, no forward points), so each pair's
live 1M-forward bar is built from the live spot price plus that instrument's
historical median spread and forward points. The features then go through the
same code as training: loader.add_features -> models.xgboost_model.build_features.
Because the live spread is that constant, the spread forecast mostly reflects
time of day and recent volatility, not live quotes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from loader import add_features, load_processed
from models.xgboost_model import FEATURES, VOL_TENOR, build_dataset, build_features, train

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "live_forecast_models.pkl"

HORIZONS = [5, 10, 15, 20, 25, 30]  # minutes; must match frontend HORIZONS_MIN
WINDOW = 30                          # realised-vol window (minutes)
CACHE_SECONDS = 15                   # how often Yahoo is re-fetched at most
MAX_FILL_MIN = 5                     # short Yahoo holes are filled flat (Yahoo skips quiet minutes)
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X?interval=1m&range=5d"
UA = "Mozilla/5.0 (fx-forecast-server)"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_bank() -> dict:
    """Vol and spread models for every horizon, plus per-pair constants."""
    series, _ = load_processed(str(DATA_DIR))
    series = {k: v for k, v in series.items() if v.attrs["product"] == "FORWARD"}
    consts = {}
    for df in series.values():
        if df.attrs["tenor"] != VOL_TENOR:
            continue
        consts[df.attrs["pair"]] = {
            "attrs": dict(df.attrs),
            "spread_bp": float(df["spread_bp"].median()),
            "fp_bp": float(df["fp_bp"].iloc[-1]),
            "days_to_settle": int(df["days_to_settle"].median()),
        }
    models: Dict[str, Dict[int, object]] = {"vol": {}, "spread": {}}
    for h in HORIZONS:
        for target in ("vol", "spread"):
            t0 = time.time()
            tr, va, _ = build_dataset(series, target, h)
            models[target][h] = train(tr, va)
            print(f"  {target:<6} h={h:>2}min  {models[target][h].best_iteration + 1:>4} trees"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return {"horizons": HORIZONS, "models": models, "consts": consts, "pairs": sorted(consts)}


def load_bank(retrain: bool = False) -> dict:
    if CACHE.exists() and not retrain:
        bank = pickle.loads(CACHE.read_bytes())
        if bank.get("horizons") == HORIZONS:
            print(f"loaded models from {CACHE}")
            return bank
    print("training XGBoost models (one-off, cached afterwards)...", flush=True)
    bank = train_bank()
    CACHE.write_bytes(pickle.dumps(bank))
    print(f"saved models to {CACHE}")
    return bank


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------

def fetch_spot(pair: str) -> tuple[pd.DataFrame, float]:
    req = urllib.request.Request(YAHOO.format(pair=pair), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open_mid": q["open"], "high_mid": q["high"], "low_mid": q["low"], "close_mid": q["close"],
    }).dropna()
    df = df[df["close_mid"] > 0].drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor("min")
    df = df.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df, float(res["meta"].get("regularMarketPrice") or df["close_mid"].iloc[-1])


def fill_short_holes(spot: pd.DataFrame) -> pd.DataFrame:
    """Yahoo leaves out minutes with no trade; fill holes up to MAX_FILL_MIN with
    flat bars so they aren't read as market gaps (real closures stay gaps)."""
    s = spot.set_index("timestamp")
    full = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="min"))
    close = full["close_mid"].ffill(limit=MAX_FILL_MIN)
    for col in ("open_mid", "high_mid", "low_mid"):
        full[col] = full[col].fillna(close)
    full["close_mid"] = close
    full = full.dropna().rename_axis("timestamp").reset_index()
    return full


def live_bars(spot: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Spot mid bars -> 1M-forward bars with the loader's columns and features."""
    df = fill_short_holes(spot)
    mid = df["close_mid"]
    half = mid * c["spread_bp"] / 2e4
    df["close_bid"] = mid - half
    df["close_ask"] = mid + half
    df["close_mid_fp"] = mid * c["fp_bp"] / 1e4
    df["close_mid_all_in"] = mid + df["close_mid_fp"]
    day = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df["settlement_date"] = day + pd.Timedelta(days=c["days_to_settle"])
    df = add_features(df)
    df.attrs.update(c["attrs"])
    return df


def status_of(last_bar: pd.Timestamp, now: pd.Timestamp) -> str:
    wd, h = now.dayofweek, now.hour  # Monday = 0
    if wd == 5 or (wd == 6 and h < 21) or (wd == 4 and h >= 22):
        return "thin"
    mins = h * 60 + now.minute
    if 21 * 60 + 6 <= mins <= 21 * 60 + 10:
        return "rollover"
    if (now - last_bar).total_seconds() > 5 * 60:
        return "stale"
    return "live"


class Engine:
    def __init__(self, bank: dict):
        self.bank = bank
        self.lock = threading.Lock()
        self.cache: Dict[str, tuple[float, dict]] = {}

    def scored(self, pair: str) -> dict:
        """Live bars for a pair with model forecasts on every row (cached CACHE_SECONDS)."""
        with self.lock:
            hit = self.cache.get(pair)
            if hit and time.time() - hit[0] < CACHE_SECONDS:
                return hit[1]
        c = self.bank["consts"][pair]
        spot, last_price = fetch_spot(pair)
        bars = live_bars(spot, c)
        X = build_features(bars)
        X["pair"] = pd.Categorical(X["pair"], categories=self.bank["pairs"])
        X = X[FEATURES]
        vol = {h: np.exp(m.predict(X)) for h, m in self.bank["models"]["vol"].items()}
        spread = {h: np.exp(m.predict(X)) for h, m in self.bank["models"]["spread"].items()}
        ret2 = bars["ret_bp"].where(~bars["is_gap"], 0.0) ** 2
        past30 = np.sqrt(ret2.rolling(WINDOW, min_periods=WINDOW // 2).sum())  # t-29 … t
        next30 = past30.shift(-WINDOW)                                          # t+1 … t+30
        gap_ahead = bars["is_gap"].astype(float).rolling(WINDOW).sum().shift(-WINDOW) > 0
        out = {"bars": bars, "vol": vol, "spread": spread, "past30": past30,
               "next30": next30.mask(gap_ahead), "last_price": last_price,
               "hist30": np.sort(vol[30][~np.isnan(vol[30])])}
        with self.lock:
            self.cache[pair] = (time.time(), out)
        return out

    @staticmethod
    def _pct(sorted_hist: np.ndarray, x: float) -> float:
        if not len(sorted_hist) or not np.isfinite(x):
            return 0.5
        return float(np.clip(np.searchsorted(sorted_hist, x) / len(sorted_hist), 0.01, 0.99))

    def pair_snapshot(self, pair: str, now: pd.Timestamp) -> dict:
        s = self.scored(pair)
        bars, i = s["bars"], len(s["bars"]) - 1
        c = self.bank["consts"][pair]["attrs"]
        vol = [round(float(s["vol"][h][i]), 3) for h in HORIZONS]
        spr = [round(float(s["spread"][h][i]), 3) for h in HORIZONS]
        mid = s["last_price"]
        spread_now = spr[0]
        half = mid * spread_now / 2e4
        rv30 = float(s["past30"].iloc[i])
        return {
            "pair": pair, "base": c["base"], "quote": c["quote"],
            "bid": mid - half, "ask": mid + half, "mid": mid, "spreadBps": spread_now,
            "realizedVol30m": round(rv30 if np.isfinite(rv30) else 0.0, 3),
            "forecast": {"horizonsMin": HORIZONS, "vol": vol, "spreadBps": spr,
                         "volLow": [round(v * 0.75, 3) for v in vol],
                         "volHigh": [round(v * 1.3, 3) for v in vol]},
            "percentile": round(self._pct(s["hist30"], vol[-1]), 3),
            "volNormal30m": round(float(np.median(s["hist30"])), 3) if len(s["hist30"]) else None,
            "status": status_of(bars["timestamp"].iloc[i], now),
        }

    def snapshot(self, base: str) -> dict:
        now = pd.Timestamp.now(tz="UTC")
        pairs, newest = [], None
        for pair in self.bank["pairs"]:
            try:
                pairs.append(self.pair_snapshot(pair, now))
                last = self.scored(pair)["bars"]["timestamp"].iloc[-1]
                newest = last if newest is None or last > newest else newest
            except Exception as e:  # one pair failing shouldn't blank the map
                print(f"[snapshot] {pair}: {e}", flush=True)
        if not pairs:
            raise RuntimeError("no pairs could be scored")
        iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"asOf": iso(now), "base": base, "lastUpdated": iso(newest), "pairs": pairs}

    def history(self, pair: str, hours: float, step_min: int = 5) -> Optional[dict]:
        if pair not in self.bank["consts"]:
            return None
        s = self.scored(pair)
        bars = s["bars"]
        c = self.bank["consts"][pair]["attrs"]
        h = HORIZONS[-1]
        ts = bars["timestamp"]
        end = pd.Timestamp.now(tz="UTC").floor(f"{step_min}min")
        grid = pd.date_range(end - pd.Timedelta(hours=hours), end, freq=f"{step_min}min")
        idx = ts.searchsorted(grid, side="right") - 1  # last bar at or before each step
        points: List[dict] = []
        for t, k in zip(grid, idx):
            iso = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            if k < 0 or (t - ts.iloc[k]).total_seconds() > 5 * 60:
                points.append({"t": iso, "mid": None, "spreadBps": None, "volForecast": None,
                               "volActual": None, "percentile": None, "status": "no_data"})
                continue
            fc = float(s["vol"][h][k])
            act = s["next30"].iloc[k]
            points.append({
                "t": iso, "mid": float(bars["close_mid"].iloc[k]),
                "spreadBps": round(float(s["spread"][HORIZONS[0]][k]), 3),
                "volForecast": round(fc, 3), "volForecastLow": round(fc * 0.75, 3),
                "volForecastHigh": round(fc * 1.3, 3),
                "volActual": None if pd.isna(act) else round(float(act), 3),
                "percentile": round(self._pct(s["hist30"], fc), 3),
                "status": "rollover" if bool(bars["is_gap"].iloc[k]) else "live",
            })
        return {"pair": pair, "base": c["base"], "quote": c["quote"], "stepMin": step_min,
                "horizonMin": h, "points": points}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/health":
                    self._send(200, {"ok": True, "pairs": engine.bank["pairs"], "horizons": HORIZONS})
                elif url.path == "/api/snapshot":
                    self._send(200, engine.snapshot(q.get("base", "USD")))
                elif url.path == "/api/history":
                    hist = engine.history(q.get("pair", ""), float(q.get("hours", 24)))
                    self._send(200 if hist else 404, hist or {"error": "unknown pair"})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                print(f"[error] {url.path}: {e!r}", flush=True)
                self._send(502, {"error": str(e)})

        def log_message(self, fmt, *args) -> None:  # keep the console quiet
            pass

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--retrain", action="store_true", help="retrain instead of using the cache")
    args = p.parse_args()
    engine = Engine(load_bank(args.retrain))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"serving XGBoost forecasts on http://localhost:{args.port}/api/snapshot", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
