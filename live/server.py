"""Live forecast server: the team's XGBoost vol/spread models scored on live
Yahoo Finance 1-minute FX bars, served as JSON in the frontend's shapes
(frontend/src/types.ts: Snapshot, PairHistory).

Run from the repo root::

    python -m live.server            # http://localhost:8000
    python -m live.server --retrain  # rebuild the cached models first

Endpoints
    GET /api/snapshot?base=USD[&asOf=2026-11-13T14:30:00Z]
    GET /api/history?pair=USDJPY&hours=120[&asOf=...]
    GET /api/replay-range
    GET /api/health

With ``asOf`` the server time-travels: it answers as it would have at that
moment, using only bars at or before it (replay mode in the frontend).

Every pair in data/processed is served. Yahoo gives spot mid OHLC only (no
bid/ask, no forward points), so each pair's live bar is built from the live spot
price plus the historical median spread and forward points of the instrument
that pair's models use: its 1M forward, or spot for a spot-only pair. The features then go through the
same code as training: loader.add_features -> models.xgboost_model.build_features.
Because the live spread is that constant, the spread forecast mostly reflects
time of day and recent volatility, not live quotes.
"""
from __future__ import annotations

import argparse
import json
import math
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
from models.signal import (DIRECTION_FEATURES, build_direction_rows, direction_dataset,
                           signal_score, steady_signal, train_direction)
from models.xgboost_model import FEATURES, build_dataset, build_features, train, vol_instruments

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "live_forecast_models.pkl"

HORIZONS = [5, 10, 15, 20, 25, 30]  # minutes; must match frontend HORIZONS_MIN
WINDOW = 30                          # realised-vol window (minutes)
CACHE_SECONDS = 15                   # how often Yahoo is re-fetched at most
REPLAY_CACHE_SECONDS = 600           # replay re-uses the same bars so the timeline stays put
REPLAY_HOURS = 24                    # "the last day of data"
STALE_SECONDS = 5 * 60               # no bar for this long -> no fresh data
SIGNAL_HORIZON = 10                  # minutes: the Buy/Sell signal looks this far ahead (models.signal default)
SIGNAL_THRESHOLD = 1.0               # expected move must beat this many predicted spreads to trade
SIGNAL_EXIT = 0.5                    # ...and a call stays on until it drops below this many (less flicker)
MAX_SCORING = 4                      # pairs scored at the same time at most (memory)
MAX_FILL_MIN = 5                    # short Yahoo holes are filled flat (Yahoo skips quiet minutes)
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X?interval=1m&range=5d"
UA = "Mozilla/5.0 (fx-forecast-server)"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_bank() -> dict:
    """Vol and spread models for every horizon, plus per-pair constants."""
    series = all_series()
    consts = {}
    for instrument_id in sorted(vol_instruments(series)):
        df = series[instrument_id]
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
    return {"horizons": HORIZONS, "models": models, "consts": consts, "pairs": sorted(consts),
            "features": list(FEATURES), "direction": train_direction_model(series)}


def all_series() -> Dict[str, pd.DataFrame]:
    """Every instrument in data/processed: forwards and spot."""
    series, _ = load_processed(str(DATA_DIR))
    return series


def data_pairs() -> List[str]:
    """Pairs listed in data/processed/instruments.csv (what the cached models must cover)."""
    return sorted(pd.read_csv(DATA_DIR / "instruments.csv")["pair"].unique())


def bank_is_current(bank: dict) -> bool:
    return (bank.get("horizons") == HORIZONS and bank.get("features") == list(FEATURES)
            and bank.get("pairs") == data_pairs())


def train_direction_model(series: Dict[str, pd.DataFrame]) -> dict:
    """P(price up over the next SIGNAL_HORIZON minutes), for the Buy/Sell signal."""
    t0 = time.time()
    tr, va, te = direction_dataset(series, SIGNAL_HORIZON)
    model = train_direction(tr, va)
    moved = te["fwd_ret_bp"] != 0
    hit = float(((model.predict_proba(te[DIRECTION_FEATURES])[:, 1] > 0.5) == te["up"].astype(bool))[moved].mean())
    print(f"  direction h={SIGNAL_HORIZON}min  {model.best_iteration + 1:>4} trees  "
          f"test hit rate {hit:.1%}  ({time.time() - t0:.0f}s)", flush=True)
    return {"model": model, "horizon": SIGNAL_HORIZON, "test_hit_rate": hit}


def load_bank(retrain: bool = False) -> dict:
    if CACHE.exists() and not retrain:
        bank = pickle.loads(CACHE.read_bytes())
        if bank_is_current(bank):
            print(f"loaded models from {CACHE}")
            if bank.get("direction", {}).get("horizon") != SIGNAL_HORIZON:
                print("training the direction model for the Buy/Sell signal...", flush=True)
                bank["direction"] = train_direction_model(all_series())
                CACHE.write_bytes(pickle.dumps(bank))
            return bank
        else:
            print("cached models don't match data/processed (pairs or features changed)", flush=True)
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


def unflatten(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's EURUSD bars come with open = high = low = close, which makes every bar's
    range 0 and the model (which leans on recent ranges) forecast almost no movement.
    For such flat bars, open from the previous close and span open..close instead, so a
    bar's range is at least the move it made. Bars with a real range are left alone."""
    flat = df["high_mid"] <= df["low_mid"]
    if not flat.any():
        return df
    df = df.copy()
    prev = df["close_mid"].shift(1).fillna(df["close_mid"])
    df.loc[flat, "open_mid"] = prev[flat]
    df.loc[flat, "high_mid"] = np.maximum(prev, df["close_mid"])[flat]
    df.loc[flat, "low_mid"] = np.minimum(prev, df["close_mid"])[flat]
    return df


def live_bars(spot: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Spot mid bars -> 1M-forward bars with the loader's columns and features."""
    df = unflatten(fill_short_holes(spot))
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
    wd, h = now.day_of_week, now.hour  # Monday = 0
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
        # One scoring per pair at a time: concurrent requests for a pair wait for it and reuse
        # the result instead of each redoing it (replay asks every second; duplicate work used
        # to pile up until the process ran out of memory). At most MAX_SCORING pairs at once.
        self.pair_locks = {p: threading.Lock() for p in bank["pairs"]}
        self.scoring = threading.Semaphore(MAX_SCORING)

    def _cached(self, pair: str, max_age: float) -> Optional[dict]:
        with self.lock:
            hit = self.cache.get(pair)
        return hit[1] if hit and time.time() - hit[0] < max_age else None

    def scored(self, pair: str, max_age: float = CACHE_SECONDS) -> dict:
        """Live bars for a pair with model forecasts on every row (cached max_age seconds)."""
        hit = self._cached(pair, max_age)
        if hit is not None:
            return hit
        with self.pair_locks[pair]:
            hit = self._cached(pair, max_age)  # another request may have just scored it
            if hit is not None:
                return hit
            with self.scoring:
                return self._score(pair)

    def warm(self) -> None:
        """Score every pair once in the background so the first page load / replay is quick."""
        def run():
            t0 = time.time()
            for pair in self.bank["pairs"]:
                try:
                    self.scored(pair)
                except Exception as e:
                    print(f"[warm] {pair}: {e}", flush=True)
            print(f"all pairs scored ({time.time() - t0:.0f}s)", flush=True)
        threading.Thread(target=run, daemon=True).start()

    def _score(self, pair: str) -> dict:
        c = self.bank["consts"][pair]
        spot, last_price = fetch_spot(pair)
        bars = live_bars(spot, c)
        vol = {h: np.exp(m.predict(self._features(bars, "vol", h)))
               for h, m in self.bank["models"]["vol"].items()}
        spread = {h: np.exp(m.predict(self._features(bars, "spread", h)))
                  for h, m in self.bank["models"]["spread"].items()}
        signal = self._signal(bars, vol, spread)
        ret2 = bars["ret_bp"].where(~bars["is_gap"], 0.0) ** 2
        past30 = np.sqrt(ret2.rolling(WINDOW, min_periods=WINDOW // 2).sum())  # t-29 … t
        next30 = past30.shift(-WINDOW)                                          # t+1 … t+30
        gap_ahead = bars["is_gap"].astype(float).rolling(WINDOW).sum().shift(-WINDOW) > 0
        out = {"bars": bars, "vol": vol, "spread": spread, "past30": past30,
               "next30": next30.mask(gap_ahead), "last_price": last_price, "signal": signal,
               "hist30": np.sort(vol[30][~np.isnan(vol[30])])}
        with self.lock:
            self.cache[pair] = (time.time(), out)
        return out

    def _features(self, bars: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
        """Model inputs, built exactly as in training (some columns depend on target and horizon)."""
        X = build_features(bars, target, horizon)
        X["pair"] = pd.Categorical(X["pair"], categories=self.bank["pairs"])
        return X[FEATURES]

    def _signal(self, bars: pd.DataFrame, vol: dict, spread: dict) -> Optional[dict]:
        """Buy/Sell/Hold on every row: the direction model's P(up) put through
        models.signal.signal_score with the vol and spread forecasts for the same horizon."""
        d = self.bank.get("direction")
        if not d:
            return None
        h = d["horizon"]
        rows = build_direction_rows(bars, h)
        rows["pair"] = pd.Categorical(rows["pair"], categories=self.bank["pairs"])
        p_up = d["model"].predict_proba(rows[DIRECTION_FEATURES])[:, 1]
        expected, _, score = signal_score(p_up, vol[h], spread[h], SIGNAL_THRESHOLD)
        hurdle = SIGNAL_THRESHOLD * spread[h]
        action = steady_signal(expected, hurdle, SIGNAL_EXIT, reset=bars["is_gap"].to_numpy())
        return {"p_up": p_up, "expected": expected, "action": action, "score": score,
                "hurdle": hurdle, "horizon": h}

    @staticmethod
    def _signal_at(s: dict, i: int, status: str) -> Optional[dict]:
        sig = s.get("signal")
        if not sig:
            return None
        # no trade call on data we wouldn't show a forecast for
        action = "hold" if status in ("rollover", "stale", "no_data") else str(sig["action"][i]).lower()
        return {"action": action, "score": round(float(sig["score"][i]), 4),
                "pUp": round(float(sig["p_up"][i]), 4),
                "expectedMoveBps": round(float(sig["expected"][i]), 3),
                "hurdleBps": round(float(sig["hurdle"][i]), 3), "horizonMin": sig["horizon"]}

    @staticmethod
    def _pct(sorted_hist: np.ndarray, x: float) -> float:
        if not len(sorted_hist) or not np.isfinite(x):
            return 0.5
        return float(np.clip(np.searchsorted(sorted_hist, x) / len(sorted_hist), 0.01, 0.99))

    def _get(self, pair: str, as_of: Optional[pd.Timestamp]) -> dict:
        # Replay re-uses whatever bars are cached (replay-range refreshes them when Replay is
        # pressed), so the timeline stays put and nothing is rescored mid-replay.
        return self.scored(pair, math.inf if as_of is not None else CACHE_SECONDS)

    @staticmethod
    def _row_at(ts: pd.Series, t: pd.Timestamp) -> int:
        """Index of the last bar at or before t (-1 if none)."""
        return int(ts.searchsorted(t, side="right")) - 1

    @staticmethod
    def _hist(s: dict, k: Optional[int]) -> np.ndarray:
        """Sorted 30-min forecasts the percentile is ranked against. In replay only rows
        up to k count, so nothing after the simulated "now" leaks in."""
        if k is None:
            return s["hist30"]
        v = s["vol"][HORIZONS[-1]][: k + 1]
        return np.sort(v[~np.isnan(v)])

    def pair_snapshot(self, pair: str, now: pd.Timestamp, as_of: Optional[pd.Timestamp] = None) -> dict:
        s = self._get(pair, as_of)
        bars = s["bars"]
        c = self.bank["consts"][pair]["attrs"]
        if as_of is None:
            i, mid, hist = len(bars) - 1, s["last_price"], s["hist30"]
        else:
            i = self._row_at(bars["timestamp"], as_of)
            if i < 0:
                raise LookupError(f"no bars at or before {as_of}")
            mid, hist = float(bars["close_mid"].iloc[i]), self._hist(s, i)
        vol = [round(float(s["vol"][h][i]), 3) for h in HORIZONS]
        spr = [round(float(s["spread"][h][i]), 3) for h in HORIZONS]
        spread_now = spr[0]
        half = mid * spread_now / 2e4
        rv30 = float(s["past30"].iloc[i])
        status = status_of(bars["timestamp"].iloc[i], now)
        return {
            "pair": pair, "base": c["base"], "quote": c["quote"],
            "bid": mid - half, "ask": mid + half, "mid": mid, "spreadBps": spread_now,
            "realizedVol30m": round(rv30 if np.isfinite(rv30) else 0.0, 3),
            "forecast": {"horizonsMin": HORIZONS, "vol": vol, "spreadBps": spr,
                         "volLow": [round(v * 0.75, 3) for v in vol],
                         "volHigh": [round(v * 1.3, 3) for v in vol]},
            "percentile": round(self._pct(hist, vol[-1]), 3),
            "volNormal30m": round(float(np.median(hist)), 3) if len(hist) else None,
            "status": status,
            "signal": self._signal_at(s, i, status),
        }

    def snapshot(self, base: str, as_of: Optional[pd.Timestamp] = None) -> dict:
        now = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
        pairs, newest = [], None
        for pair in self.bank["pairs"]:
            try:
                pairs.append(self.pair_snapshot(pair, now, as_of))
                ts = self._get(pair, as_of)["bars"]["timestamp"]
                last = ts.iloc[-1] if as_of is None else ts.iloc[self._row_at(ts, as_of)]
                newest = last if newest is None or last > newest else newest
            except Exception as e:  # one pair failing shouldn't blank the map
                print(f"[snapshot] {pair}: {e}", flush=True)
        if not pairs:
            raise RuntimeError("no pairs could be scored")
        return {"asOf": iso(now), "base": base, "lastUpdated": iso(newest), "pairs": pairs}

    def history(self, pair: str, hours: float, step_min: int = 5,
                as_of: Optional[pd.Timestamp] = None) -> Optional[dict]:
        if pair not in self.bank["consts"]:
            return None
        s = self._get(pair, as_of)
        bars = s["bars"]
        c = self.bank["consts"][pair]["attrs"]
        h = HORIZONS[-1]
        ts = bars["timestamp"]
        now = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
        end = now.floor(f"{step_min}min")
        grid = pd.date_range(end - pd.Timedelta(hours=hours), end, freq=f"{step_min}min")
        idx = ts.searchsorted(grid, side="right") - 1  # last bar at or before each step
        hist = self._hist(s, None if as_of is None else self._row_at(ts, as_of))
        horizon = pd.Timedelta(minutes=h)
        points: List[dict] = []
        for t, k in zip(grid, idx):
            if k < 0 or (t - ts.iloc[k]).total_seconds() > STALE_SECONDS:
                points.append({"t": iso(t), "mid": None, "spreadBps": None, "volForecast": None,
                               "volActual": None, "percentile": None, "status": "no_data"})
                continue
            fc = float(s["vol"][h][k])
            act = s["next30"].iloc[k]
            # what actually happened is only known once its window has fully passed
            known = not pd.isna(act) and ts.iloc[k] + horizon <= now
            points.append({
                "t": iso(t), "mid": float(bars["close_mid"].iloc[k]),
                "spreadBps": round(float(s["spread"][HORIZONS[0]][k]), 3),
                "volForecast": round(fc, 3), "volForecastLow": round(fc * 0.75, 3),
                "volForecastHigh": round(fc * 1.3, 3),
                "volActual": round(float(act), 3) if known else None,
                "percentile": round(self._pct(hist, fc), 3),
                "status": "rollover" if bool(bars["is_gap"].iloc[k]) else "live",
            })
        return {"pair": pair, "base": c["base"], "quote": c["quote"], "stepMin": step_min,
                "horizonMin": h, "points": points}

    def replay_range(self) -> dict:
        """The last REPLAY_HOURS of bars that every pair has: where replay starts and ends."""
        lasts = []
        for pair in self.bank["pairs"]:
            try:
                lasts.append(last_busy_bar(self.scored(pair, REPLAY_CACHE_SECONDS)["bars"]["timestamp"]))
            except Exception as e:
                print(f"[replay-range] {pair}: {e}", flush=True)
        if not lasts:
            raise RuntimeError("no pairs could be scored")
        end = min(lasts)
        return {"start": iso(end - pd.Timedelta(hours=REPLAY_HOURS)), "end": iso(end), "stepMin": 1}


def last_busy_bar(ts: pd.Series) -> pd.Timestamp:
    """Newest bar that isn't a lone straggler after a gap. Once the market shuts Yahoo adds
    a single closing bar hours after the last real one; replay should end on real trading."""
    i = len(ts) - 1
    while i > 0 and (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() > STALE_SECONDS:
        i -= 1
    return ts.iloc[i]


def iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_as_of(text: Optional[str]) -> Optional[pd.Timestamp]:
    """'2026-11-13T14:30:00Z' -> UTC Timestamp; None when absent. Raises ValueError if malformed."""
    if not text:
        return None
    t = pd.Timestamp(text)
    if pd.isna(t):
        raise ValueError(f"bad asOf: {text!r}")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def json_safe(x):
    """NaN / inf are not valid JSON (the browser's JSON.parse rejects them) -> null."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body) -> None:
            data = json.dumps(json_safe(body), allow_nan=False).encode()
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
                    d = engine.bank.get("direction") or {}
                    self._send(200, {"ok": True, "pairs": engine.bank["pairs"], "horizons": HORIZONS,
                                     "signal": {"horizonMin": d.get("horizon"), "testHitRate": d.get("test_hit_rate"),
                                                "enter": SIGNAL_THRESHOLD, "exit": SIGNAL_EXIT}})
                elif url.path == "/api/snapshot":
                    self._send(200, engine.snapshot(q.get("base", "USD"), parse_as_of(q.get("asOf"))))
                elif url.path == "/api/history":
                    hist = engine.history(q.get("pair", ""), float(q.get("hours", 24)),
                                          as_of=parse_as_of(q.get("asOf")))
                    self._send(200 if hist else 404, hist or {"error": "unknown pair"})
                elif url.path == "/api/replay-range":
                    self._send(200, engine.replay_range())
                else:
                    self._send(404, {"error": "not found"})
            except ValueError as e:  # bad asOf / hours
                self._send(400, {"error": str(e)})
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
    engine.warm()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"serving XGBoost forecasts on http://localhost:{args.port}/api/snapshot", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
