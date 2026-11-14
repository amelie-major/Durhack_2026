"""Live scoring: train the team's models once on the historical data, then score
live Yahoo Finance 1-minute FX bars with them.

Yahoo gives spot mid OHLC only (no bid/ask, no forward points), so for each
forward instrument the live bar is built from the live spot price plus that
instrument's historical median spread and forward points. The features then go
through the same code as training: loader.add_features -> build_features /
build_direction_rows.
"""
from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from loader import add_features, load_processed
from models.signal import (DIRECTION_FEATURES, build_direction_rows, direction_dataset,
                           make_signals, train_direction)
from models.xgboost_model import FEATURES, VOL_TENOR, build_dataset, train

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "live_models.pkl"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}=X?interval=1m&{span}"
UA = "Mozilla/5.0 (fx-signal-map)"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

YAHOO_DIR = ROOT / "data" / "yahoo"


def instrument_consts(series: Dict[str, pd.DataFrame]) -> dict:
    """Per-instrument constants used to turn spot bars into forward bars."""
    consts = {}
    for inst, df in series.items():
        consts[inst] = {
            "pair": df.attrs["pair"], "tenor": df.attrs["tenor"], "attrs": dict(df.attrs),
            "spread_bp": float(df["spread_bp"].median()),
            "fp_bp": float(df["fp_bp"].iloc[-1]),
            "days_to_settle": int(df["days_to_settle"].median()),
        }
    return consts


def yahoo_series(consts: dict, yahoo_dir: Path = YAHOO_DIR) -> Dict[str, pd.DataFrame]:
    """Saved Yahoo spot history (fetch_history.py) as one 1M-tenor frame per pair,
    built exactly like live bars, so training sees what live scoring sees."""
    out = {}
    for inst, c in consts.items():
        path = yahoo_dir / f"{c['pair']}.parquet"
        if c["tenor"] == VOL_TENOR and path.exists():
            out[f"yahoo_{inst}"] = live_bars(pd.read_parquet(path), c)
    return out


def train_bank(data_dir: str, horizon: int, use_yahoo: bool = False) -> dict:
    """Pooled spread, vol and direction models plus per-instrument constants.
    With `use_yahoo`, saved Yahoo history is added to the direction training
    set only: it has no bid/ask (so not spread), and on the Challenge1 test
    days it made vol slightly worse (see models/yahoo_eval.py)."""
    series, _ = load_processed(data_dir)
    series = {k: v for k, v in series.items() if v.attrs["product"] == "FORWARD"}
    consts = instrument_consts(series)
    extra = yahoo_series(consts) if use_yahoo else {}
    print(f"training on {len(series)} instruments (horizon {horizon} min)"
          f"{f', plus Yahoo history for {len(extra)} pairs' if extra else ''}...")
    models = {}
    for target in ("spread", "vol"):
        tr, va, _ = build_dataset(series, target, horizon)
        models[target] = train(tr, va)
        print(f"  {target} model: {models[target].best_iteration + 1} trees")
    tr, va, _ = direction_dataset(series, horizon, extra_train=extra)
    models["direction"] = train_direction(tr, va)
    print(f"  direction model: {models['direction'].best_iteration + 1} trees")

    pairs = sorted({c["pair"] for c in consts.values()})
    return {"horizon": horizon, "models": models, "consts": consts, "pairs": pairs,
            "yahoo": bool(extra)}


def load_bank(data_dir: str, horizon: int, retrain: bool = False) -> dict:
    if CACHE.exists() and not retrain:
        bank = pickle.loads(CACHE.read_bytes())
        if bank["horizon"] == horizon:
            print(f"loaded cached models from {CACHE}")
            return bank
    bank = train_bank(data_dir, horizon)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(bank))
    print(f"saved models to {CACHE}")
    return bank


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------

def fetch_spot(pair: str, span: str = "range=5d") -> pd.DataFrame:
    """1-minute spot mid bars for a pair, e.g. 'EURUSD'. `span` is Yahoo's
    "range=5d" or "period1=<unix>&period2=<unix>" (at most 8 days, last 30 only)."""
    req = urllib.request.Request(YAHOO.format(sym=pair, span=span), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open_mid": q["open"], "high_mid": q["high"], "low_mid": q["low"], "close_mid": q["close"],
    }).dropna()
    df = df[(df[["open_mid", "high_mid", "low_mid", "close_mid"]] > 0).all(axis=1)]
    # Drop the still-forming current minute.
    df = df[df["timestamp"] < pd.Timestamp.now(tz="UTC").floor("min")]
    return df.drop_duplicates("timestamp").reset_index(drop=True)


def live_bars(spot: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Turn spot bars into the loader's forward-bar shape for one instrument."""
    df = spot.copy()
    df["high_mid"] = df[["open_mid", "high_mid", "low_mid", "close_mid"]].max(axis=1)
    df["low_mid"] = df[["open_mid", "high_mid", "low_mid", "close_mid"]].min(axis=1)
    half = df["close_mid"] * c["spread_bp"] / 2e4
    df["close_bid"], df["close_ask"] = df["close_mid"] - half, df["close_mid"] + half
    df["close_mid_fp"] = df["close_mid"] * c["fp_bp"] / 1e4
    df["close_mid_all_in"] = df["close_mid"] + df["close_mid_fp"]
    day = df["timestamp"].dt.tz_localize(None).dt.normalize()
    df["settlement_date"] = day + pd.Timedelta(days=c["days_to_settle"])
    df = add_features(df)
    df.attrs.update(c["attrs"])
    return df


def score_all(bank: dict, threshold: float = 1.0) -> pd.DataFrame:
    """Fetch every pair and score every instrument; one row per instrument-minute."""
    m, horizon = bank["models"], bank["horizon"]
    spots = {p: fetch_spot(p) for p in bank["pairs"]}
    out = []
    for inst, c in bank["consts"].items():
        if spots[c["pair"]].empty:
            continue
        bars = live_bars(spots[c["pair"]], c)
        rows = build_direction_rows(bars, horizon)
        rows["pair"] = pd.Categorical(rows["pair"], categories=bank["pairs"])
        rows["pred_spread_bp"] = np.exp(m["spread"].predict(rows[FEATURES]))
        rows["pred_vol_bp"] = np.exp(m["vol"].predict(rows[FEATURES]))
        rows["p_up"] = m["direction"].predict_proba(rows[DIRECTION_FEATURES])[:, 1]
        rows["actual_spread_bp"] = c["spread_bp"]
        rows["pair"] = c["pair"]
        rows["close_mid"] = bars["close_mid"]
        rows["instrument_id"] = inst
        # Realised vol over t+1..t+H, the vol model's target; NaN until those
        # H minutes have arrived (or if the window crosses a gap).
        ahead = lambda x: x.rolling(horizon).sum().shift(-horizon)
        rows["true_vol_bp"] = np.sqrt(ahead(bars["ret_bp"] ** 2)).mask(
            ahead(bars["is_gap"].astype(float)) > 0)
        out.append(make_signals(rows, threshold, horizon))
    return pd.concat(out, ignore_index=True)


def to_payload(rows: pd.DataFrame, bank: dict, max_points: int = 600) -> dict:
    """Same JSON shape as frontend/data.json, plus live status."""
    times = sorted(rows["timestamp"].unique())[-max_points:]
    rows = rows[rows["timestamp"].isin(times)]
    out = {"live": True, "freq": "1min", "vol_kind": "predicted",
           "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
           "last_bar": pd.Timestamp(times[-1]).isoformat() if times else None,
           "timestamps": [pd.Timestamp(t).isoformat() for t in times],
           "instruments": [], "score": {}, "action": {}, "close_mid": {}, "vol": {}, "true_vol": {}}
    clean = lambda s, n: [None if pd.isna(v) else round(float(v), n) for v in s]
    for inst, sub in rows.groupby("instrument_id"):
        sub = sub.set_index("timestamp").reindex(times)
        c = bank["consts"][inst]
        out["instruments"].append({"id": inst, "pair": c["pair"], "product": "FORWARD", "tenor": c["tenor"]})
        out["score"][inst] = clean(sub["score"], 4)
        out["action"][inst] = [0 if pd.isna(v) else int(v) for v in sub["action"]]
        out["vol"][inst] = clean(sub["pred_vol_bp"], 3)
        out["close_mid"][inst] = clean(sub["close_mid"], 6)
        out["true_vol"][inst] = clean(sub["true_vol_bp"], 3)
    return out
