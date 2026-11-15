"""PRAGMA forecasts for the website, read from saved predictions (no torch needed).

PRAGMA is too big to run on a laptop, and the released checkpoint is the
pre-trained backbone only, so the server does not run the model. It serves the
test-set predictions from the training run instead::

    gh release download pragma-large-run1 -p '*_h30_scratch.parquet' -D runs/pragma

What that means for the site:
  * PRAGMA mode is a replay of the challenge data's test week (24-31 Aug 2026),
    not live Yahoo bars.
  * Predictions exist for the 30-minute horizon only. Shorter horizons are
    derived: vol scales with sqrt(h / 30), spread is taken as the same.
  * There is no Buy/Sell signal (it needs 10-minute forecasts).

`load_pragma` returns, per pair, the same dict `live.server.Engine.scored`
builds, so snapshots and history are produced by the same code as XGBoost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from loader import load_processed
from models.xgboost_model import VOL_TENOR, make_target

PRED_HORIZON = 30    # minutes: the only horizon the saved predictions cover
VARIANT = "scratch"  # best PRAGMA run on both targets (see the release's slurm log)
DOWNLOAD = "gh release download pragma-large-run1 -p '*_h30_scratch.parquet' -D runs/pragma"


def prediction_paths(pred_dir: Path) -> Dict[str, Path]:
    return {t: Path(pred_dir) / f"{t}_h{PRED_HORIZON}_{VARIANT}.parquet" for t in ("vol", "spread")}


def available(pred_dir: Path) -> bool:
    return all(p.exists() for p in prediction_paths(pred_dir).values())


def _aligned(saved: pd.DataFrame, df: pd.DataFrame, instrument_id: str, target: str) -> pd.Series:
    """One instrument's predictions in bp, indexed by timestamp, checked against its bars."""
    rows = saved[saved["instrument_id"] == instrument_id]
    if rows.empty:
        raise ValueError(f"PRAGMA {target} predictions have no rows for {instrument_id}")
    truth = pd.Series(make_target(df, target, PRED_HORIZON).to_numpy(), index=df["timestamp"])
    y = truth.reindex(rows["timestamp"]).to_numpy()
    if np.isnan(y).any() or not np.allclose(y, rows["y"].to_numpy(), atol=1e-6):
        raise ValueError(
            f"PRAGMA {target} predictions for {instrument_id} do not match data/processed "
            "(were they made from different data?)")
    return pd.Series(np.exp(rows["pred"].to_numpy(dtype=float)), index=rows["timestamp"].to_numpy())


def scale_horizons(vol30: np.ndarray, spread30: np.ndarray, horizons: Sequence[int]):
    """Fill every horizon from the 30-minute forecast (see the module docstring)."""
    vol = {h: vol30 * np.sqrt(h / PRED_HORIZON) for h in horizons}
    spread = {h: spread30.copy() for h in horizons}
    return vol, spread


def score_pair(df: pd.DataFrame, vol_bp: pd.Series, spread_bp: pd.Series,
               horizons: Sequence[int], window: int) -> dict:
    """Bars with PRAGMA forecasts on every row, in the shape of Engine.scored()."""
    ret2 = df["ret_bp"].where(~df["is_gap"], 0.0) ** 2
    past = np.sqrt(ret2.rolling(window, min_periods=window // 2).sum())  # t-29 ... t
    gap_ahead = df["is_gap"].astype(float).rolling(window).sum().shift(-window) > 0
    frame = pd.DataFrame({
        "timestamp": df["timestamp"], "close_mid": df["close_mid"], "is_gap": df["is_gap"],
        "past": past, "next": past.shift(-window).mask(gap_ahead),       # t+1 ... t+30
    })
    frame["vol"] = frame["timestamp"].map(vol_bp)
    frame["spread"] = frame["timestamp"].map(spread_bp)
    frame = frame.dropna(subset=["vol", "spread"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no bars have both a vol and a spread prediction")
    vol, spread = scale_horizons(frame["vol"].to_numpy(), frame["spread"].to_numpy(), horizons)
    return {"bars": frame[["timestamp", "close_mid", "is_gap"]], "vol": vol, "spread": spread,
            "past30": frame["past"], "next30": frame["next"],
            "last_price": float(frame["close_mid"].iloc[-1]), "signal": None,
            "hist30": np.sort(vol[PRED_HORIZON])}


def load_pragma(data_dir: Path, pred_dir: Path, horizons: Sequence[int], window: int) -> dict:
    """{"pairs": [...], "attrs": {pair: attrs}, "scored": {pair: scored dict}}."""
    paths = prediction_paths(pred_dir)
    saved = {t: pd.read_parquet(p) for t, p in paths.items()}
    series, _ = load_processed(str(data_dir))
    out = {"pairs": [], "attrs": {}, "scored": {}, "variant": VARIANT, "horizon": PRED_HORIZON}
    for instrument_id in sorted(saved["vol"]["instrument_id"].unique()):
        df = series.get(instrument_id)
        if df is None or df.attrs["tenor"] != VOL_TENOR:
            continue
        pair = df.attrs["pair"]
        preds = {t: _aligned(saved[t], df, instrument_id, t) for t in ("vol", "spread")}
        out["scored"][pair] = score_pair(df, preds["vol"], preds["spread"], horizons, window)
        out["attrs"][pair] = dict(df.attrs)
        out["pairs"].append(pair)
    if not out["pairs"]:
        raise ValueError(f"no PRAGMA predictions in {pred_dir} match the instruments in {data_dir}")
    out["pairs"].sort()
    return out
