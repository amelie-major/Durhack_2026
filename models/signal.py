"""Buy / Sell / Hold signal: direction model gated by the cost ratio.

The spread and vol models (models.xgboost_model) say how *big* the next move
is likely to be and what it costs to trade, but not which *way* it goes. This
module adds a third XGBoost model, a classifier for P(price up over the next
H minutes), and combines all three per test row:

    expected_move_bp = (2 * p_up - 1) * pred_vol_bp * sqrt(2 / pi)
                       (E|r| of a normal move with std pred_vol, signed by the edge)
    BUY   if  expected_move_bp >  threshold * pred_spread_bp
    SELL  if  expected_move_bp < -threshold * pred_spread_bp
    HOLD  otherwise

So a trade needs both a directional view and a move big enough to pay the
spread, i.e. |2p - 1| * 0.8 > threshold * cost_ratio. Each signal is then
backtested on the test split: a BUY/SELL is held for H minutes (default 10)
and earns the actual H-minute mid return in its direction minus the actual
spread. Only one position per instrument is open at a time; signals that fire
while a trade is still open are ignored.

Run from the repo root::

    python -m models.signal
    python -m models.signal --instrument EURUSD_EUR_FORWARD_1M_1000000
    python -m models.signal --export actions.csv   # 1 buy / 0 nothing / -1 sell
    python -m models.signal --horizon 30 --threshold 1.5 --save signals.csv
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from loader import load_processed, time_split
from models.cost_ratio import KEYS, cost_ratios
from models.xgboost_model import FEATURES, MAX_HORIZON, VOL_TENOR, build_features

MOMENTUM_WINDOWS = (5, 15, 60)
DIRECTION_FEATURES = FEATURES + [
    *[f"mom_{w}" for w in MOMENTUM_WINDOWS],  # past return over w bars (bp)
    "zscore_60",                              # close vs its 60-bar mean, in vol units
    "dfp_bp",                                 # change in forward points
]
EXPECTED_ABS_FACTOR = np.sqrt(2.0 / np.pi)  # E|X| / std for a normal X


# ==========================================================================
# Direction model
# ==========================================================================

def build_direction_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Features plus the forward H-minute mid return (bp) for one instrument."""
    rows = build_features(df)
    for w in MOMENTUM_WINDOWS:
        rows[f"mom_{w}"] = df["ret_bp"].rolling(w).sum()
    log_close = np.log(df["close_mid"]) * 1e4
    rows["zscore_60"] = (log_close - log_close.rolling(60, min_periods=20).mean()) / df["vol_60"]
    rows["dfp_bp"] = df["dfp_bp"]
    rows["timestamp"] = df["timestamp"]

    fwd = df["ret_bp"].rolling(horizon).sum().shift(-horizon)          # r over t+1..t+H
    crosses_gap = df["is_gap"].astype(float).rolling(horizon).sum().shift(-horizon) > 0
    rows["fwd_ret_bp"] = fwd.mask(crosses_gap)
    return rows


def direction_dataset(series: Dict[str, pd.DataFrame], horizon: int,
                      extra_train: Optional[Dict[str, pd.DataFrame]] = None):
    """(train, val, test) for the direction model; one tenor per pair, like vol.
    `extra_train` series go wholly into train (see build_dataset)."""
    parts = {"train": [], "val": [], "test": []}
    for df in (extra_train or {}).values():
        parts["train"].append(build_direction_rows(df, horizon).dropna(subset=["fwd_ret_bp"]))
    for df in series.values():
        if len(series) > 1 and df.attrs["tenor"] != VOL_TENOR:
            continue
        rows = build_direction_rows(df, horizon)
        for name, part in zip(parts, time_split(rows)):
            parts[name].append(part.dropna(subset=["fwd_ret_bp"]))

    pairs = sorted({df.attrs["pair"] for df in series.values()})
    out = []
    for name in ("train", "val", "test"):
        split = pd.concat(parts[name], ignore_index=True)
        split["pair"] = pd.Categorical(split["pair"], categories=pairs)
        split["up"] = (split["fwd_ret_bp"] > 0).astype(int)
        out.append(split)
    return tuple(out)


def train_direction(train_df: pd.DataFrame, val_df: pd.DataFrame, n_jobs: int = -1) -> XGBClassifier:
    """Fit P(up); flat bars (zero return) are dropped from training."""
    tr = train_df[train_df["fwd_ret_bp"] != 0]
    va = val_df[val_df["fwd_ret_bp"] != 0]
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=4, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, early_stopping_rounds=50, eval_metric="logloss",
        n_jobs=n_jobs,
    )
    model.fit(tr[DIRECTION_FEATURES], tr["up"],
              eval_set=[(va[DIRECTION_FEATURES], va["up"])], verbose=False)
    return model


def predict_direction(series: Dict[str, pd.DataFrame], horizon: int, n_jobs: int = -1) -> pd.DataFrame:
    train_df, val_df, test_df = direction_dataset(series, horizon)
    model = train_direction(train_df, val_df, n_jobs)
    p_up = model.predict_proba(test_df[DIRECTION_FEATURES])[:, 1]
    moved = test_df["fwd_ret_bp"] != 0
    acc = ((p_up > 0.5) == test_df["up"].astype(bool))[moved].mean()
    print(f"  direction best iteration {model.best_iteration + 1}, "
          f"test hit rate {acc:.3%} (on {moved.sum():,} rows that moved)")
    out = test_df[KEYS + ["fwd_ret_bp"]].copy()
    out["pair"] = out["pair"].astype(str)
    out["p_up"] = p_up
    return out


# ==========================================================================
# Signals and backtest
# ==========================================================================

def make_signals(rows: pd.DataFrame, threshold: float, hold_min: int) -> pd.DataFrame:
    rows = rows.copy()
    rows["expected_move_bp"] = (2 * rows["p_up"] - 1) * rows["pred_vol_bp"] * EXPECTED_ABS_FACTOR
    hurdle = threshold * rows["pred_spread_bp"]
    rows["signal"] = np.select(
        [rows["expected_move_bp"] > hurdle, rows["expected_move_bp"] < -hurdle],
        ["BUY", "SELL"], default="HOLD")
    side = rows["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
    rows["pnl_bp"] = np.where(side != 0, side * rows["fwd_ret_bp"] - rows["actual_spread_bp"], 0.0)
    rows["taken"] = _one_position_at_a_time(rows, side, hold_min)
    # For visualisers: 1 = buy, -1 = sell, 0 = do nothing. Only trades actually
    # opened count, so a signal during an open 10-minute hold is 0.
    # Continuous score for visualisers: 1 = definitely buy, 0 = definitely sell,
    # 0.5 = no view. It is the expected move measured in trading hurdles
    # (threshold x predicted spread), squashed into 0..1 with tanh, so a signal
    # just at the BUY/SELL line scores 0.88 / 0.12, half way there 0.73 / 0.27.
    hurdles = rows["expected_move_bp"] / hurdle
    rows["score"] = (0.5 + 0.5 * np.tanh(hurdles)).round(4)
    rows["action"] = np.where(rows["taken"], side, 0).astype("int8")
    return rows


def _one_position_at_a_time(rows: pd.DataFrame, side: pd.Series, hold_min: int) -> pd.Series:
    """Mark signals actually traded: per instrument, a trade is held for `hold_min`
    minutes and no new trade opens until it has closed."""
    taken = pd.Series(False, index=rows.index)
    hold = pd.Timedelta(minutes=hold_min)
    for _, g in rows[side != 0].sort_values("timestamp").groupby(["pair", "tenor_days"]):
        free_at = None
        for idx, ts in zip(g.index, g["timestamp"]):
            if free_at is None or ts >= free_at:
                taken[idx] = True
                free_at = ts + hold
    return taken


def summarise(rows: pd.DataFrame) -> None:
    print(f"\nSignals on {len(rows):,} test rows")
    print(f"  {'pair':<8}{'tenor_d':>8}{'BUY':>8}{'SELL':>8}{'HOLD':>9}"
          f"{'trades':>8}{'hit':>8}{'avg pnl':>9}{'tot pnl':>10}")
    for (pair, tenor), g in rows.groupby(["pair", "tenor_days"]):
        _line(f"{pair:<8}{tenor:>8.0f}", g)
    _line(f"{'ALL':<16}", rows)
    print("\n  BUY/SELL/HOLD count every minute's signal; 'trades' are those actually opened "
          "(one position per instrument, held for the horizon).\n"
          "  pnl in bp per trade, after paying the actual spread.")


def _line(label: str, g: pd.DataFrame) -> None:
    counts = g["signal"].value_counts()
    trades = g[g["taken"]]
    hit = (trades["pnl_bp"] > 0).mean() if len(trades) else float("nan")
    avg = trades["pnl_bp"].mean() if len(trades) else float("nan")
    print(f"  {label}{counts.get('BUY', 0):>8}{counts.get('SELL', 0):>8}{counts.get('HOLD', 0):>9}"
          f"{len(trades):>8}{hit:>8.1%}{avg:>9.3f}{trades['pnl_bp'].sum():>10.1f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Buy/Sell/Hold signals from direction, vol and spread models.")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--horizon", type=int, default=10, help="minutes each trade is held (and the models predict ahead)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="expected move must exceed this many predicted spreads to trade")
    parser.add_argument("--instrument", metavar="ID",
                        help="only this instrument, e.g. EURUSD_EUR_FORWARD_1M_1000000 "
                             "(default: every instrument, each modelled on its own)")
    parser.add_argument("--export", metavar="PATH",
                        help="write instrument_id,timestamp,close_mid,score (1 buy .. 0 sell), "
                             "action (1 buy, 0 nothing, -1 sell), pred_vol_bp to this CSV for a visualiser")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--save", metavar="PATH", help="write per-row signals to this CSV")
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON}")

    series, _ = load_processed(args.data)
    if args.instrument:
        if args.instrument not in series:
            parser.error(f"unknown instrument {args.instrument!r}; choose from: {', '.join(series)}")
        ids = [args.instrument]
    else:
        ids = list(series)

    # Each instrument gets its own spread, vol and direction models, trained
    # only on that instrument's bars.
    print(f"horizon={args.horizon}min threshold={args.threshold}, {len(ids)} instrument(s)")
    all_rows = []
    for instrument_id in ids:
        print(f"\n{instrument_id}")
        one = {instrument_id: series[instrument_id]}
        costs = cost_ratios(one, args.horizon, args.jobs)
        direction = predict_direction(one, args.horizon, args.jobs)
        rows = costs.merge(direction, on=KEYS, how="inner")
        rows.insert(0, "instrument_id", instrument_id)
        prices = series[instrument_id][["timestamp", "close_mid"]]
        rows = rows.merge(prices, on="timestamp", how="left")
        all_rows.append(make_signals(rows, args.threshold, args.horizon))
    rows = pd.concat(all_rows, ignore_index=True)
    summarise(rows)

    latest = rows.sort_values("timestamp").groupby(["pair", "tenor_days"]).tail(1)
    print("\nLatest signal per instrument")
    for _, r in latest.iterrows():
        print(f"  {r['pair']:<8}{r['tenor_days']:>5.0f}d  {r['timestamp']:%Y-%m-%d %H:%M}  "
              f"{r['signal']:<5} p_up={r['p_up']:.3f} exp={r['expected_move_bp']:+.2f}bp "
              f"spread={r['pred_spread_bp']:.2f}bp")
    if args.export:
        cols = ["instrument_id", "timestamp", "close_mid", "score", "action", "pred_vol_bp", "actual_vol_bp"]
        rows.sort_values(["instrument_id", "timestamp"])[cols].to_csv(args.export, index=False)
        print(f"\nexported {len(rows):,} rows to {args.export}")
    if args.save:
        rows.to_csv(args.save, index=False)
        print(f"\nsaved {len(rows):,} rows to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
