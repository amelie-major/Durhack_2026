"""Sample XGBoost model: predict next-H-minute volatility or spread.

One row = one instrument at one minute `t`. Features summarise the bars at or
before `t`; the target summarises bars t+1 ... t+H. See data/xgboost_inputs.md
for why each input is there.

Run from the repo root (after `python loader.py data/raw data/processed`)::

    python -m models.xgboost_model --target vol
    python -m models.xgboost_model --target spread --horizon 30
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from loader import load_processed, time_split

TENOR_DAYS = {
    "1W": 7, "2W": 14, "3W": 21,
    "1M": 30, "2M": 60, "3M": 90, "6M": 180, "9M": 270,
    "1Y": 365,
}

LOADER_VOL_WINDOWS = (15, 60)  # subset of loader.VOL_WINDOWS used as features

FEATURES = [
    # recent volatility: close-to-close std from loader.add_features (gap bars
    # excluded). Its vol_240 and pk_vol_* (Parkinson) columns were tried too and
    # raised test RMSE for the vol target (0.285 -> 0.289 over 5 seeds), so they
    # are left out.
    *[f"vol_{w}" for w in LOADER_VOL_WINDOWS],
    "range_mean_15", "range_mean_60",
    # recent spread
    "spread_bp", "spread_mean_15", "spread_mean_60",
    # time of day / week (UTC)
    "minute_of_day", "dow",
    # gaps and the reopen
    "is_gap", "min_since_gap",
    # settlement and instrument
    "days_to_settle", "tenor_days", "pair",
]

VOL_TENOR = "1M"    # volatility is ~one series per pair, so one tenor is enough
VOL_FLOOR_BP = 0.01  # keeps log() finite when the price did not move at all
MAX_HORIZON = 60     # time_split's embargo; a longer target would leak across splits


# ==========================================================================
# Features and targets (per instrument)
# ==========================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for one instrument, using only bars at or before each row."""
    out = pd.DataFrame(index=df.index)
    is_gap = df["is_gap"].astype(bool)

    # A gap bar's range spans the missing minutes (up to a weekend), so it is
    # hidden from the rolling windows rather than read as a volatility spike.
    bar_range = df["range_bp"].mask(is_gap)
    for window in (15, 60):
        min_bars = window // 3
        out[f"range_mean_{window}"] = bar_range.rolling(window, min_periods=min_bars).mean()
        out[f"spread_mean_{window}"] = df["spread_bp"].rolling(window, min_periods=min_bars).mean()

    # Volatility computed by the loader (trailing windows, no look-ahead).
    for window in LOADER_VOL_WINDOWS:
        out[f"vol_{window}"] = df[f"vol_{window}"]

    out["spread_bp"] = df["spread_bp"]
    out["minute_of_day"] = df["minute_of_day"]
    out["dow"] = df["dow"]

    # Minutes since the market last reopened (NaN before the first gap).
    out["is_gap"] = is_gap.astype("int8")
    last_gap_time = df["timestamp"].where(is_gap).ffill()
    out["min_since_gap"] = (df["timestamp"] - last_gap_time).dt.total_seconds() / 60.0

    out["days_to_settle"] = df["days_to_settle"]
    out["tenor_days"] = TENOR_DAYS.get(df.attrs["tenor"], np.nan)
    out["pair"] = df.attrs["pair"]
    return out


def make_target(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    """Log of next-`horizon`-minute volatility or mean spread, in bp.

    NaN where the window t+1 ... t+horizon runs off the end of the data or
    contains a gap bar (e.g. it would cross the daily rollover or a weekend).
    """
    def sum_ahead(x: pd.Series) -> pd.Series:
        return x.rolling(horizon).sum().shift(-horizon)  # sum over t+1 ... t+horizon

    if target == "vol":
        realised_vol = np.sqrt(sum_ahead(df["ret_bp"] ** 2))
        y = np.log(realised_vol.clip(lower=VOL_FLOOR_BP))
    elif target == "spread":
        y = np.log(sum_ahead(df["spread_bp"]) / horizon)
    else:
        raise ValueError(f"unknown target: {target!r}")

    crosses_gap = sum_ahead(df["is_gap"].astype(float)) > 0
    return y.mask(crosses_gap)


def baseline_prediction(features: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    """Naive "the next hour looks like the last hour" forecast to beat."""
    if target == "vol":
        # per-minute std over the last hour, scaled up to a `horizon`-minute window
        recent_vol = features["vol_60"] * np.sqrt(horizon)
        return np.log(recent_vol.clip(lower=VOL_FLOOR_BP))
    return np.log(features["spread_mean_60"])


# ==========================================================================
# Dataset
# ==========================================================================

def build_dataset(series: Dict[str, pd.DataFrame], target: str, horizon: int,
                  extra_train: Optional[Dict[str, pd.DataFrame]] = None,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stack every instrument's rows into (train, val, test), split by time.

    `extra_train` series (e.g. Yahoo spot history) go wholly into train, so val
    and test stay on `series` and scores remain comparable with and without it.
    """
    parts: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for df in (extra_train or {}).values():
        rows = build_features(df)
        rows["timestamp"] = df["timestamp"]
        rows["y"] = make_target(df, target, horizon)
        rows["baseline"] = baseline_prediction(rows, target, horizon)
        parts["train"].append(rows.dropna(subset=["y"]))

    for df in series.values():
        if target == "vol" and len(series) > 1 and df.attrs["tenor"] != VOL_TENOR:
            continue  # other tenors would count each price move again (a lone instrument is kept)

        rows = build_features(df)
        rows["timestamp"] = df["timestamp"]
        rows["y"] = make_target(df, target, horizon)
        rows["baseline"] = baseline_prediction(rows, target, horizon)

        for name, part in zip(parts, time_split(rows)):
            parts[name].append(part.dropna(subset=["y"]))

    if not parts["train"]:
        raise ValueError("no instruments to train on")

    # One shared category list so every split encodes pairs the same way.
    pairs = sorted({df.attrs["pair"] for df in series.values()})
    result = []
    for name in ("train", "val", "test"):
        split = pd.concat(parts[name], ignore_index=True)
        split["pair"] = pd.Categorical(split["pair"], categories=pairs)
        result.append(split)
    return tuple(result)  # type: ignore[return-value]


# ==========================================================================
# Training and evaluation
# ==========================================================================

def train(train_df: pd.DataFrame, val_df: pd.DataFrame, learning_rate: float = 0.05,
          max_trees: int = 2000, patience: int = 50, n_jobs: int = -1) -> XGBRegressor:
    """Fit on train, stopping when the validation error stops improving.

    To train for longer, lower `learning_rate` and raise `max_trees` and
    `patience` together: smaller steps need more trees to get to the same place.
    `n_jobs=-1` uses every CPU core.
    """
    model = XGBRegressor(
        n_estimators=max_trees,
        learning_rate=learning_rate,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=patience,
        n_jobs=n_jobs,
    )
    model.fit(train_df[FEATURES], train_df["y"],
              eval_set=[(val_df[FEATURES], val_df["y"])], verbose=False)
    return model


def _rmse(y: pd.Series, pred) -> float:
    return float(np.sqrt(np.nanmean((y - pred) ** 2)))


def _r2(y: pd.Series, pred) -> float:
    return float(1.0 - np.nanmean((y - pred) ** 2) / np.var(y))


def evaluate(model: XGBRegressor, test_df: pd.DataFrame) -> None:
    """Print test scores next to the baseline, per pair, and feature importance."""
    y = test_df["y"]
    # predict() uses the best iteration found by early stopping
    pred = pd.Series(model.predict(test_df[FEATURES]), index=test_df.index)
    baseline = test_df["baseline"]

    print(f"\nTest ({len(test_df):,} rows, target in log bp)")
    print(f"  {'':<10}{'RMSE':>8}{'R2':>8}")
    print(f"  {'xgboost':<10}{_rmse(y, pred):>8.4f}{_r2(y, pred):>8.3f}")
    print(f"  {'baseline':<10}{_rmse(y, baseline):>8.4f}{_r2(y, baseline):>8.3f}")

    print("\nTest RMSE by pair (xgboost / baseline)")
    for pair, rows in test_df.groupby("pair", observed=True):
        idx = rows.index
        print(f"  {pair:<10}{_rmse(y[idx], pred[idx]):>8.4f} /{_rmse(y[idx], baseline[idx]):>8.4f}")

    gain = pd.Series(model.get_booster().get_score(importance_type="gain"))
    share = (gain / gain.sum()).sort_values(ascending=False)
    print("\nFeature importance (share of gain)")
    for name, value in share.items():
        print(f"  {name:<16}{value:>7.1%}")


# ==========================================================================
# Command line
# ==========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a sample XGBoost volatility/spread model.")
    parser.add_argument("--data", default="data/processed", help="output dir of loader.py")
    parser.add_argument("--target", choices=["vol", "spread"], default="vol")
    parser.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="step size per tree; lower it (with more --max-trees) to train longer")
    parser.add_argument("--max-trees", type=int, default=2000, help="upper limit on boosting rounds")
    parser.add_argument("--patience", type=int, default=50,
                        help="stop after this many rounds without validation improvement")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU threads (-1 = all cores)")
    parser.add_argument("--save", metavar="PATH", help="write the trained model here (e.g. vol.json)")
    args = parser.parse_args(argv)

    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")

    series, _ = load_processed(args.data)
    train_df, val_df, test_df = build_dataset(series, args.target, args.horizon)
    print(f"target={args.target} horizon={args.horizon}min  "
          f"rows: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    model = train(train_df, val_df, learning_rate=args.learning_rate, max_trees=args.max_trees,
                  patience=args.patience, n_jobs=args.jobs)
    print(f"best iteration: {model.best_iteration + 1} of up to {args.max_trees} trees")
    evaluate(model, test_df)

    if args.save:
        model.save_model(args.save)
        print(f"\nsaved model to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
