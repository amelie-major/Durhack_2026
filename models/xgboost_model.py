"""Sample XGBoost model: predict next-H-minute volatility or spread.

One row = one instrument at one minute `t`. Features summarise the bars at or
before `t`; the target summarises bars t+1 ... t+H. See data/xgboost_inputs.md
for why each input is there.

Run from the repo root (after `python loader.py data/raw data/processed`)::

    python -m models.xgboost_model --target vol
    python -m models.xgboost_model --target spread --horizon 30
    python -m models.xgboost_model --target vol --cv 6    # walk-forward scores

Judge a change with `--cv`, not the single train/val/test split: the 6-day
validation set is small enough to rank the same change as both better and worse.
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

VOL_WINDOWS = (5, 30, 120, 240, 720, 1440)  # minutes; 15 and 60 come from the loader
EWM_HALFLIVES = (5, 20, 60)                 # minutes
SEASONAL_DAYS = 5                           # days in the short time-of-day average

LOADER_VOL_WINDOWS = (15, 60)  # subset of loader.VOL_WINDOWS used as features

FEATURES = [
    # recent volatility: close-to-close std from loader.add_features (gap bars
    # excluded). Its vol_240 and pk_vol_* (Parkinson) columns were tried too and
    # raised test RMSE for the vol target (0.285 -> 0.289 over 5 seeds), so they
    # are left out.
    *[f"vol_{w}" for w in LOADER_VOL_WINDOWS],
    "range_mean_15", "range_mean_60",
    # volatility over short to day-long windows, in logs (same scale as the target)
    *[f"log_vol_{w}" for w in VOL_WINDOWS],
    *[f"log_range_{w}" for w in VOL_WINDOWS],
    *[f"log_ewm_vol_{h}" for h in EWM_HALFLIVES],
    "log_parkinson_30", "abs_ret", "range_bp", "range_max_30", "flat_share_30",
    "vol_trend_5_60", "vol_trend_60_1440", "vol_of_vol_60",
    # the target itself: over the window just ended, and at this time on earlier days
    "y_last", "y_prev_day", "y_tod_mean", f"y_tod_mean_{SEASONAL_DAYS}d",
    # recent spread
    "spread_bp", "spread_mean_15", "spread_mean_60",
    # time of day / week (UTC)
    "minute_of_day", "dow", "minute_of_hour", "hour_of_week",
    # gaps and the reopen
    "is_gap", "min_since_gap", "min_since_roll",
    # settlement and instrument
    "days_to_settle", "tenor_days", "pair",
]

VOL_TENOR = "1M"    # volatility is ~one series per pair, so one tenor is enough
VOL_FLOOR_BP = 0.01  # keeps log() finite when the price did not move at all
MAX_HORIZON = 60     # time_split's embargo; a longer target would leak across splits
LOG_FLOOR = 1e-3     # keeps log() finite for the rolling features (bp)


# ==========================================================================
# Features and targets (per instrument)
# ==========================================================================

def _log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=LOG_FLOOR))


def _seasonal_features(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """The target at the same minute of the day on earlier days.

    A row's target covers t+1 ... t+horizon, so it is not known at `t`. The same
    minute on an earlier day finished at least 23 hours ago (horizon <= 60), so
    every value used here is already known.
    """
    out = pd.DataFrame(index=df.index)
    by_minute = y.groupby(df["minute_of_day"])
    out["y_prev_day"] = by_minute.shift(1)

    # Mean over all earlier days: running total minus this row's own value.
    known = y.notna()
    n_before = known.groupby(df["minute_of_day"]).cumsum() - known
    sum_before = y.fillna(0.0).groupby(df["minute_of_day"]).cumsum() - y.fillna(0.0)
    out["y_tod_mean"] = (sum_before / n_before).where(n_before >= 3)

    # Mean over the last few days only, which follows a change in regime faster.
    lags = pd.concat([by_minute.shift(k) for k in range(1, SEASONAL_DAYS + 1)], axis=1)
    out[f"y_tod_mean_{SEASONAL_DAYS}d"] = lags.mean(axis=1).where(lags.notna().sum(axis=1) >= 2)
    return out


def build_features(df: pd.DataFrame, target: str = "vol", horizon: int = 30) -> pd.DataFrame:
    """Model inputs for one instrument, using only bars at or before each row.

    `target` and `horizon` only shape the `y_*` columns, which describe past
    values of the quantity being predicted.
    """
    out = pd.DataFrame(index=df.index)
    is_gap = df["is_gap"].astype(bool)

    # A gap bar's return and range span the missing minutes (up to a weekend), so
    # they are hidden from the rolling windows rather than read as a volatility spike.
    ret = df["ret_bp"].mask(is_gap)
    bar_range = df["range_bp"].mask(is_gap)
    for window in (15, 60):
        min_bars = window // 3
        out[f"range_mean_{window}"] = bar_range.rolling(window, min_periods=min_bars).mean()
        out[f"spread_mean_{window}"] = df["spread_bp"].rolling(window, min_periods=min_bars).mean()

    # Volatility computed by the loader (trailing windows, no look-ahead).
    for window in LOADER_VOL_WINDOWS:
        out[f"vol_{window}"] = df[f"vol_{window}"]

    # The same two measures over more windows. Trees cannot take a ratio of two
    # columns themselves, so the trend features spell out "rising or calming".
    ret_sq = ret ** 2
    for window in VOL_WINDOWS:
        min_bars = max(3, window // 3)
        out[f"log_vol_{window}"] = 0.5 * _log(ret_sq.rolling(window, min_periods=min_bars).mean())
        out[f"log_range_{window}"] = _log(bar_range.rolling(window, min_periods=min_bars).mean())
    for halflife in EWM_HALFLIVES:
        ewm = ret_sq.ewm(halflife=halflife, min_periods=5, ignore_na=True).mean()
        out[f"log_ewm_vol_{halflife}"] = 0.5 * _log(ewm)
    # Parkinson: volatility from the high-low range, less noisy than close-to-close.
    range_sq = (bar_range ** 2).rolling(30, min_periods=10).mean()
    out["log_parkinson_30"] = 0.5 * _log(range_sq / (4.0 * np.log(2.0)))
    out["abs_ret"] = ret.abs()
    out["range_bp"] = bar_range
    out["range_max_30"] = bar_range.rolling(30, min_periods=10).max()
    out["flat_share_30"] = (bar_range == 0).astype(float).rolling(30, min_periods=10).mean()
    log_range_60 = _log(out["range_mean_60"])
    out["vol_trend_5_60"] = out["log_range_5"] - log_range_60
    out["vol_trend_60_1440"] = log_range_60 - out["log_range_1440"]
    out["vol_of_vol_60"] = out["log_range_5"].rolling(60, min_periods=20).std()

    # The target over the window that has just ended (t-horizon+1 ... t).
    if target == "vol":
        out["y_last"] = 0.5 * _log(ret_sq.rolling(horizon, min_periods=horizon // 2 + 1).sum())
    else:
        out["y_last"] = _log(df["spread_bp"].rolling(horizon, min_periods=horizon // 2 + 1).mean())
    out = out.join(_seasonal_features(df, make_target(df, target, horizon)))

    out["spread_bp"] = df["spread_bp"]
    out["minute_of_day"] = df["minute_of_day"]
    out["dow"] = df["dow"]
    out["minute_of_hour"] = df["minute_of_day"] % 60
    out["hour_of_week"] = df["dow"].astype(int) * 24 + df["minute_of_day"] // 60

    # Minutes since the market last reopened (NaN before the first gap).
    out["is_gap"] = is_gap.astype("int8")
    last_gap_time = df["timestamp"].where(is_gap).ffill()
    out["min_since_gap"] = (df["timestamp"] - last_gap_time).dt.total_seconds() / 60.0
    out["min_since_roll"] = df["min_since_roll"]

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

def vol_instruments(series: Dict[str, pd.DataFrame]) -> set:
    """One instrument per pair for volatility: its 1M forward, or spot if it has no 1M."""
    chosen: Dict[str, str] = {}
    for instrument_id, df in series.items():
        pair, tenor = df.attrs["pair"], df.attrs["tenor"]
        if tenor == VOL_TENOR or (tenor == "SPOT" and pair not in chosen):
            chosen[pair] = instrument_id
    return set(chosen.values())


def _instrument_rows(series: Dict[str, pd.DataFrame], target: str,
                     horizon: int) -> List[pd.DataFrame]:
    """Features, target and baseline for each instrument the target uses."""
    chosen = vol_instruments(series)
    result = []
    for instrument_id, df in series.items():
        if target == "vol" and len(series) > 1 and instrument_id not in chosen:
            continue  # other tenors would count each price move again (a lone instrument is kept)

        rows = build_features(df, target, horizon)
        rows["timestamp"] = df["timestamp"]
        rows["y"] = make_target(df, target, horizon)
        rows["baseline"] = baseline_prediction(rows, target, horizon)
        result.append(rows)
    return result


def _pair_categories(series: Dict[str, pd.DataFrame]) -> List[str]:
    # One shared category list so every split encodes pairs the same way.
    return sorted({df.attrs["pair"] for df in series.values()})


def build_dataset(series: Dict[str, pd.DataFrame], target: str,
                  horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stack every instrument's rows into (train, val, test), split by time."""
    parts: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for rows in _instrument_rows(series, target, horizon):
        for name, part in zip(parts, time_split(rows)):
            parts[name].append(part.dropna(subset=["y"]))

    if not parts["train"]:
        raise ValueError("no instruments to train on")

    pairs = _pair_categories(series)
    result = []
    for name in ("train", "val", "test"):
        split = pd.concat(parts[name], ignore_index=True)
        split["pair"] = pd.Categorical(split["pair"], categories=pairs)
        result.append(split)
    return tuple(result)  # type: ignore[return-value]


# ==========================================================================
# Training and evaluation
# ==========================================================================

def _make_model(n_trees: int, learning_rate: float, n_jobs: int,
                patience: Optional[int] = None) -> XGBRegressor:
    """Many small, cautious trees with a loss that large misses cannot dominate.

    Half of the squared error comes from the worst 5% of rows (data releases,
    the rollover, the Sunday reopen), so the Huber loss stops those rows from
    steering the fit. Shallow trees with large leaves suit a target this noisy.
    """
    return XGBRegressor(
        n_estimators=n_trees,
        learning_rate=learning_rate,
        max_depth=4,
        min_child_weight=50,
        reg_lambda=5.0,
        subsample=0.7,
        colsample_bytree=0.5,
        objective="reg:pseudohubererror",
        huber_slope=1.0,
        eval_metric="rmse",
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=patience,
        n_jobs=n_jobs,
    )


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, learning_rate: float = 0.02,
          max_trees: int = 4000, patience: int = 100, n_jobs: int = -1) -> XGBRegressor:
    """Fit on train, stopping when the validation error stops improving.

    To train for longer, lower `learning_rate` and raise `max_trees` and
    `patience` together: smaller steps need more trees to get to the same place.
    `n_jobs=-1` uses every CPU core.
    """
    model = _make_model(max_trees, learning_rate, n_jobs, patience)
    model.fit(train_df[FEATURES], train_df["y"],
              eval_set=[(val_df[FEATURES], val_df["y"])], verbose=False)
    return model


def walk_forward(series: Dict[str, pd.DataFrame], target: str, horizon: int,
                 n_folds: int = 6, fold_days: int = 5, n_trees: int = 1000,
                 learning_rate: float = 0.02, n_jobs: int = -1) -> pd.DataFrame:
    """Score the model on `n_folds` consecutive blocks of `fold_days` days.

        fold 1: [ train ......... ][gap][ test ]
        fold 2: [ train ................ ][gap][ test ]

    Each fold trains on everything before its test block, less a MAX_HORIZON
    embargo, with a fixed number of trees (no validation set is spent on early
    stopping). The last fold ends on the last day of data. Returns one row per
    fold with the model's and the baseline's RMSE.
    """
    rows = pd.concat(_instrument_rows(series, target, horizon), ignore_index=True)
    rows = rows.dropna(subset=["y"]).reset_index(drop=True)
    rows["pair"] = pd.Categorical(rows["pair"], categories=_pair_categories(series))
    day = rows["timestamp"].dt.tz_convert("UTC").dt.normalize()
    days = sorted(day.unique())
    if len(days) <= n_folds * fold_days:
        raise ValueError(f"need more than {n_folds * fold_days} days, have {len(days)}")

    embargo = pd.Timedelta(minutes=MAX_HORIZON)
    scores = []
    for fold in range(n_folds):
        last = len(days) - 1 - (n_folds - 1 - fold) * fold_days
        first = last - fold_days + 1
        train_df = rows[rows["timestamp"] < days[first] - embargo]
        test_df = rows[(day >= days[first]) & (day <= days[last])]

        model = _make_model(n_trees, learning_rate, n_jobs)
        model.fit(train_df[FEATURES], train_df["y"], verbose=False)
        scores.append({
            "fold": fold + 1,
            "test_from": pd.Timestamp(days[first]).date(),
            "test_to": pd.Timestamp(days[last]).date(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "xgboost": _rmse(test_df["y"], model.predict(test_df[FEATURES])),
            "baseline": _rmse(test_df["y"], test_df["baseline"]),
        })
    return pd.DataFrame(scores)


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
    parser.add_argument("--learning-rate", type=float, default=0.02,
                        help="step size per tree; lower it (with more --max-trees) to train longer")
    parser.add_argument("--max-trees", type=int, default=4000, help="upper limit on boosting rounds")
    parser.add_argument("--patience", type=int, default=100,
                        help="stop after this many rounds without validation improvement")
    parser.add_argument("--cv", type=int, default=0, metavar="FOLDS",
                        help="walk-forward evaluation over this many folds instead of one split")
    parser.add_argument("--cv-days", type=int, default=5, help="days in each walk-forward test block")
    parser.add_argument("--cv-trees", type=int, default=1000,
                        help="trees per walk-forward fold (fixed; no early stopping)")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU threads (-1 = all cores)")
    parser.add_argument("--save", metavar="PATH", help="write the trained model here (e.g. vol.json)")
    args = parser.parse_args(argv)

    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")

    series, _ = load_processed(args.data)
    if args.cv:
        if args.save:
            parser.error("--save needs the single split; drop --cv")
        scores = walk_forward(series, args.target, args.horizon, n_folds=args.cv,
                              fold_days=args.cv_days, n_trees=args.cv_trees,
                              learning_rate=args.learning_rate, n_jobs=args.jobs)
        print(f"target={args.target} horizon={args.horizon}min  walk-forward RMSE (log bp)")
        print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"mean  xgboost={scores['xgboost'].mean():.4f}  baseline={scores['baseline'].mean():.4f}")
        return 0

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
