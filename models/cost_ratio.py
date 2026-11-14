"""Cost ratio: predicted trading cost relative to the predicted price move.

For each test row (one instrument at one minute t) this combines the two
XGBoost models from models.xgboost_model:

    cost_ratio = predicted mean spread over t+1 ... t+H  (bp)
                 / predicted realised volatility over t+1 ... t+H  (bp)

A low ratio means the expected move is large compared with the cost of
crossing the spread; a high ratio means trading is expensive for the move on
offer. Volatility is modelled once per pair (on the 1M tenor), so every tenor
of a pair shares the same vol prediction at a given minute.

Run from the repo root::

    python -m models.cost_ratio
    python -m models.cost_ratio --horizon 15 --save cost_ratio.csv
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from loader import load_processed
from models.xgboost_model import FEATURES, MAX_HORIZON, build_dataset, train

KEYS = ["pair", "timestamp"]


def predict_test(series: Dict[str, pd.DataFrame], target: str, horizon: int,
                 n_jobs: int = -1) -> pd.DataFrame:
    """Train one model and return its test rows with predicted and actual values in bp."""
    train_df, val_df, test_df = build_dataset(series, target, horizon)
    model = train(train_df, val_df, n_jobs=n_jobs)
    print(f"  {target:<7} best iteration {model.best_iteration + 1}, test rows {len(test_df):,}")
    out = test_df[KEYS + ["tenor_days"]].copy()
    out["pair"] = out["pair"].astype(str)
    out[f"pred_{target}_bp"] = np.exp(model.predict(test_df[FEATURES]))  # targets are log bp
    out[f"actual_{target}_bp"] = np.exp(test_df["y"].to_numpy())
    return out


def cost_ratios(series: Dict[str, pd.DataFrame], horizon: int, n_jobs: int = -1) -> pd.DataFrame:
    """Per-row predicted and actual cost ratios on the test split."""
    spread = predict_test(series, "spread", horizon, n_jobs)
    vol = predict_test(series, "vol", horizon, n_jobs).drop(columns="tenor_days")
    rows = spread.merge(vol, on=KEYS, how="inner")  # rows where both targets exist
    rows["pred_ratio"] = rows["pred_spread_bp"] / rows["pred_vol_bp"]
    rows["actual_ratio"] = rows["actual_spread_bp"] / rows["actual_vol_bp"]
    return rows


def summarise(rows: pd.DataFrame) -> None:
    """Print median ratios per pair and tenor, and how well predictions rank actuals."""
    print(f"\nCost ratio = spread / expected move ({len(rows):,} test rows)")
    print(f"  {'pair':<8}{'tenor_d':>8}{'pred med':>10}{'actual med':>12}{'rank corr':>11}")
    for (pair, tenor), g in rows.groupby(["pair", "tenor_days"]):
        corr = g["pred_ratio"].corr(g["actual_ratio"], method="spearman")
        print(f"  {pair:<8}{tenor:>8.0f}{g['pred_ratio'].median():>10.3f}"
              f"{g['actual_ratio'].median():>12.3f}{corr:>11.3f}")
    corr = rows["pred_ratio"].corr(rows["actual_ratio"], method="spearman")
    log_err = np.log(rows["pred_ratio"] / rows["actual_ratio"])
    print(f"\n  overall: pred median {rows['pred_ratio'].median():.3f}, "
          f"actual median {rows['actual_ratio'].median():.3f}, rank corr {corr:.3f}, "
          f"RMSE(log ratio) {np.sqrt(np.mean(log_err ** 2)):.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predicted spread cost relative to predicted move.")
    parser.add_argument("--data", default="data/processed", help="output dir of loader.py")
    parser.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU threads (-1 = all cores)")
    parser.add_argument("--save", metavar="PATH", help="write per-row ratios to this CSV")
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")

    series, _ = load_processed(args.data)
    print(f"horizon={args.horizon}min, training spread and vol models")
    rows = cost_ratios(series, args.horizon, args.jobs)
    summarise(rows)
    if args.save:
        rows.to_csv(args.save, index=False)
        print(f"\nsaved {len(rows):,} rows to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
