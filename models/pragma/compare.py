"""Score PRAGMA next to XGBoost and the naive baseline on the same test rows.

    python -m models.pragma.compare --target vol
    python -m models.pragma.compare --target spread --horizon 30

Trains the sample XGBoost model here (it takes seconds to minutes), then reads
the PRAGMA predictions that `models.pragma.train finetune / probe` saved in
runs/pragma/. Every PRAGMA file is checked to cover exactly XGBoost's test rows.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from loader import load_processed
from models.xgboost_model import FEATURES, MAX_HORIZON, _r2, _rmse, build_dataset, train

from .config import RUN_DIR


def load_pragma_predictions(path: str, test_df: pd.DataFrame) -> pd.Series:
    """Read one prediction file and make sure it lines up with XGBoost's rows."""
    saved = pd.read_parquet(path)
    if len(saved) != len(test_df):
        raise ValueError(f"{path}: {len(saved):,} rows, XGBoost test set has {len(test_df):,}")
    same_time = (saved["timestamp"].to_numpy() == test_df["timestamp"].to_numpy()).all()
    same_pair = (saved["pair"].to_numpy() == test_df["pair"].astype(str).to_numpy()).all()
    same_y = np.allclose(saved["y"].to_numpy(), test_df["y"].to_numpy())
    if not (same_time and same_pair and same_y):
        raise ValueError(f"{path}: rows do not match XGBoost's test rows "
                         "(different --horizon, or data changed since it was written?)")
    return pd.Series(saved["pred"].to_numpy(), index=test_df.index)


def print_table(test_df: pd.DataFrame, predictions: Dict[str, pd.Series]) -> None:
    y = test_df["y"]
    pairs = list(test_df["pair"].cat.categories)
    print(f"\nTest ({len(test_df):,} rows, target in log bp; lower RMSE is better)")
    print(f"  {'model':<18}{'RMSE':>8}{'R2':>8}   " + "".join(f"{p + ' RMSE':>13}" for p in pairs))
    for name, pred in predictions.items():
        line = f"  {name:<18}{_rmse(y, pred):>8.4f}{_r2(y, pred):>8.3f}   "
        for pair in pairs:
            idx = test_df.index[test_df["pair"] == pair]
            line += f"{_rmse(y[idx], pred[idx]):>13.4f}" if len(idx) else f"{'-':>13}"
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare PRAGMA with XGBoost.")
    parser.add_argument("--data", default="data/processed", help="output dir of loader.py")
    parser.add_argument("--target", choices=["vol", "spread"], default="vol")
    parser.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")
    parser.add_argument("--preds", nargs="*",
                        help="PRAGMA prediction files (default: all in runs/pragma for this target)")
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")

    series, _ = load_processed(args.data)
    train_df, val_df, test_df = build_dataset(series, args.target, args.horizon)
    model = train(train_df, val_df)

    predictions = {
        "baseline": test_df["baseline"],
        "xgboost": pd.Series(model.predict(test_df[FEATURES]), index=test_df.index),
    }
    pattern = os.path.join(RUN_DIR, f"{args.target}_h{args.horizon}_*.parquet")
    paths = args.preds if args.preds is not None else sorted(glob.glob(pattern))
    if not paths:
        print(f"no PRAGMA predictions found ({pattern}); showing XGBoost only")
    for path in paths:
        name = os.path.basename(path).rsplit(".", 1)[0].split("_")[-1]
        predictions[f"pragma {name}"] = load_pragma_predictions(path, test_df)

    print_table(test_df, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
