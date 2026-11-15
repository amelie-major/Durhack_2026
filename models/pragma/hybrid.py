"""XGBoost with PRAGMA embeddings as extra features ("hybrid").

    python -m models.pragma.hybrid --target vol
    python -m models.pragma.hybrid --target spread --horizon 30 --pca 32

For every row XGBoost sees, this takes the PRAGMA embedding of the window
ending at that minute (from a pre-trained checkpoint, weights frozen), shrinks
it with PCA, and adds the components to XGBoost's own features. The test
predictions go to runs/pragma/ as `<target>_h<H>_hybrid.parquet`, so
`python -m models.pragma.compare` lists them next to xgboost and pragma.

The embeddings only use bars at or before each row's minute, and PCA is fitted
on training rows only, so nothing leaks from validation or test.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from loader import load_processed
from models.xgboost_model import FEATURES, MAX_HORIZON, _r2, _rmse, build_dataset, train

from .config import RUN_DIR
from .data import Bars, split_rows
from .train import load_checkpoint, pick_device, run_in_chunks, save_predictions


def check_same_rows(xgb_rows: pd.DataFrame, pragma_rows: pd.DataFrame, split: str) -> None:
    """XGBoost's and PRAGMA's row lists must be identical, in the same order."""
    same = (len(xgb_rows) == len(pragma_rows)
            and (xgb_rows["timestamp"].to_numpy() == pragma_rows["timestamp"].to_numpy()).all()
            and (xgb_rows["pair"].astype(str).to_numpy() == pragma_rows["pair"].to_numpy()).all()
            and np.allclose(xgb_rows["y"].to_numpy(), pragma_rows["y"].to_numpy()))
    if not same:
        raise ValueError(f"{split}: XGBoost and PRAGMA rows differ; cannot attach embeddings")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="XGBoost + PRAGMA embedding features.")
    parser.add_argument("--data", default="data/processed", help="output dir of loader.py")
    parser.add_argument("--checkpoint", default=os.path.join(RUN_DIR, "pretrain.pt"))
    parser.add_argument("--target", choices=["vol", "spread"], default="vol")
    parser.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")
    parser.add_argument("--pca", type=int, default=32,
                        help="embedding components to add (0 = use all, unreduced)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--name", default="hybrid", help="suffix of the predictions file")
    args = parser.parse_args(argv)
    if not 1 <= args.horizon <= MAX_HORIZON:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")

    from sklearn.decomposition import PCA

    device = pick_device(args.device)
    series, _ = load_processed(args.data)
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, device)
    bars = Bars(series, tokenizer)
    pragma_rows = split_rows(series, bars, args.target, args.horizon)
    train_df, val_df, test_df = build_dataset(series, args.target, args.horizon)
    xgb_rows = {"train": train_df, "val": val_df, "test": test_df}
    for split in xgb_rows:
        check_same_rows(xgb_rows[split], pragma_rows[split], split)

    print(f"device={device}  target={args.target} horizon={args.horizon}min  "
          f"checkpoint={args.checkpoint}  rows: train={len(train_df):,} "
          f"val={len(val_df):,} test={len(test_df):,}", flush=True)

    started = time.time()
    embeddings = {}
    for split, rows in pragma_rows.items():
        embeddings[split] = run_in_chunks(model, bars, rows["g"].to_numpy(), cfg, device,
                                          model.embedding)
        print(f"  embedded {split}: {embeddings[split].shape}  {time.time() - started:.0f}s",
              flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_components = args.pca or embeddings["train"].shape[1]
    pca = PCA(n_components=n_components, random_state=cfg.seed).fit(embeddings["train"])
    print(f"  PCA {n_components} components keep "
          f"{100 * pca.explained_variance_ratio_.sum():.1f}% of embedding variance", flush=True)
    emb_columns = [f"emb_{i}" for i in range(n_components)]
    for split, frame in xgb_rows.items():
        parts = pd.DataFrame(pca.transform(embeddings[split]).astype(np.float32),
                             columns=emb_columns, index=frame.index)
        xgb_rows[split] = pd.concat([frame, parts], axis=1)
    train_df, val_df, test_df = xgb_rows["train"], xgb_rows["val"], xgb_rows["test"]

    print("fitting XGBoost on its own features + embeddings ...", flush=True)
    hybrid = train(train_df, val_df, features=FEATURES + emb_columns)
    pred = hybrid.predict(test_df[FEATURES + emb_columns])
    print(f"hybrid test RMSE {_rmse(test_df['y'], pred):.4f}  R2 {_r2(test_df['y'], pred):.3f}  "
          f"(best round {hybrid.best_iteration})")
    print("saved", save_predictions(pragma_rows["test"], pred, args.target, args.horizon,
                                    args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
