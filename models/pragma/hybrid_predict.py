"""Predict with an exported hybrid model (XGBoost + PRAGMA embeddings).

    python -m models.pragma.hybrid_predict --model runs/pragma/hybrid_spread_h30 \
        --data data/processed --last 60 --out spread.csv

`--model` is a folder written by models.pragma.hybrid (pragma.pt, pca.npz,
xgb.json, meta.json). For every instrument (for vol only the VOL_TENOR ones,
as in training) this predicts at each of its last `--last` bars: the log of the
next-H-minute volatility or mean spread in bp, and that value in bp.

Features and embeddings are computed on each instrument's full history and then
sliced, so the rolling windows and the PRAGMA context are filled the same way
as in training. Runs on CPU (the default when there is no GPU).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from loader import load_processed
from models.xgboost_model import VOL_TENOR, build_features

from .data import Bars
from .train import load_checkpoint, pick_device, run_in_chunks


def load_model(folder: str, device):
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    with np.load(os.path.join(folder, "pca.npz"), allow_pickle=False) as pca:
        mean, components = pca["mean"], pca["components"]
    xgb = XGBRegressor()
    xgb.load_model(os.path.join(folder, "xgb.json"))
    model, tokenizer, cfg = load_checkpoint(os.path.join(folder, "pragma.pt"), device)
    return meta, mean, components, xgb, model, tokenizer, cfg


def predict(folder: str, data: str, last: int, device_name: str = "auto") -> pd.DataFrame:
    device = pick_device(device_name)
    meta, mean, components, xgb, model, tokenizer, cfg = load_model(folder, device)
    target = meta["target"]

    series, _ = load_processed(data)
    if target == "vol":
        series = {k: df for k, df in series.items() if df.attrs["tenor"] == VOL_TENOR}
    if not series:
        raise ValueError(f"no instruments in {data} to predict {target} for")

    bars = Bars(series, tokenizer)
    frames, ends = [], []
    for instrument_id, df in series.items():
        n = min(last, len(df))
        rows = build_features(df).iloc[-n:].copy()
        rows["timestamp"] = df["timestamp"].iloc[-n:].to_numpy()
        rows["instrument_id"] = instrument_id
        rows["tenor"] = df.attrs["tenor"]
        frames.append(rows)
        ends.append(bars.offset[instrument_id] + np.arange(len(df) - n, len(df)))
    rows = pd.concat(frames, ignore_index=True)
    ends = np.concatenate(ends)

    embeddings = run_in_chunks(model, bars, ends, cfg, device, model.embedding)
    reduced = (embeddings.astype(np.float64) - mean) @ components.T
    emb = pd.DataFrame(reduced.astype(np.float32), columns=meta["embedding_columns"])
    rows = pd.concat([rows, emb], axis=1)

    unknown = set(rows["pair"]) - set(meta["pair_categories"])
    if unknown:
        raise ValueError(f"pairs not seen in training: {sorted(unknown)}")
    rows["pair"] = pd.Categorical(rows["pair"], categories=meta["pair_categories"])

    pred = xgb.predict(rows[meta["features"] + meta["embedding_columns"]])
    out = rows[["instrument_id", "pair", "tenor", "timestamp"]].copy()
    out["pair"] = out["pair"].astype(str)
    out.insert(0, "target", target)
    out["pred_log_bp"] = pred.astype(np.float64)
    out["pred_bp"] = np.exp(out["pred_log_bp"])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predict with an exported hybrid model.")
    parser.add_argument("--model", required=True, help="folder written by models.pragma.hybrid")
    parser.add_argument("--data", default="data/processed", help="output dir of loader.py")
    parser.add_argument("--last", type=int, default=1,
                        help="predict at each instrument's last N bars")
    parser.add_argument("--out", required=True, help="output file (.csv or .parquet)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args(argv)
    if args.last < 1:
        parser.error("--last must be at least 1")
    if not args.out.endswith((".csv", ".parquet")):
        parser.error("--out must end in .csv or .parquet")

    out = predict(args.model, args.data, args.last, args.device)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"{len(out):,} predictions ({out['instrument_id'].nunique()} instruments, "
          f"{out['target'].iloc[0]}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
