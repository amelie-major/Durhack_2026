"""Turn the signal export (actions.csv) into a compact data.json for the map.

Usage: python frontend/export_data.py [actions.csv] [--freq 15min]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def parse_id(inst: str) -> dict:
    # PAIR_DEALTCCY_PRODUCT_TENOR_NOTIONAL
    parts = inst.split("_")
    return {"id": inst, "pair": parts[0], "product": parts[2], "tenor": parts[3]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default=str(HERE.parent / "actions.csv"))
    ap.add_argument("--freq", default="15min", help="resample bucket (default 15min)")
    ap.add_argument("--out", default=str(HERE / "data.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"])
    df = df.sort_values(["instrument_id", "timestamp"])
    if "pred_vol_bp" in df:
        vol_kind = "predicted"
        df["vol_bp"] = df["pred_vol_bp"]
    else:
        # Older exports: realised vol = rolling 10-min std of 1-min returns in bp.
        vol_kind = "realised"
        ret = df.groupby("instrument_id")["close_mid"].transform(lambda s: s.pct_change() * 1e4)
        df["vol_bp"] = ret.groupby(df["instrument_id"]).transform(lambda s: s.rolling(10, min_periods=3).std())
    df["bucket"] = df["timestamp"].dt.floor(args.freq)
    g = df.groupby(["instrument_id", "bucket"])
    agg = pd.DataFrame({
        "score": g["score"].last(),
        "close_mid": g["close_mid"].last(),
        "vol_bp": g["vol_bp"].mean(),
        # net trades in the bucket, clipped to buy/sell/nothing
        "action": g["action"].sum().clip(-1, 1),
        # realised vol over the same next-H minutes the prediction is for
        "true_vol_bp": g["actual_vol_bp"].mean() if "actual_vol_bp" in df else g["vol_bp"].mean() * float("nan"),
    }).reset_index()

    times = sorted(agg["bucket"].unique())
    out = {
        "freq": args.freq,
        "timestamps": [pd.Timestamp(t).isoformat() for t in times],
        "instruments": [],
        "vol_kind": vol_kind,
        "score": {}, "action": {}, "close_mid": {}, "vol": {}, "true_vol": {},
    }
    for inst, sub in agg.groupby("instrument_id"):
        sub = sub.set_index("bucket").reindex(times)
        out["instruments"].append(parse_id(inst))
        out["score"][inst] = [None if pd.isna(v) else round(float(v), 4) for v in sub["score"]]
        out["action"][inst] = [0 if pd.isna(v) else int(v) for v in sub["action"]]
        out["vol"][inst] = [None if pd.isna(v) else round(float(v), 3) for v in sub["vol_bp"]]
        out["true_vol"][inst] = [None if pd.isna(v) else round(float(v), 3) for v in sub["true_vol_bp"]]
        out["close_mid"][inst] = [None if pd.isna(v) else round(float(v), 6) for v in sub["close_mid"]]

    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out}: {len(out['instruments'])} instruments, {len(times)} timestamps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
