"""Does adding Yahoo spot history help? Train vol and direction with and without
it and score both on the same Challenge1 test days.

    python fetch_history.py          # first
    python -m models.yahoo_eval --horizon 10
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from live.engine import instrument_consts, live_bars, yahoo_series
from loader import load_processed
from models.signal import DIRECTION_FEATURES, direction_dataset, train_direction
from models.xgboost_model import FEATURES, _r2, _rmse, build_dataset, train


def read_histdata(zips) -> pd.DataFrame:
    """HistData.com generic-ASCII 1-minute zips (DAT_ASCII_<PAIR>_M1_<YEAR>.zip).
    Bid OHLC, stamped in EST with no daylight saving, i.e. a fixed UTC-5."""
    parts = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            df = pd.read_csv(zf.open(name), sep=";", header=None,
                             names=["ts", "open_mid", "high_mid", "low_mid", "close_mid", "vol"])
        df["timestamp"] = (pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
                           + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
        parts.append(df.drop(columns=["ts", "vol"]))
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp")
    return df[(df[["open_mid", "high_mid", "low_mid", "close_mid"]] > 0).all(axis=1)].reset_index(drop=True)


def histdata_series(consts: dict, hd_dir: str) -> dict:
    out = {}
    for inst, c in consts.items():
        zips = sorted(Path(hd_dir).glob(f"DAT_ASCII_{c['pair']}_M1_*.zip"))
        if c["tenor"] == "1M" and zips:
            out[f"hd_{inst}"] = live_bars(read_histdata(zips), c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--histdata", metavar="DIR",
                    help="also compare HistData.com zips (DAT_ASCII_<PAIR>_M1_<YEAR>.zip) in DIR")
    args = ap.parse_args()

    series, _ = load_processed(args.data)
    series = {k: v for k, v in series.items() if v.attrs["product"] == "FORWARD"}
    consts = instrument_consts(series)
    extra = yahoo_series(consts)
    if not extra:
        raise SystemExit("no Yahoo history found; run python fetch_history.py first")
    variants = [("", None), (" + yahoo", extra)]
    if args.histdata:
        hd = histdata_series(consts, args.histdata)
        variants += [(" + histdata", hd), (" + both", {**hd, **extra})]

    print(f"horizon {args.horizon} min; test = Challenge1 last 7 days\n")
    print(f"{'model':<22}{'train rows':>12}{'trees':>7}{'test RMSE':>11}{'R2':>8}{'hit rate':>10}{'logloss':>9}")
    for suffix, ext in variants:
        label = "vol" + suffix
        tr, va, te = build_dataset(series, "vol", args.horizon, extra_train=ext)
        m = train(tr, va)
        p = m.predict(te[FEATURES])
        print(f"{label:<22}{len(tr):>12,}{m.best_iteration + 1:>7}"
              f"{_rmse(te['y'], p):>11.4f}{_r2(te['y'], p):>8.3f}")
    for suffix, ext in variants:
        label = "direction" + suffix
        tr, va, te = direction_dataset(series, args.horizon, extra_train=ext)
        m = train_direction(tr, va)
        te = te[te["fwd_ret_bp"] != 0]
        p = np.clip(m.predict_proba(te[DIRECTION_FEATURES])[:, 1], 1e-6, 1 - 1e-6)
        y = te["up"].to_numpy()
        hit = ((p > 0.5) == y).mean()
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        print(f"{label:<22}{len(tr):>12,}{m.best_iteration + 1:>7}{'':>11}{'':>8}{hit:>10.3%}{ll:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
