"""Download the last ~30 days of Yahoo 1-minute spot bars and keep them on disk.

Yahoo only serves 1-minute bars for the last 30 days, at most 8 days per
request, so this pages through 7-day chunks. Each run merges into
data/yahoo/<PAIR>.parquet, so running it every few weeks grows the history.

The bars are spot mid OHLC only (no bid/ask, no forward points). They are used
to give the vol and direction models more history; the spread model stays on
the Challenge1 data. See live.engine.yahoo_series for how they become
training frames.

    python fetch_history.py                    # EURUSD USDJPY USDSGD
    python fetch_history.py --pairs EURUSD --days 14
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from live.engine import fetch_spot

OUT = Path(__file__).resolve().parent / "data" / "yahoo"
CHUNK_DAYS = 7


def fetch_history(pair: str, days: int = 29) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=days)
    parts = []
    while start < end:
        stop = min(start + pd.Timedelta(days=CHUNK_DAYS), end)
        span = f"period1={int(start.timestamp())}&period2={int(stop.timestamp())}"
        parts.append(fetch_spot(pair, span))
        start = stop
        time.sleep(0.5)  # be polite
    return pd.concat(parts, ignore_index=True)


def update(pair: str, days: int) -> pd.DataFrame:
    path = OUT / f"{pair}.parquet"
    new = fetch_history(pair, days)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    df = (new.drop_duplicates("timestamp", keep="last")
             .sort_values("timestamp").reset_index(drop=True))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=["EURUSD", "USDJPY", "USDSGD"])
    ap.add_argument("--days", type=int, default=29, help="how far back (Yahoo allows < 30)")
    args = ap.parse_args()
    for pair in args.pairs:
        df = update(pair, args.days)
        print(f"{pair}: {len(df):,} bars, {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
