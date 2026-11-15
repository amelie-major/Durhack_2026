"""Download 1-minute spot bars for extra currency pairs from Dukascopy.

Dukascopy's free datafeed has bid and ask minute candles for many pairs. This
script writes them as daily CSVs in the same shape as the Challenge1 spot
files, so loader.py reads them with no changes:

    python fetch_dukascopy.py                        # all default pairs, Jul-Aug 2026
    python fetch_dukascopy.py --pairs GBPUSD USDMXN  # just these
    python fetch_dukascopy.py --validate             # compare with Challenge1 spot
    python loader.py data/raw/Challenge1.zip data/raw/dukascopy data/processed

The data is spot only (no forward points). The server is slow and often
answers 503, so only a few requests run at once, each is retried, and every
download is cached under <out>/cache/. A rerun only fetches what is missing.
"""
from __future__ import annotations

import argparse
import http.client
import lzma
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"

# New currencies, each against USD, then crosses that close a triangle with
# pairs we already have (e.g. EURUSD x USDJPY = EURJPY).
DEFAULT_PAIRS = [
    "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
    "USDCNH", "USDHKD", "USDMXN", "USDZAR", "USDNOK", "USDSEK", "USDPLN", "USDTRY",
    "EURJPY", "EURGBP", "GBPJPY", "EURSGD", "SGDJPY",
]
# Pairs that are also in Challenge1, used only to check the two sources agree.
VALIDATE_PAIRS = ["EURUSD", "USDJPY", "USDSGD"]

START, END = "2026-07-01", "2026-08-31"
NOTIONAL = 1_000_000

# Dukascopy stamps a bar with the minute it starts; Challenge1 stamps the
# minute it ends. `--validate` reports which shift fits best.
BAR_SHIFT_MIN = 1

# One candle: seconds since midnight, open, close, low, high (integer prices), volume.
_CANDLE = np.dtype([("sec", ">u4"), ("open", ">u4"), ("close", ">u4"),
                    ("low", ">u4"), ("high", ">u4"), ("vol", ">f4")])

_SPOT_T_PLUS_1 = {"USDCAD", "USDTRY"}  # these settle in one business day, the rest in two
_OHLC = ["open", "high", "low", "close"]
FEED = "dukascopy"  # written into every CSV so loader.py knows where the quotes came from
CSV_COLS = (["timestamp", "settlement_date"]
            + [f"{p}_{s}" for p in _OHLC for s in ("bid", "ask", "mid")] + ["feed"])


def price_scale(pair: str) -> float:
    """Integer prices are in units of 1e-5, or 1e-3 when quoted in JPY."""
    return 1e3 if pair[3:] == "JPY" else 1e5


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

class Throttled(RuntimeError):
    """The server kept refusing a request after every retry."""


def _read_with_deadline(request, timeout: float = 30, deadline: float = 90) -> bytes:
    """Read a response, giving up if the whole body takes longer than `deadline` seconds.

    The socket timeout alone is not enough: the server sometimes sends a few
    bytes at a time and never finishes.
    """
    started = time.monotonic()
    chunks = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while chunk := response.read1(16384):  # read1 returns after one network read
            chunks.append(chunk)
            if time.monotonic() - started > deadline:
                raise TimeoutError("response took too long")
    return b"".join(chunks)


def download(pair: str, day: date, side: str, cache: Path, pause: float = 1.0,
             retries: int = 8) -> bytes:
    """One side ("BID" or "ASK") of one day, from the cache or the server.

    A day with no data (a Saturday, or a pair Dukascopy does not carry) is an
    empty file, which is cached like any other.
    """
    path = cache / pair / f"{day:%Y%m%d}_{side}.bi5"
    if path.exists():
        return path.read_bytes()

    url = URL.format(pair=pair, y=day.year, m=day.month - 1, d=day.day, side=side)  # month is 0-based
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    wait = pause
    for _ in range(retries):
        time.sleep(wait)
        try:
            body = _read_with_deadline(request)
            if body:
                lzma.decompress(body)  # a cut-off file must not reach the cache
        except urllib.error.HTTPError as e:
            if e.code == 404:
                body = b""
            else:  # 503 is the server throttling us: slow down and retry
                wait = min(wait * 2 + 1, 60)
                continue
        except (OSError, http.client.HTTPException, lzma.LZMAError):  # timeout, reset, cut-off body
            wait = min(wait * 2 + 1, 60)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body
    raise Throttled(f"{pair} {day} {side}: gave up after {retries} tries")


# --------------------------------------------------------------------------
# Decode and reshape
# --------------------------------------------------------------------------

def decode(raw: bytes, day: date, pair: str) -> pd.DataFrame:
    """Candles of one .bi5 file, indexed by bar-start time (UTC)."""
    if not raw:
        return pd.DataFrame(columns=_OHLC + ["vol"], index=pd.DatetimeIndex([], tz="UTC"))
    candles = np.frombuffer(lzma.decompress(raw), dtype=_CANDLE)
    start = pd.Timestamp(day, tz="UTC")
    out = pd.DataFrame({c: candles[c] / price_scale(pair) for c in _OHLC},
                       index=start + pd.to_timedelta(candles["sec"].astype("int64"), unit="s"))
    out["vol"] = candles["vol"].astype("float32")  # native byte order
    return out


def settlement_dates(timestamps: pd.DatetimeIndex, pair: str) -> np.ndarray:
    """Spot settlement date of each bar.

    The trade date rolls at 21:00 UTC (17:00 New York in summer), then
    settlement is two business days later (one for USDCAD and USDTRY).
    Weekends are skipped; bank holidays are not known and so are not.
    """
    trade_date = (timestamps + pd.Timedelta(hours=3)).tz_localize(None).normalize()
    trade_date = np.busday_offset(trade_date.values.astype("datetime64[D]"), 0, roll="forward")
    return np.busday_offset(trade_date, 1 if pair in _SPOT_T_PLUS_1 else 2)


def make_bars(bid: pd.DataFrame, ask: pd.DataFrame, pair: str,
              shift_min: int = BAR_SHIFT_MIN) -> pd.DataFrame:
    """Join the two sides into Challenge1's spot columns.

    Minutes with no trades on either side are dropped: Dukascopy pads them
    with flat candles, whereas Challenge1 leaves quiet minutes out.
    """
    both = bid.add_suffix("_bid").join(ask.add_suffix("_ask"), how="inner")
    both = both[(both["vol_bid"] > 0) | (both["vol_ask"] > 0)]
    both.index = both.index + pd.Timedelta(minutes=shift_min)

    out = pd.DataFrame({"timestamp": both.index})
    out["settlement_date"] = settlement_dates(both.index, pair)
    decimals = 3 if price_scale(pair) == 1e3 else 5
    for part in _OHLC:
        b, a = both[f"{part}_bid"].to_numpy(), both[f"{part}_ask"].to_numpy()
        out[f"{part}_bid"] = b
        out[f"{part}_ask"] = a
        out[f"{part}_mid"] = np.round((b + a) / 2, decimals + 1)
    out["feed"] = FEED
    return out[CSV_COLS]


def prefetch(pairs: List[str], days: List[date], cache: Path, pause: float = 1.0,
             workers: int = 4) -> None:
    """Fill the cache for every pair through one shared pool of downloads.

    Going pair by pair would leave most workers idle while a pair's last few
    slow files retry. Files that give up here are tried again per pair later.
    """
    jobs = [(pair, day, side) for pair in pairs for day in days for side in ("BID", "ASK")
            if not (cache / pair / f"{day:%Y%m%d}_{side}.bi5").exists()]
    if not jobs:
        return
    print(f"downloading {len(jobs):,} files with {workers} workers")
    started = time.monotonic()
    done = 0

    def one(job) -> None:
        nonlocal done
        try:
            download(*job, cache, pause)
        except Throttled:
            return
        done += 1
        if done % 50 == 0:
            rate = (time.monotonic() - started) / done
            print(f"  {done:,}/{len(jobs):,} files, {rate:.1f}s each, "
                  f"about {(len(jobs) - done) * rate / 60:.0f} min left", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))


def fetch_sides(pair: str, days: List[date], cache: Path, pause: float = 1.0,
                workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bid and ask candles of one pair over `days`, downloading a few files at a time."""
    jobs = [(day, side) for day in days for side in ("BID", "ASK")]

    def one(job):
        raw = download(pair, job[0], job[1], cache, pause)
        print(".", end="", flush=True)
        return raw

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]  # all run, even if one gives up
    print()
    raws = {job: f.result() for job, f in zip(jobs, futures)}  # raises Throttled if any gave up
    bid = pd.concat([decode(raws[(day, "BID")], day, pair) for day in days])
    ask = pd.concat([decode(raws[(day, "ASK")], day, pair) for day in days])
    return bid, ask


def fetch_pair(pair: str, days: List[date], cache: Path, pause: float = 1.0,
               workers: int = 4) -> pd.DataFrame:
    """All bars of one pair over `days`, in Challenge1's spot columns."""
    bid, ask = fetch_sides(pair, days, cache, pause, workers)
    return make_bars(bid, ask, pair)


def write_daily_csvs(bars: pd.DataFrame, pair: str, out: Path) -> int:
    """One CSV per UTC day, named like the Challenge1 spot files. Returns the file count."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    # File a bar under the day it started in, so the bar ending at midnight
    # stays with its own day instead of becoming a one-row file.
    bar_start = bars["timestamp"] - pd.Timedelta(minutes=BAR_SHIFT_MIN)
    for day, day_bars in bars.groupby(bar_start.dt.date):
        name = f"{pair}_{pair[:3]}_SPOT_SPOT_{NOTIONAL}_{day:%Y%m%d}.csv"
        day_bars.to_csv(out / name, index=False)
        n += 1
    return n


# --------------------------------------------------------------------------
# Check against Challenge1
# --------------------------------------------------------------------------

def compare(ours: pd.DataFrame, theirs: pd.DataFrame) -> Dict[str, float]:
    """Agreement between our bars and Challenge1's for the same pair.

    Both frames need timestamp, close_bid, close_ask, close_mid and settlement_date.
    """
    cols = ["close_bid", "close_ask", "close_mid", "settlement_date"]
    j = (theirs.set_index("timestamp")[cols].add_suffix("_c")
         .join(ours.set_index("timestamp")[cols].add_suffix("_d"), how="inner"))
    if j.empty:
        return {"n_common": len(j)}
    ret_c = np.log(j["close_mid_c"]).diff()
    ret_d = np.log(j["close_mid_d"]).diff()
    spread_c = (j["close_ask_c"] - j["close_bid_c"]) / j["close_mid_c"] * 1e4
    spread_d = (j["close_ask_d"] - j["close_bid_d"]) / j["close_mid_d"] * 1e4
    same_settle = (pd.to_datetime(j["settlement_date_c"]).dt.date
                   == pd.to_datetime(j["settlement_date_d"]).dt.date)
    return {
        "n_common": len(j),
        "share_of_challenge_bars": len(j) / len(theirs),
        "ret_corr": ret_c.corr(ret_d),
        "mid_diff_bp": ((j["close_mid_c"] - j["close_mid_d"]).abs() / j["close_mid_c"] * 1e4).mean(),
        "spread_bp_challenge": spread_c.mean(),
        "spread_bp_dukascopy": spread_d.mean(),
        "spread_corr": spread_c.corr(spread_d),
        "settle_match": same_settle.mean(),
    }


def spread_scale(stats: pd.DataFrame) -> Optional[float]:
    """Factor that puts Dukascopy spreads on Challenge1's scale.

    The geometric mean of each overlapping pair's ratio of mean spreads, so
    every pair counts equally whatever its spread level.
    """
    if stats.empty or "spread_bp_dukascopy" not in stats:
        return None
    ratio = stats["spread_bp_challenge"] / stats["spread_bp_dukascopy"]
    return float(np.exp(np.log(ratio).mean()))


def validate(challenge: str, days: List[date], cache: Path, pause: float, workers: int) -> None:
    from loader import load_all

    series, _ = load_all([challenge], verbose=False, product="spot")
    by_pair = {df.attrs["pair"]: df for df in series.values()}
    prefetch(VALIDATE_PAIRS, days, cache, pause, workers)
    rows = []
    for pair in VALIDATE_PAIRS:
        if pair not in by_pair:
            print(f"{pair}: no spot series in {challenge}")
            continue
        print(pair, end=" ", flush=True)
        try:
            bid, ask = fetch_sides(pair, days, cache, pause, workers)
        except Throttled as e:
            print(f"\n  skipped: {e}. Rerun later; finished days are cached.")
            continue
        for shift in (0, 1):
            ours = make_bars(bid, ask, pair, shift_min=shift)
            rows.append({"pair": pair, "shift_min": shift, **compare(ours, by_pair[pair])})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    scale = spread_scale(table[table["shift_min"] == BAR_SHIFT_MIN])
    if scale is not None:
        print(f"\nspread scale (Challenge1 / Dukascopy, geometric mean over pairs): {scale:.3f}")
        print("  this is loader.SPREAD_SCALE['dukascopy']")


# --------------------------------------------------------------------------

def _default_challenge() -> str:
    for path in ("data/raw/Challenge1.zip", "data/data.zip"):
        if Path(path).exists():
            return path
    return "data/raw/Challenge1.zip"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--out", default="data/raw/dukascopy")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--workers", type=int, default=4, help="downloads at once (keep it small)")
    ap.add_argument("--validate", action="store_true",
                    help="compare EURUSD, USDJPY and USDSGD with the Challenge1 spot files, write nothing")
    ap.add_argument("--challenge", default=_default_challenge(), help="Challenge1 zip, for --validate")
    args = ap.parse_args(argv)

    out = Path(args.out)
    cache = out / "cache"
    days = [d.date() for d in pd.date_range(args.start, args.end, freq="D")]

    if args.validate:
        validate(args.challenge, days, cache, args.pause, args.workers)
        return 0

    prefetch(args.pairs, days, cache, args.pause, args.workers)
    failed = []
    for pair in args.pairs:
        print(f"{pair} ", end="", flush=True)
        try:
            bars = fetch_pair(pair, days, cache, args.pause, args.workers)
        except Throttled as e:
            print(f"\n  stopped: {e}. Rerun later; finished days are cached.")
            failed.append(pair)
            continue
        # The last minute of the range ends at midnight; drop it so the series
        # stops on the end date instead of adding a one-bar day.
        bars = bars[bars["timestamp"] < pd.Timestamp(days[-1], tz="UTC") + pd.Timedelta(days=1)]
        if bars.empty:
            print("  no data (Dukascopy may not carry this pair)")
            failed.append(pair)
            continue
        n_files = write_daily_csvs(bars, pair, out)
        print(f"  {len(bars):,} bars in {n_files} files, "
              f"{bars['timestamp'].min():%Y-%m-%d} to {bars['timestamp'].max():%Y-%m-%d}")
    if failed:
        print("not written:", " ".join(failed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
