"""Load 1-minute FX-forward price bars into clean, feature-rich DataFrames.

Input
-----
The raw data is one CSV per instrument per day. Each filename describes the
instrument and the day, for example::

    EURUSD_EUR_FWD_1M_1000000_20240105.csv
    |      |   |   |  |       |
    pair   |   |   |  |       date (YYYYMMDD)
           |   |   |  notional
           |   |   tenor
           |   product
           dealt currency

Files can live in folders, in zip files, or in zips nested inside zips. Some
filenames have a junk prefix (e.g. "85de32c1-EURUSD_...") which is ignored.

Pipeline
--------
1. discover()      find every data file and skip duplicate copies
2. read_day()      read one CSV, drop bad rows, report quality stats
3. load_all()      join all days of each instrument into one DataFrame
4. add_features()  add returns, spreads, gaps, settlement and calendar columns
5. time_split() / split_instruments() / save_processed()
                   prepare train/val/test splits or write parquet files

Command line
------------
    python loader.py [--product all|forward|spot] <input dirs, CSVs or zips ...> <output dir>
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import date as _date
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ==========================================================================
# 1. Understanding filenames
# ==========================================================================

_FILENAME_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])                 # must not start in the middle of a word
    (?P<pair>[A-Z]{6})        _      # currency pair,  e.g. EURUSD
    (?P<dealt>[A-Z]{3})       _      # dealt currency, e.g. EUR
    (?P<product>[A-Z][A-Z0-9]*) _    # product,        e.g. FWD
    (?P<tenor>\d*[A-Z]+)      _      # tenor,          e.g. 1M, ON, SPOT
    (?P<notional>\d+)         _      # notional,       e.g. 1000000
    (?P<date>\d{8})                  # trading day,    e.g. 20240105
    \.csv$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class FileMeta:
    """Everything we can learn about a data file from its name."""

    pair: str           # "EURUSD"
    base: str           # "EUR" (first half of the pair)
    quote: str          # "USD" (second half of the pair)
    dealt: str          # "EUR"
    product: str        # "FWD"
    tenor: str          # "1M"
    notional: int       # 1000000
    date: _date         # date(2024, 1, 5)
    instrument_id: str  # "EURUSD_EUR_FWD_1M_1000000" (everything except the date)

    def attrs(self) -> dict:
        """Instrument-level fields, suitable for DataFrame.attrs (no date)."""
        fields = asdict(self)
        fields.pop("date")
        return fields


def _basename(path: str) -> str:
    """Last component of a path, accepting both / and \\ separators."""
    return re.split(r"[\\/]", str(path))[-1]


def parse_filename(name: str) -> Optional[FileMeta]:
    """Parse a data filename into a FileMeta.

    Leading folders and junk prefixes are tolerated. Returns None if the name
    is not a data file (including macOS "._" resource-fork files).
    """
    filename = _basename(name)
    if filename.startswith("._"):
        return None

    match = _FILENAME_RE.search(filename)
    if not match:
        return None

    try:
        trading_day = pd.Timestamp(match["date"]).date()
    except ValueError:  # e.g. 20241399 is eight digits but not a real date
        return None

    pair = match["pair"]
    dealt = match["dealt"]
    product = match["product"]
    tenor = match["tenor"]
    notional = int(match["notional"])

    return FileMeta(
        pair=pair,
        base=pair[:3],
        quote=pair[3:],
        dealt=dealt,
        product=product,
        tenor=tenor,
        notional=notional,
        date=trading_day,
        instrument_id=f"{pair}_{dealt}_{product}_{tenor}_{notional}",
    )


# ==========================================================================
# 2. Finding files (folders, CSVs, zips, zips inside zips)
# ==========================================================================

# A "reader" is a function with no arguments that returns a file's bytes.
# Using readers means we only open a file when we actually need its content.
Reader = Callable[[], bytes]


@dataclass
class Source:
    """One data file we have found but not yet read."""

    meta: FileMeta
    source: str          # where it came from, e.g. "data.zip!2024/EURUSD_...csv"
    read_bytes: Reader   # call this to get the CSV content


def _is_macos_junk(path: str) -> bool:
    """True for the __MACOSX folders and "._" files that macOS adds to zips."""
    parts = re.split(r"[\\/]", path)
    return "__MACOSX" in parts or parts[-1].startswith("._")


def _reader_for_disk_file(path: str) -> Reader:
    def read() -> bytes:
        with open(path, "rb") as f:
            return f.read()
    return read


def _reader_for_zip_entry(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> Reader:
    def read() -> bytes:
        return archive.read(entry)
    return read


def _walk_zip(archive: zipfile.ZipFile, label: str) -> Iterator[Tuple[str, Reader]]:
    """Yield (location, reader) for every CSV in a zip, opening nested zips too.

    Locations use "!" to show "inside of", e.g. "outer.zip!inner.zip!file.csv".
    """
    for entry in archive.infolist():
        if entry.is_dir() or _is_macos_junk(entry.filename):
            continue

        location = f"{label}!{entry.filename}"
        name = entry.filename.lower()

        if name.endswith(".zip"):
            try:
                inner_archive = zipfile.ZipFile(io.BytesIO(archive.read(entry)))
            except zipfile.BadZipFile:
                continue  # skip corrupt inner zips rather than failing the run
            yield from _walk_zip(inner_archive, location)

        elif name.endswith(".csv"):
            yield location, _reader_for_zip_entry(archive, entry)


def _walk_path(path: str) -> Iterator[Tuple[str, Reader]]:
    """Yield (location, reader) for every CSV under a folder, CSV or zip path.

    Folders are walked in sorted order so results are reproducible.
    """
    if os.path.isdir(path):
        for folder, subfolders, filenames in os.walk(path):
            # Editing subfolders in place controls where os.walk goes next.
            subfolders[:] = sorted(d for d in subfolders if d != "__MACOSX")
            for filename in sorted(filenames):
                if filename.startswith("._"):
                    continue
                yield from _walk_path(os.path.join(folder, filename))

    elif os.path.isfile(path):
        name = path.lower()
        if name.endswith(".zip"):
            # The zip is left open on purpose: its readers need it later.
            yield from _walk_zip(zipfile.ZipFile(path), path)
        elif name.endswith(".csv"):
            yield path, _reader_for_disk_file(path)

    else:
        raise FileNotFoundError(path)


PRODUCTS = ("all", "forward", "spot")


def _wanted(meta: FileMeta, product: str) -> bool:
    """True if `meta` matches the product filter ("all", "forward" or "spot")."""
    if product == "all":
        return True
    is_spot = meta.product == "SPOT"
    return is_spot if product == "spot" else not is_spot


def discover(paths: Iterable[str], skipped: Optional[List[Tuple[FileMeta, str]]] = None,
             product: str = "all", dedupe: bool = True) -> Iterator[Source]:
    """Yield every recognised data file under `paths`, once per (instrument, day).

    `product` limits the result to "forward" or "spot" files ("all" keeps both).

    The same day can appear more than once, e.g. inside a zip and also
    extracted next to it. The first copy found is used; later copies are
    appended to `skipped` (if a list is given) as (meta, location). With
    `dedupe=False` every copy is yielded and the caller decides.
    """
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {PRODUCTS}, got {product!r}")
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    already_seen = set()
    for path in paths:
        for location, reader in _walk_path(os.fspath(path)):
            filename = location.split("!")[-1]
            meta = parse_filename(filename)
            if meta is None or not _wanted(meta, product):
                continue  # not a data file, or filtered out

            key = (meta.instrument_id, meta.date)
            if dedupe and key in already_seen:
                if skipped is not None:
                    skipped.append((meta, location))
                continue

            already_seen.add(key)
            yield Source(meta, location, reader)


# ==========================================================================
# 3. Reading and cleaning one day
# ==========================================================================

# The columns we keep from each raw CSV.
KEEP_COLS = [
    "timestamp",
    "open_mid", "high_mid", "low_mid", "close_mid",
    "close_bid", "close_ask",
    "close_mid_fp",       # forward points at the close
    "close_mid_all_in",   # close mid + forward points
    "settlement_date",
]

_OHLC = ["open", "high", "low", "close"]

# Every price column that might be present, e.g. "open_bid", "close_mid".
_PRICE_COLS = [f"{bar_part}_{side}" for bar_part in _OHLC for side in ("bid", "ask", "mid")]


def _parse_and_dedupe(raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Parse date columns, sort by time and drop repeated timestamps.

    Returns (clean frame, number of duplicate rows removed).
    """
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["settlement_date"] = pd.to_datetime(raw["settlement_date"], errors="coerce")

    raw = raw.sort_values("timestamp", kind="stable")  # stable: keep file order on ties
    is_duplicate = raw["timestamp"].duplicated(keep="first")
    raw = raw[~is_duplicate].reset_index(drop=True)
    return raw, int(is_duplicate.sum())


def _find_bad_rows(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return one True/False mask per kind of problem, aligned with `df`."""
    no_problem = pd.Series(False, index=df.index)

    # Bid above ask in any of open/high/low/close.
    crossed = no_problem.copy()
    for bar_part in _OHLC:
        bid, ask = f"{bar_part}_bid", f"{bar_part}_ask"
        if bid in df and ask in df:
            crossed |= df[bid] > df[ask]

    # The bar's high is below its low.
    high_below_low = df["high_mid"] < df["low_mid"]

    # Any price that is zero or negative.
    price_cols = [c for c in _PRICE_COLS if c in df.columns]
    if price_cols:
        non_positive = (df[price_cols] <= 0).any(axis=1)
    else:
        non_positive = no_problem.copy()

    # Any kept column is missing (includes unparseable settlement dates).
    missing_value = df[KEEP_COLS].isna().any(axis=1)

    return {
        "crossed": crossed,
        "high_lt_low": high_below_low,
        "nonpositive": non_positive,
        "nan": missing_value,
    }


def _max_abs_mismatch(df: pd.DataFrame, actual: str, expected: Callable[[str], pd.Series],
                      needs: Sequence[str]) -> float:
    """Largest |actual - expected| over open/high/low/close, as a sanity check.

    `actual` and `needs` are column suffixes (e.g. "mid"); for each bar part
    such as "open" the check is skipped unless all the needed columns exist.
    Returns 0.0 if nothing could be checked.
    """
    mismatches = []
    for bar_part in _OHLC:
        if all(f"{bar_part}_{suffix}" in df for suffix in needs):
            mismatches.append(df[f"{bar_part}_{actual}"] - expected(bar_part))

    if not mismatches:
        return 0.0
    worst = pd.concat(mismatches, axis=1).abs().max().max()
    return 0.0 if pd.isna(worst) else float(worst)


# Spot has no forward points, so its CSVs omit the _fp and _all_in columns.
_FP_COLS = ["close_mid_fp", "close_mid_all_in"]


def _fill_spot_columns(raw: pd.DataFrame) -> None:
    """Add the forward-point columns to a spot file, in place.

    Spot settles at the spot date, so forward points are 0 and the all-in
    price is just the mid. Only applied when *both* columns are absent, so a
    forward file missing just one of them is still reported as broken.
    """
    if all(c not in raw.columns for c in _FP_COLS) and "close_mid" in raw.columns:
        raw["close_mid_fp"] = 0.0
        raw["close_mid_all_in"] = raw["close_mid"]


def read_day(src, meta: Optional[FileMeta] = None) -> Tuple[pd.DataFrame, dict]:
    """Read and clean one daily CSV.

    `src` can be a path, raw bytes or a file-like object. `meta` is accepted
    for convenience but not needed.

    Returns (clean_df, stats). `clean_df` has only KEEP_COLS and no bad rows.
    `stats` counts what was found and removed (see the keys at the bottom).
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    raw = pd.read_csv(src)
    _fill_spot_columns(raw)
    # Files from other feeds name themselves in a "feed" column (see fetch_dukascopy.py).
    feed = str(raw["feed"].iloc[0]) if "feed" in raw.columns and len(raw) else DEFAULT_FEED

    missing_cols = [c for c in KEEP_COLS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    n_raw_rows = len(raw)
    df, n_duplicates = _parse_and_dedupe(raw)

    problems = _find_bad_rows(df)
    is_bad = problems["crossed"] | problems["high_lt_low"] | problems["nonpositive"] | problems["nan"]

    # mid should equal (bid + ask) / 2
    max_mid_err = _max_abs_mismatch(
        df, actual="mid", needs=("bid", "ask", "mid"),
        expected=lambda p: (df[f"{p}_bid"] + df[f"{p}_ask"]) / 2,
    )
    # all-in mid should equal mid + forward points
    max_allin_err = _max_abs_mismatch(
        df, actual="mid_all_in", needs=("mid_all_in", "mid", "mid_fp"),
        expected=lambda p: df[f"{p}_mid"] + df[f"{p}_mid_fp"],
    )

    stats = {
        "n_raw": n_raw_rows,
        "n_dup_ts": n_duplicates,
        "n_crossed": int(problems["crossed"].sum()),
        "n_high_lt_low": int(problems["high_lt_low"].sum()),
        "n_nonpositive": int(problems["nonpositive"].sum()),
        "n_nan": int(problems["nan"].sum()),
        "n_dropped": int(is_bad.sum()),
        "n_kept": int((~is_bad).sum()),
        "max_mid_err": max_mid_err,
        "max_allin_err": max_allin_err,
        "feed": feed,
    }

    clean = df.loc[~is_bad, KEEP_COLS].reset_index(drop=True)
    return clean, stats


# ==========================================================================
# 4. Derived features
# ==========================================================================

BASIS_POINTS = 1e4  # 1 basis point (bp) = 0.01%

# Trailing window lengths (in bars) for the volatility columns.
VOL_WINDOWS = (15, 60, 240)


# Where an instrument's quotes come from. Challenge1 files carry no label.
DEFAULT_FEED = "challenge1"

# Other feeds quote wider spreads than Challenge1 for the same pair, so their
# spread is multiplied by a factor to put it on Challenge1's scale. The factor
# is measured on the pairs both feeds have by `python fetch_dukascopy.py
# --validate`; see "Combining the two feeds" in DATA_OVERVIEW.md.
SPREAD_SCALE: Dict[str, float] = {
    "dukascopy": 0.477,  # per pair: EURUSD 0.47, USDJPY 0.61, USDSGD 0.38
}


def calibrate_spread(raw_spread_bp: pd.Series, feed: str) -> pd.Series:
    """A feed's quoted spread (bp) on Challenge1's scale. Challenge1 itself is unchanged."""
    return raw_spread_bp * SPREAD_SCALE.get(feed, 1.0)


def add_features(df: pd.DataFrame, feed: str = DEFAULT_FEED) -> pd.DataFrame:
    """Add derived columns to one instrument's bars (sorted, no duplicate times).

    Prices are turned into basis points so instruments are comparable. `feed`
    says where the quotes came from; it only affects `spread_bp`.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]
    close = df["close_mid"]
    open_ = df["open_mid"]
    high = df["high_mid"]
    low = df["low_mid"]

    # --- Time between bars --------------------------------------------------
    minutes_since_prev = ts.diff().dt.total_seconds() / 60.0
    df["dt_prev_min"] = minutes_since_prev.fillna(1.0)  # first bar: assume normal 1 min
    df["is_gap"] = df["dt_prev_min"] > 1.0               # one or more bars missing
    # Gap length squashed so long gaps don't dominate: about equal to the gap
    # in seconds for short gaps, growing only logarithmically for long ones.
    seconds_since_prev = df["dt_prev_min"] * 60.0
    df["log_dt_prev"] = 8.0 * np.log1p(seconds_since_prev / 8.0)

    # --- Returns (bp) ---------------------------------------------------------
    df["ret_bp"] = (np.log(close).diff() * BASIS_POINTS).fillna(0.0)  # close-to-close
    df["ret_oc_bp"] = np.log(close / open_) * BASIS_POINTS             # within the bar
    df["range_bp"] = np.log(high / low) * BASIS_POINTS                 # bar high-low range
    df["ret_across_gap"] = df["is_gap"]  # ret_bp spans missing bars

    # --- Spread and forward points (bp of the close) --------------------------
    df["spread_bp_raw"] = (df["close_ask"] - df["close_bid"]) / close * BASIS_POINTS  # as quoted
    df["spread_bp"] = calibrate_spread(df["spread_bp_raw"], feed)  # on Challenge1's scale
    df["fp_bp"] = df["close_mid_fp"] / close * BASIS_POINTS
    df["dfp_bp"] = df["fp_bp"].diff().fillna(0.0)  # change in forward points

    # --- Volatility (bp per bar, trailing windows only: no look-ahead) --------
    # Returns that span missing bars are excluded so gaps don't inflate vol.
    clean_ret = df["ret_bp"].where(~df["is_gap"])
    # Parkinson estimator: per-bar variance from the high-low range.
    parkinson_var = df["range_bp"] ** 2 / (4.0 * np.log(2.0))
    for window in VOL_WINDOWS:
        min_periods = max(2, window // 2)
        df[f"vol_{window}"] = clean_ret.rolling(window, min_periods=min_periods).std()
        df[f"pk_vol_{window}"] = np.sqrt(
            parkinson_var.rolling(window, min_periods=min_periods).mean()
        )

    # --- Settlement date ------------------------------------------------------
    settle = df["settlement_date"]
    previous_settle = settle.shift()
    # A "roll" is a bar whose settlement date differs from the bar before it.
    is_roll = settle.ne(previous_settle) & previous_settle.notna()
    df["is_settle_roll"] = is_roll

    bar_day = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df["days_to_settle"] = (settle - bar_day).dt.days.astype("int32")

    # Minutes since the latest roll. Mark the time of each roll bar, treat the
    # very first bar as a roll too (so the clock has a starting point), then
    # carry each roll time forward to the bars after it.
    last_roll_time = ts.where(is_roll)
    if pd.isna(last_roll_time.iloc[0]):
        last_roll_time.iloc[0] = ts.iloc[0]
    last_roll_time = last_roll_time.ffill()
    df["min_since_roll"] = (ts - last_roll_time).dt.total_seconds() / 60.0

    # --- Calendar (UTC) -------------------------------------------------------
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int16")
    df["dow"] = ts.dt.day_of_week.astype("int8")  # Monday = 0
    df["dom"] = ts.dt.day.astype("int8")          # day of month
    df["date"] = (ts.dt.year * 10000 + ts.dt.month * 100 + ts.dt.day).astype("int32")  # YYYYMMDD
    return df


# ==========================================================================
# 5. Loading everything
# ==========================================================================

def _load_one_file(src: Source) -> Tuple[Optional[pd.DataFrame], dict]:
    """Read one source. Returns (day frame or None, manifest row). Never raises."""
    meta = src.meta
    row = {
        "instrument_id": meta.instrument_id,
        "date": meta.date,
        "source": src.source,
        "status": "ok",
        "error": "",
    }
    try:
        day_df, stats = read_day(src.read_bytes(), meta)
    except Exception as e:  # noqa: BLE001 - deliberate: one bad file must not stop the run
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        return None, row

    row.update(stats)
    if len(day_df) == 0:
        row.update(status="error", error="no usable rows")
        return None, row
    return day_df, row


def _print_summary(series: Dict[str, pd.DataFrame], manifest: pd.DataFrame, n_skipped: int) -> None:
    """Print a one-line total, one line per instrument, then any errors."""
    n_files = len(manifest)
    is_error = manifest["status"] == "error" if n_files else pd.Series(dtype=bool)
    n_errors = int(is_error.sum()) if n_files else 0

    print(f"{n_files} files loaded ({n_errors} errors), "
          f"{n_skipped} duplicates skipped, {len(series)} instruments")

    for instrument_id, df in series.items():
        first_day = df["timestamp"].iloc[0]
        last_day = df["timestamp"].iloc[-1]
        print(f"  {instrument_id:<36} days={df['date'].nunique():>3}  bars={len(df):>8,}  "
              f"{first_day:%Y-%m-%d} -> {last_day:%Y-%m-%d}  {df.attrs.get('feed', DEFAULT_FEED)}")

    n_clash = int((manifest["status"] == "skipped").sum()) if n_files else 0
    if n_clash:
        print(f"  {n_clash} files skipped: the same instrument came from two feeds "
              f"({DEFAULT_FEED} is kept)")

    if n_errors:
        for _, failed in manifest[is_error].iterrows():
            print(f"  ERROR {failed['source']}: {failed['error']}", file=sys.stderr)


def load_all(paths, verbose: bool = True, product: str = "all") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load every instrument found under `paths`.

    `product` is "all", "forward" or "spot" (see discover()).

    Returns (series, manifest):
      series   {instrument_id: DataFrame of all its bars, with features}
      manifest one row per file read, with its quality stats. Files that
               failed have status="error" and a message; they never stop the run.
    """
    skipped: List[Tuple[FileMeta, str]] = []
    manifest_rows: List[dict] = []
    days_by_instrument: Dict[str, List[pd.DataFrame]] = {}
    meta_by_instrument: Dict[str, FileMeta] = {}
    feed_by_instrument: Dict[str, str] = {}

    # discover() cannot tell feeds apart (that needs the file's contents), so
    # repeated days are sorted out here instead.
    seen = set()  # (instrument, day) already loaded
    for src in discover(paths, product=product, dedupe=False):
        instrument_id = src.meta.instrument_id
        key = (instrument_id, src.meta.date)
        current = feed_by_instrument.get(instrument_id)
        if key in seen and current == DEFAULT_FEED:
            skipped.append((src.meta, src.source))  # Challenge1 already has this day: no need to read it
            continue

        day_df, row = _load_one_file(src)
        if day_df is None:
            manifest_rows.append(row)
            continue
        feed = row["feed"]
        if current is None:
            current = feed_by_instrument[instrument_id] = feed

        if feed != current:
            # Two feeds have the same instrument. One series must not mix
            # quotes from both, so Challenge1 wins; otherwise first come, first kept.
            if feed != DEFAULT_FEED:
                row.update(status="skipped", error=f"instrument already loaded from {current}")
                manifest_rows.append(row)
                continue
            for earlier in manifest_rows:
                if earlier["instrument_id"] == instrument_id and earlier["status"] == "ok":
                    earlier.update(status="skipped", error=f"instrument also in {DEFAULT_FEED}")
            days_by_instrument[instrument_id] = []
            feed_by_instrument[instrument_id] = feed
            seen = {k for k in seen if k[0] != instrument_id}
        elif key in seen:
            skipped.append((src.meta, src.source))  # the same day twice from one feed
            continue

        seen.add(key)
        manifest_rows.append(row)
        days_by_instrument.setdefault(instrument_id, []).append(day_df)
        meta_by_instrument[instrument_id] = src.meta

    series: Dict[str, pd.DataFrame] = {}
    for instrument_id in sorted(days_by_instrument):
        df = pd.concat(days_by_instrument[instrument_id], ignore_index=True)
        df = df.sort_values("timestamp")
        df = df[~df["timestamp"].duplicated(keep="first")]  # days may overlap at midnight
        df = add_features(df, feed=feed_by_instrument[instrument_id])
        df.attrs.update(meta_by_instrument[instrument_id].attrs())
        df.attrs["feed"] = feed_by_instrument[instrument_id]
        series[instrument_id] = df

    manifest = pd.DataFrame(manifest_rows)
    if verbose:
        _print_summary(series, manifest, len(skipped))
    return series, manifest


# ==========================================================================
# 6. Train / validation / test splits
# ==========================================================================

def time_split(df: pd.DataFrame, n_val_days: int = 6, n_test_days: int = 7,
               embargo_min: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one instrument by UTC day into (train, val, test), oldest first.

        [ train ...... ][gap][ val ... ][gap][ test ... ]
                         ^ embargo_min    ^ embargo_min

    The last `n_test_days` days are test and the `n_val_days` before them are
    val; everything earlier is train. The last `embargo_min` minutes of train
    and of val are dropped, so a target that looks up to `embargo_min` minutes
    ahead never peeks into the next split.
    """
    bar_day = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
    all_days = bar_day.drop_duplicates().sort_values().tolist()
    if len(all_days) <= n_val_days + n_test_days:
        raise ValueError(f"need more than {n_val_days + n_test_days} days, have {len(all_days)}")

    val_start = all_days[-(n_val_days + n_test_days)]
    test_start = all_days[-n_test_days]
    embargo = pd.Timedelta(minutes=embargo_min)
    ts = df["timestamp"]

    train = df[ts < val_start - embargo]
    val = df[(ts >= val_start) & (ts < test_start - embargo)]
    test = df[ts >= test_start]

    result = []
    for part in (train, val, test):
        part = part.reset_index(drop=True)
        part.attrs.update(df.attrs)  # keep instrument info on each split
        result.append(part)
    return tuple(result)  # type: ignore[return-value]


def _pair_of(instrument_id: str) -> str:
    """"EURUSD_EUR_FWD_1M_1000000" -> "EURUSD"."""
    return instrument_id.split("_")[0]


def split_instruments(series, holdout_pairs: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split instrument ids into (pretrain, holdout) by currency pair.

    Holding out a pair holds out all of its tenors and notionals. `series`
    can be the dict from load_all() or any iterable of instrument ids.
    """
    holdout = {pair.upper() for pair in holdout_pairs}
    instrument_ids = sorted(series.keys() if isinstance(series, dict) else series)

    pretrain = [i for i in instrument_ids if _pair_of(i) not in holdout]
    held_out = [i for i in instrument_ids if _pair_of(i) in holdout]
    return pretrain, held_out


# ==========================================================================
# 7. Saving and loading processed data
# ==========================================================================

# Instrument fields stored in instruments.csv and restored onto df.attrs.
_ATTR_KEYS = ["pair", "base", "quote", "dealt", "product", "tenor", "notional", "instrument_id"]


def save_processed(series: Dict[str, pd.DataFrame], manifest: pd.DataFrame, out_dir: str) -> None:
    """Write the output of load_all() to `out_dir`.

    Creates:
      <instrument_id>.parquet  one file of bars per instrument
      instruments.csv          one row per instrument (metadata, size, date range)
      manifest.csv             the per-file manifest
    """
    os.makedirs(out_dir, exist_ok=True)

    instrument_rows = []
    for instrument_id, df in series.items():
        filename = f"{instrument_id}.parquet"
        df.to_parquet(os.path.join(out_dir, filename), index=False)
        instrument_rows.append({
            **df.attrs,
            "instrument_id": instrument_id,
            "file": filename,
            "n_days": int(df["date"].nunique()),
            "n_bars": len(df),
            "start": df["timestamp"].iloc[0].isoformat(),
            "end": df["timestamp"].iloc[-1].isoformat(),
        })

    pd.DataFrame(instrument_rows).to_csv(os.path.join(out_dir, "instruments.csv"), index=False)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)


def load_processed(out_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Read back what save_processed() wrote. Returns (series, manifest)."""
    instruments = pd.read_csv(os.path.join(out_dir, "instruments.csv"),
                              dtype=str, keep_default_na=False)

    series: Dict[str, pd.DataFrame] = {}
    for _, info in instruments.iterrows():
        df = pd.read_parquet(os.path.join(out_dir, info["file"]))
        df.attrs.update({key: info[key] for key in _ATTR_KEYS})
        df.attrs["feed"] = info.get("feed", DEFAULT_FEED)  # folders saved before feeds existed
        df.attrs["notional"] = int(info["notional"])  # read as text; restore the number
        series[info["instrument_id"]] = df

    manifest = pd.read_csv(os.path.join(out_dir, "manifest.csv"))
    if "error" in manifest:
        manifest["error"] = manifest["error"].fillna("")  # empty messages come back as NaN
    return series, manifest


# ==========================================================================
# Command line
# ==========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Load raw inputs and write processed parquet. Returns the exit code."""
    parser = argparse.ArgumentParser(description="Load raw FX-forward bars and write processed parquet.")
    parser.add_argument("paths", nargs="+", help="input dirs/CSVs/zips ... followed by the output dir")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-p", "--product", choices=PRODUCTS, default="all",
                        help="which instruments to load (default: all)")
    args = parser.parse_args(argv)

    if len(args.paths) < 2:
        parser.error("need at least one input and an output directory")
    *inputs, out_dir = args.paths

    series, manifest = load_all(inputs, verbose=not args.quiet, product=args.product)
    if not series:
        print("no data found", file=sys.stderr)
        return 1

    save_processed(series, manifest, out_dir)
    print(f"saved {len(series)} instruments to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
