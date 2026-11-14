"""Loader for 1-minute FX-forward bars (one CSV per instrument per day).

Pipeline: discover files (dirs / CSVs / nested zips) -> read_day (clean +
quality stats) -> concat per instrument -> add_features -> splits / parquet.

CLI:  python -m fxfm.data.loader <inputs...> <out_dir>
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

# --------------------------------------------------------------------------
# Filenames
# --------------------------------------------------------------------------

# PAIR(6)_DEALT(3)_PRODUCT_TENOR_NOTIONAL_YYYYMMDD.csv, optionally with a junk
# prefix such as "85de32c1-" (lookbehind stops it eating part of a longer token).
_FILENAME_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<pair>[A-Z]{6})_(?P<dealt>[A-Z]{3})_(?P<product>[A-Z][A-Z0-9]*)_"
    r"(?P<tenor>\d*[A-Z]+)_(?P<notional>\d+)_(?P<date>\d{8})\.csv$"
)


@dataclass(frozen=True)
class FileMeta:
    pair: str
    base: str
    quote: str
    dealt: str
    product: str
    tenor: str
    notional: int
    date: _date
    instrument_id: str  # everything except the date

    def attrs(self) -> dict:
        d = asdict(self)
        d.pop("date")
        return d


def parse_filename(name: str) -> Optional[FileMeta]:
    """Parse a data filename (path and junk prefix tolerated); None if no match."""
    base_name = re.split(r"[\\/]", str(name))[-1]
    if base_name.startswith("._"):
        return None
    m = _FILENAME_RE.search(base_name)
    if not m:
        return None
    try:
        d = pd.Timestamp(m["date"]).date()
    except ValueError:
        return None
    pair, product, tenor = m["pair"], m["product"], m["tenor"]
    notional = int(m["notional"])
    return FileMeta(
        pair=pair, base=pair[:3], quote=pair[3:], dealt=m["dealt"], product=product,
        tenor=tenor, notional=notional, date=d,
        instrument_id=f"{pair}_{m['dealt']}_{product}_{tenor}_{notional}",
    )


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

@dataclass
class Source:
    meta: FileMeta
    source: str                      # human-readable location
    read_bytes: Callable[[], bytes]  # lazily fetches the CSV content


def _is_junk(path: str) -> bool:
    parts = re.split(r"[\\/]", path)
    return any(p == "__MACOSX" for p in parts) or parts[-1].startswith("._")


def _walk_zip(zf: zipfile.ZipFile, label: str) -> Iterator[Tuple[str, Callable[[], bytes]]]:
    for info in zf.infolist():
        if info.is_dir() or _is_junk(info.filename):
            continue
        lower = info.filename.lower()
        loc = f"{label}!{info.filename}"
        if lower.endswith(".zip"):
            try:
                inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            except zipfile.BadZipFile:
                continue
            yield from _walk_zip(inner, loc)
        elif lower.endswith(".csv"):
            yield loc, (lambda zf=zf, info=info: zf.read(info))


def _walk_path(path: str) -> Iterator[Tuple[str, Callable[[], bytes]]]:
    def disk(p):
        def _r():
            with open(p, "rb") as f:
                return f.read()
        return _r

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
            for fn in sorted(files):
                if fn.startswith("._"):
                    continue
                yield from _walk_path(os.path.join(root, fn))
    elif os.path.isfile(path):
        low = path.lower()
        if low.endswith(".zip"):
            # left open on purpose: entries are read lazily by the openers
            yield from _walk_zip(zipfile.ZipFile(path), path)
        elif low.endswith(".csv"):
            yield path, disk(path)
    else:
        raise FileNotFoundError(path)


def discover(paths: Iterable[str], skipped: Optional[List[Tuple[FileMeta, str]]] = None) -> Iterator[Source]:
    """Yield every recognised data file under `paths`, once per (instrument_id, date).

    First occurrence wins; later copies (e.g. extracted next to its zip) are
    appended to `skipped` if a list is given.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    seen = set()
    for p in paths:
        for loc, reader in _walk_path(os.fspath(p)):
            meta = parse_filename(loc.split("!")[-1])
            if meta is None:
                continue
            key = (meta.instrument_id, meta.date)
            if key in seen:
                if skipped is not None:
                    skipped.append((meta, loc))
                continue
            seen.add(key)
            yield Source(meta, loc, reader)


# --------------------------------------------------------------------------
# Reading one day
# --------------------------------------------------------------------------

KEEP_COLS = [
    "timestamp", "open_mid", "high_mid", "low_mid", "close_mid",
    "close_bid", "close_ask", "close_mid_fp", "close_mid_all_in", "settlement_date",
]
_OHLC = ["open", "high", "low", "close"]
_PRICE_COLS = [f"{o}_{s}" for o in _OHLC for s in ("bid", "ask", "mid")]


def read_day(src, meta: Optional[FileMeta] = None) -> Tuple[pd.DataFrame, dict]:
    """Read one daily CSV (path, bytes or file-like). Returns (clean_df, quality_stats)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    raw = pd.read_csv(src)
    missing = [c for c in KEEP_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["settlement_date"] = pd.to_datetime(raw["settlement_date"], errors="coerce")
    n_raw = len(raw)
    raw = raw.sort_values("timestamp", kind="stable")
    dup = raw["timestamp"].duplicated(keep="first")
    n_dup = int(dup.sum())
    raw = raw[~dup].reset_index(drop=True)

    present = [c for c in _PRICE_COLS if c in raw.columns]
    # --- quality checks on the raw (de-duplicated) rows -------------------
    crossed = pd.Series(False, index=raw.index)
    for o in _OHLC:
        if f"{o}_bid" in raw and f"{o}_ask" in raw:
            crossed |= raw[f"{o}_bid"] > raw[f"{o}_ask"]
    hl = raw["high_mid"] < raw["low_mid"]
    nonpos = (raw[present] <= 0).any(axis=1) if present else pd.Series(False, index=raw.index)
    nan_row = raw[[c for c in KEEP_COLS if c != "settlement_date"]].isna().any(axis=1) | raw["settlement_date"].isna()

    def _max_abs(parts):
        errs = [e for e in parts if e is not None]
        if not errs:
            return 0.0
        v = pd.concat(errs, axis=1).abs().max().max()
        return 0.0 if pd.isna(v) else float(v)

    mid_errs = [raw[f"{o}_mid"] - (raw[f"{o}_bid"] + raw[f"{o}_ask"]) / 2
                if all(f"{o}_{s}" in raw for s in ("bid", "ask", "mid")) else None for o in _OHLC]
    allin_errs = [raw[f"{o}_mid_all_in"] - (raw[f"{o}_mid"] + raw[f"{o}_mid_fp"])
                  if all(f"{o}_{s}" in raw for s in ("mid_all_in", "mid", "mid_fp")) else None for o in _OHLC]

    bad = crossed | hl | nonpos | nan_row
    stats = {
        "n_raw": n_raw, "n_dup_ts": n_dup,
        "n_crossed": int(crossed.sum()), "n_high_lt_low": int(hl.sum()),
        "n_nonpositive": int(nonpos.sum()), "n_nan": int(nan_row.sum()),
        "n_dropped": int(bad.sum()), "n_kept": int((~bad).sum()),
        "max_mid_err": _max_abs(mid_errs), "max_allin_err": _max_abs(allin_errs),
    }
    out = raw.loc[~bad, KEEP_COLS].reset_index(drop=True)
    return out, stats


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a time-sorted, de-duplicated per-instrument frame."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]
    dt_min = ts.diff().dt.total_seconds() / 60.0
    df["dt_prev_min"] = dt_min.fillna(1.0)
    df["is_gap"] = df["dt_prev_min"] > 1.0
    df["log_dt_prev"] = 8.0 * np.log1p(df["dt_prev_min"] * 60.0 / 8.0)

    c, o, h, l = df["close_mid"], df["open_mid"], df["high_mid"], df["low_mid"]
    df["ret_bp"] = (np.log(c).diff() * 1e4).fillna(0.0)
    df["ret_oc_bp"] = np.log(c / o) * 1e4
    df["range_bp"] = np.log(h / l) * 1e4
    df["ret_across_gap"] = df["is_gap"]

    df["spread_bp"] = (df["close_ask"] - df["close_bid"]) / c * 1e4
    df["fp_bp"] = df["close_mid_fp"] / c * 1e4
    df["dfp_bp"] = df["fp_bp"].diff().fillna(0.0)

    sd = df["settlement_date"]
    roll = sd.ne(sd.shift()) & sd.shift().notna()
    df["is_settle_roll"] = roll
    day = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df["days_to_settle"] = (sd - day).dt.days.astype("int32")
    # minutes since the most recent roll bar; the first bar anchors the clock
    anchor = ts.where(roll)
    anchor.iloc[0] = ts.iloc[0] if pd.isna(anchor.iloc[0]) else anchor.iloc[0]
    df["min_since_roll"] = (ts - anchor.ffill()).dt.total_seconds() / 60.0

    df["hour"] = ts.dt.hour.astype("int8")
    df["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int16")
    df["dow"] = ts.dt.day_of_week.astype("int8")
    df["dom"] = ts.dt.day.astype("int8")
    df["date"] = (ts.dt.year * 10000 + ts.dt.month * 100 + ts.dt.day).astype("int32")
    return df


# --------------------------------------------------------------------------
# Loading everything
# --------------------------------------------------------------------------

def load_all(paths, verbose: bool = True) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all instruments found under `paths`.

    Returns ({instrument_id: DataFrame}, manifest). One bad file never aborts the
    run: it is recorded in the manifest with status="error".
    """
    skipped: List[Tuple[FileMeta, str]] = []
    rows: List[dict] = []
    frames: Dict[str, List[pd.DataFrame]] = {}
    metas: Dict[str, FileMeta] = {}

    for src in discover(paths, skipped):
        m = src.meta
        row = {"instrument_id": m.instrument_id, "date": m.date, "source": src.source,
               "status": "ok", "error": ""}
        try:
            day_df, stats = read_day(src.read_bytes(), m)
            row.update(stats)
            if len(day_df):
                frames.setdefault(m.instrument_id, []).append(day_df)
                metas[m.instrument_id] = m
            else:
                row.update(status="error", error="no usable rows")
        except Exception as e:  # noqa: BLE001 - deliberate: keep going
            row.update(status="error", error=f"{type(e).__name__}: {e}")
        rows.append(row)

    series: Dict[str, pd.DataFrame] = {}
    for iid in sorted(frames):
        df = pd.concat(frames[iid], ignore_index=True)
        df = df.sort_values("timestamp")
        df = df[~df["timestamp"].duplicated(keep="first")]
        df = add_features(df)
        df.attrs.update(metas[iid].attrs())
        series[iid] = df

    manifest = pd.DataFrame(rows)
    if verbose:
        print(f"{len(rows)} files loaded ({int((manifest['status'] == 'error').sum()) if rows else 0} errors), "
              f"{len(skipped)} duplicates skipped, {len(series)} instruments")
        for iid, df in series.items():
            print(f"  {iid:<36} days={df['date'].nunique():>3}  bars={len(df):>8,}  "
                  f"{df['timestamp'].iloc[0]:%Y-%m-%d} -> {df['timestamp'].iloc[-1]:%Y-%m-%d}")
        if rows and (manifest["status"] == "error").any():
            for _, r in manifest[manifest["status"] == "error"].iterrows():
                print(f"  ERROR {r['source']}: {r['error']}", file=sys.stderr)
    return series, manifest


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def time_split(df: pd.DataFrame, n_val_days: int = 6, n_test_days: int = 7,
               embargo_min: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train | val | test split by UTC day.

    The last `n_test_days` days are test, the `n_val_days` before are val, the rest
    train. The final `embargo_min` minutes of train and of val are dropped, so a
    target looking up to `embargo_min` ahead never reaches into the next split.
    """
    day = df["timestamp"].dt.tz_convert("UTC").dt.normalize()
    days = day.drop_duplicates().sort_values().tolist()
    if len(days) <= n_val_days + n_test_days:
        raise ValueError(f"need more than {n_val_days + n_test_days} days, have {len(days)}")
    val_start = days[-(n_val_days + n_test_days)]
    test_start = days[-n_test_days]
    emb = pd.Timedelta(minutes=embargo_min)
    ts = df["timestamp"]
    parts = (
        df[ts < val_start - emb],
        df[(ts >= val_start) & (ts < test_start - emb)],
        df[ts >= test_start],
    )
    out = []
    for p in parts:
        p = p.reset_index(drop=True)
        p.attrs.update(df.attrs)
        out.append(p)
    return tuple(out)  # type: ignore[return-value]


def split_instruments(series, holdout_pairs: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Split instrument ids into (pretrain, holdout); a held-out pair takes all its
    tenors/notionals with it. `series` may be the load_all dict or any iterable of ids."""
    holdout = {p.upper() for p in holdout_pairs}
    ids = sorted(series.keys() if isinstance(series, dict) else series)
    held = [i for i in ids if i.split("_")[0] in holdout]
    pre = [i for i in ids if i.split("_")[0] not in holdout]
    return pre, held


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_processed(series: Dict[str, pd.DataFrame], manifest: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    inst_rows = []
    for iid, df in series.items():
        fn = f"{iid}.parquet"
        df.to_parquet(os.path.join(out_dir, fn), index=False)
        inst_rows.append({**df.attrs, "instrument_id": iid, "file": fn,
                          "n_days": int(df["date"].nunique()), "n_bars": len(df),
                          "start": df["timestamp"].iloc[0].isoformat(),
                          "end": df["timestamp"].iloc[-1].isoformat()})
    pd.DataFrame(inst_rows).to_csv(os.path.join(out_dir, "instruments.csv"), index=False)
    manifest.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)


_ATTR_KEYS = ["pair", "base", "quote", "dealt", "product", "tenor", "notional", "instrument_id"]


def load_processed(out_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    inst = pd.read_csv(os.path.join(out_dir, "instruments.csv"), dtype=str, keep_default_na=False)
    series: Dict[str, pd.DataFrame] = {}
    for _, r in inst.iterrows():
        df = pd.read_parquet(os.path.join(out_dir, r["file"]))
        df.attrs.update({k: r[k] for k in _ATTR_KEYS})
        df.attrs["notional"] = int(r["notional"])
        series[r["instrument_id"]] = df
    manifest = pd.read_csv(os.path.join(out_dir, "manifest.csv"))
    if "error" in manifest:
        manifest["error"] = manifest["error"].fillna("")
    return series, manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Load raw FX-forward bars and write processed parquet.")
    ap.add_argument("paths", nargs="+", help="input dirs/CSVs/zips ... followed by the output dir")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    if len(args.paths) < 2:
        ap.error("need at least one input and an output directory")
    *inputs, out_dir = args.paths
    series, manifest = load_all(inputs, verbose=not args.quiet)
    if not series:
        print("no data found", file=sys.stderr)
        return 1
    save_processed(series, manifest, out_dir)
    print(f"saved {len(series)} instruments to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
