import io
import math
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

import loader as L

PAIRS = {"EURUSD": 1.14, "USDJPY": 150.0, "GBPUSD": 1.30}
N_DAYS = 20
START = date(2026, 7, 1)


def make_day(day: pd.Timestamp, base: float, seed: int, fp: float = 0.0004) -> pd.DataFrame:
    """Synthetic day: 00:00-21:00 and 21:05(excl.)... i.e. gap 21:01-21:05, roll at 21:06."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(day, day + pd.Timedelta(hours=23, minutes=59), freq="1min", tz="UTC")
    mins = idx.hour * 60 + idx.minute
    # bars at 21:01..21:05 are missing (5-minute gap: 21:00 -> 21:06)
    idx = idx[~((mins >= 21 * 60 + 1) & (mins <= 21 * 60 + 5))]
    n = len(idx)
    closes = base * np.exp(np.cumsum(rng.normal(0, 1e-5, n)))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) * 1.00001
    lows = np.minimum(opens, closes) * 0.99999
    half = base * 1e-5
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S+00:00")})
    mins = idx.hour * 60 + idx.minute
    settle = np.where(mins >= 21 * 60 + 6, day + pd.Timedelta(days=31), day + pd.Timedelta(days=30))
    df["settlement_date"] = pd.to_datetime(settle).strftime("%Y-%m-%d")
    for name, mid in zip(["open", "high", "low", "close"], [opens, highs, lows, closes]):
        df[f"{name}_bid"], df[f"{name}_ask"], df[f"{name}_mid"] = mid - half, mid + half, mid
        df[f"{name}_bid_fp"], df[f"{name}_ask_fp"], df[f"{name}_mid_fp"] = fp, fp, fp
        df[f"{name}_mid_all_in"] = mid + fp
    return df


def write_dataset(root, pairs=PAIRS, n_days=N_DAYS):
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, (pair, base) in enumerate(pairs.items()):
        for d in range(n_days):
            day = pd.Timestamp(START) + pd.Timedelta(days=d)
            df = make_day(day, base, seed=k * 100 + d)
            name = f"{pair}_{pair[:3]}_FORWARD_1M_1000000_{day:%Y%m%d}.csv"
            p = root / name
            df.to_csv(p, index=False)
            paths[(pair, d)] = p
    return paths


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "raw"
    paths = write_dataset(root)
    return root, paths


# ---------------------------------------------------------------- filenames
def test_parse_filename_basic():
    m = L.parse_filename("EURUSD_EUR_FORWARD_1M_1000000_20260701.csv")
    assert (m.pair, m.base, m.quote, m.dealt) == ("EURUSD", "EUR", "USD", "EUR")
    assert (m.product, m.tenor, m.notional, m.date) == ("FORWARD", "1M", 1000000, date(2026, 7, 1))
    assert m.instrument_id == "EURUSD_EUR_FORWARD_1M_1000000"


def test_parse_filename_prefix_path_and_tenors():
    m = L.parse_filename("some/dir/85de32c1-USDJPY_USD_FORWARD_ON_5000000_20260702.csv")
    assert m.pair == "USDJPY" and m.tenor == "ON" and m.date == date(2026, 7, 2)
    assert L.parse_filename("EURUSD_EUR_FORWARD_1W_1000000_20260701.csv").tenor == "1W"


@pytest.mark.parametrize("bad", [
    "README.md", "EURUSD_EUR_FORWARD_1M_1000000_20260701.txt",
    "._EURUSD_EUR_FORWARD_1M_1000000_20260701.csv",
    "EURUSD_EUR_FORWARD_1M_1000000_20261345.csv", "foo.csv",
])
def test_parse_filename_rejects(bad):
    assert L.parse_filename(bad) is None


# ---------------------------------------------------------------- discovery
def test_discover_dedup_zip_nested_and_macos(dataset, tmp_path):
    root, paths = dataset
    # zip with a copy of one file, a nested zip with another copy, and junk
    p0 = paths[("EURUSD", 0)]
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.write(paths[("EURUSD", 1)], "x/85de32c1-" + paths[("EURUSD", 1)].name)
        z.writestr("._" + paths[("EURUSD", 2)].name, b"junk")
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(p0, p0.name)
        z.writestr("inner.zip", inner.getvalue())
        z.writestr("__MACOSX/" + p0.name, b"junk")
    skipped = []
    found = list(L.discover([str(zpath), str(root)], skipped))
    keys = [(s.meta.instrument_id, s.meta.date) for s in found]
    assert len(keys) == len(set(keys)) == len(PAIRS) * N_DAYS
    assert len(skipped) == 2  # loose copies of d0 and d1 dropped
    # first-seen wins: zipped copies are used
    assert sum("bundle.zip" in s.source for s in found) == 2
    assert not any(s.source.endswith(".csv") and "._" in s.source for s in found)


# ---------------------------------------------------------------- read_day
def test_read_day_quality_and_drops(tmp_path):
    day = pd.Timestamp("2026-07-01")
    df = make_day(day, 1.14, seed=0)
    n = len(df)
    df.loc[5, "close_bid"] = df.loc[5, "close_ask"] + 1e-3        # crossed
    df.loc[6, "high_mid"] = df.loc[6, "low_mid"] - 1e-3           # high<low
    df.loc[7, "open_bid"] = -1.0                                  # non-positive
    df = pd.concat([df, df.iloc[[20]]]).sample(frac=1, random_state=0)  # dup ts + shuffled
    p = tmp_path / "EURUSD_EUR_FORWARD_1M_1000000_20260701.csv"
    df.to_csv(p, index=False)
    out, st = L.read_day(str(p))
    assert st["n_raw"] == n + 1 and st["n_dup_ts"] == 1
    assert st["n_crossed"] >= 1 and st["n_high_lt_low"] >= 1 and st["n_nonpositive"] == 1
    assert st["n_kept"] == n - 3
    assert st["n_kept"] == len(out) and out["timestamp"].is_monotonic_increasing
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert list(out.columns) == L.KEEP_COLS
    assert out["timestamp"].is_unique


def test_read_day_error_stats(tmp_path):
    df = make_day(pd.Timestamp("2026-07-01"), 1.14, seed=0)
    df.loc[8, "close_mid"] += 5e-4
    df.loc[9, "close_mid_all_in"] += 8e-4
    out, st = L.read_day(df.to_csv(index=False).encode())
    assert st["max_mid_err"] == pytest.approx(5e-4, rel=1e-3)
    assert st["max_allin_err"] == pytest.approx(8e-4, rel=1e-3)
    assert st["n_dropped"] == 0 and len(out) == len(df)


def test_load_all_survives_bad_file(dataset):
    root, paths = dataset
    paths[("EURUSD", 3)].write_text("timestamp,foo\n1,2\n")
    series, manifest = L.load_all([str(root)], verbose=False)
    assert (manifest["status"] == "error").sum() == 1
    assert "missing columns" in manifest.loc[manifest["status"] == "error", "error"].iloc[0]
    assert series["EURUSD_EUR_FORWARD_1M_1000000"]["date"].nunique() == N_DAYS - 1


# ---------------------------------------------------------------- features
@pytest.fixture
def series(dataset):
    root, _ = dataset
    s, m = L.load_all([str(root)], verbose=False)
    return s, m


def test_gap_and_roll_flags(series):
    s, _ = series
    df = s["EURUSD_EUR_FORWARD_1M_1000000"]
    gaps = df[df["is_gap"]]
    assert len(gaps) == N_DAYS  # every day's 21:06 bar; day joins are clean
    assert set(gaps["timestamp"].dt.strftime("%H:%M")) == {"21:06"}
    assert (gaps["dt_prev_min"] == 6).all()
    rolls = df[df["is_settle_roll"]]
    assert set(rolls["timestamp"].dt.strftime("%H:%M")) == {"21:06"}
    assert len(rolls) == N_DAYS
    assert df["dt_prev_min"].iloc[0] == 1 and not df["is_gap"].iloc[0]
    assert df["ret_across_gap"].equals(df["is_gap"])
    r = df[df["timestamp"] == pd.Timestamp("2026-07-02 21:06", tz="UTC")].iloc[0]
    assert r["min_since_roll"] == 0
    r2 = df[df["timestamp"] == pd.Timestamp("2026-07-03 00:00", tz="UTC")].iloc[0]
    assert r2["min_since_roll"] == pytest.approx(2 * 60 + 54)  # 21:06 -> 00:00
    assert r2["dt_prev_min"] == 1


def test_feature_values_hand_calc(series):
    s, _ = series
    df = s["EURUSD_EUR_FORWARD_1M_1000000"]
    i = df.index[df["timestamp"] == pd.Timestamp("2026-07-02 21:06", tz="UTC")][0]
    r, prev = df.iloc[i], df.iloc[i - 1]
    assert r["log_dt_prev"] == pytest.approx(8 * math.log(1 + 360 / 8))
    assert df["log_dt_prev"].iloc[1] == pytest.approx(8 * math.log(1 + 60 / 8))
    assert r["ret_bp"] == pytest.approx(math.log(r["close_mid"] / prev["close_mid"]) * 1e4)
    assert r["ret_oc_bp"] == pytest.approx(math.log(r["close_mid"] / r["open_mid"]) * 1e4)
    assert r["range_bp"] == pytest.approx(math.log(r["high_mid"] / r["low_mid"]) * 1e4)
    assert r["spread_bp"] == pytest.approx((r["close_ask"] - r["close_bid"]) / r["close_mid"] * 1e4)
    assert r["spread_bp"] == pytest.approx(2 * 1.14e-5 / r["close_mid"] * 1e4)
    assert r["fp_bp"] == pytest.approx(0.0004 / r["close_mid"] * 1e4)
    assert r["dfp_bp"] == pytest.approx(r["fp_bp"] - prev["fp_bp"])
    assert r["days_to_settle"] == 31 and prev["days_to_settle"] == 30
    assert (r["hour"], r["minute_of_day"], r["dow"], r["dom"], r["date"]) == (21, 21 * 60 + 6, 3, 2, 20260702)
    assert df["ret_bp"].iloc[0] == 0


def test_attrs(series):
    s, _ = series
    a = s["USDJPY_USD_FORWARD_1M_1000000"].attrs
    assert a["pair"] == "USDJPY" and a["base"] == "USD" and a["quote"] == "JPY"
    assert a["notional"] == 1000000 and a["tenor"] == "1M"


# ---------------------------------------------------------------- splits
def test_time_split_no_overlap_and_embargo(series):
    s, _ = series
    df = s["EURUSD_EUR_FORWARD_1M_1000000"]
    tr, va, te = L.time_split(df, n_val_days=6, n_test_days=7, embargo_min=60)
    assert len(tr) and len(va) and len(te)
    assert tr["timestamp"].max() < va["timestamp"].min() < va["timestamp"].max() < te["timestamp"].min()
    assert set(te["date"]) == set(df["date"].unique()[-7:])
    assert set(va["date"]) == set(df["date"].unique()[-13:-7])
    gap1 = va["timestamp"].min() - tr["timestamp"].max()
    gap2 = te["timestamp"].min() - va["timestamp"].max()
    assert gap1 > pd.Timedelta(minutes=60) and gap2 > pd.Timedelta(minutes=60)
    assert len(tr) + len(va) + len(te) == len(df) - 2 * 60
    assert tr.attrs["pair"] == "EURUSD"
    with pytest.raises(ValueError):
        L.time_split(df.iloc[:1000], 6, 7)


def test_split_instruments():
    ids = ["EURUSD_EUR_FORWARD_1M_1000000", "EURUSD_EUR_FORWARD_1W_1000000",
           "USDJPY_USD_FORWARD_1M_1000000", "GBPUSD_GBP_FORWARD_ON_5000000"]
    pre, held = L.split_instruments(ids, ["EURUSD"])
    assert held == ids[:2] and pre == [ids[3], ids[2]]  # sorted
    assert set(pre) | set(held) == set(ids) and not set(pre) & set(held)
    pre, held = L.split_instruments({i: None for i in ids}, [])
    assert held == [] and len(pre) == 4


# ---------------------------------------------------------------- persistence + CLI
def test_save_load_roundtrip(series, tmp_path):
    s, m = series
    out = tmp_path / "proc"
    L.save_processed(s, m, str(out))
    assert (out / "manifest.csv").exists() and (out / "instruments.csv").exists()
    assert len(list(out.glob("*.parquet"))) == len(s)
    s2, m2 = L.load_processed(str(out))
    assert set(s2) == set(s) and len(m2) == len(m)
    for k in s:
        pd.testing.assert_frame_equal(s[k], s2[k])
        assert s[k].attrs == s2[k].attrs


def test_cli(dataset, tmp_path, capsys):
    root, _ = dataset
    out = tmp_path / "cli_out"
    assert L.main([str(root), str(out)]) == 0
    assert "saved 3 instruments" in capsys.readouterr().out
    assert len(list(out.glob("*.parquet"))) == 3


def _spot_day(day):
    df = make_day(day, 1.14, seed=7)
    return df.drop(columns=[c for c in df.columns if c.endswith(("_fp", "_all_in"))])


def test_read_day_spot_fills_fp(tmp_path):
    df = _spot_day(pd.Timestamp(START))
    clean, stats = L.read_day(df.to_csv(index=False).encode())
    assert stats["n_kept"] == len(df)
    assert (clean["close_mid_fp"] == 0).all()
    assert np.allclose(clean["close_mid_all_in"], clean["close_mid"])


def test_read_day_forward_missing_one_fp_col_still_errors():
    df = make_day(pd.Timestamp(START), 1.14, seed=1).drop(columns=["close_mid_fp"])
    with pytest.raises(ValueError):
        L.read_day(df.to_csv(index=False).encode())


def test_product_flag(dataset, tmp_path):
    root, _ = dataset
    for d in range(N_DAYS):
        day = pd.Timestamp(START) + pd.Timedelta(days=d)
        _spot_day(day).to_csv(root / f"EURUSD_EUR_SPOT_SPOT_1000000_{day:%Y%m%d}.csv", index=False)
    spot_id = "EURUSD_EUR_SPOT_SPOT_1000000"

    every, man = L.load_all([root], verbose=False)
    assert spot_id in every and len(every) == 4 and (man["status"] == "ok").all()
    spot, _ = L.load_all([root], verbose=False, product="spot")
    assert list(spot) == [spot_id]
    fwd, _ = L.load_all([root], verbose=False, product="forward")
    assert spot_id not in fwd and len(fwd) == 3
    with pytest.raises(ValueError):
        list(L.discover([root], product="bogus"))

    out = tmp_path / "out"
    assert L.main(["--product", "spot", "-q", str(root), str(out)]) == 0
    assert (out / f"{spot_id}.parquet").exists()
