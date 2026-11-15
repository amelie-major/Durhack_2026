"""Tests for fetch_dukascopy.py. Nothing here touches the network."""
import lzma
from datetime import date

import numpy as np
import pandas as pd

import fetch_dukascopy as F
import loader as L

DAY = date(2026, 7, 1)  # a Wednesday


def make_bi5(rows, scale=1e5) -> bytes:
    """Encode (second, open, close, low, high, volume) rows the way Dukascopy does."""
    candles = np.zeros(len(rows), dtype=F._CANDLE)
    for i, (sec, o, c, lo, hi, vol) in enumerate(rows):
        candles[i] = (sec, round(o * scale), round(c * scale), round(lo * scale), round(hi * scale), vol)
    return lzma.compress(candles.tobytes())


# minute 0 traded, minute 1 is a flat filler (volume 0), minute 2 traded
BID = [(0, 1.3000, 1.3002, 1.2999, 1.3003, 5.0), (60, 1.3002, 1.3002, 1.3002, 1.3002, 0.0),
       (120, 1.3002, 1.3001, 1.3000, 1.3004, 2.0)]
ASK = [(s, o + 2e-5, c + 2e-5, lo + 2e-5, hi + 2e-5, v) for s, o, c, lo, hi, v in BID]


def bars(shift_min=1):
    bid = F.decode(make_bi5(BID), DAY, "GBPUSD")
    ask = F.decode(make_bi5(ASK), DAY, "GBPUSD")
    return F.make_bars(bid, ask, "GBPUSD", shift_min=shift_min)


def test_decode_reads_times_and_prices():
    df = F.decode(make_bi5(BID), DAY, "GBPUSD")
    assert list(df.index) == [pd.Timestamp("2026-07-01 00:00", tz="UTC") + pd.Timedelta(minutes=m)
                              for m in range(3)]
    assert df["close"].tolist() == [1.3002, 1.3002, 1.3001]
    assert df["high"].iloc[0] == 1.3003 and df["low"].iloc[0] == 1.2999


def test_decode_jpy_scale_and_empty_file():
    df = F.decode(make_bi5([(0, 162.633, 162.640, 162.630, 162.641, 1.0)], scale=1e3), DAY, "USDJPY")
    assert df["open"].iloc[0] == 162.633
    assert len(F.decode(b"", DAY, "USDJPY")) == 0


def test_make_bars_drops_quiet_minutes_and_stamps_bar_end():
    df = bars()
    assert list(df.columns) == F.CSV_COLS
    assert df["timestamp"].tolist() == [pd.Timestamp("2026-07-01 00:01", tz="UTC"),
                                        pd.Timestamp("2026-07-01 00:03", tz="UTC")]
    assert np.allclose(df["close_mid"], (df["close_bid"] + df["close_ask"]) / 2)
    assert np.allclose(df["close_ask"] - df["close_bid"], 2e-5)
    assert bars(shift_min=0)["timestamp"].iloc[0] == pd.Timestamp("2026-07-01 00:00", tz="UTC")


def test_settlement_rolls_at_2100_and_skips_weekends():
    ts = pd.DatetimeIndex(["2026-07-01 00:01", "2026-07-01 20:59", "2026-07-01 21:00",
                           "2026-07-03 12:00",   # Friday
                           "2026-07-05 21:30"],  # Sunday evening = Monday's trade date
                          tz="UTC")
    got = [str(d) for d in F.settlement_dates(ts, "GBPUSD")]
    assert got == ["2026-07-03", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08"]
    # USDCAD settles one day sooner
    assert str(F.settlement_dates(ts[:1], "USDCAD")[0]) == "2026-07-02"


def test_written_csv_loads_as_a_spot_instrument(tmp_path):
    assert F.write_daily_csvs(bars(), "GBPUSD", tmp_path) == 1
    path = tmp_path / "GBPUSD_GBP_SPOT_SPOT_1000000_20260701.csv"
    assert path.exists()

    series, manifest = L.load_all([str(tmp_path)], verbose=False)
    assert (manifest["status"] == "ok").all()
    df = series["GBPUSD_GBP_SPOT_SPOT_1000000"]
    assert df.attrs["pair"] == "GBPUSD" and df.attrs["product"] == "SPOT"
    assert df.attrs["feed"] == "dukascopy"  # the CSV says where it came from
    assert len(df) == 2 and (df["close_mid_fp"] == 0).all()


def test_midnight_bar_is_filed_with_the_day_it_started(tmp_path):
    late = [(23 * 3600 + 59 * 60, 1.30, 1.30, 1.30, 1.30, 1.0)]
    bid = F.decode(make_bi5(late), DAY, "GBPUSD")
    ask = F.decode(make_bi5(late), DAY, "GBPUSD")
    F.write_daily_csvs(F.make_bars(bid, ask, "GBPUSD"), "GBPUSD", tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["GBPUSD_GBP_SPOT_SPOT_1000000_20260701.csv"]


def test_download_uses_the_cache(tmp_path):
    cached = tmp_path / "GBPUSD" / "20260701_BID.bi5"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    assert F.download("GBPUSD", DAY, "BID", tmp_path) == b"cached"


def test_compare_identical_series_agree():
    ours = bars()
    stats = F.compare(ours.copy(), ours)
    assert stats["n_common"] == 2 and stats["settle_match"] == 1.0
    assert stats["mid_diff_bp"] == 0 and stats["share_of_challenge_bars"] == 1.0


def test_spread_scale_is_the_geometric_mean_of_pair_ratios():
    stats = pd.DataFrame({"spread_bp_challenge": [1.0, 1.0], "spread_bp_dukascopy": [2.0, 8.0]})
    assert F.spread_scale(stats) == 0.25  # ratios 1/2 and 1/8
    assert F.spread_scale(pd.DataFrame()) is None
