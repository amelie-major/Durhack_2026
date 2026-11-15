"""Replay (time travel) in live/server.py: answers at `asOf` must only use bars at or before it.

The Engine is built with hand-made bars and predictions, so there is no network and no training.
"""
import json

import numpy as np
import pandas as pd
import pytest

from live.server import (HORIZONS, Engine, json_safe, last_busy_bar, parse_as_of, unflatten)

N = 600
START = pd.Timestamp("2026-11-12T08:00:00Z")  # a Thursday, well away from the rollover window


def make_scored(n: int = N, start: pd.Timestamp = START) -> dict:
    ts = pd.Series(pd.date_range(start, periods=n, freq="min"))
    i = np.arange(n, dtype=float)
    bars = pd.DataFrame({"timestamp": ts, "close_mid": 100.0 + i / 100.0, "is_gap": False})
    vol = {h: 1.0 + i / 100.0 + h / 1000.0 for h in HORIZONS}       # rises over time
    spread = {h: np.full(n, 0.5 + h / 100.0) for h in HORIZONS}
    past30 = pd.Series(2.0 + i / 1000.0)
    next30 = pd.Series(3.0 + i / 1000.0)
    next30.iloc[-30:] = np.nan                                       # runs off the end of the data
    return {"bars": bars, "vol": vol, "spread": spread, "past30": past30, "next30": next30,
            "last_price": 999.0, "hist30": np.sort(vol[30])}


@pytest.fixture
def engine(monkeypatch):
    bank = {"pairs": ["USDJPY"], "consts": {"USDJPY": {"attrs": {"base": "USD", "quote": "JPY"}}}}
    eng = Engine(bank)
    scored = make_scored()
    monkeypatch.setattr(eng, "scored", lambda pair, max_age=None: scored)
    return eng


def at(minutes: int) -> pd.Timestamp:
    return START + pd.Timedelta(minutes=minutes)


def test_snapshot_at_as_of_uses_that_bar(engine):
    snap = engine.snapshot("USD", at(300))
    p = snap["pairs"][0]
    assert snap["asOf"] == "2026-11-12T13:00:00Z"
    assert snap["lastUpdated"] == "2026-11-12T13:00:00Z"
    assert p["mid"] == pytest.approx(103.0)             # the bar's close, not the live last price
    assert p["forecast"]["vol"] == [round(4.0 + h / 1000.0, 3) for h in HORIZONS]
    assert p["forecast"]["horizonsMin"] == HORIZONS
    assert p["realizedVol30m"] == pytest.approx(2.3)
    assert p["bid"] < p["mid"] < p["ask"]
    assert p["status"] == "live"


def test_live_snapshot_still_uses_latest_bar(engine):
    p = engine.pair_snapshot("USDJPY", at(N - 1))
    assert p["mid"] == 999.0
    assert p["forecast"]["vol"][-1] == round(1.0 + (N - 1) / 100.0 + 0.03, 3)


def test_percentile_ignores_later_bars(engine):
    # Forecasts rise all day, so at any moment "now" is the highest seen so far. Ranking against
    # the whole day (looking ahead) would put minute 300 near the middle instead.
    p = engine.pair_snapshot("USDJPY", at(300), at(300))
    assert p["percentile"] == 0.99
    assert p["volNormal30m"] == pytest.approx(1.0 + 150 / 100.0 + 0.03, abs=0.01)


def test_history_stops_at_as_of_and_hides_unfinished_windows(engine):
    hist = engine.history("USDJPY", 2, as_of=at(300))
    pts = hist["points"]
    assert pts[-1]["t"] == "2026-11-12T13:00:00Z"
    assert all(pd.Timestamp(p["t"]) <= at(300) for p in pts)
    for p in pts:
        finished = pd.Timestamp(p["t"]) + pd.Timedelta(minutes=30) <= at(300)
        assert (p["volActual"] is not None) == finished, p["t"]
    assert pts[-1]["volForecast"] == pytest.approx(4.03)
    assert pts[-1]["mid"] == pytest.approx(103.0)


def test_history_before_the_data_is_no_data(engine):
    pts = engine.history("USDJPY", 2, as_of=at(30))["points"]
    assert pts[0]["status"] == "no_data" and pts[0]["mid"] is None
    assert pts[-1]["status"] == "live"


def test_snapshot_before_any_bar_fails(engine):
    with pytest.raises(RuntimeError):
        engine.snapshot("USD", START - pd.Timedelta(minutes=1))


def test_replay_range_is_last_day_of_real_trading(engine):
    r = engine.replay_range()
    assert r == {"start": "2026-11-11T17:59:00Z", "end": "2026-11-12T17:59:00Z", "stepMin": 1}


def test_last_busy_bar_skips_lone_closing_bar():
    ts = pd.Series(pd.to_datetime(["2026-11-13T21:58Z", "2026-11-13T21:59Z", "2026-11-13T23:59Z"]))
    assert last_busy_bar(ts) == pd.Timestamp("2026-11-13T21:59Z")


def test_unflatten_gives_flat_bars_a_range():
    df = pd.DataFrame({"open_mid": [1.0, 1.1, 1.2], "high_mid": [1.0, 1.1, 1.3],
                       "low_mid": [1.0, 1.1, 1.1], "close_mid": [1.0, 1.1, 1.2]})
    out = unflatten(df)
    assert out.loc[1, ["open_mid", "high_mid", "low_mid"]].tolist() == [1.0, 1.1, 1.0]  # spans the move
    assert out.loc[2].tolist() == df.loc[2].tolist()                                    # real bar untouched
    assert df.loc[1, "open_mid"] == 1.1                                                 # input not modified


def test_parse_as_of():
    assert parse_as_of(None) is None
    assert parse_as_of("2026-11-13T14:30:00Z") == pd.Timestamp("2026-11-13T14:30:00Z")
    assert parse_as_of("2026-11-13T14:30:00").tzinfo is not None
    with pytest.raises(ValueError):
        parse_as_of("garbage")


def test_json_safe_turns_nan_into_null():
    body = json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "x"})
    assert json.loads(json.dumps(body, allow_nan=False)) == {"a": None, "b": [1.0, None], "c": "x"}
