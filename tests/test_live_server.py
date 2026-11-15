"""Replay (time travel) in live/server.py: answers at `asOf` must only use bars at or before it.

The Engine is built with hand-made bars and predictions, so there is no network and no training.
"""
import json

import numpy as np
import pandas as pd
import pytest

from live.server import (HORIZONS, Engine, json_safe, last_busy_bar, parse_as_of, unflatten)
from models.signal import signal_score, steady_signal

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
    # P(up) climbs through the day: sell early, hold in the middle, buy late
    p_up = np.linspace(0.0, 1.0, n)
    expected, action, score = signal_score(p_up, vol[10], spread[10], 1.0)
    signal = {"p_up": p_up, "expected": expected, "action": action, "score": score,
              "hurdle": spread[10], "horizon": 10}
    return {"bars": bars, "vol": vol, "spread": spread, "past30": past30, "next30": next30,
            "last_price": 999.0, "hist30": np.sort(vol[30]), "signal": signal}


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


def test_signal_comes_from_the_as_of_row(engine):
    early = engine.pair_snapshot("USDJPY", at(10), at(10))["signal"]
    mid = engine.pair_snapshot("USDJPY", at(300), at(300))["signal"]
    late = engine.pair_snapshot("USDJPY", at(590), at(590))["signal"]
    assert (early["action"], mid["action"], late["action"]) == ("sell", "hold", "buy")
    for sig in (early, mid, late):
        assert sig["horizonMin"] == 10 and 0.0 <= sig["score"] <= 1.0
    assert early["score"] < 0.1192 < mid["score"] < 0.8808 < late["score"]


def test_no_signal_on_stale_data(engine):
    # two hours after the last bar the pair is stale: never a buy or sell
    sig = engine.pair_snapshot("USDJPY", at(N + 120), at(N + 120))["signal"]
    assert sig["action"] == "hold"


def test_signal_score_rule():
    p = np.array([0.5, 1.0, 0.0, 0.75, 0.25])
    expected, signal, score = signal_score(p, np.full(5, 5.0), np.full(5, 1.0), threshold=1.0)
    factor = np.sqrt(2 / np.pi)
    assert expected == pytest.approx([0, 5 * factor, -5 * factor, 2.5 * factor, -2.5 * factor])
    assert signal.tolist() == ["HOLD", "BUY", "SELL", "BUY", "SELL"]
    assert score[0] == 0.5 and score[1] > 0.88 and score[2] < 0.12
    # a higher threshold (dearer hurdle) turns the weaker calls into HOLD
    assert signal_score(p, np.full(5, 5.0), np.full(5, 1.0), threshold=3.0)[1].tolist() == ["HOLD", "BUY", "SELL", "HOLD", "HOLD"]


def test_steady_signal_holds_a_call_until_it_clearly_ends():
    hurdles = [0.2, 1.2, 0.8, 0.6, 0.4, 0.9, -1.1, -0.7, -0.4, 1.5]
    got = steady_signal(np.array(hurdles) * 2.0, np.full(len(hurdles), 2.0), exit_at=0.5)
    assert got.tolist() == ["HOLD", "BUY", "BUY", "BUY", "HOLD", "HOLD", "SELL", "SELL", "HOLD", "BUY"]


def test_steady_signal_can_flip_straight_over_and_resets_on_gaps():
    assert steady_signal(np.array([1.2, -1.2]), np.ones(2)).tolist() == ["BUY", "SELL"]
    got = steady_signal(np.array([1.2, 0.8, 0.8, np.nan, 0.8]), np.ones(5), reset=[False, False, True, False, False])
    assert got.tolist() == ["BUY", "BUY", "HOLD", "HOLD", "HOLD"]


# ---------------------------------------------------------------------------
# Model selection: xgboost (live) or pragma (saved predictions, replay only)
# ---------------------------------------------------------------------------

from live import pragma_replay  # noqa: E402
from live.server import WINDOW, pragma_health  # noqa: E402
from models.xgboost_model import make_target  # noqa: E402

P_START = pd.Timestamp("2026-08-25T08:00:00Z")  # a Tuesday
P_ID = "USDJPY_USD_FORWARD_1M_1000000"


def pragma_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range(P_START, periods=n, freq="min"),
        "close_mid": 150.0 + np.arange(n) / 100.0,
        "ret_bp": rng.normal(0, 1, n),
        "spread_bp": rng.uniform(0.2, 0.4, n),
        "is_gap": False,
    })
    df.attrs.update({"pair": "USDJPY", "base": "USD", "quote": "JPY", "tenor": "1M", "product": "FORWARD"})
    return df


def write_predictions(folder, df: pd.DataFrame, shift_y: float = 0.0) -> None:
    """Predictions for bars 100..299: vol = 4 bp, spread = 0.5 bp (stored as logs)."""
    for target, bp in (("vol", 4.0), ("spread", 0.5)):
        y = make_target(df, target, pragma_replay.PRED_HORIZON)
        rows = pd.DataFrame({"instrument_id": P_ID, "pair": "USDJPY", "timestamp": df["timestamp"],
                             "y": y + shift_y, "pred": np.log(bp)}).iloc[100:300]
        rows.to_parquet(pragma_replay.prediction_paths(folder)[target], index=False)


@pytest.fixture
def pragma(tmp_path, monkeypatch):
    df = pragma_frame()
    monkeypatch.setattr(pragma_replay, "load_processed", lambda path: ({P_ID: df}, None))
    write_predictions(tmp_path, df)
    return pragma_replay.load_pragma(tmp_path, tmp_path, HORIZONS, WINDOW)


@pytest.fixture
def both(engine, pragma):
    engine.pragma = pragma
    return engine


def test_pragma_only_scores_rows_it_has_predictions_for(pragma):
    s = pragma["scored"]["USDJPY"]
    assert pragma["pairs"] == ["USDJPY"]
    assert len(s["bars"]) == 200
    assert s["bars"]["timestamp"].iloc[0] == P_START + pd.Timedelta(minutes=100)
    assert s["signal"] is None


def test_pragma_shorter_horizons_are_scaled_from_30_minutes(pragma):
    s = pragma["scored"]["USDJPY"]
    assert s["vol"][30][0] == pytest.approx(4.0)
    assert s["vol"][5][0] == pytest.approx(4.0 * np.sqrt(5 / 30))
    assert s["spread"][5][0] == pytest.approx(0.5)


def test_pragma_rejects_predictions_made_from_other_data(tmp_path, monkeypatch):
    df = pragma_frame()
    monkeypatch.setattr(pragma_replay, "load_processed", lambda path: ({P_ID: df}, None))
    write_predictions(tmp_path, df, shift_y=0.1)
    with pytest.raises(ValueError, match="do not match"):
        pragma_replay.load_pragma(tmp_path, tmp_path, HORIZONS, WINDOW)


def test_pragma_snapshot_at_as_of_uses_that_bar(both):
    t = P_START + pd.Timedelta(minutes=150)
    snap = both.snapshot("USD", t, "pragma")
    (p,) = snap["pairs"]
    assert snap["model"] == "pragma" and snap["asOf"] == "2026-08-25T10:30:00Z"
    assert p["mid"] == pytest.approx(151.5)
    assert p["forecast"]["vol"][-1] == pytest.approx(4.0)
    assert p["signal"] is None
    json.dumps(json_safe(snap), allow_nan=False)


def test_pragma_without_as_of_ends_on_its_last_bar(both):
    snap = both.snapshot("USD", None, "pragma")
    assert snap["asOf"] == snap["lastUpdated"] == "2026-08-25T12:59:00Z"
    assert both.replay_range("pragma")["end"] == "2026-08-25T12:59:00Z"
    hist = both.history("USDJPY", 1, as_of=None, model="pragma")
    assert hist["model"] == "pragma" and hist["points"][-1]["volForecast"] == pytest.approx(4.0)


def test_xgboost_is_untouched_by_pragma(both):
    snap = both.snapshot("USD", at(100))
    assert snap["model"] == "xgboost" and snap["pairs"][0]["mid"] == pytest.approx(101.0)


def test_pick_model(engine, pragma):
    assert engine.pick_model(None) == "xgboost" and engine.models() == ["xgboost"]
    with pytest.raises(ValueError, match="no PRAGMA predictions"):
        engine.pick_model("pragma")
    with pytest.raises(ValueError, match="unknown model"):
        engine.pick_model("lstm")
    engine.pragma, engine.default_model = pragma, "pragma"
    assert engine.pick_model(None) == "pragma" and engine.pick_model("xgboost") == "xgboost"
    assert pragma_health(pragma)["end"] == "2026-08-25T12:59:00Z" and pragma_health(None) is None
