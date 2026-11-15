"""The live Yahoo pairs and the globe's currency zones match the pairs in data/processed,
and the live server builds one set of constants per pair (1M forward, else spot)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import live.server as server

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENTS = ROOT / "data" / "processed" / "instruments.csv"
FRONTEND = ROOT / "frontend" / "src" / "data"


def _data_pairs() -> pd.DataFrame:
    if not INSTRUMENTS.exists():
        pytest.skip("data/processed not built")
    return pd.read_csv(INSTRUMENTS)[["pair", "base", "quote"]].drop_duplicates()


def _block(file: str, const: str) -> str:
    """Body of `export const <const> ... = [ ... ]` in a frontend data file."""
    src = (FRONTEND / file).read_text()
    start = src.index(f"export const {const}")
    return src[start:src.index("\n]", start)]


def test_live_pairs_match_processed_data():
    pairs = _data_pairs()
    live = set(re.findall(r"pair: '(\w+)', base: '(\w+)', quote: '(\w+)'", _block("yahoo.ts", "LIVE_PAIRS")))
    assert live == set(pairs.itertuples(index=False, name=None))


def test_every_currency_has_a_zone():
    pairs = _data_pairs()
    zones = set(re.findall(r"\{ code: '(\w+)'", _block("currencies.ts", "ZONES")))
    assert set(pairs["base"]) | set(pairs["quote"]) == zones


def _series(pair: str, tenor: str, product: str, spread: float) -> pd.DataFrame:
    df = pd.DataFrame({"spread_bp": np.full(10, spread), "fp_bp": np.zeros(10),
                       "days_to_settle": np.full(10, 2)})
    df.attrs.update(pair=pair, tenor=tenor, product=product, base=pair[:3], quote=pair[3:])
    return df


def test_train_bank_takes_1m_forward_else_spot(monkeypatch):
    series = {
        "EURUSD_SPOT": _series("EURUSD", "SPOT", "SPOT", 0.1),
        "EURUSD_1M": _series("EURUSD", "1M", "FORWARD", 0.3),
        "EURUSD_1W": _series("EURUSD", "1W", "FORWARD", 0.2),
        "AUDUSD_SPOT": _series("AUDUSD", "SPOT", "SPOT", 1.3),
    }

    class Fake:
        best_iteration = 0

    monkeypatch.setattr(server, "all_series", lambda: series)
    monkeypatch.setattr(server, "build_dataset", lambda s, t, h: (None, None, None))
    monkeypatch.setattr(server, "train", lambda tr, va: Fake())
    monkeypatch.setattr(server, "train_direction_model", lambda s: {})
    bank = server.train_bank()
    assert bank["pairs"] == ["AUDUSD", "EURUSD"]
    assert bank["consts"]["EURUSD"]["attrs"]["tenor"] == "1M"
    assert bank["consts"]["EURUSD"]["spread_bp"] == pytest.approx(0.3)
    assert bank["consts"]["AUDUSD"]["attrs"]["tenor"] == "SPOT"
    assert bank["features"] == list(server.FEATURES)
