"""Which instruments the XGBoost volatility model trains on."""
import pandas as pd

from models.xgboost_model import vol_instruments


def frame(pair: str, tenor: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs.update(pair=pair, tenor=tenor)
    return df


def test_vol_uses_1m_forward_or_spot_when_there_is_none():
    series = {
        "EURUSD_SPOT": frame("EURUSD", "SPOT"),  # listed before the 1M on purpose
        "EURUSD_1M": frame("EURUSD", "1M"),
        "EURUSD_1Y": frame("EURUSD", "1Y"),
        "USDJPY_1M": frame("USDJPY", "1M"),
        "USDJPY_SPOT": frame("USDJPY", "SPOT"),
        "GBPUSD_SPOT": frame("GBPUSD", "SPOT"),  # a Dukascopy pair: spot only
    }
    assert vol_instruments(series) == {"EURUSD_1M", "USDJPY_1M", "GBPUSD_SPOT"}
