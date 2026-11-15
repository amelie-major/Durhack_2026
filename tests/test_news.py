"""News features: currency attribution, pair sign, and no look-ahead. No network, no FinBERT."""
import numpy as np
import pandas as pd
import pytest

from news.features import NEWS_FEATURES, NewsStore, news_features, top_headlines
from news.sentiment import currencies_in, currency_impacts


def fake_scorer(titles):
    """+0.8 for 'gains'/'rises', -0.8 for 'falls'/'slumps', else 0."""
    out = []
    for t in titles:
        t = t.lower()
        out.append(0.8 if ("gain" in t or "rise" in t) else -0.8 if ("fall" in t or "slump" in t) else 0.0)
    return np.array(out)


def headlines(rows):
    return pd.DataFrame([{"t": pd.Timestamp(t, tz="UTC"), "title": title, "url": f"https://x/{i}",
                          "domain": "x", "ccy_query": q} for i, (t, title, q) in enumerate(rows)])


def store(rows):
    s = NewsStore(["USD", "EUR", "JPY"], scorer=fake_scorer, cache_path=None)
    s.add(headlines(rows))
    return s


def test_currencies_in():
    assert currencies_in("Dollar rises against the yen after Fed minutes") == ["USD", "JPY"]
    assert currencies_in("ECB holds rates; euro steady") == ["EUR"]
    assert currencies_in("Singapore dollar firms as MAS tightens") == ["SGD"]
    assert currencies_in("Australian dollar slips on China data") == []
    assert currencies_in("Residents fed up with traffic") == []


def test_currency_impacts():
    # two currencies + a move word: one against the other
    assert currency_impacts("Yen slumps as dollar firms", -0.6) == {"JPY": -0.6, "USD": 0.6}
    # two currencies without one: both get the tone
    assert currency_impacts("Fed and ECB meet this week", 0.2) == {"USD": 0.2, "EUR": 0.2}
    # nothing named: the currency the headline was found by
    assert currency_impacts("Inflation cools in June", 0.5, fallback="USD") == {"USD": 0.5}


def test_good_usd_news_is_up_for_usdjpy_and_down_for_eurusd():
    s = store([("2026-07-01 09:00", "US dollar gains broadly", "USD")])
    t = pd.Series([pd.Timestamp("2026-07-01 10:00", tz="UTC")])
    usdjpy = news_features(t, s.pair_events("USD", "JPY"))
    eurusd = news_features(t, s.pair_events("EUR", "USD"))
    assert usdjpy["news_sent_day"].iloc[0] == pytest.approx(0.8)
    assert eurusd["news_sent_day"].iloc[0] == pytest.approx(-0.8)
    assert usdjpy["news_count_day"].iloc[0] == pytest.approx(np.log1p(1))


def test_no_look_ahead_and_day_reset():
    s = store([("2026-07-01 10:05", "Dollar gains", "USD"),
               ("2026-07-01 10:30", "Yen falls", "JPY")])
    ev = s.pair_events("USD", "JPY")
    t = pd.Series(pd.to_datetime(["2026-07-01 10:04", "2026-07-01 10:05", "2026-07-01 10:40",
                                  "2026-07-01 11:20", "2026-07-02 00:00"], utc=True))
    f = news_features(t, ev)
    assert f["news_count_day"].iloc[0] == 0 and f["news_sent_day"].iloc[0] == 0  # 10:05 not seen yet
    assert f["news_sent_day"].iloc[1] == pytest.approx(0.8)                        # seen at its own minute
    assert f["news_sent_day"].iloc[2] == pytest.approx(0.8)                        # yen falls = + for USDJPY
    assert np.expm1(f["news_count_day"].iloc[2]) == pytest.approx(2)
    assert f["news_sent_1h"].iloc[3] == pytest.approx(0.8)                         # only 10:30 within the hour
    assert f["news_count_day"].iloc[4] == 0                                        # new day starts empty


def test_without_news_features_are_nan():
    f = news_features(pd.Series(pd.to_datetime(["2026-07-01"], utc=True)), None)
    assert list(f.columns) == NEWS_FEATURES and f.isna().all().all()


def test_store_skips_repeat_titles_and_ranks_headlines():
    s = store([("2026-07-01 09:00", "Dollar gains", "USD"), ("2026-07-01 09:30", "ECB meets", "EUR")])
    assert s.add(headlines([("2026-07-01 09:45", "Dollar gains", "USD")])) == 0
    top = top_headlines(s.pair_events("EUR", "USD"), pd.Timestamp("2026-07-01 12:00", tz="UTC"), n=1)
    assert top[0]["title"] == "Dollar gains" and top[0]["score"] == pytest.approx(-0.8)


def test_gdelt_failure_is_not_cached_as_no_news(tmp_path, monkeypatch):
    import news.gdelt as g
    monkeypatch.setattr(g, "RAW_DIR", tmp_path)
    start = pd.Timestamp("2026-07-01", tz="UTC")
    monkeypatch.setattr(g, "_call", lambda params: None)          # GDELT kept failing
    assert g.fetch_window("USD", start) == [] and not list(tmp_path.iterdir())
    monkeypatch.setattr(g, "_call", lambda params: [{"title": "x"}])
    assert g.fetch_window("USD", start) == [{"title": "x"}] and len(list(tmp_path.iterdir())) == 1


def test_gdelt_empty_answer_is_not_cached(tmp_path, monkeypatch):
    import news.gdelt as g
    monkeypatch.setattr(g, "RAW_DIR", tmp_path)
    monkeypatch.setattr(g, "_call", lambda params: [])
    assert g.fetch_window("USD", pd.Timestamp("2026-07-01", tz="UTC")) == [] and not list(tmp_path.iterdir())


def test_bigquery_export_becomes_day_cache_files(tmp_path, monkeypatch):
    import news.gdelt as g
    from news import bigquery as bq
    monkeypatch.setattr(g, "RAW_DIR", tmp_path)
    csv = tmp_path / "export.csv"
    csv.write_text("date,domain,url,title\n"
                   "20260701091500,a.com,https://a/1,Dollar gains after strong jobs data\n"
                   "20260701091500,b.com,https://a/1,Dollar gains after strong jobs data\n"   # same URL
                   "20260702100000,c.com,https://c/2,Yen &amp; euro slide as BOJ holds rates\n"
                   "20260702100000,d.com,https://d/3,short\n")                               # too short
    df = bq.read_export([str(csv)])
    assert len(df) == 2 and df["title"].iloc[1] == "Yen & euro slide as BOJ holds rates"
    written = bq.write_cache(df, tmp_path)
    assert sorted(written.values()) == [1, 1]
    bq.write_cache(df, tmp_path)                                  # re-running merges, no duplicates
    heads = g.fetch_headlines("2026-07-01", "2026-07-03", ["USD", "EUR", "JPY", "SGD"], offline=True)
    assert len(heads) == 2 and heads["t"].iloc[0] == pd.Timestamp("2026-07-01 09:15", tz="UTC")
