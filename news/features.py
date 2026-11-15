"""News features for the direction model, at every bar, using only headlines seen by then.

For a pair, each headline counts as (its score for base) - (its score for quote), from
news.sentiment.currency_impacts, so good USD news is + for USDJPY and - for EURUSD.

    news_sent_day   mean of that over headlines seen since 00:00 UTC that day, up to t
    news_sent_1h    the same over the last 60 minutes
    news_count_day  log(1 + number of those headlines that day so far)

With no headlines in a window, the sentiment is 0 (neutral). With no news at all
(store missing), every feature is NaN, which XGBoost treats as "unknown".
"""
from __future__ import annotations

import threading
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from news import SCORED
from news.gdelt import fetch_headlines
from news.sentiment import currency_impacts, score_titles

NEWS_FEATURES = ["news_sent_day", "news_sent_1h", "news_count_day"]
RECENT_MIN = 60


def _ns(ts) -> np.ndarray:
    """UTC nanoseconds as int64, for any tz-aware or naive (= UTC) datetime series."""
    s = pd.to_datetime(pd.Series(ts), utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().astype("int64")


def empty_features(n: int, index=None) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index if index is not None else range(n), columns=NEWS_FEATURES)


def news_features(timestamps: pd.Series, events: Optional[pd.DataFrame]) -> pd.DataFrame:
    """NEWS_FEATURES for each timestamp, from a pair's events (columns t, s; see
    NewsStore.pair_events). Only events with t <= timestamp count."""
    index = timestamps.index if isinstance(timestamps, pd.Series) else None
    if events is None:
        return empty_features(len(timestamps), index)
    ts = _ns(timestamps)
    et = _ns(events["t"]) if len(events) else np.array([], dtype="int64")
    order = np.argsort(et, kind="stable")
    et = et[order]
    s = events["s"].to_numpy(dtype=float)[order] if len(events) else np.array([])
    csum = np.concatenate([[0.0], np.cumsum(s)])
    cnt = np.arange(len(et) + 1, dtype=float)

    end = np.searchsorted(et, ts, side="right")                            # events at or before t
    day0 = (ts // 86_400_000_000_000) * 86_400_000_000_000                 # 00:00 UTC that day
    d = np.searchsorted(et, day0, side="left")
    h = np.searchsorted(et, ts - RECENT_MIN * 60_000_000_000, side="right")

    def mean(a, b):
        n = cnt[b] - cnt[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (csum[b] - csum[a]) / n, 0.0), n

    day, n_day = mean(d, end)
    recent, _ = mean(h, end)
    return pd.DataFrame({"news_sent_day": day, "news_sent_1h": recent, "news_count_day": np.log1p(n_day)},
                        index=index)


class NewsStore:
    """Scored headlines with each one's score per currency. Thread-safe; refresh() adds
    the newest headlines for the live server."""

    def __init__(self, currencies: Iterable[str], scorer=None, cache_path=SCORED):
        """`scorer` defaults to FinBERT; pass cache_path=None to skip the score cache."""
        self.currencies = sorted(set(currencies))
        self.scorer = scorer
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.headlines = pd.DataFrame(columns=["t", "title", "url", "domain", "ccy_query", "score"])
        self.impacts = pd.DataFrame(columns=["hid", "ccy", "impact"])

    # ---- loading ----------------------------------------------------------
    def load(self, start, end, offline: bool = False) -> "NewsStore":
        """Add headlines seen in [start, end) (cached windows only when offline)."""
        self.add(fetch_headlines(start, end, self.currencies, offline=offline))
        return self

    def refresh(self, hours: float = 24) -> int:
        """Fetch the last `hours` of headlines; returns how many were new."""
        end = pd.Timestamp.now(tz="UTC")
        return self.add(fetch_headlines(end - pd.Timedelta(hours=hours), end, self.currencies))

    def add(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        with self.lock:
            seen = set(self.headlines["title"].str.lower()) if len(self.headlines) else set()
        df = df[~df["title"].str.lower().isin(seen)]
        if df.empty:
            return 0
        df = df.assign(score=score_titles(df["title"].tolist(), self.scorer, self.cache_path))
        with self.lock:
            start = len(self.headlines)
            df = df.reset_index(drop=True)
            df.index = range(start, start + len(df))
            rows = [(hid, c, v) for hid, title, sc, q in zip(df.index, df["title"], df["score"], df["ccy_query"])
                    for c, v in currency_impacts(title, sc, q).items()]
            imp = pd.DataFrame(rows, columns=["hid", "ccy", "impact"])
            self.headlines = df if start == 0 else pd.concat([self.headlines, df])
            self.impacts = imp if not len(self.impacts) else pd.concat([self.impacts, imp], ignore_index=True)
        return len(df)

    # ---- per pair ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.headlines)

    def span(self):
        """(first, last) headline time, or None when empty."""
        if not len(self.headlines):
            return None
        return self.headlines["t"].min(), self.headlines["t"].max()

    def covers(self, start, end, min_per_day: float = 5) -> bool:
        """True if there are headlines on most days in [start, end]."""
        if not len(self.headlines):
            return False
        t = self.headlines["t"]
        t = t[(t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end))]
        days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
        return t.dt.date.nunique() >= 0.8 * days and len(t) >= min_per_day * days

    def pair_events(self, base: str, quote: str) -> pd.DataFrame:
        """One row per headline that touches base or quote: t, s (+ = good for base vs
        quote), title, url."""
        with self.lock:
            imp, heads = self.impacts, self.headlines
        if not len(imp):
            return pd.DataFrame(columns=["t", "s", "title", "url"])
        sign = imp["ccy"].map({base: 1.0, quote: -1.0})
        part = imp[sign.notna()]
        s = (part["impact"] * sign[sign.notna()]).groupby(part["hid"]).sum()
        out = heads.loc[s.index, ["t", "title", "url"]].assign(s=s.to_numpy())
        return out.sort_values("t", kind="stable").reset_index(drop=True)


def top_headlines(events: pd.DataFrame, t: pd.Timestamp, n: int = 3) -> List[dict]:
    """The n most opinionated headlines of t's UTC day, seen by t."""
    if events is None or not len(events):
        return []
    day = t.floor("D")
    e = events[(events["t"] >= day) & (events["t"] <= t)]
    e = e.reindex(e["s"].abs().sort_values(ascending=False).index).head(n)
    return [{"t": x.t.strftime("%Y-%m-%dT%H:%M:%SZ"), "title": x.title, "url": x.url,
             "score": round(float(x.s), 3)} for x in e.itertuples()]

