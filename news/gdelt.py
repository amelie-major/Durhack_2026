"""Headlines about our currencies from the GDELT DOC 2.0 API (free, no key), cached on disk.

One combined query (every currency's terms OR-ed together) per WINDOW_HOURS window,
because GDELT throttles hard: in practice about one request in 30 s gets through,
so requests are spaced PAUSE_SECONDS apart. A call returns at most 250 articles;
which currency each headline is about comes from its title (news.sentiment).
A finished window is saved to data/news/raw/ and never fetched again, so an
interrupted backfill picks up where it stopped; the running window is re-fetched.

    python -m news.gdelt --start 2026-07-01 --end 2026-08-31   # backfill + FinBERT scores
    python -m news.gdelt --days 6                              # just the last 6 days
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from typing import Iterable, List, Optional

import pandas as pd

from news import GDELT_TERMS, RAW_DIR

API = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = "Mozilla/5.0 (fx-news-client)"
WINDOW_HOURS = 24
MAX_RECORDS = 250
PAUSE_SECONDS = 40.0  # GDELT documents 5 s between requests but 429s far slower paces
BACKOFF_SECONDS = 30  # after a 429 / error, wait this x the attempt number
SETTLE_HOURS = 1      # a window is only cached once it ended this long ago
COLUMNS = ["t", "title", "url", "domain", "ccy_query"]

_last_call = 0.0


def group_key(currencies: Iterable[str]) -> str:
    """"EUR-JPY-SGD-USD": names the combined query (and its cache files)."""
    return "-".join(sorted(c for c in set(currencies) if c in GDELT_TERMS))


def query_for(group: str) -> str:
    terms = [f'"{t}"' if " " in t else t for c in group.split("-") for t in GDELT_TERMS[c]]
    return f"({' OR '.join(terms)}) sourcelang:english"


def _call(params: dict, retries: int = 5) -> Optional[list]:
    """One GDELT request (rate-limited): its articles ([] = none), or None if GDELT
    kept failing, so the caller doesn't cache a failure as "no news"."""
    global _last_call
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        wait = PAUSE_SECONDS - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception as e:  # 429s and timeouts: back off and try again
            print(f"  [gdelt] {e!r}, retrying", flush=True)
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
            continue
        try:
            return json.loads(text).get("articles", [])
        except json.JSONDecodeError:  # GDELT answers errors in plain text
            print(f"  [gdelt] {text.strip()[:120]!r}", flush=True)
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    return None


def _cache_path(group: str, start: pd.Timestamp):
    return RAW_DIR / f"{group}_{start:%Y%m%d%H}_{WINDOW_HOURS}h.json"


def fetch_window(group: str, start: pd.Timestamp, offline: bool = False) -> list:
    """Raw GDELT articles for a currency group (group_key) in the window starting at `start` (UTC)."""
    path = _cache_path(group, start)
    if path.exists():
        return json.loads(path.read_text())
    if offline:
        return []
    end = start + pd.Timedelta(hours=WINDOW_HOURS)
    arts = _call({"query": query_for(group), "mode": "artlist", "format": "json",
                  "maxrecords": MAX_RECORDS, "startdatetime": f"{start:%Y%m%d%H%M%S}",
                  "enddatetime": f"{end:%Y%m%d%H%M%S}"})
    if arts is None:
        print(f"  [gdelt] gave up on {start:%Y-%m-%d %H:%M}; it will be retried next run", flush=True)
        return []
    # an empty answer isn't cached: a whole window without one headline about four major
    # currencies isn't believable, and GDELT has been seen returning [] while throttling
    if arts and end <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=SETTLE_HOURS):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(arts))
    return arts


def windows(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    first = start.floor(f"{WINDOW_HOURS}h")
    return list(pd.date_range(first, end, freq=f"{WINDOW_HOURS}h", inclusive="left")) or [first]


def fetch_headlines(start, end, currencies: Iterable[str], offline: bool = False,
                    verbose: bool = False) -> pd.DataFrame:
    """Headlines seen between start and end about any of the currencies, one row each
    (t, title, url, domain, ccy_query = the group), with syndicated copies dropped."""
    start, end = _utc(start), _utc(end)
    group = group_key(currencies)
    rows = []
    if group:
        ws = windows(start, end)
        for i, w in enumerate(ws):
            for a in fetch_window(group, w, offline):
                rows.append((a.get("seendate"), a.get("title"), a.get("url"), a.get("domain"), group))
            if verbose and ((i + 1) % 5 == 0 or i + 1 == len(ws)):
                print(f"  {i + 1}/{len(ws)} windows", flush=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df.assign(t=pd.to_datetime(df["t"], utc=True))
    df["t"] = pd.to_datetime(df["t"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    df["title"] = df["title"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["t"].notna() & (df["title"].str.len() > 15)]
    df = df[(df["t"] >= start) & (df["t"] < end)]
    # the same story is syndicated across many sites: keep its first sighting
    key = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df.assign(_k=key).sort_values("t").drop_duplicates("_k").drop(columns="_k")
    return df.reset_index(drop=True)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def main(argv: Optional[List[str]] = None) -> int:
    global PAUSE_SECONDS
    p = argparse.ArgumentParser(description="Download (and FinBERT-score) GDELT headlines per currency.")
    p.add_argument("--start", help="UTC date, e.g. 2026-07-01")
    p.add_argument("--end", help="UTC date (exclusive end is the day after)")
    p.add_argument("--days", type=float, help="instead of --start/--end: the last N days")
    p.add_argument("--currencies", default=",".join(GDELT_TERMS))
    p.add_argument("--no-score", action="store_true", help="download only, skip FinBERT")
    p.add_argument("--pause", type=float, default=PAUSE_SECONDS, help="seconds between GDELT requests")
    args = p.parse_args(argv)
    PAUSE_SECONDS = args.pause
    if args.days:
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=args.days)
    elif args.start and args.end:
        start, end = _utc(args.start), _utc(args.end) + pd.Timedelta(days=1)
    else:
        p.error("give --start and --end, or --days")
    ccys = [c.strip().upper() for c in args.currencies.split(",")]
    print(f"GDELT headlines {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} for {', '.join(ccys)}", flush=True)
    df = fetch_headlines(start, end, ccys, verbose=True)
    print(f"{len(df):,} unique headlines")
    if len(df):
        from news.sentiment import currencies_in
        named = df["title"].map(currencies_in).explode().dropna()
        days = df["t"].dt.date.nunique()
        print(f"headlines per day: {len(df) / days:.0f}; naming each currency, per day:",
              {c: round(n / days) for c, n in named.value_counts().items()})
    if not args.no_score and len(df):
        from news.sentiment import score_titles
        scores = score_titles(df["title"].tolist(), verbose=True)
        print(f"FinBERT scores: mean {scores.mean():+.3f}, share positive {(scores > 0.2).mean():.0%}, "
              f"negative {(scores < -0.2).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
