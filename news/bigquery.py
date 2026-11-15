"""Load headlines exported from BigQuery (news/gkg_query.sql) into the news cache.

The GDELT API throttles hard, so the history comes from GDELT's BigQuery copy instead:
run news/gkg_query.sql in the BigQuery console, save the result as CSV, then::

    python -m news.bigquery data/news/gkg_export.csv            # cache + FinBERT scores
    python -m news.bigquery data/news/gkg_export.csv --no-score

Rows are written as per-day files in data/news/raw/, in the same format news.gdelt
caches API results in, so news.features, models.signal and the live server read them
without knowing where they came from. Re-running merges (by URL) with what is there.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from news import GDELT_TERMS, RAW_DIR
from news.gdelt import SETTLE_HOURS, WINDOW_HOURS, _cache_path, group_key

GROUP = group_key(GDELT_TERMS)


def read_export(paths: List[str]) -> pd.DataFrame:
    """CSV(s) with columns date (YYYYMMDDHHMMSS), domain, url, title -> one tidy frame."""
    df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"date", "domain", "url", "title"} - set(df.columns)
    if missing:
        raise SystemExit(f"export is missing columns {sorted(missing)}; use news/gkg_query.sql")
    df["t"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    df["title"] = df["title"].map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["t"].notna() & (df["title"].str.len() > 15)]
    return df.drop_duplicates("url").sort_values("t").reset_index(drop=True)


def write_cache(df: pd.DataFrame, raw_dir: Path = RAW_DIR) -> dict:
    """Per-day cache files (news.gdelt format); returns {day: articles in its file}.
    Today (and any day that ended less than SETTLE_HOURS ago) is skipped: GDELT is
    still adding to it, and a partial day cached as complete would never be refetched."""
    assert WINDOW_HOURS == 24, "cache files are per day"
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=SETTLE_HOURS)
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for day, g in df.groupby(df["t"].dt.floor("D")):
        if day + pd.Timedelta(days=1) > cutoff:
            continue
        path = raw_dir / _cache_path(GROUP, day).name
        old = json.loads(path.read_text()) if path.exists() else []
        arts = {a["url"]: a for a in old}
        for r in g.itertuples():
            arts.setdefault(r.url, {"seendate": f"{r.t:%Y%m%dT%H%M%SZ}", "title": r.title,
                                    "url": r.url, "domain": r.domain, "source": "bigquery"})
        path.write_text(json.dumps(list(arts.values())))
        written[day] = len(arts)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Load a BigQuery GDELT export into data/news.")
    p.add_argument("csv", nargs="+", help="CSV file(s) saved from news/gkg_query.sql")
    p.add_argument("--no-score", action="store_true", help="skip FinBERT scoring")
    args = p.parse_args(argv)

    df = read_export(args.csv)
    if df.empty:
        print("no usable rows in the export")
        return 1
    written = write_cache(df)
    if not written:
        print("nothing cached: every row is from today (still being added to by GDELT)")
        return 1
    days = sorted(written)
    sizes = pd.Series(written)
    print(f"{len(df):,} headlines -> {len(days)} day files in {RAW_DIR} "
          f"({days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}), per day: median {sizes.median():.0f}, "
          f"min {sizes.min()}, max {sizes.max()}")
    gaps = pd.date_range(days[0], days[-1], freq="D").difference(pd.DatetimeIndex(days))
    if len(gaps):
        print(f"days with no rows: {', '.join(f'{d:%Y-%m-%d}' for d in gaps)}")

    if not args.no_score:
        from news.gdelt import fetch_headlines
        from news.sentiment import currencies_in, score_titles
        heads = fetch_headlines(days[0], days[-1] + pd.Timedelta(days=1), GDELT_TERMS, offline=True)
        named = heads["title"].map(currencies_in).explode().dropna().value_counts()
        n_days = max(1, heads["t"].dt.date.nunique())
        print(f"{len(heads):,} unique titles; naming each currency, per day:",
              {c: round(n / n_days) for c, n in named.items()})
        scores = score_titles(heads["title"].tolist(), verbose=True)
        print(f"FinBERT: mean {scores.mean():+.3f}, positive {(scores > 0.2).mean():.0%}, "
              f"negative {(scores < -0.2).mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
