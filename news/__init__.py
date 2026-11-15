"""News sentiment for the Buy/Sell direction model.

GDELT headlines (news.gdelt) -> FinBERT score per headline (news.sentiment) ->
per-pair "how good has today's news been for base vs quote" features at each bar
(news.features), which models.signal adds to the direction model's inputs.

Backfill the history once, from the repo root::

    python -m news.gdelt --start 2026-07-01 --end 2026-08-31
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEWS_DIR = ROOT / "data" / "news"
RAW_DIR = NEWS_DIR / "raw"
SCORED = NEWS_DIR / "scored.parquet"

# GDELT search terms per currency: specific enough that the results are about that
# currency or its central bank. Multi-word terms are quoted in the query.
GDELT_TERMS = {
    "USD": ["US dollar", "greenback", "Federal Reserve", "FOMC", "dollar index"],
    "EUR": ["euro", "ECB", "eurozone", "euro area", "Lagarde"],
    "JPY": ["yen", "Bank of Japan", "BOJ"],
    "SGD": ["Singapore dollar", "Monetary Authority of Singapore"],
}
