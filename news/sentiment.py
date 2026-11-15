"""FinBERT headline scores and which currency each headline is about.

score = P(positive) - P(negative) from ProsusAI/finbert, in -1..+1. Scores are cached
in data/news/scored.parquet by title, so a headline is only ever scored once and
training, replay and live all see the same number.

FinBERT judges the tone of the headline ("yen slumps" is negative), not which way a
pair goes. currency_impacts() turns that into a score per currency, and
news.features then nets base against quote for a pair. Whether that tone actually
predicts the next 10 minutes is for the direction model to learn (or not).
"""
from __future__ import annotations

import hashlib
import re
import threading
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from news import SCORED

MODEL = "ProsusAI/finbert"
BATCH = 32

# Currency mentions in a headline. Words are matched ignoring case; ACRONYMS only as
# written, so "fed up" or "mas" in another sense don't count.
_OTHER_DOLLARS = (r"\b(?:australian|aussie|canadian|new zealand|nz|hong kong|hk|taiwan|"
                  r"taiwanese|zimbabwe|zimbabwean|liberian|namibian|jamaican|fiji)\s+dollars?\b")
WORDS = {
    "SGD": [r"singapore dollars?", r"sing dollars?", r"singdollar", r"monetary authority of singapore"],
    "USD": [r"dollars?", r"greenback", r"federal reserve", r"powell", r"u\.?s\.? treasur(?:y|ies)"],
    "EUR": [r"euros?", r"eurozone", r"euro area", r"euro zone", r"lagarde"],
    "JPY": [r"yen", r"bank of japan", r"ueda"],
}
ACRONYMS = {"USD": ["Fed", "FOMC", "USD"], "EUR": ["ECB", "EUR"], "JPY": ["BOJ", "BoJ", "JPY"],
            "SGD": ["MAS", "SGD"]}
# with exactly two currencies and one of these, the headline is one against the other
_RELATIVE = re.compile(r"\b(?:against|versus|vs\.?|rise[sd]?|gain(?:s|ed)?|climb(?:s|ed)?|jump(?:s|ed)?|"
                       r"firm(?:s|ed)?|rall(?:y|ies|ied)|surge[sd]?|soar(?:s|ed)?|strengthen(?:s|ed)?|"
                       r"fall(?:s)?|fell|drop(?:s|ped)?|slide[sd]?|slip(?:s|ped)?|slump(?:s|ed)?|"
                       r"weaken(?:s|ed)?|sink(?:s)?|sank|tumble[sd]?|plunge[sd]?|eases?|eased)\b", re.I)

_PATTERNS = {
    ccy: [re.compile(rf"\b{w}\b", re.I) for w in WORDS[ccy]] + [re.compile(rf"\b{a}\b") for a in ACRONYMS[ccy]]
    for ccy in WORDS
}


def currencies_in(title: str) -> List[str]:
    """Currencies a headline mentions, in the order they first appear."""
    text = re.sub(_OTHER_DOLLARS, " ", title, flags=re.I)
    first: Dict[str, int] = {}
    for ccy in ("SGD", "USD", "EUR", "JPY"):  # SGD first: "Singapore dollar" is not USD
        for pat in _PATTERNS[ccy]:
            m = pat.search(text)
            if m and (ccy not in first or m.start() < first[ccy]):
                first[ccy] = m.start()
        if ccy == "SGD" and "SGD" in first:
            text = re.sub(r"\bsing(?:apore)?\s+dollars?\b", " ", text, flags=re.I)
    return sorted(first, key=first.get)


def currency_impacts(title: str, score: float, fallback: Optional[str] = None) -> Dict[str, float]:
    """{currency: score} for one headline.

    - one currency (or none named: `fallback`, the currency it was found by, if it is
      a single currency) gets the score;
    - two currencies with a move word ("dollar rises against yen"): the first gets
      the score and the second its opposite;
    - otherwise every named currency gets the score.
    """
    ccys = currencies_in(title) or ([fallback] if fallback in WORDS else [])
    if len(ccys) == 2 and _RELATIVE.search(title):
        return {ccys[0]: score, ccys[1]: -score}
    return {c: score for c in ccys}


# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------

_pipe = None
_pipe_lock = threading.Lock()


def finbert(titles: Sequence[str]) -> np.ndarray:
    """P(positive) - P(negative) per title (loads the model on first use)."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            from transformers import pipeline  # heavy import, only when scoring is needed
            _pipe = pipeline("text-classification", model=MODEL, top_k=None, device=-1)
        out = _pipe(list(titles), batch_size=BATCH, truncation=True, max_length=64)
    scores = []
    for labels in out:
        p = {d["label"].lower(): d["score"] for d in labels}
        scores.append(p.get("positive", 0.0) - p.get("negative", 0.0))
    return np.asarray(scores, dtype=float)


def _key(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]


_cache: Optional[Dict[str, float]] = None
_cache_lock = threading.Lock()


def _load_cache(path) -> Dict[str, float]:
    global _cache
    if _cache is None:
        _cache = {}
        if path.exists():
            df = pd.read_parquet(path)
            _cache = dict(zip(df["key"], df["score"]))
    return _cache


def score_titles(titles: Sequence[str], scorer: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
                 cache_path=SCORED, verbose: bool = False) -> np.ndarray:
    """Cached FinBERT scores for titles. Pass `scorer` (and cache_path=None) to use
    another model, e.g. a fake one in tests."""
    keys = [_key(t) for t in titles]
    if cache_path is None:
        return np.asarray((scorer or finbert)(titles), dtype=float)
    with _cache_lock:
        cache = _load_cache(cache_path)
        todo = list(dict.fromkeys(k for k in keys if k not in cache))
        if todo:
            by_key = dict(zip(keys, titles))
            texts = [by_key[k] for k in todo]
            if verbose:
                print(f"FinBERT scoring {len(texts):,} new headlines...", flush=True)
            fn = scorer or finbert
            step = 2048
            for i in range(0, len(texts), step):
                for k, s in zip(todo[i:i + step], fn(texts[i:i + step])):
                    cache[k] = float(s)
                if verbose and len(texts) > step:
                    print(f"  {min(i + step, len(texts)):,}/{len(texts):,}", flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"key": list(cache), "score": list(cache.values())}).to_parquet(cache_path)
        return np.array([cache[k] for k in keys], dtype=float)
