"""Turn bars into integer tokens, the way PRAGMA turns banking events into tokens.

PRAGMA writes every event as key-value pairs and gives each pair one vector,
E(key) + E(value), from one shared embedding table. Numbers become percentile
buckets (with one extra bucket for exactly zero); categories get a token each.

Here an "event" is one 1-minute bar and the "profile" is what we know about
the instrument at that minute:

    event    ret_bp, ret_oc_bp, range_bp, spread_bp, dfp_bp   (bucketed)
             is_gap, is_settle_roll                           (flags)
    profile  pair, tenor                                      (categories)
             days_to_settle, fp_bp                            (bucketed)

Bucket edges are fitted on training bars only, so nothing about the
validation or test period leaks into the tokens.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

EVENT_NUMERIC = ["ret_bp", "ret_oc_bp", "range_bp", "spread_bp", "dfp_bp"]
EVENT_FLAGS = ["is_gap", "is_settle_roll"]
EVENT_KEYS = EVENT_NUMERIC + EVENT_FLAGS

PROFILE_CATEGORICAL = ["pair", "tenor"]  # read from df.attrs
PROFILE_NUMERIC = ["days_to_settle", "fp_bp"]
PROFILE_KEYS = PROFILE_CATEGORICAL + PROFILE_NUMERIC

SPECIALS = ["[PAD]", "[MASK]", "[UNK]", "[INS]", "[EVT]"]
PAD, MASK, UNK, INS, EVT = range(len(SPECIALS))


class Tokenizer:
    """Vocabulary layout (one shared table, as in the paper)::

        specials | one token per key | zero bucket + n_buckets percentile
        buckets (shared by every numeric key) | False, True | categories

    A percentile token means "this value's rank within its own field"; the
    key embedding added to it says which field.
    """

    def __init__(self, n_buckets: int, edges: Dict[str, List[float]],
                 categories: Dict[str, List[str]]):
        self.n_buckets = n_buckets
        self.edges = {key: np.asarray(e, dtype=np.float64) for key, e in edges.items()}
        self.categories = {key: list(values) for key, values in categories.items()}

        next_id = len(SPECIALS)
        self.key_id = {}
        for key in EVENT_KEYS + PROFILE_KEYS:
            self.key_id[key] = next_id
            next_id += 1
        self.zero_id = next_id
        self.bucket_base = next_id + 1
        next_id += 1 + n_buckets
        self.false_id, self.true_id = next_id, next_id + 1
        next_id += 2
        self.category_id = {}
        for key in PROFILE_CATEGORICAL:
            for value in self.categories[key]:
                self.category_id[(key, value)] = next_id
                next_id += 1
        self.vocab_size = next_id

        self.event_key_ids = np.array([self.key_id[k] for k in EVENT_KEYS])
        self.profile_key_ids = np.array([self.key_id[k] for k in PROFILE_KEYS])

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, train_frames: Iterable[pd.DataFrame], n_buckets: int) -> "Tokenizer":
        """Learn bucket edges and category lists from training bars only."""
        frames = list(train_frames)
        edges = {}
        inner_quantiles = np.arange(1, n_buckets) / n_buckets
        for key in EVENT_NUMERIC + PROFILE_NUMERIC:
            values = np.concatenate([f[key].to_numpy(dtype=np.float64) for f in frames])
            values = values[np.isfinite(values) & (values != 0.0)]  # zero has its own bucket
            # Repeated edges (heavily tied values) are merged, leaving fewer buckets.
            edges[key] = np.unique(np.quantile(values, inner_quantiles)).tolist()
        categories = {key: sorted({str(f.attrs[key]) for f in frames})
                      for key in PROFILE_CATEGORICAL}
        return cls(n_buckets, edges, categories)

    def _bucket(self, key: str, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        ids = self.bucket_base + np.searchsorted(self.edges[key], values, side="right")
        ids = np.where(values == 0.0, self.zero_id, ids)
        return np.where(np.isfinite(values), ids, UNK)

    def encode_events(self, df: pd.DataFrame) -> np.ndarray:
        """(n_bars, 7) value tokens, one column per EVENT_KEYS entry."""
        columns = [self._bucket(key, df[key]) for key in EVENT_NUMERIC]
        columns += [np.where(df[key].to_numpy(dtype=bool), self.true_id, self.false_id)
                    for key in EVENT_FLAGS]
        return np.stack(columns, axis=1).astype(np.int16)

    def encode_profile(self, df: pd.DataFrame) -> np.ndarray:
        """(n_bars, 4) value tokens: the instrument's state at each bar."""
        columns = [np.full(len(df), self.category_id.get((key, str(df.attrs[key])), UNK))
                   for key in PROFILE_CATEGORICAL]  # unseen pair/tenor -> [UNK]
        columns += [self._bucket(key, df[key]) for key in PROFILE_NUMERIC]
        return np.stack(columns, axis=1).astype(np.int16)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_buckets": self.n_buckets,
            "edges": {key: e.tolist() for key, e in self.edges.items()},
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, state: dict) -> "Tokenizer":
        return cls(state["n_buckets"], state["edges"], state["categories"])
