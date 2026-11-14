"""Windows of bars for PRAGMA, and the masking used in pre-training.

One training example ("record" in the paper) is one instrument's last L bars
ending at minute t, plus the instrument's profile at t. The model may use
everything at or before t and nothing after it.

Which split an example belongs to is decided by its end minute t. A test
window may reach back into validation or training bars: that is past data the
model would really have at time t, not leakage.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from loader import time_split
from models.xgboost_model import VOL_TENOR, make_target

from .config import Config
from .tokenizer import EVENT_KEYS, MASK, Tokenizer

SPLITS = ("train", "val", "test")
DAY_HARMONICS = (1, 2, 4, 8)  # time-of-day waves: 24h, 12h, 6h, 3h
N_CALENDAR = 2 * len(DAY_HARMONICS) + 2


def fit_tokenizer(series: Dict[str, pd.DataFrame], n_buckets: int) -> Tokenizer:
    """Fit bucket edges on the training part of every instrument."""
    train_frames = []
    for df in series.values():
        train = time_split(df)[0]
        train_frames.append(train)
    return Tokenizer.fit(train_frames, n_buckets)


def _calendar_features(df: pd.DataFrame) -> np.ndarray:
    """Sine/cosine of time of day and day of week (fixed calendar periods).

    The paper uses one wave per cycle. A few extra time-of-day harmonics are
    added so sharp intraday events (the 21:00 rollover) are easy to locate.
    Day of month is left out: with two months of data it would be memorised.
    """
    day_angle = 2 * np.pi * df["minute_of_day"].to_numpy(dtype=np.float64) / 1440.0
    week_angle = 2 * np.pi * df["dow"].to_numpy(dtype=np.float64) / 7.0
    columns = []
    for k in DAY_HARMONICS:
        columns += [np.sin(k * day_angle), np.cos(k * day_angle)]
    columns += [np.sin(week_angle), np.cos(week_angle)]
    return np.stack(columns, axis=1).astype(np.float32)


class Bars:
    """Every instrument's tokenised bars stacked into flat arrays.

    A bar is addressed by its global row number `g`. `start[g]` is the first
    row of the instrument that bar `g` belongs to, so a window never runs
    into the previous instrument.
    """

    def __init__(self, series: Dict[str, pd.DataFrame], tokenizer: Tokenizer):
        events, profile, seconds, calendar, start = [], [], [], [], []
        self.offset: Dict[str, int] = {}
        n_rows = 0
        for instrument_id, df in series.items():
            self.offset[instrument_id] = n_rows
            events.append(tokenizer.encode_events(df))
            profile.append(tokenizer.encode_profile(df))
            seconds.append(df["timestamp"].astype("int64").to_numpy() / _ticks_per_second(df))
            calendar.append(_calendar_features(df))
            start.append(np.full(len(df), n_rows, dtype=np.int64))
            n_rows += len(df)

        self.events = np.concatenate(events)      # (N, 7) value tokens
        self.profile = np.concatenate(profile)    # (N, 4) value tokens
        self.seconds = np.concatenate(seconds)    # (N,)   unix seconds
        self.calendar = np.concatenate(calendar)  # (N, N_CALENDAR)
        self.start = np.concatenate(start)        # (N,)

    def batch(self, ends: np.ndarray, context: int, device) -> Dict[str, torch.Tensor]:
        """Build the windows that end at the global rows `ends`.

        Returns tensors with B = len(ends), L = context:
          events   (B, L, 7) value tokens, oldest bar first, bar t last
          valid    (B, L)    False where the window starts before the data does
          time_pos (B, L)    8*ln(1 + seconds_before_t / 8), the paper's
                             relative-time coordinate (0 for bar t itself)
          calendar (B, L, N_CALENDAR)
          profile  (B, 4)    profile value tokens at bar t
        """
        ends = np.asarray(ends, dtype=np.int64)
        rows = ends[:, None] - np.arange(context - 1, -1, -1)[None, :]
        first_row = self.start[ends][:, None]
        valid = rows >= first_row
        rows = np.maximum(rows, first_row)

        seconds_before_end = self.seconds[ends][:, None] - self.seconds[rows]
        time_pos = 8.0 * np.log1p(seconds_before_end / 8.0)

        return {
            "events": torch.from_numpy(self.events[rows].astype(np.int64)).to(device),
            "valid": torch.from_numpy(valid).to(device),
            "time_pos": torch.from_numpy(time_pos.astype(np.float32)).to(device),
            "calendar": torch.from_numpy(self.calendar[rows]).to(device),
            "profile": torch.from_numpy(self.profile[ends].astype(np.int64)).to(device),
        }


def _ticks_per_second(df: pd.DataFrame) -> float:
    """Timestamps may be stored in ns or us; find how many ticks make a second."""
    unit = np.datetime_data(df["timestamp"].dt.tz_localize(None).to_numpy().dtype)[0]
    return {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]


def split_rows(series: Dict[str, pd.DataFrame], bars: Bars, target: Optional[str] = None,
               horizon: int = 30) -> Dict[str, pd.DataFrame]:
    """Which bars are the end of an example in train / val / test.

    Returns one frame per split with columns instrument_id, pair, timestamp,
    g (global row in `bars`) and y.

    With a `target` ("vol" or "spread") this mirrors
    models.xgboost_model.build_dataset step for step - same instruments, same
    time_split, same dropped rows, same order - so both models are scored on
    identical rows. With target=None (pre-training) every bar is kept and y is 0.
    """
    parts: Dict[str, List[pd.DataFrame]] = {name: [] for name in SPLITS}
    for instrument_id, df in series.items():
        if target == "vol" and len(series) > 1 and df.attrs["tenor"] != VOL_TENOR:
            continue  # same rule as build_dataset: a lone instrument is kept

        rows = pd.DataFrame({
            "timestamp": df["timestamp"],
            "g": bars.offset[instrument_id] + np.arange(len(df)),
        })
        rows["y"] = make_target(df, target, horizon) if target else 0.0
        rows["instrument_id"] = instrument_id
        rows["pair"] = df.attrs["pair"]

        for name, part in zip(SPLITS, time_split(rows)):
            parts[name].append(part.dropna(subset=["y"]))

    if not parts["train"]:
        raise ValueError("no instruments to train on")
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def make_mask(valid: torch.Tensor, cfg: Config, generator: torch.Generator) -> torch.Tensor:
    """Choose which value tokens to hide. Returns a (B, L, 7) boolean mask.

    Three kinds of masking are combined, as in the paper:
      token  single values
      event  whole bars - the model must rebuild a bar from its neighbours
      key    one field across the whole window - no copying it from other bars
    Padded bars are never masked.
    """
    batch_size, length = valid.shape
    n_keys = len(EVENT_KEYS)

    def chance(*shape):
        return torch.rand(*shape, generator=generator)

    mask = chance(batch_size, length, n_keys) < cfg.mask_token
    mask |= chance(batch_size, length, 1) < cfg.mask_event
    mask |= chance(batch_size, 1, n_keys) < cfg.mask_key
    return mask.to(valid.device) & valid[:, :, None]


def apply_mask(events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the masked value tokens with [MASK]."""
    return events.masked_fill(mask, MASK)
