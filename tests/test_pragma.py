"""Cheap checks for the PRAGMA pipeline: tokens, windows, rows and shapes."""
from dataclasses import replace

import numpy as np
import pytest
import torch

import loader as L
from models.pragma.config import PRESETS
from models.pragma.data import Bars, apply_mask, fit_tokenizer, make_mask, split_rows
from models.pragma.model import Pragma
from models.pragma.tokenizer import EVENT_KEYS, MASK, PROFILE_KEYS, Tokenizer
from models.xgboost_model import build_dataset
from test_loader import write_dataset

CFG = replace(PRESETS["smoke"], context=16, n_buckets=8)


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    write_dataset(root)
    loaded, _ = L.load_all([str(root)], verbose=False)
    return loaded


@pytest.fixture(scope="module")
def tokenizer(series):
    return fit_tokenizer(series, CFG.n_buckets)


def test_tokens_are_valid_and_survive_a_save(series, tokenizer):
    df = next(iter(series.values()))
    events, profile = tokenizer.encode_events(df), tokenizer.encode_profile(df)
    assert events.shape == (len(df), len(EVENT_KEYS))
    assert profile.shape == (len(df), len(PROFILE_KEYS))
    assert events.min() >= 0 and events.max() < tokenizer.vocab_size
    for edges in tokenizer.edges.values():
        assert (np.diff(edges) > 0).all()

    # An exact zero gets the dedicated zero token.
    zero_rows = df["ret_bp"].to_numpy() == 0.0
    assert zero_rows.any() and (events[zero_rows, 0] == tokenizer.zero_id).all()

    restored = Tokenizer.from_dict(tokenizer.to_dict())
    assert (restored.encode_events(df) == events).all()


def test_bucket_edges_ignore_validation_and_test_bars(series, tokenizer):
    changed = {}
    for instrument_id, df in series.items():
        train = L.time_split(df)[0]
        df = df.copy()
        df.attrs.update(series[instrument_id].attrs)
        later = df["timestamp"] > train["timestamp"].iloc[-1]
        df.loc[later, "spread_bp"] *= 50.0
        changed[instrument_id] = df
    refit = fit_tokenizer(changed, CFG.n_buckets)
    assert np.array_equal(refit.edges["spread_bp"], tokenizer.edges["spread_bp"])


def test_window_does_not_see_the_future(series, tokenizer):
    bars = Bars(series, tokenizer)
    end = 500
    before = bars.batch(np.array([end]), CFG.context, "cpu")

    bars.events[end + 1:] = MASK       # scramble everything after bar `end`
    bars.calendar[end + 1:] = 9.0
    bars.seconds[end + 1:] += 1e6
    after = bars.batch(np.array([end]), CFG.context, "cpu")
    for name in before:
        assert torch.equal(before[name], after[name]), name

    assert before["time_pos"][0, -1] == 0                   # bar t is "now"
    assert (before["time_pos"][0, :-1] > 0).all()            # older bars are further away
    assert (before["time_pos"][0].diff() < 0).all()


def test_window_is_padded_at_the_start_of_an_instrument(series, tokenizer):
    bars = Bars(series, tokenizer)
    second_instrument_start = sorted(bars.offset.values())[1]
    batch = bars.batch(np.array([second_instrument_start + 2]), CFG.context, "cpu")
    assert batch["valid"][0].tolist() == [False] * (CFG.context - 3) + [True] * 3


def test_same_test_rows_as_xgboost(series, tokenizer):
    bars = Bars(series, tokenizer)
    for target in ("vol", "spread"):
        ours = split_rows(series, bars, target, horizon=30)
        theirs = dict(zip(("train", "val", "test"), build_dataset(series, target, 30)))
        for name in ours:
            assert len(ours[name]) == len(theirs[name])
            assert (ours[name]["timestamp"].to_numpy() == theirs[name]["timestamp"].to_numpy()).all()
            assert np.allclose(ours[name]["y"], theirs[name]["y"])


def test_model_shapes_and_masking(series, tokenizer):
    torch.manual_seed(0)
    bars = Bars(series, tokenizer)
    model = Pragma(CFG, tokenizer).eval()
    # includes a window that is mostly padding
    batch = bars.batch(np.array([1, 300, 5000, 9000]), CFG.context, "cpu")

    mask = make_mask(batch["valid"], CFG, torch.Generator().manual_seed(0))
    assert mask.any() and not mask[~batch["valid"]].any()   # padding is never a target
    masked = apply_mask(batch["events"], mask)
    assert (masked[mask] == MASK).all() and torch.equal(masked[~mask], batch["events"][~mask])

    with torch.no_grad():
        logits = model.mlm_logits(model.encode(batch, events=masked), mask)
        prediction = model(batch)
        embedding = model.embedding(batch)
    assert logits.shape == (int(mask.sum()), tokenizer.vocab_size)
    assert prediction.shape == (4,) and torch.isfinite(prediction).all()
    assert embedding.shape == (4, 2 * CFG.d_model)

    # What is in the padded slots must not matter.
    scrambled = dict(batch, events=batch["events"].masked_fill(~batch["valid"][:, :, None], MASK))
    with torch.no_grad():
        assert torch.allclose(model(scrambled), prediction, atol=1e-5)
