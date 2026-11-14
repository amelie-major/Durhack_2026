"""Train the PRAGMA-style model. Run from the repo root.

    python -m models.pragma.train pretrain --preset gpu
    python -m models.pragma.train finetune --target vol
    python -m models.pragma.train probe    --target vol
    python -m models.pragma.compare        --target vol

pretrain  hide some values of each window and learn to predict them (no
          targets needed, all 27 instruments, training bars only)
finetune  start from the pre-trained weights and train everything, plus a
          small regression head, on the volatility or spread target
probe     freeze the pre-trained model and fit a linear model on its embeddings

`finetune` and `probe` write their test-set predictions to runs/pragma/, which
`compare` reads. Use `--preset smoke` for a few-minute CPU run that only
checks the pipeline works.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import replace
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from loader import load_processed
from models.xgboost_model import MAX_HORIZON

from .config import PRESETS, RUN_DIR, Config
from .data import Bars, apply_mask, fit_tokenizer, make_mask, split_rows
from .model import Pragma
from .tokenizer import EVENT_KEYS, Tokenizer


# ==========================================================================
# Shared helpers
# ==========================================================================

def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device: torch.device):
    """Mixed precision on GPU (bf16, as in the paper); plain float32 on CPU."""
    use = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use)


def lr_at(step: int, total_steps: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: Pragma, cfg: Config, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)


def save_checkpoint(path: str, model: Pragma, tokenizer: Tokenizer, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"config": cfg.to_dict(), "tokenizer": tokenizer.to_dict(),
                "model": model.state_dict()}, path)


def load_checkpoint(path: str, device: torch.device):
    """Returns (model, tokenizer, cfg) exactly as they were when saved."""
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = Config(**state["config"])
    tokenizer = Tokenizer.from_dict(state["tokenizer"])
    model = Pragma(cfg, tokenizer).to(device)
    model.load_state_dict(state["model"])
    return model, tokenizer, cfg


@torch.no_grad()
def run_in_chunks(model: Pragma, bars: Bars, ends: np.ndarray, cfg: Config,
                  device: torch.device, fn: Callable) -> np.ndarray:
    """Apply `fn(batch)` to the windows ending at `ends`, a chunk at a time."""
    model.eval()
    outputs = []
    chunk = 2 * cfg.batch_size
    for i in range(0, len(ends), chunk):
        batch = bars.batch(ends[i:i + chunk], cfg.context, device)
        with autocast(device):
            outputs.append(fn(batch).float().cpu().numpy())
    return np.concatenate(outputs)


def subsample(values: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= max_rows:
        return values
    return values[np.sort(rng.choice(len(values), max_rows, replace=False))]


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def save_predictions(test_rows: pd.DataFrame, pred: np.ndarray, target: str, horizon: int,
                     name: str) -> str:
    out = test_rows[["instrument_id", "pair", "timestamp", "y"]].copy()
    out["pred"] = pred
    os.makedirs(RUN_DIR, exist_ok=True)
    path = os.path.join(RUN_DIR, f"{target}_h{horizon}_{name}.parquet")
    out.to_parquet(path, index=False)
    return path


# ==========================================================================
# Pre-training: masked modelling
# ==========================================================================

def unigram_loss(bars: Bars, rows: np.ndarray) -> float:
    """Loss of always guessing each field's overall frequencies (no context).

    A model that has learned nothing about context scores about this. The
    pre-trained model must score clearly lower to have learned anything.
    """
    entropies = []
    for column in range(len(EVENT_KEYS)):
        counts = np.bincount(bars.events[rows, column])
        p = counts[counts > 0] / counts.sum()
        entropies.append(-(p * np.log(p)).sum())
    return float(np.mean(entropies))


def masked_loss(model: Pragma, batch, cfg: Config, generator, smoothing: float) -> torch.Tensor:
    mask = make_mask(batch["valid"].cpu(), cfg, generator).to(batch["events"].device)
    enc = model.encode(batch, events=apply_mask(batch["events"], mask))
    logits = model.mlm_logits(enc, mask)
    return F.cross_entropy(logits.float(), batch["events"][mask], label_smoothing=smoothing)


def pretrain(args) -> None:
    cfg = PRESETS[args.preset]
    if args.steps:
        cfg = replace(cfg, pretrain_steps=args.steps)
    device = pick_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    series, _ = load_processed(args.data)
    tokenizer = fit_tokenizer(series, cfg.n_buckets)
    bars = Bars(series, tokenizer)
    rows = split_rows(series, bars)  # every bar; split by time
    train_ends = rows["train"]["g"].to_numpy()
    val_ends = subsample(rows["val"]["g"].to_numpy(), 8 * cfg.batch_size, rng)

    model = Pragma(cfg, tokenizer).to(device)
    optimizer = make_optimizer(model, cfg, cfg.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params / 1e6:.2f}M  vocab={tokenizer.vocab_size}  "
          f"train bars={len(train_ends):,}  context={cfg.context}")
    print(f"no-context loss to beat: {unigram_loss(bars, train_ends):.3f}")

    def validation_loss() -> float:
        model.eval()
        generator = torch.Generator().manual_seed(1234)  # same masks every time
        losses = []
        with torch.no_grad():
            for i in range(0, len(val_ends), cfg.batch_size):
                batch = bars.batch(val_ends[i:i + cfg.batch_size], cfg.context, device)
                with autocast(device):
                    losses.append(masked_loss(model, batch, cfg, generator, 0.0).item())
        return float(np.mean(losses))

    generator = torch.Generator().manual_seed(cfg.seed)
    started, recent = time.time(), []
    for step in range(cfg.pretrain_steps):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg.pretrain_steps, cfg.lr, cfg.warmup_steps)
        batch = bars.batch(rng.choice(train_ends, cfg.batch_size), cfg.context, device)
        with autocast(device):
            loss = masked_loss(model, batch, cfg, generator, cfg.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(loss.item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.pretrain_steps:
            print(f"step {step + 1:>6}/{cfg.pretrain_steps}  train {np.mean(recent):.3f} "
                  f"(smoothed labels)  val {validation_loss():.3f}  "
                  f"{time.time() - started:.0f}s", flush=True)
            recent = []
            save_checkpoint(args.checkpoint, model, tokenizer, cfg)
    print(f"saved {args.checkpoint}")


# ==========================================================================
# Fine-tuning: regression on the volatility / spread target
# ==========================================================================

def finetune(args) -> None:
    device = pick_device()
    series, _ = load_processed(args.data)

    if args.from_scratch:  # same architecture, random weights: does pre-training help?
        cfg = PRESETS[args.preset]
        tokenizer = fit_tokenizer(series, cfg.n_buckets)
        torch.manual_seed(cfg.seed)
        model = Pragma(cfg, tokenizer).to(device)
        name = "scratch"
    else:
        model, tokenizer, cfg = load_checkpoint(args.checkpoint, device)
        name = "finetune"
    if args.steps:
        cfg = replace(cfg, finetune_steps=args.steps)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    bars = Bars(series, tokenizer)
    rows = split_rows(series, bars, args.target, args.horizon)
    train_ends, train_y = rows["train"]["g"].to_numpy(), rows["train"]["y"].to_numpy()
    val = rows["val"].iloc[subsample(np.arange(len(rows["val"])), cfg.max_eval_rows, rng)]
    val_ends, val_y = val["g"].to_numpy(), val["y"].to_numpy()
    print(f"device={device}  target={args.target} horizon={args.horizon}min  rows: "
          f"train={len(train_ends):,} val={len(rows['val']):,} test={len(rows['test']):,}")

    # The network predicts a standardised target; predictions are scaled back.
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    train_z = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32)

    def predict(ends: np.ndarray) -> np.ndarray:
        return run_in_chunks(model, bars, ends, cfg, device, model) * y_std + y_mean

    optimizer = make_optimizer(model, cfg, cfg.finetune_lr)
    best_rmse, best_state, evals_since_best = float("inf"), None, 0
    started, recent = time.time(), []
    for step in range(cfg.finetune_steps):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg.finetune_steps, cfg.finetune_lr, cfg.warmup_steps)
        picks = rng.choice(len(train_ends), cfg.batch_size)
        batch = bars.batch(train_ends[picks], cfg.context, device)
        with autocast(device):
            loss = F.mse_loss(model(batch).float(), train_z[picks].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(loss.item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.finetune_steps:
            val_rmse = rmse(val_y, predict(val_ends))
            train_rmse = y_std * math.sqrt(np.mean(recent))
            print(f"step {step + 1:>6}/{cfg.finetune_steps}  train RMSE {train_rmse:.4f}  "
                  f"val RMSE {val_rmse:.4f}  {time.time() - started:.0f}s", flush=True)
            recent = []
            if val_rmse < best_rmse:
                best_rmse, evals_since_best = val_rmse, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                evals_since_best += 1
                if evals_since_best >= cfg.patience:
                    print("validation stopped improving; stopping early")
                    break

    model.load_state_dict(best_state)
    test_pred = predict(rows["test"]["g"].to_numpy())
    print(f"best val RMSE {best_rmse:.4f}  "
          f"test RMSE {rmse(rows['test']['y'].to_numpy(), test_pred):.4f}")
    print("saved", save_predictions(rows["test"], test_pred, args.target, args.horizon, name))


# ==========================================================================
# Linear probe on frozen embeddings (the paper's main evaluation)
# ==========================================================================

def probe(args) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    device = pick_device()
    series, _ = load_processed(args.data)
    model, tokenizer, cfg = load_checkpoint(args.checkpoint, device)
    rng = np.random.default_rng(cfg.seed)

    bars = Bars(series, tokenizer)
    rows = split_rows(series, bars, args.target, args.horizon)
    train = rows["train"].iloc[subsample(np.arange(len(rows["train"])), cfg.max_probe_rows, rng)]
    val = rows["val"].iloc[subsample(np.arange(len(rows["val"])), cfg.max_eval_rows, rng)]
    test = rows["test"]

    def embed(part: pd.DataFrame) -> np.ndarray:
        return run_in_chunks(model, bars, part["g"].to_numpy(), cfg, device, model.embedding)

    print(f"device={device}  embedding train={len(train):,} val={len(val):,} test={len(test):,}")
    scaler = StandardScaler().fit(embed(train))
    x_train, x_val = scaler.transform(embed(train)), scaler.transform(embed(val))

    # Choose the ridge strength on validation, then predict the test set.
    best_rmse, best_model = float("inf"), None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        linear = Ridge(alpha=alpha).fit(x_train, train["y"].to_numpy())
        val_rmse = rmse(val["y"].to_numpy(), linear.predict(x_val))
        print(f"  ridge alpha={alpha:<8g} val RMSE {val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse, best_model = val_rmse, linear

    test_pred = best_model.predict(scaler.transform(embed(test)))
    print(f"test RMSE {rmse(test['y'].to_numpy(), test_pred):.4f}")
    print("saved", save_predictions(test, test_pred, args.target, args.horizon, "probe"))


# ==========================================================================
# Command line
# ==========================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the PRAGMA-style model.")
    commands = parser.add_subparsers(dest="command", required=True)
    default_checkpoint = os.path.join(RUN_DIR, "pretrain.pt")

    def common(sub, with_target: bool):
        sub.add_argument("--data", default="data/processed", help="output dir of loader.py")
        sub.add_argument("--checkpoint", default=default_checkpoint)
        if with_target:
            sub.add_argument("--target", choices=["vol", "spread"], default="vol")
            sub.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")

    sub = commands.add_parser("pretrain", help="masked pre-training")
    common(sub, with_target=False)
    sub.add_argument("--preset", choices=sorted(PRESETS), default="gpu")
    sub.add_argument("--steps", type=int, help="override the preset's number of steps")
    sub.set_defaults(run=pretrain)

    sub = commands.add_parser("finetune", help="train on a target, from the checkpoint")
    common(sub, with_target=True)
    sub.add_argument("--steps", type=int, help="override the number of steps")
    sub.add_argument("--from-scratch", action="store_true",
                     help="ignore the checkpoint and start from random weights (ablation)")
    sub.add_argument("--preset", choices=sorted(PRESETS), default="gpu",
                     help="only used with --from-scratch; otherwise the checkpoint decides")
    sub.set_defaults(run=finetune)

    sub = commands.add_parser("probe", help="linear model on frozen embeddings")
    common(sub, with_target=True)
    sub.set_defaults(run=probe)

    args = parser.parse_args(argv)
    if getattr(args, "horizon", 1) > MAX_HORIZON or getattr(args, "horizon", 1) < 1:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
