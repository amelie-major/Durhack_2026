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
from tqdm import tqdm

from loader import load_processed
from models.xgboost_model import MAX_HORIZON, VOL_TENOR

from .config import PRESETS, RUN_DIR, Config, with_current_finetune_settings
from .data import Bars, apply_mask, fit_tokenizer, make_mask, split_rows
from .model import Pragma
from .tokenizer import EVENT_KEYS, Tokenizer


# ==========================================================================
# Shared helpers
# ==========================================================================

def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # faster float32 maths on newer GPUs
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Which reduced precision this device runs fast, or None for plain float32.

    bf16 (what the paper uses) is only fast on Ampere and newer cards. Older
    cards with tensor cores (Colab's T4, V100, RTX 20xx) are fast in fp16 and
    only *emulate* bf16, slowly. Pascal cards and CPUs gain nothing from either.
    """
    if device.type != "cuda":
        return None
    generation = torch.cuda.get_device_capability(device)[0]
    if generation >= 8:
        return torch.bfloat16
    return torch.float16 if generation == 7 else None


def autocast(device: torch.device):
    dtype = amp_dtype(device)
    return torch.autocast(device_type=device.type, dtype=dtype or torch.bfloat16,
                          enabled=dtype is not None)


def make_scaler(device: torch.device) -> "torch.amp.GradScaler":
    """fp16 needs its gradients rescaled to avoid underflow; other modes do not."""
    return torch.amp.GradScaler(device.type, enabled=amp_dtype(device) is torch.float16)


def describe(device: torch.device) -> str:
    dtype = amp_dtype(device)
    return f"device={device} precision={str(dtype).replace('torch.', '') if dtype else 'float32'}"


def lr_at(step: int, total_steps: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def check_finite(loss: torch.Tensor, step: int, stage: str, debug: bool, batch=None) -> float:
    """Fail loudly (with context) the moment the loss goes NaN/inf."""
    value = loss.item()
    if not math.isfinite(value):
        msg = f"{stage}: non-finite loss ({value}) at step {step + 1}"
        if batch is not None:
            msg += "  batch shapes: " + ", ".join(
                f"{k}={tuple(v.shape)}" for k, v in batch.items() if hasattr(v, "shape"))
        raise FloatingPointError(msg)
    return value


def backward_and_step(loss: torch.Tensor, model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer, scaler) -> float:
    """Backward pass, gradient clipping and optimiser step. Returns the gradient norm."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # so clipping sees the real gradient sizes
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    scaler.step(optimizer)
    scaler.update()
    return grad_norm


def log_progress(step: int, total: int, every: int, recent: list, lr: float,
                 grad_norm: float, started: float, label: str) -> None:
    """One plain-text line every `every` steps (readable in slurm logs, unlike the bar)."""
    if every and (step + 1) % every == 0:
        window = recent[-every:]
        tqdm.write(f"  [{step + 1:>6}/{total}] {label} {np.mean(window):.4f}  "
                   f"lr {lr:.2e}  grad norm {grad_norm:.2f}  {time.time() - started:.0f}s")


def set_dropout(model: torch.nn.Module, p: float) -> None:
    """Change every dropout rate in the model (the layers store it in two ways)."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
        if isinstance(getattr(module, "dropout", None), float):
            module.dropout = p


def make_optimizer(model: Pragma, cfg: Config, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)


def save_checkpoint(path: str, model: Pragma, tokenizer: Tokenizer, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"config": cfg.to_dict(), "tokenizer": tokenizer.to_dict(),
                "model": model.state_dict()}, path)


def load_checkpoint(path: str, device: torch.device):
    """Returns (model, tokenizer, cfg): the saved model, with today's fine-tuning settings."""
    state = torch.load(path, map_location=device, weights_only=True)
    known = Config.__dataclass_fields__
    cfg = with_current_finetune_settings(
        Config(**{k: v for k, v in state["config"].items() if k in known}))
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
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    device = pick_device(args.device)
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
    scaler = make_scaler(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{describe(device)}  params={n_params / 1e6:.2f}M  vocab={tokenizer.vocab_size}  "
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

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    generator = torch.Generator().manual_seed(cfg.seed)
    started, recent = time.time(), []
    bar = tqdm(range(cfg.pretrain_steps), desc="pretrain", unit="step", dynamic_ncols=True)
    for step in bar:
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg.pretrain_steps, cfg.lr, cfg.warmup_steps)
        batch = bars.batch(rng.choice(train_ends, cfg.batch_size), cfg.context, device)
        with autocast(device):
            loss = masked_loss(model, batch, cfg, generator, cfg.label_smoothing)
        recent.append(check_finite(loss, step, "pretrain", args.debug, batch))
        grad_norm = backward_and_step(loss, model, optimizer, scaler)
        bar.set_postfix(loss=f"{recent[-1]:.3f}", lr=f"{group['lr']:.1e}",
                        gnorm=f"{grad_norm:.2f}", refresh=False)
        log_progress(step, cfg.pretrain_steps, args.log_every, recent, group["lr"], grad_norm,
                     started, "train loss")
        if args.debug and step < 3:
            tqdm.write(f"[debug] step {step + 1}: loss {recent[-1]:.4f}  grad norm "
                       f"{grad_norm:.3f}  batch " + ", ".join(
                           f"{k}={tuple(v.shape)}" for k, v in batch.items()
                           if hasattr(v, "shape")))

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.pretrain_steps:
            tqdm.write(f"step {step + 1:>6}/{cfg.pretrain_steps}  train {np.mean(recent):.3f} "
                       f"(smoothed labels)  val {validation_loss():.3f}  "
                       f"{time.time() - started:.0f}s")
            recent = []
            save_checkpoint(args.checkpoint, model, tokenizer, cfg)
    bar.close()
    print(f"saved {args.checkpoint}")


# ==========================================================================
# Fine-tuning: regression on the volatility / spread target
# ==========================================================================

def finetune(args) -> None:
    device = pick_device(args.device)
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
    print(f"{describe(device)}  target={args.target} horizon={args.horizon}min  rows: "
          f"train={len(train_ends):,} val={len(rows['val']):,} test={len(rows['test']):,}")

    # The network predicts a standardised target; predictions are scaled back.
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    train_z = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32)

    def predict(ends: np.ndarray) -> np.ndarray:
        return run_in_chunks(model, bars, ends, cfg, device, model) * y_std + y_mean

    # Gentle fine-tune (the full-size model overfits the small target sets):
    # more dropout; the new head first learns alone on frozen layers (LP-FT), then
    # the pre-trained layers move with a small learning rate, the head with a larger one.
    set_dropout(model, cfg.finetune_dropout)
    backbone = [p for n, p in model.named_parameters() if not n.startswith("regressor")]
    head = list(model.regressor.parameters())
    if args.from_scratch:  # random weights: no head-only phase, one learning rate
        freeze, backbone_lr, head_lr = 0, cfg.finetune_scratch_lr, cfg.finetune_scratch_lr
    else:
        freeze, backbone_lr, head_lr = cfg.finetune_freeze_steps, cfg.finetune_lr, cfg.finetune_head_lr
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "peak": backbone_lr, "start": freeze},
         {"params": head, "peak": head_lr, "start": 0}],
        lr=backbone_lr, weight_decay=cfg.finetune_weight_decay)
    print(f"fine-tune: backbone lr {backbone_lr:g} (after {freeze} head-only steps)  "
          f"head lr {head_lr:g}  dropout {cfg.finetune_dropout}  "
          f"weight decay {cfg.finetune_weight_decay}  eval every {cfg.finetune_eval_every}")
    scaler = make_scaler(device)
    best_rmse, best_state, evals_since_best = float("inf"), None, 0
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    started, recent = time.time(), []
    bar = tqdm(range(cfg.finetune_steps), desc="finetune", unit="step", dynamic_ncols=True)
    for step in bar:
        frozen = step < freeze
        for p in backbone:
            p.requires_grad_(not frozen)
        model.train(not frozen)  # head-only phase: embeddings without dropout
        for group in optimizer.param_groups:
            s0 = group["start"]
            group["lr"] = lr_at(step - s0, cfg.finetune_steps - s0, group["peak"],
                                cfg.finetune_warmup_steps) if step >= s0 else 0.0
        group = optimizer.param_groups[-1]  # head: shown in the progress line
        picks = rng.choice(len(train_ends), cfg.batch_size)
        batch = bars.batch(train_ends[picks], cfg.context, device)
        with autocast(device):
            loss = F.mse_loss(model(batch).float(), train_z[picks].to(device))
        recent.append(check_finite(loss, step, "finetune", args.debug, batch))
        grad_norm = backward_and_step(loss, model, optimizer, scaler)
        bar.set_postfix(loss=f"{recent[-1]:.3f}", lr=f"{group['lr']:.1e}",
                        gnorm=f"{grad_norm:.2f}", refresh=False)
        log_progress(step, cfg.finetune_steps, args.log_every, recent, group["lr"], grad_norm,
                     started, "train mse (standardised)")
        if args.debug and step < 3:
            tqdm.write(f"[debug] step {step + 1}: loss {recent[-1]:.4f}  grad norm "
                       f"{grad_norm:.3f}")

        if (step + 1) % cfg.finetune_eval_every == 0 or step + 1 == cfg.finetune_steps:
            val_rmse = rmse(val_y, predict(val_ends))
            train_rmse = y_std * math.sqrt(np.mean(recent))
            tqdm.write(f"step {step + 1:>6}/{cfg.finetune_steps}  train RMSE {train_rmse:.4f}  "
                       f"val RMSE {val_rmse:.4f}  {time.time() - started:.0f}s")
            recent = []
            if val_rmse < best_rmse:
                best_rmse, evals_since_best = val_rmse, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                evals_since_best += 1
                if evals_since_best >= cfg.patience:
                    tqdm.write("validation stopped improving; stopping early")
                    break
    bar.close()

    if best_state is None:
        raise RuntimeError("no finite validation RMSE was ever recorded; nothing to restore")
    model.load_state_dict(best_state)
    test_pred = predict(rows["test"]["g"].to_numpy())
    print(f"best val RMSE {best_rmse:.4f}  "
          f"test RMSE {rmse(rows['test']['y'].to_numpy(), test_pred):.4f}")
    print("saved", save_predictions(rows["test"], test_pred, args.target, args.horizon, name))

    # The fine-tuned model itself, so predictions can be made later (on CPU too).
    model_path = os.path.join(RUN_DIR, f"{args.target}_h{args.horizon}_{name}.pt")
    os.makedirs(RUN_DIR, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "tokenizer": tokenizer.to_dict(),
                "model": model.state_dict(),
                "head": {"target": args.target, "horizon": args.horizon,
                         "y_mean": y_mean, "y_std": y_std}}, model_path)
    print("saved", model_path, " (use: python -m models.pragma.train predict --model ...)")


# ==========================================================================
# Linear probe on frozen embeddings (the paper's main evaluation)
# ==========================================================================

def probe(args) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    device = pick_device(args.device)
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

    print(f"{describe(device)}  embedding train={len(train):,} val={len(val):,} test={len(test):,}")
    embedded_train = embed(train)
    scaler = StandardScaler().fit(embedded_train)
    x_train, x_val = scaler.transform(embedded_train), scaler.transform(embed(val))

    # Choose the ridge strength on validation, then predict the test set.
    best_rmse, best_model, best_alpha = float("inf"), None, None
    alphas = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7)
    for alpha in alphas:
        linear = Ridge(alpha=alpha).fit(x_train, train["y"].to_numpy())
        val_rmse = rmse(val["y"].to_numpy(), linear.predict(x_val))
        print(f"  ridge alpha={alpha:<8g} val RMSE {val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse, best_model, best_alpha = val_rmse, linear, alpha
    if best_alpha in (alphas[0], alphas[-1]):
        print(f"  note: best alpha {best_alpha:g} is at the edge of the grid; widen it")

    test_pred = best_model.predict(scaler.transform(embed(test)))
    print(f"test RMSE {rmse(test['y'].to_numpy(), test_pred):.4f}")
    print("saved", save_predictions(test, test_pred, args.target, args.horizon, "probe"))


# ==========================================================================
# Predictions from a fine-tuned model (works on CPU)
# ==========================================================================

def predict_command(args) -> None:
    """Predict the target for the latest bar(s) of every instrument in --data."""
    device = pick_device(args.device)
    head = torch.load(args.model, map_location="cpu", weights_only=True).get("head")
    if head is None:
        raise SystemExit(f"{args.model} is a pre-training checkpoint, not a fine-tuned model; "
                         "use one written by `finetune` (e.g. runs/pragma/vol_h30_finetune.pt)")
    model, tokenizer, cfg = load_checkpoint(args.model, device)
    series, _ = load_processed(args.data)
    bars = Bars(series, tokenizer)

    parts = []
    for instrument_id, df in series.items():
        if head["target"] == "vol" and df.attrs["tenor"] != VOL_TENOR:
            continue  # the vol model was trained on one tenor per pair
        n = len(df)
        keep = n if args.last == 0 else min(args.last, n)
        parts.append(pd.DataFrame({
            "instrument_id": instrument_id, "pair": df.attrs["pair"],
            "tenor": df.attrs["tenor"],
            "timestamp": df["timestamp"].iloc[n - keep:].to_numpy(),
            "g": bars.offset[instrument_id] + np.arange(n - keep, n),
        }))
    out = pd.concat(parts, ignore_index=True)

    z = run_in_chunks(model, bars, out["g"].to_numpy(), cfg, device, model)
    out = out.drop(columns="g")
    out["pred_log_bp"] = z * head["y_std"] + head["y_mean"]
    out["pred_bp"] = np.exp(out["pred_log_bp"])  # back from the log scale the model predicts
    out.insert(0, "target", f"{head['target']}_h{head['horizon']}")

    if args.out.endswith(".csv"):
        out.to_csv(args.out, index=False)
    else:
        out.to_parquet(args.out, index=False)
    with pd.option_context("display.width", 120, "display.max_columns", 10):
        print(out.tail(10).to_string(index=False))
    print(f"\n{len(out):,} predictions ({head['target']}, next {head['horizon']} min) "
          f"-> {args.out}")


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
        sub.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                         help="auto picks cuda if available, else cpu")
        if with_target:
            sub.add_argument("--target", choices=["vol", "spread"], default="vol")
            sub.add_argument("--horizon", type=int, default=30, help="minutes ahead to predict")

    sub = commands.add_parser("pretrain", help="masked pre-training")
    common(sub, with_target=False)
    sub.add_argument("--preset", choices=sorted(PRESETS), default="gpu")
    sub.add_argument("--steps", type=int, help="override the preset's number of steps")
    sub.add_argument("--batch-size", type=int, help="override the preset's batch size "
                                                    "(lower it if the GPU runs out of memory)")
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

    sub = commands.add_parser("predict", help="predict with a fine-tuned model (CPU is fine)")
    sub.add_argument("--model", required=True,
                     help="a fine-tuned model, e.g. runs/pragma/vol_h30_finetune.pt")
    sub.add_argument("--data", default="data/processed", help="output dir of loader.py")
    sub.add_argument("--last", type=int, default=1,
                     help="predict the last N bars of each instrument (0 = every bar)")
    sub.add_argument("--out", default=os.path.join(RUN_DIR, "predictions.csv"),
                     help=".csv or .parquet")
    sub.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    sub.set_defaults(run=predict_command)

    for name in ("pretrain", "finetune"):
        commands.choices[name].add_argument(
            "--log-every", type=int, default=100,
            help="print a progress line every N steps (0 to turn off)")
        commands.choices[name].add_argument(
            "--debug", action="store_true",
            help="anomaly detection + print shapes/grad norms for the first steps")

    args = parser.parse_args(argv)
    if getattr(args, "horizon", 1) > MAX_HORIZON or getattr(args, "horizon", 1) < 1:
        parser.error(f"--horizon must be between 1 and {MAX_HORIZON} (the split embargo)")
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
