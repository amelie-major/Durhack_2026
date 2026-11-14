"""All PRAGMA settings in one place. Pick a preset with `--preset`."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

RUN_DIR = "runs/pragma"  # checkpoints and predictions (gitignored)


@dataclass(frozen=True)
class Config:
    # --- data ---------------------------------------------------------------
    context: int = 240      # bars of history the model sees (L)
    n_buckets: int = 64     # quantile buckets per numeric field

    # --- model (PRAGMA-S width; depth moved from events to history) ----------
    d_model: int = 192
    n_heads: int = 3
    d_ffn: int = 768
    profile_layers: int = 1
    event_layers: int = 2
    history_layers: int = 4
    dropout: float = 0.1

    # --- masked pre-training (rates from the paper) --------------------------
    mask_token: float = 0.15   # single values
    mask_event: float = 0.10   # whole bars
    mask_key: float = 0.10     # one field across the whole window
    label_smoothing: float = 0.1

    # --- optimisation ---------------------------------------------------------
    batch_size: int = 256
    lr: float = 5e-4
    finetune_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    pretrain_steps: int = 15_000
    finetune_steps: int = 6_000
    eval_every: int = 500
    patience: int = 6           # evals without val improvement before stopping
    max_eval_rows: int = 30_000  # val rows used for early stopping
    max_probe_rows: int = 200_000  # train rows embedded for the linear probe
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # Real runs, on a GPU.
    "gpu": Config(),
    # Wider and deeper than "gpu" (about 2x slower per step). Same data and schedule.
    "gpu-medium": replace(
        Config(),
        d_model=256, n_heads=4, d_ffn=1024, history_layers=6,
    ),
    # Minutes on a laptop CPU. Only proves the pipeline runs end to end.
    "smoke": replace(
        Config(),
        context=32, d_model=64, n_heads=2, d_ffn=256,
        event_layers=1, history_layers=2,
        batch_size=32, warmup_steps=30,
        pretrain_steps=1500, finetune_steps=600, eval_every=100, patience=3,
        max_eval_rows=2_000, max_probe_rows=5_000,
    ),
}
