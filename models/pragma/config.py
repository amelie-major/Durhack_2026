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
    weight_decay: float = 0.01
    warmup_steps: int = 500
    pretrain_steps: int = 15_000
    eval_every: int = 500
    max_eval_rows: int = 30_000  # val rows used for early stopping
    max_probe_rows: int = 200_000  # train rows embedded for the linear probe
    seed: int = 0

    # --- fine-tuning (set here, not read from the checkpoint; see FINETUNE_FIELDS) ---
    # The 20M model overfit 137k volatility rows (train RMSE 0.07, val 0.32), so the
    # fine-tune is gentle: head-only warm-up, a small backbone LR, more dropout and
    # weight decay, and frequent early stopping.
    finetune_steps: int = 3_000
    finetune_lr: float = 3e-5          # pre-trained layers
    finetune_head_lr: float = 3e-4     # the new regression head
    finetune_scratch_lr: float = 2e-4  # --from-scratch: random weights need a larger LR
    finetune_freeze_steps: int = 300   # train only the head for this long first (LP-FT)
    finetune_warmup_steps: int = 100
    finetune_weight_decay: float = 0.05
    finetune_dropout: float = 0.2
    finetune_eval_every: int = 100
    patience: int = 8           # fine-tune evals without val improvement before stopping

    def to_dict(self) -> dict:
        return asdict(self)


# Settings that belong to the training recipe, not the model. A checkpoint
# remembers the recipe it was pre-trained with; these are re-read from the
# matching preset below so fine-tuning changes apply to old checkpoints too.
FINETUNE_FIELDS = (
    "finetune_steps", "finetune_lr", "finetune_head_lr", "finetune_scratch_lr",
    "finetune_freeze_steps", "finetune_warmup_steps", "finetune_weight_decay",
    "finetune_dropout", "finetune_eval_every", "patience", "max_eval_rows", "max_probe_rows",
)
ARCHITECTURE_FIELDS = (
    "context", "n_buckets", "d_model", "n_heads", "d_ffn", "profile_layers",
    "event_layers", "history_layers",
)


def with_current_finetune_settings(cfg: "Config") -> "Config":
    """Give `cfg` the fine-tuning settings of the preset with the same architecture."""
    for preset in PRESETS.values():
        if all(getattr(preset, f) == getattr(cfg, f) for f in ARCHITECTURE_FIELDS):
            return replace(cfg, **{f: getattr(preset, f) for f in FINETUNE_FIELDS})
    return cfg  # custom architecture: keep what the checkpoint says


PRESETS = {
    # Real runs, on a GPU.
    "gpu": Config(),
    # Free Colab GPU (T4): half the history and about half the steps, so
    # pre-training costs roughly a quarter of "gpu". Expect slightly worse scores.
    "colab": replace(Config(), context=120, pretrain_steps=8_000, finetune_steps=4_000),
    # Wider and deeper than "gpu" (about 2x slower per step). Same data and schedule.
    "gpu-medium": replace(
        Config(),
        d_model=256, n_heads=4, d_ffn=1024, history_layers=6,
    ),
    # Bigger again: for an A100, with fewer steps (pass --steps / PRETRAIN_STEPS).
    "gpu-large": replace(
        Config(),
        d_model=384, n_heads=6, d_ffn=1536, history_layers=8,
    ),
    # Minutes on a laptop CPU. Only proves the pipeline runs end to end.
    "smoke": replace(
        Config(),
        context=32, d_model=64, n_heads=2, d_ffn=256,
        event_layers=1, history_layers=2,
        batch_size=32, warmup_steps=30,
        pretrain_steps=1500, eval_every=100, patience=3,
        finetune_steps=600, finetune_freeze_steps=50, finetune_warmup_steps=30,
        finetune_eval_every=100,
        # the tiny model does not overfit, so it keeps the original, bolder recipe
        finetune_lr=2e-4, finetune_head_lr=2e-4, finetune_weight_decay=0.01,
        finetune_dropout=0.1,
        max_eval_rows=2_000, max_probe_rows=5_000,
    ),
}
