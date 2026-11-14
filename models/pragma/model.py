"""A small PRAGMA-style encoder.

Three Transformer encoders, as in the paper::

    profile tokens --> Profile encoder --> [INS] vector --------.
                                                                 v
    bar tokens -----> Event encoder ---> one [EVT] vector --> History encoder
    (each bar alone)                     per bar, + calendar   (bars see each other)

The history encoder is the only place bars interact. It knows *when* each bar
happened through rotary position encoding (RoPE) on the paper's relative-time
coordinate, 8*ln(1 + seconds_before_t / 8).

Differences from the paper, all because our events are tiny and regular:
  * every bar has the same 7 fields, so bars are a plain (B*L, 8, d) tensor
    and the paper's variable-length packing is not needed;
  * each value is a single token, so the within-field position encoding
    (used for multi-token text values) would be a constant and is dropped;
  * most of the depth is in the history encoder rather than the event encoder.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import Config
from .data import N_CALENDAR
from .tokenizer import EVT, INS, Tokenizer


def rotate(x: torch.Tensor, position: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """RoPE: rotate pairs of channels of `x` by an angle proportional to `position`.

    x (B, heads, T, head_dim), position (B, T). After this, the dot product
    between a query and a key depends on the *difference* of their positions.
    """
    angle = position[:, None, :, None].float() * inv_freq  # (B, 1, T, head_dim/2)
    cos, sin = angle.cos(), angle.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class Block(nn.Module):
    """Pre-norm Transformer block (GELU), with optional RoPE in the attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.d_ffn), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(cfg.d_ffn, d))
        self.drop = nn.Dropout(cfg.dropout)

        head_dim = d // cfg.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position=None, valid=None):
        batch, length, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.n_heads, -1).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, heads, T, hd)
        if position is not None:
            q, k = rotate(q, position, self.inv_freq), rotate(k, position, self.inv_freq)
        attn_mask = None if valid is None else valid[:, None, None, :]  # True = may attend
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(attended.transpose(1, 2).reshape(batch, length, d)))
        return x + self.drop(self.ffn(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, cfg: Config, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, position=None, valid=None):
        for block in self.blocks:
            x = block(x, position, valid)
        return self.norm(x)


class Pragma(nn.Module):
    def __init__(self, cfg: Config, tokenizer: Tokenizer):
        super().__init__()
        d = cfg.d_model
        if d % cfg.n_heads or (d // cfg.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even whole number")
        self.cfg = cfg

        # One shared table for keys, values and special tokens.
        self.embed = nn.Embedding(tokenizer.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.register_buffer("event_keys", torch.as_tensor(tokenizer.event_key_ids))
        self.register_buffer("profile_keys", torch.as_tensor(tokenizer.profile_key_ids))

        self.profile_encoder = Encoder(cfg, cfg.profile_layers)
        self.event_encoder = Encoder(cfg, cfg.event_layers)
        self.history_encoder = Encoder(cfg, cfg.history_layers)
        self.calendar = nn.Sequential(nn.Linear(N_CALENDAR, d), nn.GELU(), nn.Linear(d, d))

        # Masked-token head: three views of a hidden value -> a vocabulary guess.
        self.mlm_proj = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.mlm_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))

        # Regression head for fine-tuning.
        self.regressor = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    # ------------------------------------------------------------------
    def encode(self, batch: Dict[str, torch.Tensor],
               events: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Run the three encoders. Pass `events` to override (e.g. a masked copy).

        Returns
          tokens   (B, L, 7, d) each value as seen within its own bar
          history  (B, L, d)    each bar as seen within the whole window
          record   (B, d)       the [INS] summary of the window
        """
        events = batch["events"] if events is None else events
        batch_size, length, n_keys = events.shape
        d = self.cfg.d_model

        # Profile: [INS] + (key + value) rows -> the [INS] output.
        profile_rows = self.embed(self.profile_keys) + self.embed(batch["profile"])
        ins = self.embed.weight[INS].expand(batch_size, 1, d)
        profile_vec = self.profile_encoder(torch.cat([ins, profile_rows], dim=1))[:, 0]

        # Events: every bar on its own. [EVT] + 7 (key + value) rows.
        event_rows = self.embed(self.event_keys) + self.embed(events)  # (B, L, 7, d)
        evt = self.embed.weight[EVT].expand(batch_size, length, 1, d)
        per_bar = torch.cat([evt, event_rows], dim=2).view(batch_size * length, n_keys + 1, d)
        per_bar = self.event_encoder(per_bar).view(batch_size, length, n_keys + 1, d)
        tokens = per_bar[:, :, 1:]
        bar_vec = per_bar[:, :, 0] + self.calendar(batch["calendar"])

        # History: [INS] then the bars, positioned by how long before t they were.
        sequence = torch.cat([profile_vec[:, None], bar_vec], dim=1)
        position = F.pad(batch["time_pos"], (1, 0))  # [INS] sits at position 0
        valid = F.pad(batch["valid"], (1, 0), value=True)
        out = self.history_encoder(sequence, position, valid)
        return {"tokens": tokens, "history": out[:, 1:], "record": out[:, 0]}

    def mlm_logits(self, enc: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits for the masked values only: (n_masked, vocab).

        Each guess uses the value's own bar (tokens), the bar in the context
        of the window (history) and the window summary (record), then is
        compared with the same embedding table the inputs came from.
        """
        bar, time, key = mask.nonzero(as_tuple=True)
        views = torch.cat([enc["tokens"][bar, time, key],
                           enc["history"][bar, time],
                           enc["record"][bar]], dim=-1)
        return self.mlm_proj(views) @ self.embed.weight.T + self.mlm_bias

    def embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, 2d) summary of the window: [INS] output and the last bar's output."""
        enc = self.encode(batch)
        return torch.cat([enc["record"], enc["history"][:, -1]], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression prediction, one number per window."""
        return self.regressor(self.embedding(batch)).squeeze(-1)
