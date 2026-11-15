"""Small PRAGMA-style model (arXiv 2604.08649) for 1-minute FX-forward bars.

    tokenizer.py  bars -> integer tokens (quantile buckets, fitted on train only)
    data.py       windows of the last L bars ending at minute t, and masking
    model.py      profile / event / history encoders, MLM and regression heads
    train.py      pretrain, finetune, probe
    compare.py    score PRAGMA next to XGBoost on the same test rows
"""
