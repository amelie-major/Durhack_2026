# Durhack 2026

Models for 1-minute FX-forward bars. See [DATA_OVERVIEW.md](DATA_OVERVIEW.md) for the data and [CONTRIBUTING.md](CONTRIBUTING.md) for how we work.

## Setup

You need Python 3.9 or newer. With conda (or miniforge):

```bash
conda create -n fxfm python=3.12 -y
conda activate fxfm
python -m pip install -r requirements.txt
```

Always use `python -m pip`, not bare `pip`. On some machines `pip` points at an older Python, and the install fails with `No matching distribution found for pandas>=1.5`.

## Get the data

Put `Challenge1.zip` in `data/raw/`. Don't unzip it. The `data/` folder is gitignored, so never commit it.

## Run the data loader

From the repo root:

```bash
python loader.py data/raw/Challenge1.zip data/processed
```

The last argument is the output folder. Everything before it is input, and an input can be a zip, a folder or a CSV. Add `-q` to hide the per-instrument summary.

It writes the following to `data/processed/`:

- `<instrument_id>.parquet`: one file per instrument, with cleaned bars and features.
- `instruments.csv`: one row per instrument, with day and bar counts and the date range.
- `manifest.csv`: one row per input file, with its status, quality stats and any error.

Expected result: 27 forward instruments and 159 errors. The errors are all the SPOT files, which the loader can't read yet because they have no forward-point columns.

## Use it from Python

From a script or notebook in the repo root:

```python
from loader import load_processed, time_split, split_instruments

series, manifest = load_processed("data/processed")    # {instrument_id: DataFrame}
train, val, test = time_split(series["EURUSD_EUR_FORWARD_1M_1000000"])
pretrain_ids, holdout_ids = split_instruments(series, holdout_pairs=["USDSGD"])
```

## Tests

```bash
python -m pytest
```

## PRAGMA-style model

A small reimplementation of Revolut's PRAGMA ([arXiv 2604.08649](https://arxiv.org/abs/2604.08649)) in `models/pragma/`. One bar is one "event"; the model sees the last 240 bars of an instrument and is scored on exactly the same test rows as the XGBoost model. Settings live in `models/pragma/config.py`.

Run on a GPU machine (Colab, NCC) from the repo root, after the data loader:

```bash
python -m models.pragma.train pretrain --preset gpu     # masked pre-training, writes runs/pragma/pretrain.pt
python -m models.pragma.train finetune --target vol     # fine-tune a regression head
python -m models.pragma.train probe    --target vol     # linear model on frozen embeddings
python -m models.pragma.train finetune --target vol --from-scratch   # optional: no pre-training
python -m models.pragma.compare        --target vol     # table: baseline / XGBoost / PRAGMA
```

Repeat the last four with `--target spread`. Use `--preset smoke` on `pretrain` for a few-minute CPU run that only checks the pipeline works; its scores are not meaningful. `runs/` is gitignored.

On NCC, `sbatch scripts/ncc_pragma.sh` runs all of the above for both targets in one job (copy `data/processed/` over first; see the top of the script).
