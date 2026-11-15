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

Expected result: 30 instruments (27 forward, 3 spot) and 0 errors. Spot files have no forward-point columns, so the loader fills `close_mid_fp = 0` and `close_mid_all_in = close_mid`.

Use `--product forward` or `--product spot` (`-p`) to load only one kind. From Python: `load_all(paths, product="spot")`.

## More currency pairs (optional)

Challenge1 has three pairs. `fetch_dukascopy.py` downloads 1-minute spot bars with bid and ask for 18 more from Dukascopy's free datafeed (GBP, AUD, NZD, CHF, CAD, CNH, HKD, MXN, ZAR, NOK, SEK, PLN and TRY against USD, plus five crosses) and writes them in the Challenge1 spot format:

```bash
python fetch_dukascopy.py                        # all default pairs, same dates as Challenge1
python fetch_dukascopy.py --pairs GBPUSD USDMXN  # or just some
python loader.py data/raw/Challenge1.zip data/raw/dukascopy data/processed
```

The server is slow and throttles (2 to 10 seconds per file, 124 files per pair), so the full list takes a few hours. `--workers 16` is about six times faster than the default of 4. Downloads are cached in `data/raw/dukascopy/cache/`; if it stops, run it again and it carries on. To skip the download, take `dukascopy.zip` (22 MB) from the shared drive, put it in `data/raw/` and pass it to the loader in place of `data/raw/dukascopy`.

XGBoost trains on the new pairs for both targets (`python -m models.xgboost_model --target vol`); for volatility it uses a pair's 1M forward, or its spot series when the pair has no forwards. The PRAGMA and signal models still use 1M forwards only, so they ignore the new pairs.

`python fetch_dukascopy.py --validate` compares the three overlapping pairs with Challenge1. See [DATA_OVERVIEW.md](DATA_OVERVIEW.md#extra-pairs-from-dukascopy) for what matches and what doesn't.

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

A small reimplementation of Revolut's PRAGMA ([arXiv 2604.08649](https://arxiv.org/abs/2604.08649)) in `models/pragma/`. Full write-up: [PRAGMA.md](PRAGMA.md). One bar is one "event"; the model sees the last 240 bars of an instrument and is scored on exactly the same test rows as the XGBoost model. Settings live in `models/pragma/config.py`.

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
