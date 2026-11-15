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
python fetch_dukascopy.py --workers 16           # all default pairs, same dates as Challenge1
python fetch_dukascopy.py --pairs GBPUSD USDMXN  # or just some
```

The server is slow and throttles (2 to 10 seconds per file, 124 files per pair), so the full list takes about 2.5 hours. Downloads are cached in `data/raw/dukascopy/cache/`; if it stops, run it again and it carries on. To skip the download, take `dukascopy.zip` (23 MB) from the shared drive and put it in `data/raw/`.

To use both feeds together, give the loader both:

```bash
python loader.py data/raw/Challenge1.zip data/raw/dukascopy.zip data/processed
```

That writes 48 instruments. Each has a `feed` (`challenge1` or `dukascopy`) in `df.attrs`, and Dukascopy's `spread_bp` is rescaled to Challenge1's level (the quoted value is kept in `spread_bp_raw`). XGBoost trains on the combined folder for both targets; the PRAGMA and signal models still use 1M forwards only for volatility and direction, so they ignore the new pairs there. Details and caveats are in [DATA_OVERVIEW.md](DATA_OVERVIEW.md#combining-the-two-feeds).

## Use it from Python

From a script or notebook in the repo root:

```python
from loader import load_processed, time_split, split_instruments

series, manifest = load_processed("data/processed")    # {instrument_id: DataFrame}
train, val, test = time_split(series["EURUSD_EUR_FORWARD_1M_1000000"])
pretrain_ids, holdout_ids = split_instruments(series, holdout_pairs=["USDSGD"])
```

## Run the website with live forecasts

Two terminals, both started from the repo root:

```bash
python -m live.server                # XGBoost forecasts on live Yahoo bars, http://localhost:8000
cd frontend && npm ci && npm run dev # website, http://localhost:5173
```

The first `live.server` run trains the models from `data/processed` (about 3 minutes) and caches them in
`data/live_forecast_models.pkl`. Re-run `loader.py` first if `data/processed` is older than the loader:
the models need its `vol_15` / `vol_60` columns.

**Replay last day** (button at the bottom of the site) plays back the last 24 hours of Yahoo bars. The
server answers `/api/snapshot` and `/api/history` with `&asOf=<time>` using only bars at or before that
moment, so every number is what the model said then. Useful when the market is closed.

### Choose the model: XGBoost or PRAGMA

The site has an `XGBoost | PRAGMA` switch in the top bar. It appears once the server has PRAGMA's saved
predictions, which are downloaded once from the release:

```bash
gh release download pragma-large-run1 -p '*_h30_scratch.parquet' -D runs/pragma
python -m live.server                 # XGBoost by default, PRAGMA on the switch
python -m live.server --model pragma  # start on PRAGMA instead
```

The API takes `&model=xgboost` or `&model=pragma` on `/api/snapshot`, `/api/history` and
`/api/replay-range`; `/api/health` lists what is available.

The server does not run PRAGMA: the model is too big for a laptop, and the release holds the pre-trained
backbone, not a forecasting model. It serves the predictions PRAGMA made in its training run, so:

- PRAGMA is a replay of the challenge data's test week (24-31 Aug 2026), not live prices. Choosing it
  starts the replay; "Back to live" returns to XGBoost.
- Only 30-minute forecasts were saved. Shorter horizons are derived from them (vol scaled by
  sqrt(h / 30), spread kept the same).
- There is no Buy / Sell signal in PRAGMA mode.
- The two models use different data. XGBoost trains on all of `data/processed` (Challenge1 plus
  Dukascopy pairs). PRAGMA covers only the three Challenge1 pairs it was trained on (EURUSD, USDJPY,
  USDSGD 1M forwards). By default PRAGMA's bars also come from `data/processed`, whose Challenge1
  instruments are unchanged by the Dukascopy feed. To read them from a Challenge1-only folder, use
  `--pragma-data path/to/processed`. The server refuses predictions that don't match the bars.

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

## Hybrid model: XGBoost + PRAGMA embeddings

`models/pragma/hybrid.py` gives XGBoost its usual features plus 32 PCA components of the frozen PRAGMA embedding of each row's window. It is our best spread model (test RMSE in log bp, h=30, lower is better):

| target | hybrid | XGBoost alone | PRAGMA fine-tuned |
|---|---|---|---|
| spread | **0.1387** | 0.1403 | 0.1663 |
| vol | 0.2853 | **0.2840** | 0.2978 |

### Train and export (NCC, GPU)

```bash
CHECKPOINT=runs/pragma/pretrain_large.pt sbatch scripts/ncc_hybrid.sh
```

For each target this writes the test predictions (`runs/pragma/<target>_h30_hybrid.parquet`, read by `compare.py`) and a self-contained model folder `runs/pragma/hybrid_<target>_h30/`:

| file | contents |
|---|---|
| `pragma.pt` | copy of the pre-trained PRAGMA checkpoint (82 MB) |
| `pca.npz` | PCA `mean` and `components` as plain arrays |
| `xgb.json` | the XGBoost model |
| `meta.json` | target, horizon, feature and embedding column names, pair categories, test scores |

The trained folders are committed on the `models/hybrid-h30` branch under `exported_models/` (`git checkout models/hybrid-h30`).

### Predict (any laptop, CPU is fine)

After the data loader has built `data/processed/`:

```bash
python -m models.pragma.hybrid_predict --model exported_models/hybrid_spread_h30 \
    --data data/processed --last 1 --out spread_now.csv --device cpu
```

`--last N` predicts at each instrument's last N bars (vol uses only the 1M tenor, as in training). The output (`.csv` or `.parquet`) has `target, instrument_id, pair, tenor, timestamp, pred_log_bp, pred_bp`; `pred_bp` is the forecast for the next 30 minutes in bp (spread: mean spread; vol: realised volatility). Use `--model exported_models/hybrid_vol_h30` for volatility. Progress is printed with timings; a run takes about 10 s on a laptop.

Predictions are only as fresh as `data/processed/`: for live forecasts, re-run the data loader on the newest bars first.

**macOS: if it hangs** with no new progress line for a minute, torch and XGBoost are probably clashing over OpenMP (their parallel-processing library). Run with:

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python -m models.pragma.hybrid_predict ...
```
