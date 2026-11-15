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

## Tests

```bash
python -m pytest
```
