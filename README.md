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

## Tests

```bash
python -m pytest
```

## Visualiser (world map)

```bash
python models/signal.py ... --export actions.csv   # produce model output
python frontend/export_data.py actions.csv         # writes frontend/data.json (15-min buckets; --freq to change)
python -m http.server -d frontend 8000             # open http://localhost:8000
```

Each currency pair is an arc between its two financial centres, coloured by the model score (red = sell, green = buy). Pick a tenor, scrub or play through time, and click a pair for details.
