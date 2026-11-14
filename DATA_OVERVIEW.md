# Data overview

Source: `Challenge1.zip` (put it in `data/raw/`, which is gitignored). Loaded with `loader.py`.

## What it is

1-minute bars for FX instruments, one CSV per instrument per day, with 1,543 files loaded in total. The zip contains nested zips; the loader handles them and skips macOS `._` files.

Filename format: `PAIR_DEALTCCY_PRODUCT_TENOR_NOTIONAL_YYYYMMDD.csv`, for example `EURUSD_EUR_FORWARD_1M_1000000_20260701.csv`.

## Instruments

| Pair | Product | Tenors | Days |
|---|---|---|---|
| EURUSD | FORWARD | 1W, 2W, 3W, 1M, 2M, 3M, 6M, 9M, 1Y | 53 each |
| USDJPY | FORWARD | same 9 | 53 each |
| USDSGD | FORWARD | same 9 | 45–53 each (missing days) |
| EURUSD, USDJPY, USDSGD | SPOT | SPOT | 53 each |

- Date range: 2026-07-01 to 2026-08-31.
- Notional is 1,000,000 everywhere.
- About 63k bars per EURUSD or USDJPY forward instrument.

**Spot vs forward:** spot is the price for exchange now (about 2 business days). A forward is the price for exchange at a fixed future date. Forward price = spot + forward points, which reflect the interest-rate difference between the two currencies.

## Raw columns

`timestamp` (UTC), `settlement_date`, then `{open,high,low,close}_{bid,ask,mid}`, the same 12 with an `_fp` suffix (forward points), and `{open,high,low,close}_mid_all_in`.

Most are redundant (`mid = (bid+ask)/2`, `all_in = mid + mid_fp`; verified to within 1e-6), so the loader keeps only: `open/high/low/close_mid`, `close_bid`, `close_ask`, `close_mid_fp`, `close_mid_all_in`, `settlement_date`.

Spot files have no `_fp` or `all_in` columns; the loader fills forward points with 0 and the all-in price with the mid.

## Quality findings

- Across 1.66M raw bars: 0 crossed quotes, 0 non-positive prices, 0 duplicate timestamps, 0 NaNs.
- 294 bars (about 0.02%) have high below low and are dropped.
- Consecutive days join cleanly, as expected.
- Bars per day average about 1,190, not 1,435. Weekend and holiday days are short (one has 169 bars).
- USDSGD forwards are missing 0–8 days per tenor (the files are absent, not corrupt).
- The daily rollover gap is not fixed. For EURUSD 1M it starts at 21:06 on 25 of 49 gap days and at 21:11 on 15. Flags are derived from the data, not from clock times.

## Processed output

`python loader.py data/raw data/processed` writes one parquet per instrument, `manifest.csv` (one row per file, with status, error and quality stats) and `instruments.csv`.

Derived features (all price-based ones in basis points, so pairs are comparable):

- **Time:** `dt_prev_min`, `is_gap`, `log_dt_prev` (8·ln(1+s/8)), `ret_across_gap`
- **Returns and range:** `ret_bp`, `ret_oc_bp`, `range_bp`
- **Spread and forward points:** `spread_bp`, `fp_bp`, `dfp_bp`
- **Settlement:** `is_settle_roll`, `days_to_settle`, `min_since_roll`
- **Calendar (raw ints):** `hour`, `minute_of_day`, `dow`, `dom`, `date`

## Splits

- `time_split(df, n_val_days=6, n_test_days=7, embargo_min=60)`: train, then val, then the last days as test. The last 60 minutes of train and of val are dropped so forward-looking targets can't leak.
- `split_instruments(series, holdout_pairs)`: holding out a pair removes all its tenors.

## Pre-training check: are the 9 tenors really 9 series?

Checked on the full data by aligning every tenor to the 1M tenor of the same pair on timestamp.

| Pair | Return corr vs 1M | Range corr vs 1M | Spread corr vs 1M | Mean spread (1W → 1Y) |
|---|---|---|---|---|
| EURUSD | 0.983–0.991 | 0.994–0.998 | 0.913–0.992 | 0.24 → 0.50 bp |
| USDJPY | 0.995–0.998 | 0.998–0.999 | 0.905–0.983 | 0.29 → 0.48 bp |
| USDSGD | 0.955–0.964 | 0.967–0.982 | 0.879–0.978 | 0.50 → 0.77 bp |

- **Volatility is almost one series per pair.** Std of 1-minute returns differs by under 1% across tenors (EURUSD 0.798–0.805 bp), and return/range correlations are 0.96–1.00. The tenors are not byte-identical (the mid differs by up to 3 bp for EURUSD, 7 bp for USDJPY and 10 bp for USDSGD), but the volatility targets are effectively the same.
- **Spread is different.** The level rises with tenor and the correlation with 1M is lower (down to 0.88), so spread carries real tenor-specific information.
- **Forward points are tenor-specific**, as expected (`fp_bp` for EURUSD runs from +2.8 bp at 1W to +146 bp at 1Y).
- **Consequence:** treat the data as 3 price series for volatility. An unseen-tenor test is not a meaningful volatility test, so holding out a pair (`split_instruments`) is the honest generalisation test. Unseen tenor remains a fair test for spread and forward points.
