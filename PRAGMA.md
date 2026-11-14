# PRAGMA-style model

A small reimplementation of Revolut's PRAGMA foundation model ([arXiv 2604.08649](https://arxiv.org/abs/2604.08649)) for our 1-minute FX-forward bars, scored against the sample XGBoost model on the same rows. Revolut released no code or weights, so everything in `models/pragma/` is written from the paper.

See [DATA_OVERVIEW.md](DATA_OVERVIEW.md) for the data and `data/xgboost_inputs.md` for the XGBoost features.

## The idea in one paragraph

PRAGMA reads a customer's history as a sequence of *events*, each a timestamp plus key-value pairs. It is first **pre-trained** without labels: some values are hidden and the model learns to fill them in from the rest of the history. The trained model then turns any history into a vector (an *embedding*) that is reused for real tasks, either by fitting a simple linear model on it (a *probe*) or by training the whole network a little further (*fine-tuning*). XGBoost, by contrast, sees one row at a time and relies on rolling features we design by hand.

## How PRAGMA maps to FX bars

| PRAGMA concept | In the paper | Here | Why |
|---|---|---|---|
| Record | one user's history at an evaluation point | one instrument's last `L` bars ending at minute `t` | `t` is exactly one XGBoost row |
| Event | a transaction or app event | one 1-minute bar | gaps simply become larger time differences |
| Event keys | Direction, Amount, MCC, ... | `ret_bp`, `ret_oc_bp`, `range_bp`, `spread_bp`, `dfp_bp`, `is_gap`, `is_settle_roll` | basis-point columns are comparable across pairs; raw prices are not |
| Numeric values | percentile-bucket tokens, plus a bucket for zero | 64 quantile buckets per field plus a zero bucket | as in the paper |
| Profile state `[USR]` | plan, region, balance quantile | `[INS]`: `pair`, `tenor`, bucketed `days_to_settle` and `fp_bp` at `t` | slow-moving facts about the instrument |
| Relative time | `8*ln(1 + seconds/8)` back from the latest event, as a rotary position | the same, back from bar `t` | a weekend gap is just a jump in position; no special case |
| Calendar | hour, day of week, day of month | time of day, day of week | day of month would be memorised from two months of data |
| Pre-training | hide 15% of tokens, 10% of whole events, 10% of keys | the same rates | hiding whole bars forces the model to use history |
| Adaptation | linear probe; LoRA fine-tune | Ridge probe; full fine-tune with a regression head | the model is small, so LoRA saves nothing |

**Tokens.** Every value becomes one integer token; the model input for a value is `E(key) + E(value)` from one shared embedding table. The vocabulary has 95 tokens: 5 special (`[PAD] [MASK] [UNK] [INS] [EVT]`), 11 keys, 65 bucket tokens shared by all numeric fields (a bucket token means "this rank within its own field"; the key says which field), 2 flag values, 3 pairs and 9 tenors.

## Architecture

```
profile tokens ──> Profile encoder ──> [INS] vector ─────────┐
                                                             v
bar tokens ─────> Event encoder ───> one [EVT] vector ──> History encoder ──> window embedding
(each bar alone)                     per bar, + calendar   (bars see each other,
                                                            placed by relative time)
```

- **Event encoder** turns the 7 fields of one bar into one vector. Bars do not see each other here.
- **History encoder** is the only place bars interact. It knows when each bar happened through rotary position encoding on the relative-time coordinate.
- **Masked-token head** (pre-training) guesses a hidden value from three views: the value's own bar, that bar in the context of the window, and the window summary.
- **Regression head** (fine-tuning) predicts the target from the `[INS]` output joined with the last bar's output. The probe uses the same joined vector.

| Preset | Parameters | `d_model` | Heads | Layers (profile / event / history) | Context | Use |
|---|---|---|---|---|---|---|
| `gpu` | 3.36M | 192 | 3 | 1 / 2 / 4 | 240 bars | real runs |
| `smoke` | 0.23M | 64 | 2 | 1 / 1 / 2 | 32 bars | checking the pipeline on a laptop |

### Where this differs from the paper

1. **Depth is in the history encoder.** PRAGMA-S uses 5 event layers and 2 history layers because banking events are long and varied. Our bars are 7 fixed fields and the signal is across time, so the split is reversed.
2. **No variable-length packing.** Every bar has the same shape, so bars are a plain tensor.
3. **No text tokens or within-field positions.** We have no text fields and every value is a single token.
4. **Full fine-tuning instead of LoRA.** At 3M parameters there is nothing to save.
5. **Extra time-of-day waves** (24h, 12h, 6h, 3h) so sharp events such as the 21:00 rollover are easy to locate. The paper uses one wave per calendar cycle.
6. **Optimiser is AdamW.** The paper combines Muon and AdamW but gives no details.

## Why the comparison is fair

- **Same targets.** `train.py` calls `make_target` from `models/xgboost_model.py`: log of next-`H`-minute realised volatility, or log of mean spread.
- **Same rows.** `split_rows` in `models/pragma/data.py` repeats the steps of `build_dataset` (same instruments, same `time_split` with its 60-minute embargo, same dropped rows, same order). `compare.py` refuses a prediction file whose timestamps, pairs or targets differ from XGBoost's test rows, and `tests/test_pragma.py` checks it too.
- **No look-ahead.** A window ends at `t` and only reaches backwards. A test window may include earlier validation or training bars; that is past data the model would really have.
- **Nothing fitted on the test period.** Bucket edges and pre-training use training bars only.
- **Volatility uses the 1M tenor only**, as XGBoost does, because the nine tenors of a pair are nearly the same price series.

One difference to keep in mind: pre-training uses all 27 instruments, including tenors the volatility XGBoost model never sees. That is extra unlabelled data, not extra labels.

## Running it

All commands run from the repo root after the data loader has written `data/processed/`.

### On a laptop (pipeline check only)

```bash
python -m models.pragma.train pretrain --preset smoke
python -m models.pragma.train finetune --target vol
python -m models.pragma.train probe    --target vol
python -m models.pragma.compare        --target vol
```

This takes a few minutes on CPU. The scores are not meaningful.

### On NCC

```bash
# once, from your laptop: data/ is gitignored, so copy it
rsync -av data/processed/ <user>@ncc1.clients.dur.ac.uk:<repo path>/data/processed/

# on NCC, in the repo root
sbatch scripts/ncc_pragma.sh
squeue -u $USER
tail -f slurm-<jobid>.out
```

The job creates `.venv-ncc` on first use, pre-trains, then for each target runs fine-tune, probe, a from-scratch run and the comparison. Options are environment variables set before `sbatch`:

| Variable | Default | Meaning |
|---|---|---|
| `TARGETS` | `vol spread` | which targets to run |
| `HORIZON` | `30` | minutes ahead to predict (at most 60, the split embargo) |
| `SKIP_PRETRAIN` | `0` | set to `1` to reuse `runs/pragma/pretrain.pt` |

NCC requires a GPU type in the request. The script asks for `ampere`; override with `sbatch --gres=gpu:turing:1 scripts/ncc_pragma.sh` if none are free. If the job runs out of GPU memory, halve `batch_size` in `models/pragma/config.py`.

### Individual commands

| Command | What it does | Writes |
|---|---|---|
| `train pretrain --preset gpu` | masked pre-training on all instruments | `runs/pragma/pretrain.pt` (every 500 steps) |
| `train finetune --target vol` | trains the whole model on the target, starting from the checkpoint | `runs/pragma/vol_h30_finetune.parquet` |
| `train probe --target vol` | Ridge regression on frozen embeddings | `runs/pragma/vol_h30_probe.parquet` |
| `train finetune --target vol --from-scratch --preset gpu` | same model, random starting weights | `runs/pragma/vol_h30_scratch.parquet` |
| `compare --target vol` | trains XGBoost, prints the table | nothing |

Each is run as `python -m models.pragma.<command>`. Add `--horizon N` to change the horizon and `--steps N` to override the number of training steps. `runs/` is gitignored.

## Reading the output

**Pre-training** prints a line like `no-context loss to beat: 2.980`. That is the loss of always guessing each field's overall frequencies. The `val` loss should fall clearly below it; if it does not, the model has learned nothing from context. The `train` number is higher than `val` because it includes label smoothing, so compare `val` with the no-context line, not `train` with `val`.

**Fine-tuning** prints validation RMSE every 500 steps and stops early when it stops improving. The best weights are used for the test predictions.

**The comparison table** (target in log basis points; lower RMSE is better):

| Row | What it tells you |
|---|---|
| `baseline` | "the next hour looks like the last hour". Anything useful must beat this. |
| `xgboost` | the hand-built-features model. The number to beat. |
| `pragma finetune` | the best PRAGMA can do on this task. |
| `pragma probe` | how much of the answer is already in the pre-trained embedding, with no task-specific training of the network. |
| `pragma scratch` | the same network without pre-training. If `finetune` is not better than this, pre-training is not helping yet. |

## Settings

Everything is in `models/pragma/config.py`.

| Field | Default (`gpu`) | Change it when |
|---|---|---|
| `context` | 240 | you want the model to see more or less history. Memory and time grow with it. |
| `n_buckets` | 64 | values need more precision (first thing to try if spread lags). Needs a new pre-training run. |
| `d_model`, `n_heads`, `d_ffn`, `*_layers` | 192, 3, 768, 1/2/4 | changing model size. `d_model / n_heads` must be even. |
| `mask_token`, `mask_event`, `mask_key` | 0.15, 0.10, 0.10 | experimenting with the pre-training task. |
| `batch_size` | 256 | the GPU runs out of memory (halve it). |
| `lr`, `finetune_lr`, `warmup_steps` | 5e-4, 2e-4, 500 | training is unstable or too slow. |
| `pretrain_steps`, `finetune_steps` | 15,000, 6,000 | the losses are still falling at the end. |
| `eval_every`, `patience` | 500, 6 | early stopping triggers too soon or too late. |
| `max_eval_rows`, `max_probe_rows` | 30,000, 200,000 | trading evaluation speed for precision. |

A checkpoint stores its own config and tokenizer, so `finetune` and `probe` always rebuild the model that was pre-trained, whatever the file says now.

## Results

### NCC run (`gpu` preset)

Not recorded yet. Paste the tables from the end of `slurm-<jobid>.out` here when the job finishes.

### Laptop smoke run (not a benchmark)

`smoke` preset, volatility at 30 minutes, 24,832 test rows. The model is 0.23M parameters with a 32-bar context and a few minutes of training, while XGBoost uses 60-minute features, so this only shows the pipeline works.

| Model | RMSE | R² |
|---|---|---|
| baseline | 0.3678 | 0.343 |
| xgboost | 0.2840 | 0.608 |
| pragma finetune | 0.3183 | 0.508 |
| pragma probe | 0.3253 | 0.486 |
| pragma scratch | 0.3164 | 0.513 |

Pre-training validation loss reached 2.13 against a no-context level of 2.98.

## Limitations

- **Little data for a foundation model.** 1.66M bars, and for volatility effectively three price series. PRAGMA was trained on 24 billion events.
- **Bucketing loses precision.** The model sees which of 64 ranks a value falls in, not the value. XGBoost sees the exact current spread, its best spread feature.
- **Short test period.** About six trading days in late August, a quiet time for FX. Small differences between models may be noise; nothing here has been run with more than one seed.
- **Each instrument is read alone.** The model cannot look at another tenor or pair at the same minute.
- **SPOT files are not used**, because the loader does not read them yet.

## Next experiments

Ranked by expected payoff for the effort. None is implemented.

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | PRAGMA embeddings as extra XGBoost features | The paper notes boosted trees work well on its embeddings. Combines learned sequence features with the rolling features. | Small |
| 2 | Exact values alongside buckets | Add a projection of the scaled raw value to `E(key) + E(value)` so precision is not lost. Departs from the paper. | Small to medium |
| 3 | Hold out a whole pair (USDSGD) | Pre-train on two pairs and score on an unseen one, using `loader.split_instruments`. The honest foundation-model test. | Small |
| 4 | Forecast-shaped masking | Also hide the last few bars of the window in pre-training, so the pre-training task resembles forecasting. | Small |
| 5 | One model, several outputs | Predict volatility and spread, at several horizons, from one fine-tune. | Medium |
| 6 | Tuning | `n_buckets` 128, `context` 480, more pre-training steps, several seeds for error bars. | Small, costs GPU time |
| 7 | SPOT files for pre-training | Three more series; needs the loader to accept files without forward points. | Medium |

Which to try first depends on the NCC table:

- `finetune` no better than `scratch`: pre-training is not adding anything yet, so try 4 and 6.
- Close to XGBoost on volatility but behind on spread: try 2.
- Behind everywhere: 1 and 3 are the most useful regardless.

## Files

| File | Contents |
|---|---|
| `models/pragma/config.py` | settings and the `gpu` / `smoke` presets |
| `models/pragma/tokenizer.py` | bars to tokens; bucket edges fitted on training bars |
| `models/pragma/data.py` | windows ending at `t`, train/val/test rows, masking |
| `models/pragma/model.py` | encoders, rotary attention, masked-token and regression heads |
| `models/pragma/train.py` | `pretrain`, `finetune`, `probe` |
| `models/pragma/compare.py` | the table against XGBoost and the baseline |
| `scripts/ncc_pragma.sh` | SLURM job for NCC |
| `tests/test_pragma.py` | tokens, no look-ahead, row parity with XGBoost, shapes |
