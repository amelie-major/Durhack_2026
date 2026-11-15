"""Evaluate XGBoost FX models and generate comprehensive performance reports.

Evaluates Volatility (vol) and Spread (spread) models across prediction horizons,
reporting:
  - Regression metrics in model log-space (RMSE, MAE, MedAE, R2, Pearson r)
  - Physical basis points (bp) metrics (RMSE_bp, MAE_bp, target_mean_bp)
  - Baseline comparison against trailing 60-min benchmark (% improvement)
  - Directional accuracy (expansion vs contraction)
  - Performance by currency pair and tenor
  - Volatility regime error breakdown (quiet vs spikes)
  - Feature importances
  - Optional walk-forward cross-validation folds
  - Optional Buy/Sell direction model evaluation

Usage:
    python scripts/evaluate_xgboost.py
    python scripts/evaluate_xgboost.py --target both --horizons 5,15,30
    python scripts/evaluate_xgboost.py --cv 6 --cv-days 5
    python scripts/evaluate_xgboost.py --no-plots
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from loader import load_processed
from models.metrics import (
    calc_classification_metrics,
    calc_directional_accuracy,
    calc_physical_bp_metrics,
    calc_regime_metrics,
    calc_regression_metrics,
)
from models.signal import DIRECTION_FEATURES, direction_dataset, train_direction
from models.xgboost_model import (
    FEATURES,
    build_dataset,
    train,
    walk_forward,
)


def evaluate_target_horizon(
    series: Dict[str, pd.DataFrame],
    target: str,
    horizon: int,
    learning_rate: float = 0.02,
    max_trees: int = 4000,
    patience: int = 100,
    n_jobs: int = -1,
    plot_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train on train/val splits and evaluate in-depth on the test split.

    With `plot_dir`, charts for this target/horizon are written there and their
    paths returned under "figures".
    """
    t0 = time.time()
    train_df, val_df, test_df = build_dataset(series, target, horizon)
    prep_time = time.time() - t0

    t1 = time.time()
    model = train(
        train_df,
        val_df,
        learning_rate=learning_rate,
        max_trees=max_trees,
        patience=patience,
        n_jobs=n_jobs,
    )
    train_time = time.time() - t1
    train_rows, val_rows = len(train_df), len(val_df)
    del train_df, val_df  # only the test split is needed from here on

    y_test = test_df["y"].to_numpy()
    pred_test = model.predict(test_df[FEATURES])
    baseline_test = test_df["baseline"].to_numpy()
    y_last = test_df["y_last"].to_numpy() if "y_last" in test_df else baseline_test

    # Overall log-space metrics
    reg_metrics = calc_regression_metrics(y_test, pred_test, baseline_test)

    # Physical basis points metrics
    bp_metrics = calc_physical_bp_metrics(y_test, pred_test, baseline_test)

    # Directional change accuracy
    dir_acc = calc_directional_accuracy(y_test, pred_test, y_last)

    # Regime breakdown
    regimes = calc_regime_metrics(y_test, pred_test, n_bins=5)

    # Per-pair breakdown
    pair_metrics: List[Dict[str, Any]] = []
    for pair, rows in test_df.groupby("pair", observed=True):
        idx = rows.index
        y_p = rows["y"].to_numpy()
        pred_p = pred_test[idx]
        base_p = rows["baseline"].to_numpy()
        pm = calc_regression_metrics(y_p, pred_p, base_p)
        pm_bp = calc_physical_bp_metrics(y_p, pred_p, base_p)
        pair_metrics.append({
            "pair": str(pair),
            "rows": len(rows),
            "rmse": pm["rmse"],
            "mae": pm["mae"],
            "r2": pm["r2"],
            "baseline_rmse": pm.get("baseline_rmse", float("nan")),
            "improvement_pct": pm.get("rmse_improvement_pct", 0.0),
            "rmse_bp": pm_bp["rmse_bp"],
            "mae_bp": pm_bp["mae_bp"],
            "improvement_pct_bp": pm_bp.get("rmse_improvement_pct_bp", 0.0),
        })

    # Per-tenor breakdown (spread target has meaningful tenors)
    tenor_metrics: List[Dict[str, Any]] = []
    if "tenor_days" in test_df and target == "spread":
        for tenor, rows in test_df.groupby("tenor_days", observed=True):
            idx = rows.index
            y_t = rows["y"].to_numpy()
            pred_t = pred_test[idx]
            base_t = rows["baseline"].to_numpy()
            tm = calc_regression_metrics(y_t, pred_t, base_t)
            tm_bp = calc_physical_bp_metrics(y_t, pred_t, base_t)
            tenor_metrics.append({
                "tenor_days": int(tenor) if pd.notna(tenor) else "SPOT",
                "rows": len(rows),
                "rmse": tm["rmse"],
                "mae": tm["mae"],
                "r2": tm["r2"],
                "rmse_bp": tm_bp["rmse_bp"],
                "mae_bp": tm_bp["mae_bp"],
                "improvement_pct": tm.get("rmse_improvement_pct", 0.0),
            })

    # Feature importance
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    total_gain = sum(gain_scores.values()) or 1.0

    features_ranking = []
    for f in sorted(gain_scores, key=gain_scores.get, reverse=True)[:15]:
        features_ranking.append({
            "feature": f,
            "gain_share": gain_scores[f] / total_gain,
            "splits": weight_scores.get(f, 0),
        })

    result = {
        "target": target,
        "horizon": horizon,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": len(test_df),
        "best_iteration": int(model.best_iteration + 1),
        "prep_seconds": round(prep_time, 2),
        "train_seconds": round(train_time, 2),
        "overall_log": reg_metrics,
        "overall_bp": bp_metrics,
        "directional_accuracy": dir_acc,
        "regimes": regimes,
        "pairs": pair_metrics,
        "tenors": tenor_metrics,
        "feature_importance": features_ranking,
        "figures": {},
    }
    if plot_dir is not None:
        result["figures"] = plot_results(result, test_df, pred_test, plot_dir)
    return result


# ==========================================================================
# Charts
# ==========================================================================

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_results(
    res: Dict[str, Any],
    test_df: pd.DataFrame,
    pred: np.ndarray,
    out_dir: Path,
) -> Dict[str, str]:
    """Write the per-target/horizon charts as PNGs; returns {kind: path}."""
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{res['target']}_{res['horizon']}m"
    title = f"{res['target'].upper()} {res['horizon']}m"
    y = test_df["y"].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    figures: Dict[str, str] = {}

    def save(fig, kind: str) -> None:
        path = out_dir / f"{tag}_{kind}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        figures[kind] = str(path)

    # Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hb = ax.hexbin(y[ok], pred[ok], gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lo = float(np.nanmin([y[ok].min(), pred[ok].min()]))
    hi = float(np.nanmax([y[ok].max(), pred[ok].max()]))
    ax.plot([lo, hi], [lo, hi], color="crimson", lw=1, ls="--", label="perfect forecast")
    ax.set_xlabel("actual (log bp)")
    ax.set_ylabel("predicted (log bp)")
    ax.set_title(f"{title}: predicted vs actual (R² {res['overall_log']['r2']:.3f})")
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="rows (log scale)")
    save(fig, "pred_vs_actual")

    # Residuals by regime (quintiles of the actual value)
    resid = (y - pred)[ok]
    edges = np.quantile(y[ok], np.linspace(0, 1, 6))
    bins = np.clip(np.searchsorted(edges, y[ok], side="right") - 1, 0, 4)
    labels = ["Calm", "Low", "Moderate", "High", "Spikes"]
    groups = [resid[bins == i] for i in range(5)]
    keep = [i for i, g in enumerate(groups) if len(g)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([groups[i] for i in keep], showfliers=False)
    ax.set_xticks(range(1, len(keep) + 1), [labels[i] for i in keep])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("residual: actual − predicted (log bp)")
    ax.set_title(f"{title}: error by market regime")
    save(fig, "residuals_by_regime")

    # Improvement over the baseline per pair
    pairs = sorted(res["pairs"], key=lambda p: p["improvement_pct"])
    if pairs:
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(pairs) + 1.5))
        vals = [p["improvement_pct"] for p in pairs]
        ax.barh([p["pair"] for p in pairs], vals,
                color=["seagreen" if v >= 0 else "indianred" for v in vals])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("RMSE improvement vs 60-min baseline (%)")
        ax.set_title(f"{title}: improvement by pair")
        save(fig, "pair_improvement")

    # Feature importance
    feats = res["feature_importance"][::-1]
    if feats:
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(feats) + 1.5))
        ax.barh([f["feature"] for f in feats], [100 * f["gain_share"] for f in feats],
                color="steelblue")
        ax.set_xlabel("share of total gain (%)")
        ax.set_title(f"{title}: top features")
        save(fig, "feature_importance")

    return figures


def plot_summary(eval_results: List[Dict[str, Any]], out_dir: Path) -> Optional[str]:
    """RMSE improvement over the baseline against horizon, one line per target."""
    if not eval_results:
        return None
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for target in sorted({r["target"] for r in eval_results}):
        rows = sorted((r for r in eval_results if r["target"] == target), key=lambda r: r["horizon"])
        ax.plot([r["horizon"] for r in rows],
                [r["overall_log"].get("rmse_improvement_pct", 0.0) for r in rows],
                marker="o", label=target.upper())
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("horizon (minutes)")
    ax.set_ylabel("RMSE improvement vs baseline (%)")
    ax.set_title("XGBoost improvement over the trailing-60-min baseline")
    ax.legend()
    path = out_dir / "improvement_by_horizon.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _figure_link(path: Optional[str], md_dir: Optional[Path], alt: str) -> List[str]:
    if not path:
        return []
    rel = os.path.relpath(path, md_dir) if md_dir is not None else path
    return [f"![{alt}]({Path(rel).as_posix()})", ""]


def _markdown_table(df: pd.DataFrame) -> List[str]:
    """DataFrame as a Markdown table (pandas' to_markdown needs `tabulate`)."""
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return lines


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (dt.date, dt.datetime, pd.Timestamp)):
        return obj.isoformat()
    return str(obj)


def evaluate_direction_signal(
    series: Dict[str, pd.DataFrame],
    horizon: int = 10,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Evaluate directional Buy/Sell classification model."""
    t0 = time.time()
    train_df, val_df, test_df = direction_dataset(series, horizon)
    model = train_direction(train_df, val_df, n_jobs=n_jobs)
    train_sec = time.time() - t0

    moved = test_df["fwd_ret_bp"] != 0
    test_moved = test_df[moved]
    probs = model.predict_proba(test_moved[DIRECTION_FEATURES])[:, 1]
    y_true = test_moved["up"].to_numpy()

    metrics = calc_classification_metrics(y_true, probs)
    metrics["horizon"] = horizon
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_moved)
    metrics["train_seconds"] = round(train_sec, 2)
    metrics["best_iteration"] = int(model.best_iteration + 1)
    return metrics


def render_performance_markdown_report(
    eval_results: List[Dict[str, Any]],
    cv_results: Optional[Dict[str, pd.DataFrame]] = None,
    direction_result: Optional[Dict[str, Any]] = None,
    md_dir: Optional[Path] = None,
    summary_figure: Optional[str] = None,
) -> str:
    """Format full performance evaluation results into rich Markdown.

    Figure links are written relative to `md_dir` (the report's folder).
    """
    horizons = sorted({r["horizon"] for r in eval_results})
    horizon_set = ", ".join(str(h) for h in horizons)
    figs = lambda res, kind: res.get("figures", {}).get(kind)  # noqa: E731
    lines = [
        "# XGBoost Model Performance & Accuracy Report",
        "",
        "> [!NOTE]",
        "> **Evaluation Benchmark**: Walk-forward & out-of-sample test splits on 1-minute FX datasets.",
        "> **Targets**: Realised Volatility (`vol`) & Mean Bid-Ask Spread (`spread`) "
        f"over forward horizons $H \\in \\{{{horizon_set}\\}}$ minutes.",
        "",
        "## 1. Executive Summary & Overall Accuracy",
        "",
        "| Target | Horizon | Test Rows | Trees | RMSE (log) | Baseline RMSE | **RMSE Imprv %** | MAE (log) | $R^2$ | RMSE (bp) | Baseline (bp) | **Imprv % (bp)** | Dir Acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for res in eval_results:
        log_m = res["overall_log"]
        bp_m = res["overall_bp"]
        lines.append(
            f"| **{res['target'].upper()}** | {res['horizon']}m | {res['test_rows']:,} | "
            f"{res['best_iteration']} | {log_m['rmse']:.4f} | {log_m.get('baseline_rmse', 0):.4f} | "
            f"**{log_m.get('rmse_improvement_pct', 0):+.2f}%** | {log_m['mae']:.4f} | {log_m['r2']:.3f} | "
            f"{bp_m['rmse_bp']:.3f} bp | {bp_m.get('baseline_rmse_bp', 0):.3f} bp | "
            f"**{bp_m.get('rmse_improvement_pct_bp', 0):+.2f}%** | {res['directional_accuracy']:.1%} |"
        )

    lines.extend([
        "",
        "> [!TIP]",
        "> - **RMSE (log)**: Represents relative/multiplicative error ($~0.25$ indicates approx 25% relative spread/volatility prediction accuracy).",
        "> - **RMSE (bp)**: Represents physical pricing deviation in basis points ($0.01\\% = 1$ bp).",
        "> - **Imprv %**: Relative percentage reduction in RMSE compared to the standard 60-minute trailing rolling benchmark.",
        "",
    ])
    lines.extend(_figure_link(summary_figure, md_dir, "Improvement by horizon"))
    for res in eval_results:
        lines.extend(_figure_link(figs(res, "pred_vs_actual"), md_dir,
                                  f"{res['target']} {res['horizon']}m predicted vs actual"))
    lines.extend([
        "## 2. Accuracy Breakdown by Currency Pair",
        "",
    ])

    for res in eval_results:
        lines.append(f"### Target: `{res['target'].upper()}` ({res['horizon']}m Horizon)")
        lines.append("")
        lines.append("| Pair | Rows | XGBoost RMSE | Baseline RMSE | **Imprv %** | XGBoost MAE | $R^2$ | RMSE (bp) | Imprv % (bp) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in res["pairs"]:
            lines.append(
                f"| **{p['pair']}** | {p['rows']:,} | {p['rmse']:.4f} | {p['baseline_rmse']:.4f} | "
                f"**{p['improvement_pct']:+.2f}%** | {p['mae']:.4f} | {p['r2']:.3f} | "
                f"{p['rmse_bp']:.3f} bp | {p['improvement_pct_bp']:+.2f}% |"
            )
        lines.append("")
        lines.extend(_figure_link(figs(res, "pair_improvement"), md_dir,
                                  f"{res['target']} {res['horizon']}m improvement by pair"))

    # Tenor breakdown if available
    spread_res = next((r for r in eval_results if r["target"] == "spread" and r.get("tenors")), None)
    if spread_res and spread_res["tenors"]:
        lines.extend([
            "## 3. Spread Accuracy Breakdown by Tenor",
            "",
            f"Evaluated on {spread_res['horizon']}m prediction horizon:",
            "",
            "| Tenor Days | Rows | RMSE (log) | MAE (log) | $R^2$ | RMSE (bp) | MAE (bp) | Imprv % |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for t in spread_res["tenors"]:
            tenor_label = f"{t['tenor_days']}d" if isinstance(t['tenor_days'], int) else str(t['tenor_days'])
            lines.append(
                f"| {tenor_label} | {t['rows']:,} | {t['rmse']:.4f} | {t['mae']:.4f} | "
                f"{t['r2']:.3f} | {t['rmse_bp']:.3f} bp | {t['mae_bp']:.3f} bp | {t['improvement_pct']:+.2f}% |"
            )
        lines.append("")

    # Market Regime breakdown
    lines.extend([
        "## 4. Market Regime Analysis (Calm vs High Volatility)",
        "",
        "Evaluates how prediction error behaves across quintiles from quiet/tight market states to extreme volatility spikes.",
        "",
    ])

    for res in eval_results:
        lines.append(f"#### Target: `{res['target'].upper()}` ({res['horizon']}m Horizon)")
        lines.append("")
        lines.append("| Regime | Rows | Range (log bp) | RMSE | MAE | MedAE | Bias (Mean Residual) |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in res["regimes"]:
            lines.append(
                f"| {r['regime']} | {r['count']:,} | [{r['y_min']:.2f}, {r['y_max']:.2f}] | "
                f"{r['rmse']:.4f} | {r['mae']:.4f} | {r['medae']:.4f} | {r['bias']:+.4f} |"
            )
        lines.append("")
        lines.extend(_figure_link(figs(res, "residuals_by_regime"), md_dir,
                                  f"{res['target']} {res['horizon']}m residuals by regime"))

    # Feature Importance
    lines.extend([
        "## 5. Top Predictive Features (Share of Gain)",
        "",
    ])
    for res in eval_results:
        lines.append(f"#### Target: `{res['target'].upper()}` ({res['horizon']}m Horizon)")
        lines.append("")
        lines.append("| Rank | Feature | Gain Share | Tree Splits |")
        lines.append("|---|---|---|---|")
        for rank, f in enumerate(res["feature_importance"], 1):
            lines.append(f"| {rank} | `{f['feature']}` | {f['gain_share']:.2%} | {f['splits']} |")
        lines.append("")
        lines.extend(_figure_link(figs(res, "feature_importance"), md_dir,
                                  f"{res['target']} {res['horizon']}m feature importance"))

    # Walk-forward CV if present
    if cv_results:
        lines.extend([
            "## 6. Walk-Forward Cross-Validation (Time Stability)",
            "",
            "Evaluated over rolling out-of-sample time blocks (no lookahead, embargo applied):",
            "",
        ])
        for target, df in cv_results.items():
            lines.append(f"### Target: `{target.upper()}` Walk-Forward Folds")
            lines.append("")
            lines.extend(_markdown_table(df))
            mean_xgb = df["xgboost"].mean()
            mean_base = df["baseline"].mean()
            imprv = (1.0 - mean_xgb / mean_base) * 100.0
            lines.append("")
            lines.append(f"**Mean Fold RMSE**: XGBoost = `{mean_xgb:.4f}`, Baseline = `{mean_base:.4f}` (**{imprv:+.2f}% Improvement**)")
            lines.append("")

    # Direction Model if present
    if direction_result:
        d = direction_result
        lines.extend([
            "## 7. Directional Buy / Sell Classification Model",
            "",
            f"Predicts whether price will move UP over the next {d['horizon']} minutes (P(price up) > 0.5):",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Test Rows Evaluated | {d['test_rows']:,} |",
            f"| Directional Accuracy | **{d['accuracy']:.2%}** |",
            f"| Precision (Up) | {d['precision']:.2%} |",
            f"| Recall (Up) | {d['recall']:.2%} |",
            f"| F1 Score | {d['f1']:.3f} |",
            f"| Log-Loss | {d['log_loss']:.4f} |",
            f"| Brier Score | {d['brier_score']:.4f} |",
            f"| Confusion Matrix (TP / FP / TN / FN) | {d['tp']} / {d['fp']} / {d['tn']} / {d['fn']} |",
            "",
        ])

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost FX models and create detailed performance reports.")
    parser.add_argument("--data", default="data/processed", help="Path to processed parquet data")
    parser.add_argument("--target", choices=["vol", "spread", "both"], default="both", help="Target variable(s)")
    parser.add_argument("--horizons", default="5,15,30", help="Comma-separated horizons in minutes (e.g. 5,15,30)")
    parser.add_argument("--cv", type=int, default=0, metavar="FOLDS", help="Run walk-forward CV over N folds")
    parser.add_argument("--cv-days", type=int, default=5, help="Days per walk-forward fold")
    parser.add_argument("--cv-trees", type=int, default=1000, help="Trees per walk-forward fold (fixed)")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--max-trees", type=int, default=4000, help="Max trees with early stopping")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience rounds")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU threads")
    parser.add_argument("--include-direction", action="store_true", help="Evaluate direction model as well")
    parser.add_argument("--output-md", default="reports/xgboost_performance_report.md", help="Path to save markdown report")
    parser.add_argument("--output-json", default="reports/xgboost_metrics.json", help="Path to save raw JSON results")
    parser.add_argument("--figures-dir", default="reports/figures", help="Where to write chart PNGs")
    parser.add_argument("--no-plots", action="store_true", help="Skip the charts")
    args = parser.parse_args(argv)

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    targets = ["vol", "spread"] if args.target == "both" else [args.target]

    print(f"Loading data from '{args.data}'...")
    t0 = time.time()
    series, _ = load_processed(args.data)
    print(f"Loaded {len(series)} instruments in {time.time() - t0:.1f}s\n")

    plot_dir = None if args.no_plots else Path(args.figures_dir)
    eval_results: List[Dict[str, Any]] = []

    for target in targets:
        for h in horizons:
            print(f"--- Evaluating Target: {target.upper()} | Horizon: {h}min ---")
            res = evaluate_target_horizon(
                series=series,
                target=target,
                horizon=h,
                learning_rate=args.learning_rate,
                max_trees=args.max_trees,
                patience=args.patience,
                n_jobs=args.jobs,
                plot_dir=plot_dir,
            )
            eval_results.append(res)
            log_m = res["overall_log"]
            bp_m = res["overall_bp"]
            print(f"  Trees: {res['best_iteration']} | Train: {res['train_seconds']:.1f}s")
            print(f"  RMSE (log): {log_m['rmse']:.4f} vs Baseline {log_m['baseline_rmse']:.4f} "
                  f"({log_m['rmse_improvement_pct']:+.2f}%) | R2: {log_m['r2']:.3f}")
            print(f"  RMSE (bp):  {bp_m['rmse_bp']:.3f} bp vs Baseline {bp_m['baseline_rmse_bp']:.3f} bp "
                  f"({bp_m['rmse_improvement_pct_bp']:+.2f}%)")
            print(f"  Dir Accuracy: {res['directional_accuracy']:.1%}\n")

    cv_results: Optional[Dict[str, pd.DataFrame]] = None
    if args.cv > 0:
        cv_results = {}
        for target in targets:
            # evaluate walk-forward on representative 30m horizon
            h = 30 if 30 in horizons else horizons[0]
            print(f"--- Running Walk-Forward CV ({args.cv} folds) for {target.upper()} (h={h}m) ---")
            df_cv = walk_forward(
                series,
                target=target,
                horizon=h,
                n_folds=args.cv,
                fold_days=args.cv_days,
                n_trees=args.cv_trees,
                learning_rate=args.learning_rate,
                n_jobs=args.jobs,
            )
            cv_results[f"{target}_{h}m"] = df_cv
            print(df_cv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"Mean RMSE: XGBoost = {df_cv['xgboost'].mean():.4f}, Baseline = {df_cv['baseline'].mean():.4f}\n")

    direction_result = None
    if args.include_direction:
        print("--- Evaluating Buy/Sell Direction Model (h=10m) ---")
        direction_result = evaluate_direction_signal(series, horizon=10, n_jobs=args.jobs)
        print(f"  Direction Accuracy: {direction_result['accuracy']:.2%} | F1: {direction_result['f1']:.3f} "
              f"| Log-Loss: {direction_result['log_loss']:.4f}\n")

    # Generate Markdown report
    summary_figure = plot_summary(eval_results, plot_dir) if plot_dir is not None else None
    md_dir = Path(args.output_md).parent if args.output_md else None
    md_content = render_performance_markdown_report(
        eval_results, cv_results, direction_result, md_dir=md_dir, summary_figure=summary_figure,
    )
    if args.output_md:
        out_p = Path(args.output_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_content, encoding="utf-8")
        print(f"Markdown performance report generated: {args.output_md}")

    # Generate JSON results
    if args.output_json:
        json_p = Path(args.output_json)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        json_dump_data = {
            "eval_results": eval_results,
            "cv_results": {k: v.to_dict(orient="records") for k, v in cv_results.items()} if cv_results else None,
            "direction_result": direction_result,
            "summary_figure": summary_figure,
        }
        json_p.write_text(json.dumps(json_dump_data, indent=2, default=_json_default), encoding="utf-8")
        print(f"JSON metrics exported: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
