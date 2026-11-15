"""Evaluation metrics for FX volatility, spread, and direction models.

Provides regression metrics (RMSE, MAE, R2, MedAE, Pearson r) both in the
model's log space and in physical basis points (bp), error distribution percentiles,
regime-based performance breakdowns, and directional accuracy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error, ignoring NaN values."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((y_t[valid] - y_p[valid]) ** 2)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error, ignoring NaN values."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(y_t[valid] - y_p[valid])))


def medae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Median Absolute Error, robust to extreme outlier spikes."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return float("nan")
    return float(np.median(np.abs(y_t[valid] - y_p[valid])))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Percentage Error (%), clipped to prevent division by zero."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p) & (np.abs(y_t) > 1e-6)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((y_t[valid] - y_p[valid]) / y_t[valid])) * 100.0)


def r2(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Coefficient of Determination R^2."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return float("nan")
    ss_tot = np.sum((y_t[valid] - np.mean(y_t[valid])) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y_t[valid] - y_p[valid]) ** 2)
    return float(1.0 - (ss_res / ss_tot))


def pearson_r(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Pearson correlation coefficient between ground truth and predictions."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if np.sum(valid) < 2:
        return float("nan")
    std_t = np.std(y_t[valid])
    std_p = np.std(y_p[valid])
    if std_t == 0 or std_p == 0:
        return 0.0
    corr = np.corrcoef(y_t[valid], y_p[valid])[0, 1]
    return float(corr)


def error_percentiles(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    percentiles: Sequence[int] = (5, 25, 50, 75, 95),
) -> Dict[str, float]:
    """Percentiles of residual errors (y_true - y_pred)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return {f"p{p}": float("nan") for p in percentiles}
    residuals = y_t[valid] - y_p[valid]
    res_vals = np.percentile(residuals, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, res_vals)}


def calc_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    baseline_pred: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, float]:
    """Calculate standard regression metrics, error spread, and baseline comparison."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if baseline_pred is not None:
        y_b = np.asarray(baseline_pred, dtype=float)
        valid = valid & np.isfinite(y_b)
    else:
        y_b = None

    metrics = {
        "count": int(np.sum(valid)),
        "rmse": rmse(y_t[valid], y_p[valid]),
        "mae": mae(y_t[valid], y_p[valid]),
        "medae": medae(y_t[valid], y_p[valid]),
        "r2": r2(y_t[valid], y_p[valid]),
        "pearson_r": pearson_r(y_t[valid], y_p[valid]),
    }

    if np.any(valid):
        residuals = y_t[valid] - y_p[valid]
        metrics["bias"] = float(np.mean(residuals))
        metrics["std_err"] = float(np.std(residuals))
        metrics.update(error_percentiles(y_t[valid], y_p[valid]))
    else:
        metrics["bias"] = float("nan")
        metrics["std_err"] = float("nan")

    if y_b is not None and np.any(valid):
        b_rmse = rmse(y_t[valid], y_b[valid])
        b_mae = mae(y_t[valid], y_b[valid])
        b_r2 = r2(y_t[valid], y_b[valid])
        metrics["baseline_rmse"] = b_rmse
        metrics["baseline_mae"] = b_mae
        metrics["baseline_r2"] = b_r2
        metrics["rmse_improvement_pct"] = (
            float((1.0 - metrics["rmse"] / b_rmse) * 100.0) if b_rmse > 0 else 0.0
        )
        metrics["mae_improvement_pct"] = (
            float((1.0 - metrics["mae"] / b_mae) * 100.0) if b_mae > 0 else 0.0
        )

    return metrics


def calc_physical_bp_metrics(
    y_true_log: np.ndarray | pd.Series,
    y_pred_log: np.ndarray | pd.Series,
    baseline_pred_log: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, float]:
    """Evaluate accuracy in physical basis points (bp) by exponentiating log predictions.

    e^y_true vs e^y_pred
    """
    y_t_bp = np.exp(np.asarray(y_true_log, dtype=float))
    y_p_bp = np.exp(np.asarray(y_pred_log, dtype=float))
    y_b_bp = (
        np.exp(np.asarray(baseline_pred_log, dtype=float))
        if baseline_pred_log is not None
        else None
    )

    base_metrics = calc_regression_metrics(y_t_bp, y_p_bp, y_b_bp)
    bp_metrics = {
        "rmse_bp": base_metrics["rmse"],
        "mae_bp": base_metrics["mae"],
        "medae_bp": base_metrics["medae"],
        "r2_bp": base_metrics["r2"],
        "target_mean_bp": float(np.nanmean(y_t_bp)),
        "pred_mean_bp": float(np.nanmean(y_p_bp)),
    }
    if "baseline_rmse" in base_metrics:
        bp_metrics["baseline_rmse_bp"] = base_metrics["baseline_rmse"]
        bp_metrics["baseline_mae_bp"] = base_metrics["baseline_mae"]
        bp_metrics["rmse_improvement_pct_bp"] = base_metrics["rmse_improvement_pct"]
        bp_metrics["mae_improvement_pct_bp"] = base_metrics["mae_improvement_pct"]
    return bp_metrics


def calc_regime_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 5,
) -> List[Dict[str, Union[str, float, int]]]:
    """Breakdown evaluation metrics across quantile regimes (e.g. Quiet vs Volatile)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if np.sum(valid) < n_bins * 2:
        return []

    y_t_valid = y_t[valid]
    y_p_valid = y_p[valid]

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_t_valid, quantiles)
    # Ensure distinct bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    regimes = []
    labels = ["Lowest / Calm", "Low", "Moderate", "High", "Highest / Spikes"]
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_t_valid > low) & (y_t_valid <= high) if i > 0 else (y_t_valid <= high)
        if not np.any(in_bin):
            continue
        label = labels[i] if i < len(labels) else f"Q{i+1}"
        regimes.append({
            "regime": label,
            "count": int(np.sum(in_bin)),
            "y_min": float(np.min(y_t_valid[in_bin])),
            "y_max": float(np.max(y_t_valid[in_bin])),
            "rmse": rmse(y_t_valid[in_bin], y_p_valid[in_bin]),
            "mae": mae(y_t_valid[in_bin], y_p_valid[in_bin]),
            "medae": medae(y_t_valid[in_bin], y_p_valid[in_bin]),
            "bias": float(np.mean(y_t_valid[in_bin] - y_p_valid[in_bin])),
        })
    return regimes


def calc_directional_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_last: np.ndarray | pd.Series,
) -> float:
    """Accuracy of predicting change direction relative to current window (y_last).

    Checks whether sign(y_pred - y_last) == sign(y_true - y_last).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_l = np.asarray(y_last, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p) & np.isfinite(y_l)
    if not np.any(valid):
        return float("nan")

    delta_true = y_t[valid] - y_l[valid]
    delta_pred = y_p[valid] - y_l[valid]

    # Non-zero moves
    moved = delta_true != 0
    if not np.any(moved):
        return 1.0

    correct = np.sign(delta_true[moved]) == np.sign(delta_pred[moved])
    return float(np.mean(correct))


def calc_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate classification metrics for directional predictions P(price up)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(valid):
        return {}

    y_t = y_t[valid]
    y_p = y_p[valid]
    pred_cls = (y_p >= threshold).astype(int)

    tp = int(np.sum((y_t == 1) & (pred_cls == 1)))
    fp = int(np.sum((y_t == 0) & (pred_cls == 1)))
    tn = int(np.sum((y_t == 0) & (pred_cls == 0)))
    fn = int(np.sum((y_t == 1) & (pred_cls == 0)))

    acc = float((tp + tn) / len(y_t)) if len(y_t) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Clip probabilities for log loss
    eps = 1e-15
    y_p_clipped = np.clip(y_p, eps, 1 - eps)
    log_loss = -float(np.mean(y_t * np.log(y_p_clipped) + (1 - y_t) * np.log(1 - y_p_clipped)))
    brier = float(np.mean((y_p - y_t) ** 2))

    return {
        "count": len(y_t),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_loss": log_loss,
        "brier_score": brier,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
