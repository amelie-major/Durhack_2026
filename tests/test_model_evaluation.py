"""Unit tests for metrics calculation, data audit, and model performance reporting."""
import numpy as np
import pandas as pd
import pytest

from models.metrics import (
    calc_classification_metrics,
    calc_directional_accuracy,
    calc_physical_bp_metrics,
    calc_regime_metrics,
    calc_regression_metrics,
    mae,
    medae,
    pearson_r,
    r2,
    rmse,
)
from scripts.audit_data import audit_processed_data, render_markdown_report
from scripts.evaluate_xgboost import render_performance_markdown_report


def test_basic_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.9, 3.1, 3.8, 5.2])
    baseline = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # RMSE: sqrt(mean([0.04, 0.01, 0.01, 0.04, 0.04])) = sqrt(0.028) ~ 0.16733
    r = rmse(y_true, y_pred)
    assert pytest.approx(r, 0.001) == 0.16733

    # MAE: mean([0.2, 0.1, 0.1, 0.2, 0.2]) = 0.16
    m = mae(y_true, y_pred)
    assert pytest.approx(m, 0.001) == 0.16

    # MedAE: median([0.2, 0.1, 0.1, 0.2, 0.2]) = 0.2
    assert pytest.approx(medae(y_true, y_pred), 0.001) == 0.2

    # R2
    val_r2 = r2(y_true, y_pred)
    assert val_r2 > 0.95

    # Pearson r
    corr = pearson_r(y_true, y_pred)
    assert corr > 0.99

    metrics = calc_regression_metrics(y_true, y_pred, baseline)
    assert "rmse_improvement_pct" in metrics
    assert metrics["rmse_improvement_pct"] > 50.0  # model error is 0.167 vs baseline 0.5


def test_nan_handling_in_metrics():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, np.nan, 4.0])
    # valid indices are only index 0 and index 3
    r = rmse(y_true, y_pred)
    assert pytest.approx(r, 0.0001) == 0.0

    all_nan = np.array([np.nan, np.nan])
    assert np.isnan(rmse(all_nan, all_nan))


def test_physical_bp_metrics():
    # log space: [0, 1] -> bp space: [1.0, e ~ 2.718]
    y_true_log = np.array([0.0, 1.0])
    y_pred_log = np.array([0.0, 1.0])
    bp_m = calc_physical_bp_metrics(y_true_log, y_pred_log)
    assert pytest.approx(bp_m["rmse_bp"], 0.001) == 0.0
    assert pytest.approx(bp_m["target_mean_bp"], 0.01) == (1.0 + np.e) / 2


def test_regime_metrics():
    y_true = np.linspace(1.0, 10.0, 100)
    y_pred = y_true + 0.1
    regimes = calc_regime_metrics(y_true, y_pred, n_bins=5)
    assert len(regimes) == 5
    for reg in regimes:
        assert reg["count"] == 20
        assert pytest.approx(reg["rmse"], 0.001) == 0.1


def test_directional_accuracy():
    y_true = np.array([2.0, 4.0, 1.0])
    y_pred = np.array([2.5, 5.0, 0.5])
    y_last = np.array([1.0, 3.0, 2.0])
    # True changes: +1, +1, -1
    # Pred changes: +1.5, +2.0, -1.5
    # All match!
    acc = calc_directional_accuracy(y_true, y_pred, y_last)
    assert acc == 1.0

    # Wrong direction
    y_wrong_pred = np.array([0.5, 2.0, 3.0])
    acc_wrong = calc_directional_accuracy(y_true, y_wrong_pred, y_last)
    assert acc_wrong == 0.0


def test_classification_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    m = calc_classification_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["tn"] == 2


def test_audit_data_runner():
    # Audit real processed data directory
    res = audit_processed_data("data/processed", check_features=True, sample_instruments=2)
    assert res["summary"]["loaded_instruments"] > 0
    assert res["summary"]["total_bars"] > 100_000
    assert res["quote_anomalies"]["total_crossed_quotes"] == 0
    assert res["quote_anomalies"]["total_negative_spreads"] == 0

    md = render_markdown_report(res)
    assert "FX Dataset Accuracy & Quality Audit Report" in md
    assert "Executive Summary" in md


def test_performance_markdown_render():
    sample_eval = [{
        "target": "vol",
        "horizon": 30,
        "test_rows": 50000,
        "best_iteration": 450,
        "prep_seconds": 1.2,
        "train_seconds": 5.4,
        "overall_log": {
            "rmse": 0.280,
            "baseline_rmse": 0.350,
            "rmse_improvement_pct": 20.0,
            "mae": 0.210,
            "r2": 0.360,
        },
        "overall_bp": {
            "rmse_bp": 1.45,
            "baseline_rmse_bp": 1.80,
            "rmse_improvement_pct_bp": 19.4,
        },
        "directional_accuracy": 0.62,
        "regimes": [
            {"regime": "Low", "count": 10000, "y_min": 0.1, "y_max": 0.5, "rmse": 0.2, "mae": 0.15, "medae": 0.12, "bias": 0.01}
        ],
        "pairs": [
            {"pair": "EURUSD", "rows": 20000, "rmse": 0.26, "baseline_rmse": 0.33, "improvement_pct": 21.2, "mae": 0.19, "r2": 0.4, "rmse_bp": 1.2, "improvement_pct_bp": 20.0}
        ],
        "tenors": [],
        "feature_importance": [
            {"feature": "vol_60", "gain_share": 0.25, "splits": 120}
        ],
    }]

    md = render_performance_markdown_report(sample_eval)
    assert "XGBoost Model Performance & Accuracy Report" in md
    assert "EURUSD" in md
    assert "vol_60" in md


def _sample_cv_frame():
    import datetime as dt
    return pd.DataFrame([
        {"fold": 1, "test_from": dt.date(2026, 1, 5), "test_to": dt.date(2026, 1, 9),
         "train_rows": 1000, "test_rows": 200, "xgboost": 0.28, "baseline": 0.35},
        {"fold": 2, "test_from": dt.date(2026, 1, 12), "test_to": dt.date(2026, 1, 16),
         "train_rows": 1200, "test_rows": 200, "xgboost": 0.27, "baseline": 0.34},
    ])


def test_performance_markdown_with_cv_needs_no_tabulate():
    from scripts.evaluate_xgboost import _markdown_table
    md = "\n".join(_markdown_table(_sample_cv_frame()))
    assert "| fold | test_from |" in md
    assert "2026-01-05" in md
    assert "0.2800" in md


def test_json_export_handles_dates_and_numpy():
    import json
    from scripts.evaluate_xgboost import _json_default
    payload = {"cv": _sample_cv_frame().to_dict(orient="records"), "n": np.int64(3), "x": np.float32(0.5)}
    back = json.loads(json.dumps(payload, default=_json_default))
    assert back["cv"][0]["test_from"] == "2026-01-05"
    assert back["n"] == 3


def test_plot_results_writes_pngs(tmp_path):
    from scripts.evaluate_xgboost import plot_results, plot_summary
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    test_df = pd.DataFrame({"y": y})
    pred = y + rng.normal(scale=0.3, size=500)
    res = {
        "target": "vol", "horizon": 30,
        "overall_log": {"r2": 0.9, "rmse_improvement_pct": 12.0},
        "pairs": [{"pair": "EURUSD", "improvement_pct": 10.0}, {"pair": "USDJPY", "improvement_pct": -2.0}],
        "feature_importance": [{"feature": "vol_60", "gain_share": 0.3, "splits": 10}],
    }
    figs = plot_results(res, test_df, pred, tmp_path)
    assert set(figs) == {"pred_vs_actual", "residuals_by_regime", "pair_improvement", "feature_importance"}
    for path in figs.values():
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0
    assert plot_summary([res], tmp_path).endswith("improvement_by_horizon.png")
