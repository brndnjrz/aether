"""
Regression suite for analysis/prediction_performance.py — the classification-
metrics layer (precision/recall/F1/FPR/FNR/profit-per-signal/holding-time/
calibration) on top of the accuracy/win-rate already computed by
ml_prediction.evaluate_model().

Core-function tests build synthetic history DataFrames directly (no
storage, no fixtures) since compute_prediction_metrics() is pure. Wrapper
tests use the existing isolated_storage/isolated_intraday_storage fixtures
from conftest.py to confirm the auto-fetch path matches calling the core
function directly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.prediction_performance import compute_prediction_metrics


def _row(direction, correct, actual_outcome, probability=0.7, confidence="high", horizon_days=5):
    return {
        "direction": direction, "probability": probability, "confidence": confidence,
        "actual_outcome": actual_outcome, "correct": correct, "horizon_days": horizon_days,
    }


def test_prediction_performance_module_imports_cleanly():
    import analysis.prediction_performance as pp

    for name in (
        "compute_prediction_metrics", "compute_daily_prediction_metrics",
        "compute_intraday_prediction_metrics",
    ):
        assert hasattr(pp, name), f"analysis.prediction_performance is missing {name}"


def test_pages_model_lab_imports_cleanly():
    """
    pages/model_lab.py calls render() at import time (like every page in this
    app), which in turn calls get_prediction_history()/
    get_intraday_prediction_history() — both already tolerate network
    failure in resolve_predictions()/resolve_intraday_predictions() by
    catching and log-warning rather than raising, so this is safe offline.
    """
    import pages.model_lab  # noqa: F401 — import success is the assertion


# ── Core metrics ─────────────────────────────────────────────────────────────

def test_perfect_predictions_yield_full_marks():
    rows = (
        [_row("bullish", True, 1.0)] * 4
        + [_row("bearish", True, -1.0)] * 4
    )
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["accuracy"] == 1.0
    assert m["win_rate"] == 1.0
    for cls in ("bullish", "bearish", "macro"):
        assert m["precision"][cls] == 1.0
        assert m["recall"][cls] == 1.0
        assert m["f1"][cls] == 1.0
        assert m["false_positive_rate"][cls] == 0.0
        assert m["false_negative_rate"][cls] == 0.0


def test_all_wrong_predictions_yield_zero_marks():
    rows = (
        [_row("bullish", False, -1.0)] * 4
        + [_row("bearish", False, 1.0)] * 4
    )
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["accuracy"] == 0.0
    assert m["precision"]["bullish"] == 0.0
    assert m["recall"]["bullish"] == 0.0
    assert m["false_positive_rate"]["bullish"] == 1.0
    assert m["false_negative_rate"]["bullish"] == 1.0
    assert m["f1"]["bullish"] == 0.0


def test_mixed_predictions_hand_computed_confusion_matrix():
    rows = (
        [_row("bullish", True, 1.0)] * 4
        + [_row("bullish", False, -0.5)] * 1
        + [_row("bearish", True, -1.0)] * 2
        + [_row("bearish", False, 0.5)] * 3
    )
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["accuracy"] == 0.6
    assert m["precision"]["bullish"] == 0.8
    assert m["recall"]["bullish"] == pytest.approx(0.5714, abs=1e-3)
    assert m["f1"]["bullish"] == pytest.approx(0.6667, abs=1e-3)
    assert m["precision"]["bearish"] == 0.4
    assert m["recall"]["bearish"] == pytest.approx(0.6667, abs=1e-3)
    assert m["f1"]["bearish"] == 0.5
    assert m["false_positive_rate"]["bullish"] == pytest.approx(0.3333, abs=1e-3)
    assert m["false_negative_rate"]["bullish"] == pytest.approx(0.4286, abs=1e-3)
    assert m["false_positive_rate"]["bearish"] == pytest.approx(0.4286, abs=1e-3)
    assert m["false_negative_rate"]["bearish"] == pytest.approx(0.3333, abs=1e-3)

    # Cross-class invariant that always holds in a 2-class confusion matrix.
    assert m["false_positive_rate"]["bullish"] == m["false_negative_rate"]["bearish"]
    assert m["false_positive_rate"]["bearish"] == m["false_negative_rate"]["bullish"]


def test_no_resolved_predictions_returns_none_metrics_not_errors():
    rows = [_row("bullish", None, None)] * 3
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["n_resolved"] == 0
    assert m["n_unresolved"] == 3
    assert m["accuracy"] is None
    assert m["win_rate"] is None
    for cls in ("bullish", "bearish", "macro"):
        assert m["precision"][cls] is None
        assert m["recall"][cls] is None
        assert m["f1"][cls] is None
        assert m["false_positive_rate"][cls] is None
        assert m["false_negative_rate"][cls] is None
    assert m["avg_profit_per_signal"] is None
    assert m["holding_time"]["mean"] is None
    assert m["warnings"]


def test_empty_history_dataframe_returns_documented_shape():
    empty = pd.DataFrame(columns=[
        "direction", "probability", "confidence", "actual_outcome", "correct", "horizon_days",
    ])
    m = compute_prediction_metrics(empty)

    assert m["n_total"] == 0
    assert m["n_resolved"] == 0
    assert m["accuracy"] is None
    assert m["holding_time"]["mean"] is None


def test_single_class_only_history_distinguishes_undefined_from_zero():
    # Sub-case A: only bullish rows, all correct — zero reconstructed
    # bearish ground truth at all.
    rows_a = [_row("bullish", True, 1.0)] * 6
    m_a = compute_prediction_metrics(pd.DataFrame(rows_a))
    assert m_a["precision"]["bearish"] is None
    assert m_a["recall"]["bearish"] is None

    # Sub-case B: only bullish rows, some wrong — the wrong bullish calls
    # reconstruct real "true bearish" instances the model never called.
    rows_b = [_row("bullish", True, 1.0)] * 4 + [_row("bullish", False, -1.0)] * 2
    m_b = compute_prediction_metrics(pd.DataFrame(rows_b))
    assert m_b["precision"]["bearish"] is None  # still 0 predicted-bearish instances
    assert m_b["recall"]["bearish"] == 0.0        # real number, not None
    assert m_b["false_negative_rate"]["bearish"] == 1.0
    assert m_b["f1"]["bearish"] is None            # needs precision, which is undefined


def test_avg_profit_per_signal_flips_sign_for_bearish():
    rows = [
        _row("bullish", True, 2.0),
        _row("bearish", True, 1.0),
        _row("bearish", False, -3.0),
    ]
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["avg_profit_per_signal"] == pytest.approx(4 / 3, abs=1e-4)
    assert m["avg_profit_per_signal_by_direction"] == {"bullish": 2.0, "bearish": 1.0}


def test_avg_holding_time_daily_flags_mixed_horizons():
    rows = [_row("bullish", True, 1.0, horizon_days=h) for h in (5, 5, 5, 10)]
    m = compute_prediction_metrics(pd.DataFrame(rows), horizon_col="horizon_days")

    assert m["holding_time"]["mean"] == 6.25
    assert m["holding_time"]["min"] == 5.0
    assert m["holding_time"]["max"] == 10.0
    assert m["holding_time"]["unit"] == "days"
    assert m["holding_time"]["distinct_horizons"] == [5.0, 10.0]


def test_avg_holding_time_intraday_uses_horizon_minutes():
    rows = [
        {"direction": "bullish", "probability": 0.7, "confidence": "high",
         "actual_outcome": 1.0, "correct": True, "horizon_minutes": h}
        for h in (75, 75, 150)
    ]
    m = compute_prediction_metrics(pd.DataFrame(rows), horizon_col="horizon_minutes")

    assert m["holding_time"]["mean"] == 100.0
    assert m["holding_time"]["unit"] == "minutes"


def test_confidence_calibration_buckets_match_hand_counts():
    rows = (
        [_row("bullish", True, 1.0, confidence="high")] * 3
        + [_row("bullish", False, -1.0, confidence="high")] * 1
        + [_row("bearish", True, -1.0, confidence="medium")] * 1
        + [_row("bearish", False, 1.0, confidence="medium")] * 1
        + [_row("bullish", False, -1.0, confidence="low")] * 2
    )
    m = compute_prediction_metrics(pd.DataFrame(rows))

    by_conf = m["calibration"]["by_confidence"]
    assert by_conf["high"]["n"] == 4
    assert by_conf["high"]["realized_accuracy"] == 0.75
    assert by_conf["medium"]["n"] == 2
    assert by_conf["medium"]["realized_accuracy"] == 0.5
    assert by_conf["low"]["n"] == 2
    assert by_conf["low"]["realized_accuracy"] == 0.0


def test_probability_decile_calibration_uses_directional_probability_not_raw():
    # 0.9 P(bullish) called bullish, and 0.1 P(bullish) called bearish, are
    # the SAME 90% confidence in the direction actually called — they must
    # land in the same calibration bucket.
    rows = [
        _row("bullish", True, 1.0, probability=0.9),
        _row("bearish", True, -1.0, probability=0.1),
    ]
    m = compute_prediction_metrics(pd.DataFrame(rows), min_decile_sample=2)

    buckets = m["calibration"]["by_probability_decile"]
    assert len(buckets) == 1
    assert buckets[0]["n"] == 2
    assert buckets[0]["avg_predicted_probability"] == pytest.approx(0.9, abs=1e-6)


def test_probability_decile_calibration_skipped_below_minimum_sample_size():
    rows = [_row("bullish", True, 1.0)] * 6
    m = compute_prediction_metrics(pd.DataFrame(rows))  # default min_decile_sample=10

    assert m["calibration"]["by_probability_decile"] == []
    assert m["calibration"]["note"]
    # Coarser 3-bucket calibration still runs even when the decile curve is skipped.
    assert m["calibration"]["by_confidence"]["high"]["n"] == 6


def test_win_rate_equals_accuracy_by_design():
    rows = [_row("bullish", True, 1.0)] * 3 + [_row("bearish", False, 1.0)] * 1
    m = compute_prediction_metrics(pd.DataFrame(rows))

    assert m["win_rate"] == m["accuracy"]


# ── Wrapper functions (daily / intraday auto-fetch) ─────────────────────────

def test_compute_daily_prediction_metrics_reads_real_history_when_none_passed(isolated_storage):
    import json
    from analysis.ml_prediction import _predictions_path, get_prediction_history
    from analysis.prediction_performance import compute_daily_prediction_metrics

    records = [
        {"predicted_at": "2026-07-28T10:00:00-04:00", "date": "2026-07-28T10:00:00-04:00",
         "ticker": "ZZLAB", "direction": "bullish", "probability": 0.7, "confidence": "high",
         "model_accuracy": 0.6, "expected_move_pct": 1.0, "horizon_days": 5,
         "price_at_prediction": 100.0, "actual_outcome": 2.0, "correct": True},
        {"predicted_at": "2026-07-29T10:00:00-04:00", "date": "2026-07-29T10:00:00-04:00",
         "ticker": "ZZLAB", "direction": "bearish", "probability": 0.3, "confidence": "medium",
         "model_accuracy": 0.6, "expected_move_pct": -1.0, "horizon_days": 5,
         "price_at_prediction": 100.0, "actual_outcome": 1.0, "correct": False},
    ]
    path = _predictions_path("ZZLAB")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    via_wrapper = compute_daily_prediction_metrics("ZZLAB")
    via_direct = compute_prediction_metrics(get_prediction_history("ZZLAB"), horizon_col="horizon_days")

    assert via_wrapper["ticker"] == "ZZLAB"
    assert via_wrapper["accuracy"] == via_direct["accuracy"]
    assert via_wrapper["n_resolved"] == via_direct["n_resolved"] == 2


def test_compute_intraday_prediction_metrics_reads_real_history_when_none_passed(isolated_intraday_storage):
    import json
    from analysis.intraday_prediction import _predictions_path, get_intraday_prediction_history
    from analysis.prediction_performance import compute_intraday_prediction_metrics

    records = [
        {"predicted_at": "2026-07-28T10:00:00-04:00", "date": "2026-07-28T10:00:00-04:00",
         "ticker": "ZZLAB", "interval": "15m", "bar_timestamp": "2026-07-28T10:00:00-04:00",
         "direction": "bullish", "probability": 0.7, "confidence": "high",
         "horizon_bars": 5, "horizon_minutes": 75, "model_accuracy": 0.6,
         "price_at_prediction": 500.0, "actual_outcome": 1.0, "correct": True},
        {"predicted_at": "2026-07-29T10:00:00-04:00", "date": "2026-07-29T10:00:00-04:00",
         "ticker": "ZZLAB", "interval": "15m", "bar_timestamp": "2026-07-29T10:00:00-04:00",
         "direction": "bullish", "probability": 0.6, "confidence": "medium",
         "horizon_bars": 5, "horizon_minutes": 75, "model_accuracy": 0.6,
         "price_at_prediction": 500.0, "actual_outcome": -0.5, "correct": False},
    ]
    path = _predictions_path("ZZLAB", "15m")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    via_wrapper = compute_intraday_prediction_metrics("ZZLAB", "15m")
    via_direct = compute_prediction_metrics(
        get_intraday_prediction_history("ZZLAB", "15m"), horizon_col="horizon_minutes",
    )

    assert via_wrapper["ticker"] == "ZZLAB"
    assert via_wrapper["interval"] == "15m"
    assert via_wrapper["accuracy"] == via_direct["accuracy"]
    assert via_wrapper["n_resolved"] == via_direct["n_resolved"] == 2
