"""
Regression suite for analysis/prediction_errors.py — rule-based failure
categorization for resolved-incorrect predictions.

Core-rule tests build synthetic snapshot dicts directly (no fixtures — the
categorization rules are pure functions over a plain dict). The
categorize_incorrect_predictions tests exercise the three snapshot sources
(logged / recomputed via an injected fetcher / unavailable) and the
correct == False filter, mirroring analysis/prediction_performance.py's
"never re-derive correctness" convention.
"""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.prediction_errors import (
    aggregate_failure_categories,
    build_indicator_snapshot,
    categorize_failure,
    categorize_incorrect_predictions,
)


def test_prediction_errors_module_imports_cleanly():
    import analysis.prediction_errors as pe

    for name in (
        "build_indicator_snapshot", "categorize_failure",
        "categorize_incorrect_predictions", "aggregate_failure_categories",
    ):
        assert hasattr(pe, name), f"analysis.prediction_errors is missing {name}"


# ── build_indicator_snapshot ─────────────────────────────────────────────────

def test_build_indicator_snapshot_from_synthetic_indicators_df(synthetic_indicators_df):
    snap = build_indicator_snapshot(synthetic_indicators_df)

    assert isinstance(snap, dict)
    for key in ("above_200ma", "above_50ma", "trend", "day_of_week"):
        assert key in snap
    assert "vix_regime" in snap  # present even if None when VIX is unreachable


def test_build_indicator_snapshot_returns_empty_dict_on_empty_df():
    assert build_indicator_snapshot(pd.DataFrame()) == {}
    assert build_indicator_snapshot(None) == {}


# ── categorize_failure ───────────────────────────────────────────────────────

def test_categorize_rejects_neutral_direction():
    with pytest.raises(ValueError):
        categorize_failure({}, "neutral")


def test_categorize_returns_empty_list_for_empty_snapshot():
    assert categorize_failure({}, "bullish") == []


def test_categorize_counter_trend_bullish_call_below_200ma():
    snap = {"above_200ma": False, "above_50ma": False}
    assert "counter_trend" in categorize_failure(snap, "bullish")


def test_categorize_counter_trend_bearish_call_above_200ma():
    snap = {"above_200ma": True, "above_50ma": True}
    assert "counter_trend" in categorize_failure(snap, "bearish")


def test_categorize_low_conviction_zone_bullish_call_overbought():
    snap = {"rsi_zone": "overbought"}
    assert "low_conviction_zone" in categorize_failure(snap, "bullish")


def test_categorize_low_conviction_zone_bearish_call_oversold():
    snap = {"rsi_zone": "oversold"}
    assert "low_conviction_zone" in categorize_failure(snap, "bearish")


def test_categorize_volume_anomaly_when_volume_surge_true():
    assert "volume_anomaly" in categorize_failure({"volume_surge": True}, "bullish")


def test_categorize_choppy_market_when_adx_below_25():
    assert "choppy_market" in categorize_failure({"strong_trend": False}, "bullish")


def test_categorize_rsi_divergence_present_bullish_call_bearish_divergence():
    snap = {"rsi_divergence": "bearish_divergence"}
    assert "rsi_divergence_present" in categorize_failure(snap, "bullish")


def test_categorize_elevated_vol_regime_from_vix_crisis():
    assert "elevated_vol_regime" in categorize_failure({"vix_regime": "Crisis"}, "bullish")
    assert "elevated_vol_regime" in categorize_failure({"vix_regime": "Elevated Fear"}, "bearish")
    assert "elevated_vol_regime" not in categorize_failure({"vix_regime": "Normal"}, "bullish")


def test_categorize_returns_uncategorized_only_via_caller_not_categorize_failure():
    # categorize_failure itself may return [] — the "uncategorized" fallback
    # is categorize_incorrect_predictions' job, not this function's.
    snap = {"above_200ma": True, "above_50ma": True, "rsi_zone": "neutral",
            "volume_surge": False, "strong_trend": True, "vix_regime": "Normal"}
    assert categorize_failure(snap, "bullish") == []


def test_categorize_can_return_multiple_categories():
    snap = {
        "above_200ma": False, "above_50ma": False, "rsi_zone": "neutral",
        "volume_surge": True, "strong_trend": False,
        "rsi_divergence": "bearish_divergence", "vix_regime": "Crisis",
    }
    cats = categorize_failure(snap, "bullish")
    assert set(cats) == {
        "counter_trend", "volume_anomaly", "choppy_market",
        "rsi_divergence_present", "elevated_vol_regime",
    }


# ── categorize_incorrect_predictions ─────────────────────────────────────────

def _history_row(direction, correct, snapshot=None, date="2026-07-01T10:00:00+00:00"):
    return {"direction": direction, "correct": correct, "date": date, "indicator_snapshot": snapshot}


def test_categorize_incorrect_predictions_filters_on_correct_false_only():
    history = pd.DataFrame([
        _history_row("bullish", True, {"rsi_zone": "overbought"}),
        _history_row("bearish", False, {"rsi_zone": "oversold"}),
        _history_row("bullish", None, {"rsi_zone": "overbought"}),
    ])
    result = categorize_incorrect_predictions(history)

    assert len(result) == 1
    assert result.iloc[0]["direction"] == "bearish"


def test_categorize_incorrect_predictions_uses_logged_snapshot_when_present():
    history = pd.DataFrame([
        _history_row("bullish", False, {"rsi_zone": "overbought"}),
    ])
    result = categorize_incorrect_predictions(history)

    assert result.iloc[0]["snapshot_source"] == "logged"
    assert "low_conviction_zone" in result.iloc[0]["failure_categories"]


def test_categorize_incorrect_predictions_recomputes_when_snapshot_missing_and_fetcher_given(synthetic_price_df):
    history = pd.DataFrame([
        _history_row("bullish", False, None, date=str(synthetic_price_df.index[300])),
    ])

    def fake_fetcher(ticker):
        return synthetic_price_df

    result = categorize_incorrect_predictions(
        history, price_history_fetcher=fake_fetcher, ticker="ZZFAKE",
    )

    assert result.iloc[0]["snapshot_source"] == "recomputed"
    assert isinstance(result.iloc[0]["failure_categories"], list)
    assert len(result.iloc[0]["failure_categories"]) >= 1


def test_categorize_incorrect_predictions_marks_unavailable_when_no_fetcher_and_no_snapshot():
    history = pd.DataFrame([_history_row("bullish", False, None)])
    result = categorize_incorrect_predictions(history)

    assert result.iloc[0]["snapshot_source"] == "unavailable"
    assert result.iloc[0]["failure_categories"] == ["uncategorized"]


def test_categorize_incorrect_predictions_empty_history_returns_empty_frame():
    result = categorize_incorrect_predictions(pd.DataFrame(columns=["direction", "correct", "date"]))
    assert result.empty
    assert "failure_categories" in result.columns


def test_categorize_incorrect_predictions_no_incorrect_rows_returns_empty_frame():
    history = pd.DataFrame([_history_row("bullish", True, {"rsi_zone": "overbought"})])
    result = categorize_incorrect_predictions(history)
    assert result.empty


# ── aggregate_failure_categories ─────────────────────────────────────────────

def test_aggregate_failure_categories_hand_computed_counts():
    categorized = pd.DataFrame({
        "failure_categories": [
            ["counter_trend"],
            ["counter_trend", "volume_anomaly"],
            ["uncategorized"],
        ],
    })
    agg = aggregate_failure_categories(categorized)

    assert agg["n_incorrect"] == 3
    assert agg["category_counts"]["counter_trend"] == 2
    assert agg["category_counts"]["volume_anomaly"] == 1
    assert agg["category_counts"]["uncategorized"] == 1
    assert agg["category_rates"]["counter_trend"] == pytest.approx(2 / 3, abs=1e-4)
    assert agg["top_category"] == "counter_trend"


def test_aggregate_failure_categories_empty_input_returns_zeroed_shape():
    agg = aggregate_failure_categories(pd.DataFrame())

    assert agg["n_incorrect"] == 0
    assert agg["top_category"] is None
    assert all(v == 0 for v in agg["category_counts"].values())
