"""
Minimal regression suite for analysis/ml_prediction.py.

Purpose: this suite exists to catch exactly the class of outage that
happened on 2026-07 — an import-time crash in the ml_prediction dependency
chain (sklearn/xgboost/narwhals) that silently killed the whole Trading
page for 11 days before anyone noticed the prediction history had gone
stale. It is deliberately NOT exhaustive; it covers:

  - the module and its transitive dependencies import cleanly
  - train_model() / predict() / get_prediction_history() / evaluate_model()
    all run without error on synthetic data and return sane shapes
  - predict() -> save_prediction() -> get_prediction_history() persistence
    round-trip actually reflects a new entry (this is the exact mechanism
    behind "prediction history table stopped getting new rows")
  - the reliability gate isn't a rubber stamp: a pure random walk (no real
    directional signal) must not be reported as is_reliable=True

No network access is used or required — synthetic OHLCV data is run
through the real calculate_indicators() pipeline (see conftest.py), since
yfinance is unreachable in sandboxed/CI environments.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ── Import-time regression guard ────────────────────────────────────────────
# This is the single test that would have caught the outage immediately: if
# analysis.ml_prediction (or anything it imports — sklearn, xgboost, narwhals,
# data.feature_engineering) is broken at import time, this fails loudly in CI
# instead of silently killing pages/trading.py for 11 days in production.
def test_ml_prediction_module_imports_cleanly():
    import analysis.ml_prediction as ml_prediction

    for name in ("predict", "train_model", "get_prediction_history", "evaluate_model"):
        assert hasattr(ml_prediction, name), f"analysis.ml_prediction is missing {name}"


def test_pages_trading_imports_cleanly():
    """
    pages/trading.py imports predict/train_model/get_prediction_history/
    evaluate_model at module level (unguarded, unlike research.py, which wraps
    the same import in try/except). If that import
    chain breaks, the entire Trading page — not just ML — goes down.
    """
    import pages.trading  # noqa: F401 — import success is the assertion


# ── train_model() ───────────────────────────────────────────────────────────

def test_train_model_happy_path(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model

    result = train_model("ZZTEST", synthetic_indicators_df)

    assert result["error"] is None
    assert result["ticker"] == "ZZTEST"
    assert result["n_train"] > 0
    assert 0.0 <= result["directional_accuracy"] <= 1.0
    assert result["trained_at"] is not None
    assert (isolated_storage / "ZZTEST_xgb.pkl").exists()
    assert (isolated_storage / "ZZTEST_rf.pkl").exists()
    assert (isolated_storage / "ZZTEST_accuracy.json").exists()


def test_train_model_pure_random_walk_is_not_reliable(isolated_storage):
    """
    Reliability-gate sanity check: a pure random walk (drift=0, no real
    directional signal) must not clear the is_reliable bar. If this ever
    starts returning is_reliable=True, the gate has become a rubber stamp
    and can no longer be trusted to flag weak models.
    """
    from analysis.indicators import calculate_indicators
    from analysis.ml_prediction import train_model
    from tests.conftest import _make_synthetic_ohlcv

    random_walk_raw = _make_synthetic_ohlcv(seed=7, drift=0.0)
    random_walk_df = calculate_indicators(random_walk_raw)

    result = train_model("ZZRANDOM", random_walk_df)

    assert result["error"] is None
    assert result["is_reliable"] is False


# ── predict() ────────────────────────────────────────────────────────────────

def test_predict_auto_trains_when_no_model_on_disk(synthetic_indicators_df, isolated_storage):
    """predict() must train a model on the fly if none exists yet on disk."""
    from analysis.ml_prediction import predict

    assert not (isolated_storage / "ZZAUTO_xgb.pkl").exists()

    result = predict("ZZAUTO", synthetic_indicators_df)

    assert result["error"] is None
    assert result["direction"] in ("bullish", "bearish", "neutral")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["confidence"] in ("high", "medium", "low")
    assert (isolated_storage / "ZZAUTO_xgb.pkl").exists()


@pytest.mark.parametrize(
    "bad_df",
    [
        pytest.param(pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), id="empty"),
        pytest.param(
            pd.DataFrame(
                {"Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [1000]},
                index=[pd.Timestamp("2026-07-01")],
            ),
            id="single_row",
        ),
        pytest.param(None, id="none"),
    ],
)
def test_train_model_edge_cases_fail_with_structured_error(bad_df, isolated_storage, monkeypatch):
    """
    train_model() must never raise for known-bad inputs — it must always
    return a dict with `error` populated. These are the inputs a caller is
    most likely to accidentally pass (empty history, a brand-new ticker with
    one bar, or a data-fetch that returned nothing).
    """
    from analysis.ml_prediction import train_model

    if bad_df is None:
        # Avoid a real network call — get_price_history() would try to hit
        # yfinance for a None df; short-circuit it to return None directly,
        # which is what "no price data available" looks like in production.
        import analysis.ml_prediction as ml_prediction
        monkeypatch.setattr(
            "data.price_data.get_price_history", lambda *a, **k: None
        )

    result = train_model("ZZEDGE", bad_df)

    assert result["error"] is not None
    assert isinstance(result["error"], str)


def test_train_model_raw_ohlcv_without_indicators_fails_with_structured_error(isolated_storage):
    """
    Regression test: train_model()/select_label_scheme() used to let a raw
    KeyError (e.g. "'RSI'") escape uncaught when df skipped
    calculate_indicators() — the caller got an unhandled exception instead
    of the documented {"error": ...} dict. Any df missing the indicator
    columns build_features() expects must degrade to a structured error.
    """
    from analysis.ml_prediction import train_model

    dates = pd.bdate_range("2024-06-01", periods=520)
    raw_no_indicators = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000},
        index=dates,
    )

    result = train_model("ZZRAWCOLS", raw_no_indicators)

    assert result["error"] is not None
    assert "RSI" in result["error"] or "calculate_indicators" in result["error"]


def test_predict_raw_ohlcv_without_indicators_fails_with_structured_error(isolated_storage):
    """Same regression as above, exercised through predict()'s auto-train fallback."""
    from analysis.ml_prediction import predict

    dates = pd.bdate_range("2024-06-01", periods=520)
    raw_no_indicators = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000_000},
        index=dates,
    )

    result = predict("ZZRAWCOLSPREDICT", raw_no_indicators)

    assert result["error"] is not None


def test_predict_on_truncated_df_fails_with_structured_error(synthetic_indicators_df, isolated_storage):
    """
    predict() with fewer than the ~60-bar warm-up build_predict_row() needs
    must return a structured error, not crash or silently produce garbage.
    """
    from analysis.ml_prediction import predict, train_model

    train_model("ZZTRUNC", synthetic_indicators_df)
    truncated_df = synthetic_indicators_df.iloc[:30]

    result = predict("ZZTRUNC", truncated_df)

    assert result["error"] is not None


def test_predict_reuses_existing_model(synthetic_indicators_df, isolated_storage):
    """A second predict() call should reuse the saved model, not retrain from scratch."""
    from analysis.ml_prediction import predict, train_model

    train_model("ZZREUSE", synthetic_indicators_df)
    mtime_before = (isolated_storage / "ZZREUSE_xgb.pkl").stat().st_mtime

    result = predict("ZZREUSE", synthetic_indicators_df)

    mtime_after = (isolated_storage / "ZZREUSE_xgb.pkl").stat().st_mtime
    assert result["error"] is None
    assert mtime_after == mtime_before  # model file untouched — no retrain happened


# ── get_prediction_history() persistence round-trip ─────────────────────────

def test_prediction_history_empty_before_any_predictions(isolated_storage):
    from analysis.ml_prediction import get_prediction_history

    history = get_prediction_history("ZZNEVER")

    assert isinstance(history, pd.DataFrame)
    assert len(history) == 0
    assert list(history.columns) == [
        "date", "direction", "probability", "confidence", "actual_outcome", "correct",
    ]


def test_predict_persists_to_history(synthetic_indicators_df, isolated_storage):
    """
    This is the core regression test for the outage: predict() must call
    save_prediction() internally, and the very next get_prediction_history()
    call must reflect the new row. If this round-trip silently breaks again
    (independent of any import crash), it would reproduce a gap identical to
    the one that went unnoticed for 11 days.
    """
    from analysis.ml_prediction import predict, get_prediction_history

    before = get_prediction_history("ZZHIST")
    assert len(before) == 0

    result = predict("ZZHIST", synthetic_indicators_df)
    assert result["error"] is None

    after = get_prediction_history("ZZHIST")
    assert len(after) == 1
    assert after.iloc[0]["direction"] == result["direction"]
    assert after.iloc[0]["probability"] == pytest.approx(result["probability"])


def test_predict_appends_without_overwriting(synthetic_indicators_df, isolated_storage):
    """Multiple predict() calls should append, not overwrite, the JSONL log."""
    from analysis.ml_prediction import predict, get_prediction_history

    predict("ZZAPPEND", synthetic_indicators_df)
    predict("ZZAPPEND", synthetic_indicators_df)
    predict("ZZAPPEND", synthetic_indicators_df)

    history = get_prediction_history("ZZAPPEND")
    assert len(history) == 3


# ── evaluate_model() ─────────────────────────────────────────────────────────

def test_evaluate_model_happy_path(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import evaluate_model, train_model

    train_model("ZZEVAL", synthetic_indicators_df)
    result = evaluate_model("ZZEVAL", synthetic_indicators_df)

    assert result["error"] is None
    assert result["ticker"] == "ZZEVAL"
    assert 0.0 <= result["directional_accuracy"] <= 1.0
    assert isinstance(result["class_balance"], dict)
    assert result["horizon_days"] is not None


def test_evaluate_model_reflects_logged_predictions(synthetic_indicators_df, isolated_storage):
    """evaluate_model()'s total_predictions must match what's actually in the log."""
    from analysis.ml_prediction import evaluate_model, predict

    predict("ZZEVALHIST", synthetic_indicators_df)
    predict("ZZEVALHIST", synthetic_indicators_df)

    result = evaluate_model("ZZEVALHIST", synthetic_indicators_df)

    assert result["error"] is None
    assert result["total_predictions"] == 2
