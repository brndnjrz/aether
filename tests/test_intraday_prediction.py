"""
Tests for analysis/intraday_prediction.py.

Two priorities:

1. **Isolation.** The whole reason this module exists separately is that
   ml_prediction.predict() feeds four pages. These tests assert that nothing
   here writes to a daily model's files and that the daily module's public
   surface is untouched.

2. **The two label-correctness fixes**, which are the difference between a
   model worth believing and one that merely looks plausible:
   - forward-return labels must not span the overnight gap
   - the neutral band must scale with volatility, not be a fixed 0.5%

All data is synthetic; no network.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.intraday_prediction import (
    DEFAULT_INTERVAL,
    HORIZON_SEARCH_GRID,
    INTERVAL_SPECS,
    INTRADAY_FEATURE_NAMES,
    assess_tradeability,
    build_intraday_features,
    build_intraday_labels,
    get_intraday_prediction_history,
    load_metadata,
    model_exists,
    normalize_intraday_features,
    predict_intraday,
    save_intraday_prediction,
    train_intraday_model,
    _accuracy_path,
    _predictions_path,
    _rf_path,
    _to_market_tz,
    _xgb_path,
)
from tests.conftest import make_intraday_history


# ── Isolation from the daily model ──────────────────────────────────────────

def test_storage_paths_never_collide_with_daily_model():
    """
    The daily module writes SPY_xgb.pkl / SPY_predictions.jsonl. If this module
    produced the same names it would overwrite a trained daily model and mix two
    incompatible horizons into one history file.
    """
    for interval in INTERVAL_SPECS:
        assert _xgb_path("SPY", interval).name == f"SPY_{interval}_xgb.pkl"
        assert _rf_path("SPY", interval).name == f"SPY_{interval}_rf.pkl"
        assert _predictions_path("SPY", interval).name == f"SPY_{interval}_predictions.jsonl"
        assert _accuracy_path("SPY", interval).name == f"SPY_{interval}_accuracy.json"

    daily_names = {"SPY_xgb.pkl", "SPY_rf.pkl", "SPY_predictions.jsonl", "SPY_accuracy.json"}
    intraday_names = {
        p("SPY", i).name
        for i in INTERVAL_SPECS
        for p in (_xgb_path, _rf_path, _predictions_path, _accuracy_path)
    }
    assert daily_names.isdisjoint(intraday_names)


def test_daily_module_public_surface_is_untouched():
    """
    This module reuses ml_prediction's helpers read-only. If a future change
    renames or removes one, fail here rather than at runtime in the UI.
    """
    import analysis.ml_prediction as mlp

    for name in ("_run_walk_forward", "_xgb_config", "_rf_config",
                 "_to_binary_labels", "_filter_directional", "_directional_accuracy",
                 "predict", "train_model", "get_prediction_history", "_STORAGE_DIR"):
        assert hasattr(mlp, name), f"ml_prediction is missing {name}"


def test_training_writes_only_interval_scoped_files(intraday_indicators_df, isolated_intraday_storage):
    """A full training run must leave no file lacking the interval tag."""
    result = train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    assert result.get("error") is None, result.get("error")

    written = sorted(p.name for p in isolated_intraday_storage.iterdir())
    assert written, "training wrote nothing"
    for name in written:
        assert "_15m_" in name, f"{name} is not interval-scoped and could clobber a daily file"


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        _xgb_path("SPY", "1d")
    with pytest.raises(ValueError):
        _xgb_path("SPY", "7m")


# ── Session-boundary label masking ──────────────────────────────────────────

def test_labels_never_span_the_overnight_gap():
    """
    A bar near the session close has no in-session forward window, so its label
    must be NaN. Without this the model is trained to predict overnight gaps
    from intraday features, which is noise.
    """
    df = _to_market_tz(make_intraday_history(n_sessions=10, bars_per_session=26))
    horizon = 5
    labels, info = build_intraday_labels(df, horizon_bars=horizon, sigma_multiple=0.75)

    sessions = df.index.normalize()
    days = sorted(set(sessions))
    for day in days:
        day_labels = labels[sessions == day]
        # The last `horizon` bars of each session cannot have a label.
        assert day_labels.iloc[-horizon:].isna().all(), (
            f"session {day.date()} has labels within {horizon} bars of the close"
        )

    # Skip the first two sessions — the trailing volatility estimate needs a
    # 30-bar warm-up before any label can be decided at all.
    for day in days[2:]:
        day_labels = labels[sessions == day]
        assert day_labels.iloc[:-horizon].notna().any(), (
            f"session {day.date()} produced no labels outside the close window"
        )

    assert info["dropped_to_session_mask"] > 0


def test_session_mask_drop_count_scales_with_horizon():
    df = _to_market_tz(make_intraday_history(n_sessions=10, bars_per_session=26))
    dropped = {}
    for h in (3, 5, 10):
        _labels, info = build_intraday_labels(df, horizon_bars=h, sigma_multiple=0.75)
        dropped[h] = info["dropped_to_session_mask"]
    assert dropped[3] < dropped[5] < dropped[10]


def test_labels_do_not_depend_on_bars_beyond_the_trailing_window():
    """
    Bar t's label must depend only on closes at t and t+h, plus a *trailing*
    volatility estimate. Perturbing a bar far in the future must not change it.

    A full-series sigma estimate would fail this: the threshold would shift and
    with it every earlier label. That is label-definition leakage, and it is why
    the sigma estimate is a rolling window rather than a single global std.
    """
    df = _to_market_tz(make_intraday_history(n_sessions=20, bars_per_session=26, seed=9))
    horizon, vol_window = 3, 130
    base, _ = build_intraday_labels(df, horizon, 0.75, vol_window)

    tampered = df.copy()
    # Perturb a bar well past both the horizon and the trailing vol window of
    # the bar we assert on (index 200).
    tampered.iloc[420, tampered.columns.get_loc("Close")] *= 1.10
    after, _ = build_intraday_labels(tampered, horizon, 0.75, vol_window)

    assert base.iloc[200] == after.iloc[200]
    # And an early bar, far from the tampered region, is likewise untouched.
    assert base.iloc[150] == after.iloc[150]


# ── Volatility-scaled thresholds ────────────────────────────────────────────

def test_threshold_scales_with_realized_volatility():
    """
    A calm and a volatile series must get different neutral bands. A fixed 0.5%
    band would treat them identically and label almost everything neutral in the
    calm one.
    """
    calm = _to_market_tz(make_intraday_history(n_sessions=10, daily_vol=0.004, seed=1))
    wild = _to_market_tz(make_intraday_history(n_sessions=10, daily_vol=0.025, seed=1))

    _, calm_info = build_intraday_labels(calm, horizon_bars=5, sigma_multiple=0.75)
    _, wild_info = build_intraday_labels(wild, horizon_bars=5, sigma_multiple=0.75)

    assert wild_info["threshold"] > calm_info["threshold"] * 2


def test_threshold_scales_with_sqrt_horizon():
    df = _to_market_tz(make_intraday_history(n_sessions=10))
    _, h4 = build_intraday_labels(df, horizon_bars=4, sigma_multiple=0.75)
    _, h16 = build_intraday_labels(df, horizon_bars=16, sigma_multiple=0.75)
    # sqrt(16)/sqrt(4) == 2
    assert h16["threshold"] == pytest.approx(h4["threshold"] * 2, rel=0.01)


def _fixed_band_directional(df: pd.DataFrame, horizon: int, threshold: float) -> int:
    """Directional sample count under a fixed percentage band, session-masked."""
    fwd = df["Close"].pct_change(horizon).shift(-horizon)
    sess = pd.Series(df.index.normalize(), index=df.index)
    fwd = fwd.where(sess.shift(-horizon) == sess).dropna()
    return int((fwd.abs() > threshold).sum())


def test_vol_scaled_band_is_stable_across_volatility_regimes():
    """
    The property that matters. A fixed +/-0.5% band calibrated for daily bars
    behaves wildly differently depending on intraday volatility — near-usable in
    a busy market, catastrophic in a calm one, where it discards almost every
    row before training. The scaled band holds a roughly constant directional
    share in both, which is what makes the training sample unbiased.
    """
    calm = _to_market_tz(make_intraday_history(n_sessions=40, daily_vol=0.005, seed=3))
    busy = _to_market_tz(make_intraday_history(n_sessions=40, daily_vol=0.020, seed=3))

    calm_labels, calm_info = build_intraday_labels(calm, horizon_bars=5, sigma_multiple=0.75)
    busy_labels, busy_info = build_intraday_labels(busy, horizon_bars=5, sigma_multiple=0.75)

    calm_share = (calm_labels.dropna() != 0).mean()
    busy_share = (busy_labels.dropna() != 0).mean()

    # Scaled band: directional share is essentially regime-independent.
    assert abs(calm_share - busy_share) < 0.10, (
        f"scaled band drifted across regimes: calm {calm_share:.0%} vs busy {busy_share:.0%}"
    )
    for share in (calm_share, busy_share):
        assert 0.30 < share < 0.70, f"directional share {share:.0%} is unusable"

    # Fixed band: collapses in the calm regime.
    calm_fixed = _fixed_band_directional(calm, 5, 0.005)
    busy_fixed = _fixed_band_directional(busy, 5, 0.005)
    calm_scaled = int((calm_labels.dropna() != 0).sum())

    assert calm_fixed < busy_fixed / 3, "expected the fixed band to collapse when volatility is low"
    assert calm_scaled > calm_fixed * 3, (
        f"in the calm regime the scaled band kept {calm_scaled} directional samples "
        f"vs only {calm_fixed} for the fixed 0.5% band"
    )
    assert calm_info["threshold"] < busy_info["threshold"]


def test_bar_sigma_excludes_the_overnight_gap():
    """
    Sigma must be estimated from within-session returns only. Including the
    first return of each session (which spans the overnight gap) inflates the
    estimate and over-widens the neutral band, silently discarding real signal.
    """
    df = _to_market_tz(make_intraday_history(
        n_sessions=30, daily_vol=0.008, overnight_gap_vol=0.05, seed=5,
    ))
    _labels, info = build_intraday_labels(df, horizon_bars=5, sigma_multiple=0.75)

    naive_sigma = float(df["Close"].pct_change().std())   # includes the gaps
    assert info["bar_sigma"] < naive_sigma / 2, (
        "bar_sigma looks contaminated by overnight gaps"
    )


def test_larger_sigma_multiple_yields_more_neutral_rows():
    df = _to_market_tz(make_intraday_history(n_sessions=20))
    shares = []
    for m in (0.5, 0.75, 1.0):
        labels, _ = build_intraday_labels(df, horizon_bars=5, sigma_multiple=m)
        labelled = labels.dropna()
        shares.append((labelled == 0).mean())
    assert shares[0] < shares[1] < shares[2]


def test_flat_series_raises_rather_than_dividing_by_zero():
    df = _to_market_tz(make_intraday_history(n_sessions=3))
    df["Close"] = 100.0
    with pytest.raises(ValueError, match="volatility"):
        build_intraday_labels(df, horizon_bars=5, sigma_multiple=0.75)


# ── Features ────────────────────────────────────────────────────────────────

def test_feature_matrix_has_expected_columns_and_no_nan(intraday_indicators_df):
    X, y, info = build_intraday_features(intraday_indicators_df, ticker="TEST", horizon_bars=5)

    assert list(X.columns) == INTRADAY_FEATURE_NAMES
    assert not X.isna().any().any()
    assert np.isfinite(X.values).all()
    assert set(y.unique()) <= {-1, 0, 1}
    assert len(X) == len(y) == info["n_samples"]


def test_intraday_feature_list_is_independent_of_the_frozen_daily_list():
    """
    feature_engineering.FEATURE_NAMES is frozen for saved daily-model
    compatibility. This module must not reuse or mutate it.
    """
    from data.feature_engineering import FEATURE_NAMES

    assert INTRADAY_FEATURE_NAMES is not FEATURE_NAMES
    assert "day_of_week" not in INTRADAY_FEATURE_NAMES, "day_of_week is near-useless intraday"
    for expected in ("minutes_since_open_norm", "price_vs_vwap", "pos_in_session_range"):
        assert expected in INTRADAY_FEATURE_NAMES


def test_minutes_since_open_is_bounded_and_rises_within_a_session(intraday_indicators_df):
    X, _y, _info = build_intraday_features(intraday_indicators_df, horizon_bars=5)
    col = X["minutes_since_open_norm"]
    assert col.min() >= 0.0 and col.max() <= 1.0
    assert col.nunique() > 5, "time-of-day feature is not varying"


def test_normalization_leaves_no_nan_and_skips_categorical_columns():
    df = _to_market_tz(make_intraday_history(n_sessions=20))
    from analysis.indicators import calculate_indicators
    from analysis.intraday_prediction import _add_intraday_features

    featured = _add_intraday_features(calculate_indicators(df))[INTRADAY_FEATURE_NAMES]
    normed = normalize_intraday_features(featured.dropna(), window=260)

    assert not normed.isna().any().any()
    # macd_hist_sign is a sign flag — normalizing it would destroy its meaning.
    assert set(np.unique(normed["macd_hist_sign"])) <= {-1.0, 0.0, 1.0}


def test_features_reject_too_little_history():
    small = _to_market_tz(make_intraday_history(n_sessions=2, bars_per_session=26))
    from analysis.indicators import calculate_indicators
    with pytest.raises(ValueError, match="at least"):
        build_intraday_features(calculate_indicators(small))


def test_naive_and_utc_indexes_produce_identical_features():
    """
    A naive index must be treated as market-local, not converted from UTC —
    otherwise 09:30 ET reads as 05:30 and every session feature is wrong.
    """
    from analysis.indicators import calculate_indicators

    aware = calculate_indicators(make_intraday_history(n_sessions=20))
    naive = aware.copy()
    naive.index = naive.index.tz_localize(None)
    utc = aware.copy()
    utc.index = utc.index.tz_convert("UTC")

    Xa, _, _ = build_intraday_features(aware, horizon_bars=5)
    Xn, _, _ = build_intraday_features(naive, horizon_bars=5)
    Xu, _, _ = build_intraday_features(utc, horizon_bars=5)

    assert len(Xa) == len(Xn) == len(Xu)
    np.testing.assert_allclose(
        Xa["minutes_since_open_norm"].values, Xn["minutes_since_open_norm"].values
    )
    np.testing.assert_allclose(
        Xa["minutes_since_open_norm"].values, Xu["minutes_since_open_norm"].values
    )


# ── Cost-aware tradeability ─────────────────────────────────────────────────

def test_marginal_accuracy_is_not_tradeable_after_costs():
    """
    A 53%-accurate model over a 0.35% sigma move is real but unprofitable once
    spread and commission are paid. Reporting accuracy alone would hide that.
    """
    verdict = assess_tradeability(0.53, horizon_sigma=0.0035, round_trip_cost_pct=0.02)
    assert verdict["gross_edge_pct"] > 0
    assert verdict["net_edge_pct"] < verdict["gross_edge_pct"]
    assert verdict["breakeven_accuracy"] > 0.5


def test_higher_accuracy_becomes_tradeable():
    weak = assess_tradeability(0.505, horizon_sigma=0.0035)
    strong = assess_tradeability(0.60, horizon_sigma=0.0035)
    assert not weak["is_tradeable"]
    assert strong["is_tradeable"]
    assert strong["net_edge_pct"] > weak["net_edge_pct"]


def test_costs_raise_the_breakeven_accuracy_bar():
    cheap = assess_tradeability(0.55, horizon_sigma=0.0035, round_trip_cost_pct=0.01)
    pricey = assess_tradeability(0.55, horizon_sigma=0.0035, round_trip_cost_pct=0.10)
    assert pricey["breakeven_accuracy"] > cheap["breakeven_accuracy"]
    assert pricey["net_edge_pct"] < cheap["net_edge_pct"]


# ── Train / predict ─────────────────────────────────────────────────────────

def test_train_returns_metrics_and_persists_a_loadable_model(intraday_indicators_df, isolated_intraday_storage):
    result = train_intraday_model("TEST", "15m", df=intraday_indicators_df)

    assert result["error"] is None
    assert result["horizon_bars"] in HORIZON_SEARCH_GRID
    assert 0.0 <= result["directional_accuracy"] <= 1.0
    assert result["horizon_minutes"] == result["horizon_bars"] * 15
    assert "tradeability" in result
    assert model_exists("TEST", "15m")

    meta = load_metadata("TEST", "15m")
    assert meta["interval"] == "15m"
    assert meta["horizon_bars"] == result["horizon_bars"]


def test_train_reports_structured_error_on_thin_history(isolated_intraday_storage):
    from analysis.indicators import calculate_indicators

    thin = calculate_indicators(make_intraday_history(n_sessions=3))
    result = train_intraday_model("TEST", "15m", df=thin)

    assert result["error"] is not None
    assert not model_exists("TEST", "15m")


def test_predict_auto_trains_then_returns_a_signal(intraday_indicators_df, isolated_intraday_storage):
    assert not model_exists("TEST", "15m")
    result = predict_intraday("TEST", "15m", df=intraday_indicators_df)

    assert result["error"] is None
    assert result["direction"] in ("bullish", "bearish", "neutral")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["confidence"] in ("low", "medium", "high")
    assert result["interval"] == "15m"
    assert result["price_at_prediction"] > 0
    assert model_exists("TEST", "15m")


def test_predict_without_auto_train_errors_cleanly(intraday_indicators_df, isolated_intraday_storage):
    result = predict_intraday("TEST", "15m", df=intraday_indicators_df, auto_train=False)
    assert result["error"] is not None
    assert "not trained" in result["error"].lower() or "no 15m model" in result["error"].lower()


def test_predict_reuses_an_existing_model(intraday_indicators_df, isolated_intraday_storage):
    train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    mtime = _xgb_path("TEST", "15m").stat().st_mtime

    predict_intraday("TEST", "15m", df=intraday_indicators_df)
    assert _xgb_path("TEST", "15m").stat().st_mtime == mtime, "predict retrained instead of loading"


def test_two_intervals_coexist_without_overwriting(isolated_intraday_storage):
    from analysis.indicators import calculate_indicators

    df15 = calculate_indicators(make_intraday_history(n_sessions=40, bars_per_session=26, interval_minutes=15))
    df30 = calculate_indicators(make_intraday_history(n_sessions=40, bars_per_session=13, interval_minutes=30))

    r15 = train_intraday_model("TEST", "15m", df=df15)
    r30 = train_intraday_model("TEST", "30m", df=df30)

    assert r15["error"] is None and r30["error"] is None
    assert model_exists("TEST", "15m") and model_exists("TEST", "30m")
    assert load_metadata("TEST", "15m")["interval"] == "15m"
    assert load_metadata("TEST", "30m")["interval"] == "30m"


# ── History ─────────────────────────────────────────────────────────────────

def test_history_empty_before_any_prediction(isolated_intraday_storage):
    hist = get_intraday_prediction_history("TEST", "15m", resolve=False)
    assert hist.empty
    assert "direction" in hist.columns


def test_prediction_round_trips_into_history(intraday_indicators_df, isolated_intraday_storage):
    predict_intraday("TEST", "15m", df=intraday_indicators_df)
    hist = get_intraday_prediction_history("TEST", "15m", resolve=False)

    assert len(hist) == 1
    assert hist.iloc[0]["direction"] in ("bullish", "bearish", "neutral")
    assert pd.notna(hist.iloc[0]["date"])


def test_history_appends_rather_than_overwrites(intraday_indicators_df, isolated_intraday_storage):
    predict_intraday("TEST", "15m", df=intraday_indicators_df)
    predict_intraday("TEST", "15m", df=intraday_indicators_df)
    assert len(get_intraday_prediction_history("TEST", "15m", resolve=False)) == 2


def test_history_parses_a_mixed_naive_and_aware_log(isolated_intraday_storage):
    """
    Regression guard for the bug that hid weeks of daily predictions: a log
    mixing naive and tz-aware timestamps must parse every row, not coerce the
    newer ones to NaT.
    """
    path = _predictions_path("TEST", "15m")
    rows = [
        {"date": "2026-07-06T10:00:00", "direction": "bullish", "probability": 0.7,
         "confidence": "high", "horizon_minutes": 75},
        {"date": "2026-07-07T10:00:00-04:00", "direction": "bearish", "probability": 0.3,
         "confidence": "high", "horizon_minutes": 75},
    ]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    hist = get_intraday_prediction_history("TEST", "15m", resolve=False)
    assert len(hist) == 2
    assert hist["date"].notna().all(), "a tz-aware row was coerced to NaT"


def test_history_tolerates_malformed_lines(isolated_intraday_storage):
    path = _predictions_path("TEST", "15m")
    with open(path, "w") as f:
        f.write('{"date": "2026-07-06T10:00:00-04:00", "direction": "bullish"}\n')
        f.write("not json at all\n")
        f.write("\n")
    assert len(get_intraday_prediction_history("TEST", "15m", resolve=False)) == 1


def test_save_records_the_interval_so_resolution_uses_the_right_bars(isolated_intraday_storage):
    save_intraday_prediction("TEST", "15m", {
        "direction": "bullish", "probability": 0.7, "confidence": "high",
        "horizon_bars": 5, "horizon_minutes": 75, "price_at_prediction": 500.0,
        "bar_timestamp": "2026-07-06T10:00:00-04:00",
    })
    with open(_predictions_path("TEST", "15m")) as f:
        record = json.loads(f.readline())
    assert record["interval"] == "15m"
    assert record["horizon_bars"] == 5
    assert record["actual_outcome"] is None
