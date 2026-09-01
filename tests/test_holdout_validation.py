"""
Tests for the out-of-sample holdout added to train_model.

train_model runs three sequential argmax searches (label scheme, XGB params, RF
params), all scored on the same history with no data withheld. The winner of many
tries is not an unbiased estimate — measured at roughly +7 points on random walks
containing no signal at all. The holdout exists to produce one number that no
search stage influenced, and to gate reliability on it.

These tests cover the split, the sample-size guards, and the significance
requirement. They do not assert particular accuracy values: the point of the
holdout is that its value is noisy, and pinning it would encode the noise.
"""
import numpy as np
import pandas as pd
import pytest

import analysis.ml_prediction as ml
from analysis.indicators import calculate_indicators
from config.settings import MIN_HOLDOUT_SAMPLES


def _series(n=560, seed=3, drift=0.0004, vol=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return calculate_indicators(pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.008,
        "Low": close * 0.992, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx))


# ── The split itself ─────────────────────────────────────────────────────────

def test_holdout_is_produced_and_recorded(isolated_storage):
    result = ml.train_model("HOLD", df=_series())
    assert result["error"] is None
    assert result["holdout_accuracy"] is not None, result.get("holdout_note")
    assert result["holdout_n"] >= MIN_HOLDOUT_SAMPLES
    assert 0.0 <= result["holdout_accuracy"] <= 1.0
    assert result["holdout_baseline_accuracy"] is not None
    assert result["holdout_ci95_halfwidth"] > 0


def test_holdout_metrics_are_persisted_for_later_reads(isolated_storage):
    """Model Lab reads these from the accuracy JSON long after training, so they
    have to survive the round trip rather than only appear in the return value."""
    result = ml.train_model("HOLD", df=_series())
    assert result["holdout_accuracy"] is not None, result.get("holdout_note")

    meta = ml._load_model_metadata("HOLD")
    for key in (
        "holdout_accuracy", "holdout_n", "holdout_baseline_accuracy",
        "holdout_edge_over_baseline", "holdout_ci95_halfwidth",
    ):
        assert key in meta, f"{key} missing from persisted metadata"

    assert meta["holdout_accuracy"] == pytest.approx(result["holdout_accuracy"])
    assert meta["holdout_n"] == result["holdout_n"]
    assert meta["selection_lift"] == result["selection_lift"]


def test_deployed_model_is_refit_on_all_data_including_the_holdout(isolated_storage):
    """
    The holdout produces an estimate; it is not a reason to ship a model fitted on
    80% of the history. n_train must therefore exceed the search-set size.
    """
    df = _series()
    result = ml.train_model("HOLD", df=df)
    assert result["error"] is None
    # The search set is ~80% of the frame; the final fit spans the whole thing.
    # Compare against the walk-forward's own training population as a lower bound.
    assert result["n_train"] > 0
    split_idx = int(len(df) * (1 - ml.HOLDOUT_FRACTION))
    X_search, y_search = ml.build_features(
        df.iloc[:split_idx], ticker="HOLD",
        forward_bars=result["horizon_days"],
        neutral_threshold=result["neutral_threshold_pct"] / 100,
    )
    n_search = len(ml._filter_directional(X_search, y_search)[0])
    assert result["n_train"] > n_search, (
        f"final fit used {result['n_train']} rows, no more than the "
        f"{n_search}-row search set — the holdout was not folded back in"
    )


def test_short_history_skips_the_holdout_rather_than_failing(isolated_storage):
    """Under 300 bars there isn't enough to spare, so reporting falls back to
    walk-forward and says so instead of erroring."""
    result = ml.train_model("SHORT", df=_series(n=280))
    assert result["error"] is None
    assert result["holdout_accuracy"] is None
    assert result["directional_accuracy"] is not None
    assert "holdout" in (result["holdout_note"] or "").lower()


# ── _evaluate_on_holdout directly ────────────────────────────────────────────

def _cfgs():
    return (
        ml._xgb_config(scale_pos_weight=1.0),
        ml._rf_config(n_features=len(ml.FEATURE_NAMES)),
    )


def test_holdout_evaluator_reports_too_few_rows_rather_than_a_number(isolated_storage):
    df = _series(n=560)
    xgb_cfg, rf_cfg = _cfgs()
    # Split all the way at the end, leaving almost nothing after the gap.
    out = ml._evaluate_on_holdout(
        df, "HOLD", 5, 0.005, xgb_cfg, rf_cfg,
        {"xgb": 0.65, "rf": 0.35}, search_end=df.index[-3],
    )
    assert out["accuracy"] is None
    assert "holdout" in out["reason"].lower()


@pytest.mark.parametrize("horizon", [3, 5, 10])
def test_holdout_starts_strictly_after_the_label_gap(isolated_storage, horizon):
    """
    The last training rows' label windows extend `horizon` days forward, so rows
    immediately after the split resolve against outcomes the model was fitted on.
    Every holdout row must therefore sit beyond search_end + horizon.

    Asserted on the boundary rather than on the row count: horizon changes the row
    count in two opposing directions at once (a wider gap removes rows, but longer
    forward returns clear the neutral band more often and add them), so counts are
    not a clean signal about the gap.
    """
    df = _series(n=560)
    xgb_cfg, rf_cfg = _cfgs()
    search_end = df.index[int(len(df) * 0.8)]

    out = ml._evaluate_on_holdout(
        df, "HOLD", horizon, 0.005, xgb_cfg, rf_cfg,
        {"xgb": 0.65, "rf": 0.35}, search_end,
    )
    assert out["holdout_start"] is not None, out["reason"]
    assert pd.Timestamp(out["holdout_start"]) > search_end + pd.Timedelta(days=horizon)


def test_holdout_evaluator_degrades_to_a_reason_on_bad_input(isolated_storage):
    xgb_cfg, rf_cfg = _cfgs()
    raw = pd.DataFrame({"Close": [1.0, 2.0, 3.0]},
                       index=pd.bdate_range("2024-01-01", periods=3))
    out = ml._evaluate_on_holdout(
        raw, "HOLD", 5, 0.005, xgb_cfg, rf_cfg,
        {"xgb": 0.65, "rf": 0.35}, search_end=raw.index[0],
    )
    assert out["accuracy"] is None
    assert out["reason"]


# ── The reliability gate ─────────────────────────────────────────────────────

def test_reliability_requires_beating_the_baseline_by_more_than_the_error_bar(
    isolated_storage,
):
    """
    A 2-point floor is meaningless at n=75, where the 95% interval on the accuracy
    is around +/-11 points — a model with no edge lands several points above
    baseline by luck routinely. Whenever a model is marked reliable, its holdout
    edge must exceed its own error bar.
    """
    for seed in (7, 11, 13, 21, 33):
        result = ml.train_model(f"GATE{seed}", df=_series(seed=seed, drift=0.0))
        if result["error"] or result["holdout_accuracy"] is None:
            continue
        if not result["is_reliable"]:
            continue
        edge = result["holdout_edge_over_baseline"]
        ci = result["holdout_ci95_halfwidth"]
        assert edge >= ci, (
            f"seed {seed} marked reliable on a {edge:+.3f} edge inside its own "
            f"+/-{ci:.3f} error bar"
        )
        assert result["holdout_accuracy"] >= 0.52


def test_a_model_losing_to_the_baseline_is_never_reliable(isolated_storage):
    """
    Strongly trending history makes the majority class dominant. A model can post a
    high raw accuracy there and still be worse than a constant guess — the case the
    flat 52% floor could not catch.
    """
    result = ml.train_model("TREND", df=_series(seed=21, drift=0.0025, vol=0.008))
    if result["error"] or result["holdout_accuracy"] is None:
        pytest.skip(f"no usable holdout: {result.get('holdout_note')}")
    if result["holdout_accuracy"] < result["holdout_baseline_accuracy"]:
        assert not result["is_reliable"], (
            f"accuracy {result['holdout_accuracy']:.3f} is below the "
            f"{result['holdout_baseline_accuracy']:.3f} baseline yet marked reliable"
        )


def test_reliability_reason_names_the_holdout_when_one_was_used(isolated_storage):
    result = ml.train_model("HOLD", df=_series())
    if result["holdout_accuracy"] is None:
        pytest.skip("no usable holdout")
    assert "holdout" in result["reliability_reason"].lower()
