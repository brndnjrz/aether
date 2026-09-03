"""
Tests for analysis/meta_labeling.py.

The central assertion is the refusal. A meta model cannot manufacture signal: if
the primary model's errors are unpredictable, there is nothing to filter on, and
saying so is the correct result. A meta filter that always claims an improvement
would be worse than none, because it would suppress signals for no reason while
looking like a sophistication.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.meta_labeling import (
    MIN_META_SAMPLES,
    MIN_META_SAMPLES_FOR_USE,
    MIN_PRECISION_GAIN,
    build_meta_features,
    build_meta_labels,
    evaluate_meta_model,
    fit_meta_model,
    meta_verdict,
)


def _primary(n=900, seed=4, high_conv_acc=0.80, low_conv_acc=0.40):
    """
    A primary model whose failures ARE predictable: it is right `high_conv_acc`
    of the time on high-conviction calls and `low_conv_acc` on low-conviction
    ones. f2 correlates with conviction, so a meta model has something to find.
    """
    rng = np.random.default_rng(seed)
    conviction = rng.uniform(0, 1, n)
    correct = (
        rng.uniform(0, 1, n) < np.where(conviction > 0.5, high_conv_acc, low_conv_acc)
    ).astype(int)
    sign = np.where(rng.uniform(0, 1, n) < 0.5, 1, -1)
    prob = 0.5 + sign * conviction / 2
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": conviction * 3 + rng.normal(0, 0.3, n),
    })
    return X, prob, correct


# ── features and labels ──────────────────────────────────────────────────────

def test_conviction_is_symmetric_around_a_coin_flip():
    """
    A bullish 0.9 and a bearish 0.1 are equally strong calls. Feeding raw
    probability would teach the meta model "bearish calls fail more" when the real
    pattern is "weak calls fail more" — different claims.
    """
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    out = build_meta_features(X, np.array([0.9, 0.1, 0.5]))
    assert out["primary_conviction"].iloc[0] == pytest.approx(0.8)
    assert out["primary_conviction"].iloc[1] == pytest.approx(0.8)
    assert out["primary_conviction"].iloc[2] == pytest.approx(0.0)


def test_meta_label_is_was_the_primary_right_not_which_way_price_went():
    prob = np.array([0.9, 0.9, 0.1, 0.1])      # bull, bull, bear, bear
    truth = np.array([1, 0, 0, 1])             # up,   down, down, up
    assert list(build_meta_labels(prob, truth)) == [1, 0, 1, 0]


def test_features_carry_the_primary_output_alongside_market_state():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = build_meta_features(X, np.array([0.7, 0.3]))
    assert {"a", "b", "primary_prob", "primary_conviction", "primary_is_bullish"} <= set(out.columns)
    assert list(out["primary_is_bullish"]) == [1, 0]


# ── it finds a filter when one exists ────────────────────────────────────────

def test_finds_a_filter_when_the_primary_fails_predictably():
    X, prob, correct = _primary()
    out = evaluate_meta_model(build_meta_features(X, prob), correct, gap=5)

    assert out["usable"], out["reason"]
    assert out["best"]["gain"] >= MIN_PRECISION_GAIN
    assert out["best"]["precision"] > out["base_precision"]


def test_the_filter_trades_coverage_for_precision():
    """Taking fewer signals is the mechanism, so coverage must fall below 1.0 and
    be reported next to the precision it bought."""
    X, prob, correct = _primary()
    out = evaluate_meta_model(build_meta_features(X, prob), correct, gap=5)
    assert out["usable"]
    assert 0.0 < out["best"]["coverage"] < 1.0
    assert out["best"]["n_taken"] > 0


# ── it refuses when nothing is there ─────────────────────────────────────────

def test_refuses_when_the_primary_fails_at_random():
    """
    The load-bearing test. Errors with no structure must produce usable=False,
    not a filter fitted to noise.
    """
    X, prob, _ = _primary()
    rng = np.random.default_rng(99)
    random_errors = (rng.uniform(0, 1, len(prob)) < 0.55).astype(int)

    out = evaluate_meta_model(build_meta_features(X, prob), random_errors, gap=5)
    assert not out["usable"]
    assert out["reason"]


def test_refuses_on_too_few_calls():
    X, prob, correct = _primary(n=MIN_META_SAMPLES - 50)
    out = evaluate_meta_model(build_meta_features(X, prob), correct, gap=5)
    assert not out["usable"]
    assert str(MIN_META_SAMPLES) in out["reason"]


def test_refuses_when_the_primary_was_right_every_time():
    """One class means nothing to separate."""
    X, prob, _ = _primary()
    out = evaluate_meta_model(
        build_meta_features(X, prob), np.ones(len(prob), dtype=int), gap=5,
    )
    assert not out["usable"]
    assert "one class" in out["reason"].lower()


def test_a_threshold_keeping_almost_nothing_is_not_selected():
    """
    Raw argmax over precision drifts to the highest cutoff, where a couple of
    surviving signals happened to both be right. Selection is restricted to
    thresholds retaining a workable share.
    """
    X, prob, correct = _primary()
    out = evaluate_meta_model(build_meta_features(X, prob), correct, gap=5)
    if out["best"]:
        assert out["best"]["coverage"] >= 0.20


def test_base_precision_is_always_reported_for_comparison():
    """Precision means nothing without the take-everything number beside it."""
    X, prob, correct = _primary()
    out = evaluate_meta_model(build_meta_features(X, prob), correct, gap=5)
    assert out["base_precision"] is not None
    for row in out["by_threshold"]:
        if row["precision"] is not None:
            assert row["gain"] == pytest.approx(row["precision"] - out["base_precision"], abs=1e-4)


# ── inference defaults to taking the signal ──────────────────────────────────

def test_missing_model_passes_the_signal_through():
    """
    A filter that silently suppressed signals because it failed to load would be
    worse than no filter. The safe direction is always to leave the primary
    model's call alone.
    """
    out = meta_verdict(None, pd.DataFrame({"a": [1.0]}), threshold=0.5, n_training_samples=999)
    assert out["take"] is True
    assert out["meta_probability"] is None


def test_a_model_trained_on_too_little_is_not_acted_on():
    X, prob, correct = _primary()
    model = fit_meta_model(build_meta_features(X, prob), correct)
    assert model is not None

    row = build_meta_features(X.head(1), prob[:1])
    out = meta_verdict(model, row, threshold=0.5,
                       n_training_samples=MIN_META_SAMPLES_FOR_USE - 1)
    assert out["take"] is True
    assert str(MIN_META_SAMPLES_FOR_USE) in out["reason"]


def test_a_broken_model_passes_the_signal_through():
    class Exploding:
        def predict_proba(self, _X):
            raise RuntimeError("boom")

    out = meta_verdict(Exploding(), pd.DataFrame({"a": [1.0]}),
                       threshold=0.5, n_training_samples=999)
    assert out["take"] is True
    assert "boom" in out["reason"]


def test_verdict_respects_the_threshold():
    X, prob, correct = _primary()
    X_meta = build_meta_features(X, prob)
    model = fit_meta_model(X_meta, correct)
    row = X_meta.head(1)

    lenient = meta_verdict(model, row, threshold=0.0, n_training_samples=999)
    strict = meta_verdict(model, row, threshold=1.01, n_training_samples=999)
    assert lenient["take"] is True
    assert strict["take"] is False
    assert lenient["meta_probability"] == strict["meta_probability"]


def test_fit_returns_none_rather_than_a_useless_model():
    X, prob, correct = _primary(n=MIN_META_SAMPLES - 10)
    assert fit_meta_model(build_meta_features(X, prob), correct) is None
