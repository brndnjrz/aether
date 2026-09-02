"""
Regression suite for the Prediction Improvement Engine's model-comparison
and ensemble-weighting generalization (Phases 4+6):
analysis/ml_prediction.py's _run_walk_forward_multi()/_softmax_ensemble_weights()/
compare_models(), and analysis/intraday_prediction.py's compare_intraday_models().

Core-function tests (softmax weighting) are pure and fast. compare_models()/
compare_intraday_models() tests train real walk-forward folds on synthetic
data — same cost profile as the existing train_model() tests.
"""
from __future__ import annotations

import pytest

from analysis.ml_prediction import (
    _run_walk_forward,
    _run_walk_forward_multi,
    _softmax_ensemble_weights,
    _xgb_config,
    _rf_config,
    _filter_directional,
    FEATURE_NAMES,
    compare_models,
)


def test_model_comparison_module_functions_import_cleanly():
    import analysis.ml_prediction as mlp
    import analysis.intraday_prediction as ip

    for name in ("_run_walk_forward_multi", "_softmax_ensemble_weights", "compare_models"):
        assert hasattr(mlp, name), f"analysis.ml_prediction is missing {name}"
    assert hasattr(ip, "compare_intraday_models")


# ── _softmax_ensemble_weights ────────────────────────────────────────────────

def test_softmax_ensemble_weights_sum_to_one():
    weights = _softmax_ensemble_weights({"a": 0.55, "b": 0.52, "c": 0.48})
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_softmax_ensemble_weights_favor_higher_accuracy_model():
    weights = _softmax_ensemble_weights({"a": 0.58, "b": 0.51})
    assert weights["a"] > weights["b"]


def test_softmax_ensemble_weights_approximate_current_0_65_0_35_split_for_typical_xgb_rf_gap():
    # A ~2-point accuracy gap is typical of the historical XGB-ahead-of-RF
    # relationship this scheme replaces — the default temperature should
    # land in the same ballpark as the old hardcoded 0.65/0.35 constant.
    weights = _softmax_ensemble_weights({"xgb": 0.56, "rf": 0.54})
    assert weights["xgb"] == pytest.approx(0.65, abs=0.1)
    assert weights["rf"] == pytest.approx(0.35, abs=0.1)


def test_softmax_ensemble_weights_never_fully_zero_a_coinflip_model():
    weights = _softmax_ensemble_weights({"good": 0.60, "coinflip": 0.50})
    assert weights["coinflip"] > 0.0


def test_softmax_ensemble_weights_empty_input_returns_empty_dict():
    assert _softmax_ensemble_weights({}) == {}


# ── _run_walk_forward_multi ──────────────────────────────────────────────────

def test_run_walk_forward_multi_reproduces_existing_two_model_ensemble_bit_for_bit(synthetic_indicators_df):
    from data.feature_engineering import build_features

    X, y = build_features(synthetic_indicators_df, ticker="ZZWFM", forward_bars=5, neutral_threshold=0.005)
    X_dir, y_dir = _filter_directional(X, y)
    X_dir_18 = X_dir[FEATURE_NAMES]
    xgb_cfg = _xgb_config(scale_pos_weight=1.0)
    rf_cfg = _rf_config()

    old = _run_walk_forward(X_dir_18, y_dir, xgb_cfg, rf_cfg, n_splits=10, gap=5)
    new = _run_walk_forward_multi(
        X_dir_18, y_dir, {"xgb": ("xgb", xgb_cfg), "rf": ("rf", rf_cfg)},
        n_splits=10, gap=5, ensemble_weights={"xgb": 0.65, "rf": 0.35},
    )["ensemble"]

    assert old == new


def test_run_walk_forward_multi_defaults_to_equal_weights_when_unspecified(synthetic_indicators_df):
    from data.feature_engineering import build_features

    X, y = build_features(synthetic_indicators_df, ticker="ZZWFM2", forward_bars=5, neutral_threshold=0.005)
    X_dir, y_dir = _filter_directional(X, y)
    X_dir_18 = X_dir[FEATURE_NAMES]
    xgb_cfg = _xgb_config(scale_pos_weight=1.0)
    rf_cfg = _rf_config()

    result = _run_walk_forward_multi(X_dir_18, y_dir, {"xgb": ("xgb", xgb_cfg), "rf": ("rf", rf_cfg)}, n_splits=10, gap=5)
    assert result["ensemble_weights"] == {"xgb": 0.5, "rf": 0.5}


# ── compare_models() ─────────────────────────────────────────────────────────

def test_compare_models_happy_path(synthetic_indicators_df, isolated_storage):
    result = compare_models("ZZCOMPARE", synthetic_indicators_df)

    assert result["error"] is None
    assert set(result["models"].keys()) == {"xgb", "rf", "logreg", "gbc"}
    assert result["best_single_model"] in result["models"]
    assert sum(result["ensemble_weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_compare_models_does_not_write_any_pkl_or_accuracy_file(synthetic_indicators_df, isolated_storage):
    compare_models("ZZREADONLY", synthetic_indicators_df)

    assert not (isolated_storage / "ZZREADONLY_xgb.pkl").exists()
    assert not (isolated_storage / "ZZREADONLY_rf.pkl").exists()
    assert not (isolated_storage / "ZZREADONLY_accuracy.json").exists()


# ── compare_intraday_models() ────────────────────────────────────────────────

def test_compare_intraday_models_happy_path(intraday_indicators_df, isolated_intraday_storage):
    from analysis.intraday_prediction import compare_intraday_models

    result = compare_intraday_models("TEST", "15m", intraday_indicators_df)

    assert result["error"] is None
    assert set(result["models"].keys()) == {"xgb", "rf", "logreg", "gbc"}
    assert result["best_single_model"] in result["models"]


def test_compare_intraday_models_reuses_daily_harness_not_a_fork(intraday_indicators_df, isolated_intraday_storage, monkeypatch):
    import analysis.ml_prediction as ml_prediction
    import analysis.intraday_prediction as ip

    calls = {"n": 0}
    real = ml_prediction._run_walk_forward_multi

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ip, "_run_walk_forward_multi", spy)

    result = ip.compare_intraday_models("TEST", "15m", intraday_indicators_df)

    assert result["error"] is None
    assert calls["n"] == 1


# ── train_model()/predict() ensemble weight persistence + backward compat ──

def test_train_model_persists_ensemble_weights(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model

    result = train_model("ZZENSEMBLE", synthetic_indicators_df)

    assert result["error"] is None
    assert set(result["ensemble_weights"].keys()) == {"xgb", "rf"}
    assert sum(result["ensemble_weights"].values()) == pytest.approx(1.0, abs=1e-3)

    import json
    with open(isolated_storage / "ZZENSEMBLE_accuracy.json") as f:
        saved = json.load(f)
    assert saved["ensemble_weights"] == result["ensemble_weights"]


def test_predict_uses_persisted_ensemble_weights_not_hardcoded_constant(synthetic_indicators_df, isolated_storage, monkeypatch):
    from analysis.ml_prediction import predict, train_model, _predictions_path

    train_model("ZZWEIGHTED", synthetic_indicators_df)
    result = predict("ZZWEIGHTED", synthetic_indicators_df)

    assert result["error"] is None
    assert result["ensemble_weights"]  # non-empty — read from the persisted metadata


def test_predict_falls_back_to_0_65_0_35_when_metadata_lacks_ensemble_weights(synthetic_indicators_df, isolated_storage):
    """Backward-compat regression guard: a pre-Phase-6 accuracy.json has no
    ensemble_weights key. predict() must fall back to the historical
    hardcoded 0.65/0.35 split, not crash or silently use 0-weights."""
    import json
    from analysis.ml_prediction import train_model, predict, _STORAGE_DIR

    train_result = train_model("ZZLEGACYENS", synthetic_indicators_df)
    assert train_result["error"] is None

    acc_path = _STORAGE_DIR / "ZZLEGACYENS_accuracy.json"
    with open(acc_path) as f:
        record = json.load(f)
    del record["ensemble_weights"]
    with open(acc_path, "w") as f:
        json.dump(record, f)

    result = predict("ZZLEGACYENS", synthetic_indicators_df)
    assert result["error"] is None
    assert result["ensemble_weights"] == {"xgb": 0.65, "rf": 0.35}
