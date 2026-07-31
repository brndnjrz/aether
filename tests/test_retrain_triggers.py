"""
Regression suite for analysis/retrain_triggers.py (Prediction Improvement
Engine, Phase 8).
"""
from __future__ import annotations

import json
import time

import pytest


def test_retrain_triggers_module_imports_cleanly():
    import analysis.retrain_triggers as rt

    for name in (
        "check_staleness_trigger", "check_performance_drop_trigger",
        "check_regime_change_trigger", "check_all_retrain_triggers",
    ):
        assert hasattr(rt, name), f"analysis.retrain_triggers is missing {name}"


# ── check_staleness_trigger ──────────────────────────────────────────────────

def test_check_staleness_trigger_no_model_returns_untriggered(isolated_storage):
    from analysis.retrain_triggers import check_staleness_trigger

    result = check_staleness_trigger("ZZNOMODEL")
    assert result["triggered"] is False
    assert result["age_days"] is None


def test_check_staleness_trigger_fires_past_threshold(isolated_storage):
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR
    from analysis.retrain_triggers import check_staleness_trigger
    from config.settings import RETRAIN_STALENESS_DAYS

    _xgb_path("ZZSTALE").write_bytes(b"x")
    _rf_path("ZZSTALE").write_bytes(b"x")
    acc_path = _STORAGE_DIR / "ZZSTALE_accuracy.json"
    acc_path.write_text(json.dumps({"directional_accuracy": 0.55}))
    old_time = time.time() - (RETRAIN_STALENESS_DAYS + 5) * 86400
    import os
    os.utime(acc_path, (old_time, old_time))

    result = check_staleness_trigger("ZZSTALE")
    assert result["triggered"] is True
    assert result["age_days"] > RETRAIN_STALENESS_DAYS


def test_check_staleness_trigger_within_threshold_does_not_fire(isolated_storage):
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR
    from analysis.retrain_triggers import check_staleness_trigger

    _xgb_path("ZZFRESH").write_bytes(b"x")
    _rf_path("ZZFRESH").write_bytes(b"x")
    (_STORAGE_DIR / "ZZFRESH_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))

    result = check_staleness_trigger("ZZFRESH")
    assert result["triggered"] is False


def test_check_staleness_trigger_uses_settings_constant_not_hardcoded_30(isolated_storage, monkeypatch):
    """Regression guard: the threshold must come from config.settings, not a
    bare literal — proven by changing the setting and observing the
    trigger's own reported threshold_days change with it."""
    import analysis.retrain_triggers as rt
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR

    _xgb_path("ZZCONST").write_bytes(b"x")
    _rf_path("ZZCONST").write_bytes(b"x")
    acc_path = _STORAGE_DIR / "ZZCONST_accuracy.json"
    acc_path.write_text(json.dumps({"directional_accuracy": 0.55}))
    old_time = time.time() - 10 * 86400
    import os
    os.utime(acc_path, (old_time, old_time))

    monkeypatch.setattr(rt, "RETRAIN_STALENESS_DAYS", 5)
    result = rt.check_staleness_trigger("ZZCONST")

    assert result["threshold_days"] == 5
    assert result["triggered"] is True  # 10 days old > new 5-day threshold


# ── check_performance_drop_trigger ───────────────────────────────────────────

def test_check_performance_drop_trigger_requires_minimum_resolved_sample(isolated_storage):
    from analysis.ml_prediction import _predictions_path, _STORAGE_DIR
    from analysis.retrain_triggers import check_performance_drop_trigger

    (_STORAGE_DIR / "ZZFEWRESOLVED_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.65}))
    records = [
        {"predicted_at": "2026-07-01T10:00:00-04:00", "date": "2026-07-01T10:00:00-04:00",
         "ticker": "ZZFEWRESOLVED", "direction": "bullish", "probability": 0.6, "confidence": "high",
         "model_accuracy": 0.65, "expected_move_pct": 1.0, "horizon_days": 5,
         "price_at_prediction": 100.0, "actual_outcome": -1.0, "correct": False},
    ]
    with open(_predictions_path("ZZFEWRESOLVED"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    result = check_performance_drop_trigger("ZZFEWRESOLVED")
    assert result["triggered"] is False
    assert "resolved" in result["reason"].lower()


def test_check_performance_drop_trigger_fires_on_synthetic_accuracy_gap(isolated_storage):
    from analysis.ml_prediction import _predictions_path, _STORAGE_DIR
    from analysis.retrain_triggers import check_performance_drop_trigger
    from config.settings import RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK

    (_STORAGE_DIR / "ZZDROP_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.70}))

    records = []
    for i in range(RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK):
        records.append({
            "predicted_at": f"2026-07-{(i % 28) + 1:02d}T10:00:00-04:00",
            "date": f"2026-07-{(i % 28) + 1:02d}T10:00:00-04:00",
            "ticker": "ZZDROP", "direction": "bullish", "probability": 0.6, "confidence": "high",
            "model_accuracy": 0.70, "expected_move_pct": 1.0, "horizon_days": 5,
            "price_at_prediction": 100.0, "actual_outcome": -1.0, "correct": False,
        })
    with open(_predictions_path("ZZDROP"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    result = check_performance_drop_trigger("ZZDROP")
    assert result["triggered"] is True
    assert result["trained_accuracy"] == 0.70
    assert result["live_accuracy"] == 0.0
    assert result["n_resolved"] == RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK


def test_check_performance_drop_trigger_no_gap_does_not_fire(isolated_storage):
    from analysis.ml_prediction import _predictions_path, _STORAGE_DIR
    from analysis.retrain_triggers import check_performance_drop_trigger
    from config.settings import RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK

    (_STORAGE_DIR / "ZZNODROP_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))
    records = []
    for i in range(RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK):
        records.append({
            "predicted_at": f"2026-07-{(i % 28) + 1:02d}T10:00:00-04:00",
            "date": f"2026-07-{(i % 28) + 1:02d}T10:00:00-04:00",
            "ticker": "ZZNODROP", "direction": "bullish", "probability": 0.6, "confidence": "high",
            "model_accuracy": 0.55, "expected_move_pct": 1.0, "horizon_days": 5,
            "price_at_prediction": 100.0, "actual_outcome": 1.0, "correct": True,
        })
    with open(_predictions_path("ZZNODROP"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    result = check_performance_drop_trigger("ZZNODROP")
    assert result["triggered"] is False


def test_check_performance_drop_trigger_no_data_returns_untriggered(isolated_storage):
    from analysis.retrain_triggers import check_performance_drop_trigger

    result = check_performance_drop_trigger("ZZNODATA")
    assert result["triggered"] is False
    assert result["trained_accuracy"] is None


# ── check_regime_change_trigger ──────────────────────────────────────────────

def test_check_regime_change_trigger_flags_elevated_vix(monkeypatch):
    import analysis.retrain_triggers as rt

    monkeypatch.setattr(
        "data.macro_data.get_vix_data",
        lambda *a, **k: {"current": 30.0, "regime": "Crisis"},
    )
    result = rt.check_regime_change_trigger("ZZVIX")

    assert result["triggered"] is True
    assert result["vix_elevated"] is True
    assert result["vix_regime"] == "Crisis"


def test_check_regime_change_trigger_normal_regime_does_not_fire(monkeypatch):
    import analysis.retrain_triggers as rt

    monkeypatch.setattr(
        "data.macro_data.get_vix_data",
        lambda *a, **k: {"current": 18.0, "regime": "Normal"},
    )
    result = rt.check_regime_change_trigger("ZZCALM")

    assert result["triggered"] is False
    assert result["vix_elevated"] is False


# ── check_all_retrain_triggers ────────────────────────────────────────────────

def test_check_all_retrain_triggers_should_retrain_true_if_any_fires(isolated_storage, monkeypatch):
    import analysis.retrain_triggers as rt

    monkeypatch.setattr(rt, "check_staleness_trigger", lambda ticker, interval=None: {"triggered": True, "reason": "stale"})
    monkeypatch.setattr(rt, "check_performance_drop_trigger", lambda ticker, interval=None: {"triggered": False, "reason": "fine"})
    monkeypatch.setattr(rt, "check_regime_change_trigger", lambda ticker, df=None: {"triggered": False, "reason": "calm"})

    result = rt.check_all_retrain_triggers("ZZANY")
    assert result["should_retrain"] is True


def test_check_all_retrain_triggers_should_retrain_false_if_none_fire(isolated_storage, monkeypatch):
    import analysis.retrain_triggers as rt

    monkeypatch.setattr(rt, "check_staleness_trigger", lambda ticker, interval=None: {"triggered": False, "reason": "fresh"})
    monkeypatch.setattr(rt, "check_performance_drop_trigger", lambda ticker, interval=None: {"triggered": False, "reason": "fine"})
    monkeypatch.setattr(rt, "check_regime_change_trigger", lambda ticker, df=None: {"triggered": False, "reason": "calm"})

    result = rt.check_all_retrain_triggers("ZZNONE")
    assert result["should_retrain"] is False
