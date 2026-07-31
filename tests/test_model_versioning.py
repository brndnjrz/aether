"""
Regression suite for the Prediction Improvement Engine's model versioning
(Phase 7): analysis/ml_prediction.py's _archive_current_version()/
get_version_history()/rollback_to_version(), mirrored in
analysis/intraday_prediction.py.

Training is real (walk-forward on synthetic data) — same cost profile as
the existing train_model()/train_intraday_model() tests.
"""
from __future__ import annotations

import pytest


# ── Daily (ml_prediction.py) ─────────────────────────────────────────────────

def test_versioning_module_functions_import_cleanly():
    import analysis.ml_prediction as mlp
    import analysis.intraday_prediction as ip

    for name in ("_archive_current_version", "get_version_history", "rollback_to_version"):
        assert hasattr(mlp, name), f"analysis.ml_prediction is missing {name}"
        assert hasattr(ip, name), f"analysis.intraday_prediction is missing {name}"


def test_first_ever_train_creates_no_version_history(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model, get_version_history

    result = train_model("ZZVFIRST", synthetic_indicators_df)

    assert result["error"] is None
    assert result["archived_version"] is None
    assert get_version_history("ZZVFIRST").empty


def test_second_train_archives_the_first_model_as_v1(synthetic_indicators_df, isolated_storage):
    import joblib
    from analysis.ml_prediction import train_model, get_version_history, _versions_dir, _xgb_path

    train_model("ZZVSECOND", synthetic_indicators_df)
    v1_bytes_path = _xgb_path("ZZVSECOND")
    original_n_estimators = joblib.load(v1_bytes_path).n_estimators

    result = train_model("ZZVSECOND", synthetic_indicators_df)

    assert result["archived_version"] == 1
    history = get_version_history("ZZVSECOND")
    assert len(history) == 1
    assert history.iloc[0]["version"] == 1

    archived_xgb_path = _versions_dir("ZZVSECOND") / "v1_xgb.pkl"
    assert archived_xgb_path.exists()
    assert joblib.load(archived_xgb_path).n_estimators == original_n_estimators


def test_third_train_archives_as_v2_history_grows(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model, get_version_history

    train_model("ZZVTHIRD", synthetic_indicators_df)
    train_model("ZZVTHIRD", synthetic_indicators_df)
    result = train_model("ZZVTHIRD", synthetic_indicators_df)

    assert result["archived_version"] == 2
    history = get_version_history("ZZVTHIRD")
    assert sorted(history["version"].tolist()) == [1, 2]
    # sorted version-descending, newest first
    assert history.iloc[0]["version"] == 2


def test_get_version_history_empty_for_never_retrained_ticker(isolated_storage):
    from analysis.ml_prediction import get_version_history

    history = get_version_history("ZZNEVERTRAINED")
    assert history.empty
    assert "version" in history.columns


def test_rollback_restores_an_older_versions_files(synthetic_indicators_df, isolated_storage):
    import joblib
    from analysis.ml_prediction import train_model, rollback_to_version, _xgb_path

    train_model("ZZROLLBACK", synthetic_indicators_df)
    v1_n_estimators = joblib.load(_xgb_path("ZZROLLBACK")).n_estimators

    train_model("ZZROLLBACK", synthetic_indicators_df)  # archives v1, deploys a new model as latest

    result = rollback_to_version("ZZROLLBACK", 1)
    assert result["error"] is None
    assert result["rolled_back_to_version"] == 1

    restored_n_estimators = joblib.load(_xgb_path("ZZROLLBACK")).n_estimators
    assert restored_n_estimators == v1_n_estimators


def test_rollback_archives_the_pre_rollback_current_as_a_new_version(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model, rollback_to_version, get_version_history

    train_model("ZZROLLARCH", synthetic_indicators_df)
    train_model("ZZROLLARCH", synthetic_indicators_df)  # archives the first as v1

    result = rollback_to_version("ZZROLLARCH", 1)
    assert result["new_current_version_archived"] == 2

    history = get_version_history("ZZROLLARCH")
    # v1 archived, v2 archived (pre-rollback current), v1 rollback event = 3 rows
    assert len(history) == 3
    assert 2 in history["version"].tolist()
    rollback_rows = history[history["rolled_back_from_latest"] == True]  # noqa: E712
    assert len(rollback_rows) == 1
    assert rollback_rows.iloc[0]["version"] == 1


def test_rollback_to_nonexistent_version_returns_structured_error(synthetic_indicators_df, isolated_storage):
    from analysis.ml_prediction import train_model, rollback_to_version

    train_model("ZZBADROLL", synthetic_indicators_df)
    result = rollback_to_version("ZZBADROLL", 999)

    assert result["error"] is not None
    assert result["rolled_back_to_version"] is None


def test_xgb_path_and_rf_path_unchanged_by_versioning(synthetic_indicators_df, isolated_storage):
    """Regression guard: versioning must never change what _xgb_path()/
    _rf_path() point to — every other reader depends on those exact paths."""
    from analysis.ml_prediction import train_model, _xgb_path, _rf_path

    before_xgb, before_rf = _xgb_path("ZZPATHCHECK"), _rf_path("ZZPATHCHECK")
    train_model("ZZPATHCHECK", synthetic_indicators_df)
    train_model("ZZPATHCHECK", synthetic_indicators_df)
    after_xgb, after_rf = _xgb_path("ZZPATHCHECK"), _rf_path("ZZPATHCHECK")

    assert before_xgb == after_xgb
    assert before_rf == after_rf


# ── Intraday (intraday_prediction.py) ────────────────────────────────────────

def test_intraday_second_train_archives_the_first_model_as_v1(intraday_indicators_df, isolated_intraday_storage):
    from analysis.intraday_prediction import train_intraday_model, get_version_history

    r1 = train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    assert r1["archived_version"] is None

    r2 = train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    assert r2["archived_version"] == 1

    history = get_version_history("TEST", "15m")
    assert len(history) == 1
    assert history.iloc[0]["version"] == 1


def test_intraday_rollback_restores_an_older_versions_files(intraday_indicators_df, isolated_intraday_storage):
    import joblib
    from analysis.intraday_prediction import train_intraday_model, rollback_to_version, _xgb_path

    train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    v1_n_estimators = joblib.load(_xgb_path("TEST", "15m")).n_estimators

    train_intraday_model("TEST", "15m", df=intraday_indicators_df)

    result = rollback_to_version("TEST", "15m", 1)
    assert result["error"] is None

    restored_n_estimators = joblib.load(_xgb_path("TEST", "15m")).n_estimators
    assert restored_n_estimators == v1_n_estimators


def test_intraday_rollback_to_nonexistent_version_returns_structured_error(intraday_indicators_df, isolated_intraday_storage):
    from analysis.intraday_prediction import train_intraday_model, rollback_to_version

    train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    result = rollback_to_version("TEST", "15m", 999)

    assert result["error"] is not None


def test_intraday_xgb_path_and_rf_path_unchanged_by_versioning(intraday_indicators_df, isolated_intraday_storage):
    from analysis.intraday_prediction import train_intraday_model, _xgb_path, _rf_path

    before_xgb, before_rf = _xgb_path("TEST", "15m"), _rf_path("TEST", "15m")
    train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    train_intraday_model("TEST", "15m", df=intraday_indicators_df)
    after_xgb, after_rf = _xgb_path("TEST", "15m"), _rf_path("TEST", "15m")

    assert before_xgb == after_xgb
    assert before_rf == after_rf


def test_intraday_and_daily_versions_never_collide(intraday_indicators_df, isolated_intraday_storage):
    """Isolation guard mirroring the module's existing storage-path-isolation
    tests: intraday's versions/ tree must be interval-scoped, never colliding
    with a same-named daily ticker's versions/ tree."""
    from analysis.intraday_prediction import _versions_dir as intraday_versions_dir

    assert "_15m" in str(intraday_versions_dir("TEST", "15m"))
