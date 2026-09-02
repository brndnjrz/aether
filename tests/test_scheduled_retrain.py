"""
Regression suite for scripts/scheduled_retrain.py (Prediction Improvement
Engine, Phase 8) — the standalone, OS-cron-triggered retraining sweep.
"""
from __future__ import annotations

import json

import pytest


def test_scheduled_retrain_script_imports_cleanly():
    import scripts.scheduled_retrain  # noqa: F401 — import success is the assertion


def test_discover_daily_tickers_matches_only_daily_accuracy_files(isolated_storage):
    from scripts.scheduled_retrain import _discover_daily_tickers

    (isolated_storage / "AAPL_accuracy.json").write_text("{}")
    (isolated_storage / "SPY_accuracy.json").write_text("{}")
    (isolated_storage / "SPY_15m_accuracy.json").write_text("{}")  # must NOT be picked up as a daily ticker

    tickers = _discover_daily_tickers(isolated_storage)
    assert tickers == ["AAPL", "SPY"]


def test_discover_intraday_pairs_matches_only_interval_suffixed_files(isolated_intraday_storage):
    from scripts.scheduled_retrain import _discover_intraday_pairs

    (isolated_intraday_storage / "SPY_15m_accuracy.json").write_text("{}")
    (isolated_intraday_storage / "SPY_1h_accuracy.json").write_text("{}")
    (isolated_intraday_storage / "AAPL_accuracy.json").write_text("{}")  # daily-shaped, must NOT match

    pairs = _discover_intraday_pairs(isolated_intraday_storage)
    assert pairs == [("SPY", "15m"), ("SPY", "1h")]


def test_scheduled_retrain_dry_run_does_not_call_train_model(isolated_storage, monkeypatch, capsys):
    import scripts.scheduled_retrain as sr
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR

    _xgb_path("ZZDRY").write_bytes(b"x")
    _rf_path("ZZDRY").write_bytes(b"x")
    (_STORAGE_DIR / "ZZDRY_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))

    calls = {"n": 0}
    monkeypatch.setattr("analysis.ml_prediction.train_model", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    # _process_daily lazily imports check_all_retrain_triggers from
    # analysis.retrain_triggers each call — patch it there so a real,
    # otherwise-untriggered ticker still exercises the dry-run path.
    monkeypatch.setattr(
        "analysis.retrain_triggers.check_all_retrain_triggers",
        lambda *a, **k: {"should_retrain": True, "triggers": {}},
    )

    rc = sr.main(["--dry-run", "--tickers", "ZZDRY"])

    assert rc == 0
    assert calls["n"] == 0
    captured = capsys.readouterr()
    assert "Sweep complete" in captured.out


def test_scheduled_retrain_force_retrains_regardless_of_trigger(isolated_storage, monkeypatch):
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR
    import scripts.scheduled_retrain as sr

    _xgb_path("ZZFORCE").write_bytes(b"x")
    _rf_path("ZZFORCE").write_bytes(b"x")
    (_STORAGE_DIR / "ZZFORCE_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))

    calls = {"n": 0}

    def fake_train_model(ticker, df=None):
        calls["n"] += 1
        return {"error": None, "directional_accuracy": 0.6, "is_reliable": True}

    monkeypatch.setattr("analysis.ml_prediction.train_model", fake_train_model)

    rc = sr.main(["--force", "--tickers", "ZZFORCE"])

    assert rc == 0
    assert calls["n"] == 1


def test_scheduled_retrain_continues_after_one_ticker_raises(isolated_storage, monkeypatch, capsys):
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR
    import scripts.scheduled_retrain as sr

    for ticker in ("ZZFAIL", "ZZOK"):
        _xgb_path(ticker).write_bytes(b"x")
        _rf_path(ticker).write_bytes(b"x")
        (_STORAGE_DIR / f"{ticker}_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))

    calls = []

    def fake_train_model(ticker, df=None):
        calls.append(ticker)
        if ticker == "ZZFAIL":
            raise RuntimeError("boom")
        return {"error": None, "directional_accuracy": 0.6, "is_reliable": True}

    monkeypatch.setattr("analysis.ml_prediction.train_model", fake_train_model)

    rc = sr.main(["--force", "--tickers", "ZZFAIL,ZZOK"])

    assert rc == 0
    assert set(calls) == {"ZZFAIL", "ZZOK"}  # both attempted despite ZZFAIL raising


def test_scheduled_retrain_writes_append_only_log(isolated_storage, monkeypatch):
    from analysis.ml_prediction import _xgb_path, _rf_path, _STORAGE_DIR
    import scripts.scheduled_retrain as sr

    _xgb_path("ZZLOG").write_bytes(b"x")
    _rf_path("ZZLOG").write_bytes(b"x")
    (_STORAGE_DIR / "ZZLOG_accuracy.json").write_text(json.dumps({"directional_accuracy": 0.55}))

    monkeypatch.setattr(
        "analysis.ml_prediction.train_model",
        lambda ticker, df=None: {"error": None, "directional_accuracy": 0.6, "is_reliable": True},
    )

    sr.main(["--force", "--tickers", "ZZLOG"])
    sr.main(["--force", "--tickers", "ZZLOG"])

    log_path = _STORAGE_DIR / "retrain_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2  # append-only — second run adds, doesn't overwrite
    assert all(line["ticker"] == "ZZLOG" for line in lines)
