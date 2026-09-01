"""
Tests for the grading arithmetic in resolve_predictions() /
resolve_intraday_predictions().

These two functions compute the `correct` field that every accuracy number in the
app is derived from — live win rate, the retrain performance-drop trigger, Model
Lab, and both attribution modules all read it as ground truth. Before this file
they had no direct test: every existing test that reads prediction history
pre-writes `correct` into its JSONL fixture, so `pending` came back empty and the
resolver returned before doing any math.

Each test writes an unresolved JSONL log plus a synthetic price series and asserts
the resulting correct/actual_outcome directly. No network: both resolvers take
their prices from data.price_data.get_price_history, monkeypatched here.
"""
import json

import pandas as pd
import pytest

import analysis.ml_prediction as ml
import analysis.intraday_prediction as intra


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_log(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_log(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _daily_prices(closes, start="2026-06-01"):
    idx = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=idx)


def _patch_daily_prices(monkeypatch, df):
    import data.price_data as pd_mod
    monkeypatch.setattr(pd_mod, "get_price_history", lambda *a, **k: df)


# ── Daily resolver ───────────────────────────────────────────────────────────

def test_bullish_call_that_rose_is_marked_correct(isolated_storage, monkeypatch):
    # Entry 100, +5% by the 5th forward bar.
    _patch_daily_prices(monkeypatch, _daily_prices([100, 101, 102, 103, 104, 105, 106]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    assert ml.resolve_predictions("TEST") == 1
    rec = _read_log(path)[0]
    assert rec["correct"] is True
    assert rec["actual_outcome"] == pytest.approx(5.0, abs=0.01)


def test_bullish_call_that_fell_is_marked_incorrect(isolated_storage, monkeypatch):
    _patch_daily_prices(monkeypatch, _daily_prices([100, 99, 98, 97, 96, 95, 94]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    assert ml.resolve_predictions("TEST") == 1
    assert _read_log(path)[0]["correct"] is False


def test_bearish_call_that_fell_is_marked_correct(isolated_storage, monkeypatch):
    _patch_daily_prices(monkeypatch, _daily_prices([100, 99, 98, 97, 96, 95, 94]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bearish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    assert ml.resolve_predictions("TEST") == 1
    assert _read_log(path)[0]["correct"] is True


def test_outcome_inside_the_neutral_band_is_excluded_not_scored(isolated_storage, monkeypatch):
    """
    Training drops neutral rows before validation, so directional_accuracy is
    conditional on the move clearing the band. Grading a sub-band outcome here
    would measure a strictly harder question and open a permanent trained-vs-live
    gap that the retrain trigger reads as degradation.
    """
    # +0.2% over the horizon, inside a 0.5% band.
    _patch_daily_prices(monkeypatch, _daily_prices([100, 100, 100, 100, 100, 100.2, 100.2]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    ml.resolve_predictions("TEST")
    rec = _read_log(path)[0]
    assert rec["correct"] is None
    assert rec["neutral_outcome"] is True
    assert rec["actual_outcome"] == pytest.approx(0.2, abs=0.01)


def test_neutral_outcome_is_not_retried_on_a_second_pass(isolated_storage, monkeypatch):
    """Without the neutral_outcome guard in `pending`, a correct=None row would
    be re-resolved on every read forever."""
    _patch_daily_prices(monkeypatch, _daily_prices([100, 100, 100, 100, 100, 100.2, 100.2]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    ml.resolve_predictions("TEST")
    assert ml.resolve_predictions("TEST") == 0


def test_tiny_move_is_not_forced_to_incorrect_by_rounding(isolated_storage, monkeypatch):
    """
    Regression: actual_return_pct was rounded to 2dp *before* the sign test, so
    any move under 0.005% became exactly 0.0, satisfied neither branch, and was
    recorded incorrect regardless of direction.

    neutral_threshold is pinned to 0 here so the band cannot exclude the row and
    the sign path is what's under test. (With a band set, this move would be
    excluded as neutral instead — see the neutral-band test above.)
    """
    # +0.001% — real, positive, below the old rounding resolution.
    _patch_daily_prices(monkeypatch, _daily_prices([100, 100, 100, 100, 100, 100.001, 100.001]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0,
        "actual_outcome": None, "correct": None,
    }])

    ml.resolve_predictions("TEST")
    rec = _read_log(path)[0]
    assert rec.get("neutral_outcome") is None
    assert rec["correct"] is True, "a genuine up-move must not round to incorrect"


def test_absent_neutral_threshold_falls_back_to_the_model_default(
    isolated_storage, monkeypatch
):
    """A record written before threshold_pct was persisted still gets graded
    against a band — the module default — rather than on a bare sign test."""
    _patch_daily_prices(monkeypatch, _daily_prices([100, 100, 100, 100, 100, 100.2, 100.2]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0,          # no neutral_threshold key
        "actual_outcome": None, "correct": None,
    }])

    ml.resolve_predictions("TEST")
    rec = _read_log(path)[0]
    # +0.2% sits inside the default 0.5% band.
    assert rec["neutral_outcome"] is True
    assert rec["correct"] is None


def test_horizon_not_yet_elapsed_stays_unresolved(isolated_storage, monkeypatch):
    _patch_daily_prices(monkeypatch, _daily_prices([100, 101, 102]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    assert ml.resolve_predictions("TEST") == 0
    assert _read_log(path)[0]["correct"] is None


def test_prediction_older_than_the_fetch_window_is_not_graded(isolated_storage, monkeypatch):
    """
    Regression: `dates > predicted_at` selected the whole series for a prediction
    predating the fetch window, so iloc[horizon-1] graded it against a close
    roughly two years later.
    """
    _patch_daily_prices(monkeypatch, _daily_prices([100, 101, 102, 103, 104, 105, 106],
                                                   start="2026-06-01"))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2019-01-02T10:00:00-05:00",   # long before the window
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    assert ml.resolve_predictions("TEST") == 0
    assert _read_log(path)[0]["correct"] is None


def test_resolution_preserves_a_prediction_appended_during_the_fetch(
    isolated_storage, monkeypatch
):
    """
    The resolver reads every record, makes a network call, then rewrites the file.
    It must apply its updates to a fresh read, not to the pre-fetch snapshot —
    otherwise anything appended inside that window is erased.
    """
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    prices = _daily_prices([100, 101, 102, 103, 104, 105, 106])

    def fetch_and_append(*_a, **_k):
        # Simulates a concurrent save_prediction() landing mid-fetch.
        with open(path, "a") as f:
            f.write(json.dumps({
                "predicted_at": "2026-06-05T10:00:00-04:00",
                "direction": "bearish", "horizon_days": 5,
                "price_at_prediction": 104.0, "neutral_threshold": 0.005,
                "actual_outcome": None, "correct": None,
            }) + "\n")
        return prices

    import data.price_data as pd_mod
    monkeypatch.setattr(pd_mod, "get_price_history", fetch_and_append)

    ml.resolve_predictions("TEST")

    records = _read_log(path)
    assert len(records) == 2, "the concurrently-appended prediction was lost"
    by_ts = {r["predicted_at"]: r for r in records}
    assert by_ts["2026-06-01T10:00:00-04:00"]["correct"] is True
    assert by_ts["2026-06-05T10:00:00-04:00"]["correct"] is None


def test_no_temp_file_is_left_behind(isolated_storage, monkeypatch):
    _patch_daily_prices(monkeypatch, _daily_prices([100, 101, 102, 103, 104, 105, 106]))
    path = ml._predictions_path("TEST")
    _write_log(path, [{
        "predicted_at": "2026-06-01T10:00:00-04:00",
        "direction": "bullish", "horizon_days": 5,
        "price_at_prediction": 100.0, "neutral_threshold": 0.005,
        "actual_outcome": None, "correct": None,
    }])

    ml.resolve_predictions("TEST")
    leftovers = list(isolated_storage.glob("*.tmp"))
    assert leftovers == []


# ── Intraday resolver ────────────────────────────────────────────────────────

def _intraday_prices(closes, date="2026-06-01", interval_minutes=15):
    idx = pd.date_range(
        start=f"{date} 09:30", periods=len(closes),
        freq=f"{interval_minutes}min", tz="America/New_York",
    )
    return pd.DataFrame({"Close": closes}, index=idx)


def test_intraday_bullish_call_that_rose_is_correct(isolated_intraday_storage, monkeypatch):
    prices = _intraday_prices([100, 100.5, 101, 101.5, 102, 102.5, 103])
    import data.price_data as pd_mod
    monkeypatch.setattr(pd_mod, "get_price_history", lambda *a, **k: prices)

    path = intra._predictions_path("TEST", "15m")
    _write_log(path, [{
        "predicted_at": "2026-06-01T09:30:00-04:00",
        "bar_timestamp": prices.index[0].isoformat(),
        "direction": "bullish", "horizon_bars": 5,
        "price_at_prediction": 100.0, "threshold_pct": 0.1,
        "actual_outcome": None, "correct": None,
    }])

    assert intra.resolve_intraday_predictions("TEST", "15m") == 1
    rec = _read_log(path)[0]
    assert rec["correct"] is True
    assert rec["actual_outcome"] == pytest.approx(2.5, abs=0.01)


def test_intraday_bar_missing_from_the_window_is_not_snapped_to_a_neighbour(
    isolated_intraday_storage, monkeypatch
):
    """
    Regression: get_indexer(method="nearest") only returns -1 when a tolerance is
    supplied. Without one, a prediction whose bar has aged out of the refetch
    window snapped to the nearest surviving bar — in either direction — and was
    graded against unrelated prices.
    """
    prices = _intraday_prices([100, 100.5, 101, 101.5, 102, 102.5, 103],
                              date="2026-06-10")
    import data.price_data as pd_mod
    monkeypatch.setattr(pd_mod, "get_price_history", lambda *a, **k: prices)

    path = intra._predictions_path("TEST", "15m")
    _write_log(path, [{
        "predicted_at": "2026-06-01T09:30:00-04:00",
        # A bar from nine days earlier — not in the fetched window at all.
        "bar_timestamp": "2026-06-01T09:30:00-04:00",
        "direction": "bullish", "horizon_bars": 5,
        "price_at_prediction": 100.0, "threshold_pct": 0.1,
        "actual_outcome": None, "correct": None,
    }])

    assert intra.resolve_intraday_predictions("TEST", "15m") == 0
    assert _read_log(path)[0]["correct"] is None


def test_intraday_outcome_inside_the_band_is_excluded(isolated_intraday_storage, monkeypatch):
    # +0.05% against a 0.1% band.
    prices = _intraday_prices([100, 100, 100, 100, 100, 100.05, 100.05])
    import data.price_data as pd_mod
    monkeypatch.setattr(pd_mod, "get_price_history", lambda *a, **k: prices)

    path = intra._predictions_path("TEST", "15m")
    _write_log(path, [{
        "predicted_at": "2026-06-01T09:30:00-04:00",
        "bar_timestamp": prices.index[0].isoformat(),
        "direction": "bullish", "horizon_bars": 5,
        "price_at_prediction": 100.0, "threshold_pct": 0.1,
        "actual_outcome": None, "correct": None,
    }])

    intra.resolve_intraday_predictions("TEST", "15m")
    rec = _read_log(path)[0]
    assert rec["correct"] is None
    assert rec["neutral_outcome"] is True
