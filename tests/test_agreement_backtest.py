"""
Tests for interval_consensus.backtest_agreement() and
ml_prediction._summarize_similar_setups() — Roadmap Items 7 and 6.

No network. `isolated_intraday_storage` keeps writes in tmp_path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.interval_consensus import (
    MIN_PAIRS_PER_BUCKET,
    backtest_agreement,
)
from config.tz import MARKET_TZ


def _utc(local: str) -> str:
    return pd.Timestamp(local).tz_localize(MARKET_TZ).tz_convert("UTC").isoformat()


def _rec(local, direction, *, correct=None, horizon_minutes=75):
    return {
        "date": _utc(local),
        "bar_timestamp": pd.Timestamp(local).tz_localize(MARKET_TZ).isoformat(),
        "direction": direction,
        "probability": 0.61 if direction == "bullish" else 0.39,
        "confidence": "high",
        "horizon_minutes": horizon_minutes,
        "price_at_prediction": 600.0,
        "actual_outcome": (0.2 if correct else -0.2) if correct is not None else None,
        "correct": correct,
    }


def _install(storage, ticker, interval, rows):
    (storage / f"{ticker}_{interval}_xgb.pkl").write_bytes(b"stub")
    (storage / f"{ticker}_{interval}_rf.pkl").write_bytes(b"stub")
    (storage / f"{ticker}_{interval}_accuracy.json").write_text(
        json.dumps({"directional_accuracy": 0.558, "horizon_minutes": 75})
    )
    with open(storage / f"{ticker}_{interval}_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def storage(isolated_intraday_storage):
    return isolated_intraday_storage


def _times(n, start_hour=10, step_minutes=5, day=3):
    """Distinct timestamps far enough apart that merge_asof pairs them 1:1."""
    base = pd.Timestamp(f"2026-08-{day:02d} {start_hour:02d}:00")
    return [(base + pd.Timedelta(minutes=i * step_minutes)).strftime("%Y-%m-%d %H:%M")
            for i in range(n)]


# ── shape / empty cases ───────────────────────────────────────────────────────

def test_no_base_predictions_returns_empty_without_raising(storage):
    out = backtest_agreement("SPY")
    assert out["n_base_resolved"] == 0
    assert out["baseline_accuracy"] is None
    assert any("No 15m predictions" in w for w in out["warnings"])


def test_self_selection_caveat_is_always_present(storage):
    """
    The sample is irregular and self-selected. That caveat must be impossible to
    miss regardless of how good the numbers look.
    """
    out = backtest_agreement("SPY")
    assert any("self-selected" in w for w in out["warnings"])


def test_unresolved_base_predictions_are_excluded(storage):
    _install(storage, "SPY", "15m", [_rec(t, "bullish") for t in _times(5)])
    out = backtest_agreement("SPY")
    assert out["n_base_resolved"] == 0
    assert any("No resolved directional" in w for w in out["warnings"])


def test_neutral_base_predictions_are_excluded(storage):
    rows = [_rec(t, "neutral", correct=None) for t in _times(5)]
    _install(storage, "SPY", "15m", rows)
    out = backtest_agreement("SPY")
    assert out["n_base_resolved"] == 0


def test_missing_confirm_interval_marks_everything_unconfirmed(storage):
    n = MIN_PAIRS_PER_BUCKET
    _install(storage, "SPY", "15m",
             [_rec(t, "bullish", correct=True) for t in _times(n)])
    out = backtest_agreement("SPY")
    assert out["buckets"]["unconfirmed"]["n"] == n
    assert out["buckets"]["agree"]["n"] == 0
    assert any("No directional 30m predictions" in w for w in out["warnings"])


# ── bucketing ─────────────────────────────────────────────────────────────────

def test_agreeing_pairs_land_in_the_agree_bucket(storage):
    n = MIN_PAIRS_PER_BUCKET
    times = _times(n)
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=True) for t in times])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in times])

    out = backtest_agreement("SPY")
    assert out["buckets"]["agree"]["n"] == n
    assert out["buckets"]["conflict"]["n"] == 0
    assert out["buckets"]["agree"]["accuracy"] == pytest.approx(1.0)


def test_opposing_pairs_land_in_the_conflict_bucket(storage):
    n = MIN_PAIRS_PER_BUCKET
    times = _times(n)
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=False) for t in times])
    _install(storage, "SPY", "30m", [_rec(t, "bearish") for t in times])

    out = backtest_agreement("SPY")
    assert out["buckets"]["conflict"]["n"] == n
    assert out["buckets"]["agree"]["n"] == 0


def test_confirm_outside_the_tolerance_window_is_unconfirmed(storage):
    n = MIN_PAIRS_PER_BUCKET
    base_times = _times(n, start_hour=10)
    # Same count, but hours away -- far outside the default 30-minute tolerance.
    confirm_times = _times(n, start_hour=14)
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=True) for t in base_times])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in confirm_times])

    out = backtest_agreement("SPY", tolerance_minutes=5)
    assert out["buckets"]["unconfirmed"]["n"] == n
    assert out["buckets"]["agree"]["n"] == 0


def test_tolerance_is_configurable(storage):
    times = _times(MIN_PAIRS_PER_BUCKET, start_hour=10, step_minutes=60)
    shifted = [
        (pd.Timestamp(t) + pd.Timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M")
        for t in times
    ]
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=True) for t in times])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in shifted])

    tight = backtest_agreement("SPY", tolerance_minutes=5)
    loose = backtest_agreement("SPY", tolerance_minutes=30)
    assert tight["buckets"]["unconfirmed"]["n"] == MIN_PAIRS_PER_BUCKET
    assert loose["buckets"]["agree"]["n"] == MIN_PAIRS_PER_BUCKET


# ── thin samples ──────────────────────────────────────────────────────────────

def test_bucket_below_the_floor_reports_n_but_no_accuracy(storage):
    n = MIN_PAIRS_PER_BUCKET - 1
    times = _times(n)
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=True) for t in times])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in times])

    out = backtest_agreement("SPY")
    assert out["buckets"]["agree"]["n"] == n
    assert out["buckets"]["agree"]["accuracy"] is None       # not 1.0
    assert any(f"only {n} pair" in w for w in out["warnings"])


def test_lift_is_none_when_the_agree_bucket_is_thin(storage):
    times = _times(MIN_PAIRS_PER_BUCKET - 1)
    _install(storage, "SPY", "15m", [_rec(t, "bullish", correct=True) for t in times])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in times])
    assert backtest_agreement("SPY")["agreement_lift"] is None


# ── the actual question ───────────────────────────────────────────────────────

def test_agreement_lift_is_positive_when_confirmation_helps(storage):
    """
    Construct a history where confirmed calls hit and unconfirmed ones miss. The
    lift must come out positive -- this is the measurement the panel exists for.
    """
    n = MIN_PAIRS_PER_BUCKET
    confirmed_times = _times(n, start_hour=10, step_minutes=5, day=3)
    lone_times = _times(n, start_hour=14, step_minutes=5, day=3)

    base_rows = (
        [_rec(t, "bullish", correct=True) for t in confirmed_times]
        + [_rec(t, "bullish", correct=False) for t in lone_times]
    )
    _install(storage, "SPY", "15m", base_rows)
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in confirmed_times])

    out = backtest_agreement("SPY", tolerance_minutes=5)
    assert out["baseline_accuracy"] == pytest.approx(0.5)
    assert out["buckets"]["agree"]["accuracy"] == pytest.approx(1.0)
    assert out["buckets"]["unconfirmed"]["accuracy"] == pytest.approx(0.0)
    assert out["agreement_lift"] == pytest.approx(0.5)


def test_agreement_lift_is_negative_when_confirmation_does_not_help(storage):
    """The honest negative result must be reportable too."""
    n = MIN_PAIRS_PER_BUCKET
    confirmed_times = _times(n, start_hour=10, step_minutes=5)
    lone_times = _times(n, start_hour=14, step_minutes=5)

    base_rows = (
        [_rec(t, "bullish", correct=False) for t in confirmed_times]
        + [_rec(t, "bullish", correct=True) for t in lone_times]
    )
    _install(storage, "SPY", "15m", base_rows)
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in confirmed_times])

    out = backtest_agreement("SPY", tolerance_minutes=5)
    assert out["agreement_lift"] < 0


def test_baseline_uses_every_resolved_row_not_just_confirmed_ones(storage):
    n = MIN_PAIRS_PER_BUCKET
    confirmed = _times(n, start_hour=10, step_minutes=5)
    lone = _times(n, start_hour=14, step_minutes=5)
    _install(storage, "SPY", "15m",
             [_rec(t, "bullish", correct=True) for t in confirmed]
             + [_rec(t, "bullish", correct=False) for t in lone])
    _install(storage, "SPY", "30m", [_rec(t, "bullish") for t in confirmed])

    out = backtest_agreement("SPY", tolerance_minutes=5)
    assert out["n_base_resolved"] == 2 * n
    assert out["baseline_accuracy"] == pytest.approx(0.5)


def test_intervals_are_configurable(storage):
    n = MIN_PAIRS_PER_BUCKET
    times = _times(n)
    _install(storage, "SPY", "5m", [_rec(t, "bullish", correct=True) for t in times])
    _install(storage, "SPY", "1h", [_rec(t, "bullish") for t in times])

    out = backtest_agreement("SPY", base_interval="5m", confirm_interval="1h")
    assert out["base_interval"] == "5m"
    assert out["confirm_interval"] == "1h"
    assert out["buckets"]["agree"]["n"] == n


# ── similar setups (Item 6) ───────────────────────────────────────────────────

def _summarize(returns_pct, direction="bullish"):
    from analysis.ml_prediction import _summarize_similar_setups
    return _summarize_similar_setups(np.array(returns_pct) / 100.0, direction)


def test_similar_setups_reports_the_full_distribution():
    out = _summarize([1.0, 2.0, 3.0, -1.0, 4.0])
    assert out["n"] == 5
    assert out["median_pct"] == pytest.approx(2.0)
    assert out["mean_pct"] == pytest.approx(1.8)
    assert out["worst_pct"] == pytest.approx(-1.0)
    assert out["best_pct"] == pytest.approx(4.0)


def test_similar_setups_win_rate_is_suppressed_below_the_floor():
    from analysis.ml_prediction import MIN_SIMILAR_SETUPS_FOR_WIN_RATE

    thin = _summarize([1.0] * (MIN_SIMILAR_SETUPS_FOR_WIN_RATE - 1))
    assert thin["win_rate"] is None
    assert thin["is_thin"] is True
    assert thin["n"] == MIN_SIMILAR_SETUPS_FOR_WIN_RATE - 1     # n still reported


def test_similar_setups_win_rate_appears_once_the_sample_is_adequate():
    from analysis.ml_prediction import MIN_SIMILAR_SETUPS_FOR_WIN_RATE

    n = MIN_SIMILAR_SETUPS_FOR_WIN_RATE
    out = _summarize([1.0] * (n // 2) + [-1.0] * (n - n // 2))
    assert out["is_thin"] is False
    assert out["win_rate"] == pytest.approx(0.5, abs=0.05)


def test_similar_setups_win_rate_is_direction_aware():
    """
    A bearish call wins when price falls. Counting positive returns as wins for
    both directions would invert the bearish win rate entirely.
    """
    returns = [-1.0] * 20 + [1.0] * 5
    bullish = _summarize(returns, "bullish")
    bearish = _summarize(returns, "bearish")
    assert bullish["win_rate"] == pytest.approx(5 / 25)
    assert bearish["win_rate"] == pytest.approx(20 / 25)


def test_similar_setups_is_always_flagged_in_sample():
    assert _summarize([1.0] * 30)["is_in_sample"] is True


def test_similar_setups_quartiles_bracket_the_median():
    out = _summarize(list(range(1, 41)))
    assert out["p25_pct"] < out["median_pct"] < out["p75_pct"]
