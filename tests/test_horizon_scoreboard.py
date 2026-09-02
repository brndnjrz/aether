"""
Tests for prediction_performance.compare_horizons() — the cross-horizon
scoreboard behind Model Lab's "which horizon should I trade" panel.

No network. `isolated_intraday_storage` keeps every write in tmp_path.

The verdict rule chain is the substance here: each branch gets its own test,
because the distinction between "Do not trade" (no real edge) and "Uneconomic"
(real edge, costs eat it) is the whole diagnostic value of the column — they have
different fixes.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.prediction_performance import (
    MIN_N_PER_CONFIDENCE_BUCKET,
    compare_horizons,
)
from config.settings import (
    RETRAIN_ACCURACY_DROP_THRESHOLD,
    RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK,
)


def _meta(*, accuracy=0.558, std=0.041, horizon_minutes=75, avg_move_pct=0.35,
          net_edge=0.008, tradeable=True):
    return {
        "directional_accuracy": accuracy,
        "accuracy_std": std,
        "horizon_minutes": horizon_minutes,
        "horizon_bars": max(int(horizon_minutes // 15), 1),
        "is_reliable": True,
        "tradeability": {
            "avg_move_pct": avg_move_pct,
            "net_edge_pct": net_edge,
            "is_tradeable": tradeable,
        },
    }


def _row(*, correct, confidence="high", prob=0.61, direction="bullish",
         horizon_minutes=75):
    return {
        "date": "2026-08-03T14:30:00+00:00",
        "bar_timestamp": "2026-08-03T10:30:00-04:00",
        "direction": direction,
        "probability": prob,
        "confidence": confidence,
        "horizon_minutes": horizon_minutes,
        "price_at_prediction": 600.0,
        "actual_outcome": 0.2 if correct else -0.2,
        "correct": correct,
    }


def _install(storage, ticker, interval, *, meta, rows):
    (storage / f"{ticker}_{interval}_xgb.pkl").write_bytes(b"stub")
    (storage / f"{ticker}_{interval}_rf.pkl").write_bytes(b"stub")
    (storage / f"{ticker}_{interval}_accuracy.json").write_text(json.dumps(meta))
    with open(storage / f"{ticker}_{interval}_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _resolved(n, *, n_correct, **kw):
    return [_row(correct=(i < n_correct), **kw) for i in range(n)]


LADDER = {
    0: {"mid": 3.00, "bid": 2.99, "ask": 3.01, "delta": 0.50,
        "days_to_expiry": 300 / 1440, "expiry_date": "2026-08-03"},
    30: {"mid": 16.00, "bid": 15.97, "ask": 16.03, "delta": 0.50,
         "theta_per_day": -0.20, "days_to_expiry": 30.0, "expiry_date": "2026-09-02"},
}


@pytest.fixture
def storage(isolated_intraday_storage):
    return isolated_intraday_storage


def _find(out, horizon):
    return next(r for r in out["rows"] if r["horizon"] == horizon)


# ── shape ─────────────────────────────────────────────────────────────────────

def test_untrained_ticker_returns_a_row_per_horizon(storage):
    out = compare_horizons("SPY", include_daily=False)
    assert [r["horizon"] for r in out["rows"]] == ["5m", "15m", "30m", "1h"]
    assert all(r["verdict"] == "Not trained" for r in out["rows"])
    assert out["grid"] == {}


def test_daily_row_is_included_when_requested(storage):
    out = compare_horizons("SPY", include_daily=True)
    assert "daily" in [r["horizon"] for r in out["rows"]]


def test_trained_horizon_reports_accuracy_and_sample_size(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    row = _find(compare_horizons("SPY", include_daily=False), "15m")
    assert row["trained_accuracy"] == pytest.approx(0.558)
    assert row["accuracy_std"] == pytest.approx(0.041)
    assert row["n_resolved"] == 30
    assert row["live_accuracy"] == pytest.approx(17 / 30, abs=1e-3)


# ── verdict rule chain ────────────────────────────────────────────────────────

def test_insufficient_data_when_below_the_resolved_threshold(storage):
    n = RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK - 1
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(n, n_correct=n))
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["verdict"] == "Insufficient data"


def test_do_not_trade_when_no_real_edge_and_costs_fail(storage):
    """Accuracy at or below the 0.52 reliability gate AND negative net edge."""
    _install(storage, "SPY", "5m",
             meta=_meta(accuracy=0.51, net_edge=-0.011, tradeable=False, horizon_minutes=15),
             rows=_resolved(30, n_correct=15))
    assert _find(compare_horizons("SPY", include_daily=False), "5m")["verdict"] == "Do not trade"


def test_uneconomic_when_a_real_edge_exists_but_costs_eat_it(storage):
    """
    The distinction that matters most: accuracy clears the reliability gate, so
    the model is not broken -- the expiry or the instrument is wrong. Fixing that
    is a different action from abandoning the horizon.
    """
    _install(storage, "SPY", "15m",
             meta=_meta(accuracy=0.558, net_edge=-0.9, tradeable=False),
             rows=_resolved(30, n_correct=17))
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["verdict"] == "Uneconomic"


def test_degraded_when_live_accuracy_falls_far_below_trained(storage):
    trained = 0.60
    live_target = trained - RETRAIN_ACCURACY_DROP_THRESHOLD - 0.05
    n = 40
    _install(storage, "SPY", "15m", meta=_meta(accuracy=trained),
             rows=_resolved(n, n_correct=int(round(live_target * n))))
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["verdict"] == "Degraded, retrain"


def test_healthy_horizon_is_ranked_primary(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["verdict"] == "Primary"


def test_survivors_are_ranked_primary_then_secondary_by_net_edge(storage):
    _install(storage, "SPY", "15m", meta=_meta(net_edge=0.008), rows=_resolved(30, n_correct=17))
    _install(storage, "SPY", "30m",
             meta=_meta(net_edge=0.021, horizon_minutes=150), rows=_resolved(30, n_correct=17))
    out = compare_horizons("SPY", include_daily=False)
    assert _find(out, "30m")["verdict"] == "Primary"      # higher net edge
    assert _find(out, "15m")["verdict"] == "Secondary"


def test_verdict_chain_short_circuits_in_order(storage):
    """
    A horizon that is BOTH thin on data and uneconomic reports the data problem
    first — you cannot judge economics on 3 resolved predictions.
    """
    _install(storage, "SPY", "15m", meta=_meta(net_edge=-0.9, tradeable=False),
             rows=_resolved(3, n_correct=1))
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["verdict"] == "Insufficient data"


def test_daily_is_labelled_swing_context_not_ranked_against_intraday(storage, isolated_storage):
    """
    Ranking a 5-day call against a 75-minute one would invite treating them as
    substitutes. The daily row is context.
    """
    (isolated_storage / "SPY_xgb.pkl").write_bytes(b"stub")
    (isolated_storage / "SPY_rf.pkl").write_bytes(b"stub")
    (isolated_storage / "SPY_accuracy.json").write_text(json.dumps({
        "directional_accuracy": 0.572, "accuracy_std": 0.039,
        "horizon_days": 5, "neutral_threshold": 0.005,
    }))
    hist = [{
        "date": "2026-07-20T14:30:00+00:00", "direction": "bullish", "probability": 0.59,
        "confidence": "medium", "horizon_days": 5, "price_at_prediction": 600.0,
        "actual_outcome": 0.8, "correct": True,
    } for _ in range(30)]
    with open(isolated_storage / "SPY_predictions.jsonl", "w") as f:
        for r in hist:
            f.write(json.dumps(r) + "\n")

    row = _find(compare_horizons("SPY", include_daily=True), "daily")
    assert row["verdict"] == "Swing context"


# ── calibration column ────────────────────────────────────────────────────────

def test_calibration_unknown_when_a_bucket_is_underpopulated(storage):
    _install(storage, "SPY", "15m", meta=_meta(),
             rows=_resolved(30, n_correct=17, confidence="high"))
    row = _find(compare_horizons("SPY", include_daily=False), "15m")
    assert row["calibrated"].startswith("unknown")     # zero LOW-confidence rows


def test_calibration_yes_when_high_beats_low(storage):
    n = MIN_N_PER_CONFIDENCE_BUCKET
    rows = (
        [_row(correct=True, confidence="high") for _ in range(n)]
        + [_row(correct=False, confidence="low") for _ in range(n)]
    )
    _install(storage, "SPY", "15m", meta=_meta(), rows=rows)
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["calibrated"] == "yes"


def test_calibration_no_when_high_does_not_beat_low(storage):
    n = MIN_N_PER_CONFIDENCE_BUCKET
    rows = (
        [_row(correct=False, confidence="high") for _ in range(n)]
        + [_row(correct=True, confidence="low") for _ in range(n)]
    )
    _install(storage, "SPY", "15m", meta=_meta(), rows=rows)
    assert _find(compare_horizons("SPY", include_daily=False), "15m")["calibrated"] == "no"


# ── cost model / grid ─────────────────────────────────────────────────────────

def test_without_quotes_net_edge_is_the_stored_shares_verdict(storage):
    _install(storage, "SPY", "15m", meta=_meta(net_edge=0.008), rows=_resolved(30, n_correct=17))
    out = compare_horizons("SPY", include_daily=False)
    row = _find(out, "15m")
    assert row["cost_model"] == "shares"
    assert row["net_edge_pct"] == pytest.approx(0.008)
    assert out["grid"] == {}
    assert any("shares model" in w for w in out["warnings"])


def test_with_quotes_the_grid_is_populated_per_horizon_and_rung(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    out = compare_horizons(
        "SPY", include_daily=False, ladder_quotes=LADDER, underlying_price=600.0,
    )
    assert out["ladder"] == [0, 30]
    assert set(out["grid"]["15m"]) == {0, 30}
    assert _find(out, "15m")["cost_model"] == "options"


def test_options_sweep_flips_a_shares_tradeable_verdict_to_uneconomic(storage):
    """
    A 0DTE-only ladder must turn a green shares verdict red once theta is priced.
    This is the correction the whole options cost model exists to make.
    """
    _install(storage, "SPY", "15m", meta=_meta(net_edge=0.008, tradeable=True),
             rows=_resolved(30, n_correct=17))
    out = compare_horizons(
        "SPY", include_daily=False,
        ladder_quotes={0: LADDER[0]}, underlying_price=600.0,
    )
    row = _find(out, "15m")
    assert row["net_edge_pct"] < 0
    assert row["dominant_cost"] == "theta"
    assert row["verdict"] == "Uneconomic"       # accuracy is fine; costs are not


def test_best_rung_and_expiry_date_are_reported(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    row = _find(
        compare_horizons("SPY", include_daily=False, ladder_quotes=LADDER,
                         underlying_price=600.0),
        "15m",
    )
    assert row["best_dte"] == 30
    assert row["expiry_date"] == "2026-09-02"


def test_missing_volatility_anchor_leaves_the_shares_fallback_in_place(storage):
    """No stored avg_move_pct means no sigma to price against — do not invent one."""
    meta = _meta()
    meta["tradeability"].pop("avg_move_pct")
    _install(storage, "SPY", "15m", meta=meta, rows=_resolved(30, n_correct=17))
    out = compare_horizons(
        "SPY", include_daily=False, ladder_quotes=LADDER, underlying_price=600.0,
    )
    assert _find(out, "15m")["cost_model"] == "shares"
    assert "15m" not in out["grid"]          # no sweep ran, so no grid row


def test_quotes_without_underlying_price_do_not_produce_a_bogus_verdict(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    out = compare_horizons("SPY", include_daily=False, ladder_quotes=LADDER)
    assert _find(out, "15m")["cost_model"] == "shares"
    assert out["grid"] == {}


# ── robustness ────────────────────────────────────────────────────────────────

def test_one_unreadable_horizon_does_not_take_down_the_scoreboard(storage):
    _install(storage, "SPY", "15m", meta=_meta(), rows=_resolved(30, n_correct=17))
    (storage / "SPY_30m_xgb.pkl").write_bytes(b"stub")
    (storage / "SPY_30m_rf.pkl").write_bytes(b"stub")
    (storage / "SPY_30m_accuracy.json").write_text("{not valid json")

    out = compare_horizons("SPY", include_daily=False)
    assert _find(out, "15m")["verdict"] == "Primary"
    assert _find(out, "30m")["trained_accuracy"] is None


def test_unresolved_predictions_report_none_live_accuracy_not_zero(storage):
    rows = [_row(correct=None) for _ in range(5)]
    for r in rows:
        r["correct"] = None
        r["actual_outcome"] = None
    _install(storage, "SPY", "15m", meta=_meta(), rows=rows)
    row = _find(compare_horizons("SPY", include_daily=False), "15m")
    assert row["live_accuracy"] is None
    assert row["n_resolved"] == 0
