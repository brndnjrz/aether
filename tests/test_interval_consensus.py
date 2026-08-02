"""
Tests for analysis/interval_consensus.py.

No network, and `isolated_storage` / `isolated_intraday_storage` keep every write
inside tmp_path — storage/ holds real trained models and real trade history.

Every test injects `now` explicitly. A consensus view is entirely about what is
live versus expired, so a test that reads the wall clock would pass or fail
depending on the hour it ran.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.interval_consensus import build_consensus
from config.tz import MARKET_TZ

NOW = pd.Timestamp("2026-08-03 11:00").tz_localize(MARKET_TZ)


def _iso_utc(local: str) -> str:
    return pd.Timestamp(local).tz_localize(MARKET_TZ).tz_convert("UTC").isoformat()


def _write_intraday(storage_dir, ticker, interval, *, rows, meta):
    (storage_dir / f"{ticker}_{interval}_xgb.pkl").write_bytes(b"stub")
    (storage_dir / f"{ticker}_{interval}_rf.pkl").write_bytes(b"stub")
    (storage_dir / f"{ticker}_{interval}_accuracy.json").write_text(json.dumps(meta))
    with open(storage_dir / f"{ticker}_{interval}_predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pred(bar_local, *, direction="bullish", prob=0.61, horizon_minutes=75,
          confidence="high", correct=None, tradeability=None):
    return {
        "date": _iso_utc(bar_local),
        "bar_timestamp": pd.Timestamp(bar_local).tz_localize(MARKET_TZ).isoformat(),
        "direction": direction,
        "probability": prob,
        "confidence": confidence,
        "horizon_minutes": horizon_minutes,
        "horizon_bars": horizon_minutes // 15,
        "price_at_prediction": 600.0,
        "actual_outcome": 0.2 if correct else (-0.2 if correct is False else None),
        "correct": correct,
        "tradeability": tradeability or {
            "avg_move_pct": 0.35, "net_edge_pct": 0.008, "is_tradeable": True,
        },
    }


META = {"directional_accuracy": 0.558, "accuracy_std": 0.041,
        "is_reliable": True, "horizon_minutes": 75, "horizon_bars": 5}


@pytest.fixture
def storage(isolated_intraday_storage):
    """isolated_intraday_storage monkeypatches the intraday module's storage dir."""
    return isolated_intraday_storage


# ── empty / untrained ─────────────────────────────────────────────────────────

def test_no_models_yields_all_blank_rows_without_raising(storage):
    out = build_consensus("SPY", include_daily=False, now=NOW)
    assert len(out["horizons"]) == 4                      # 5m/15m/30m/1h
    assert all(h["has_model"] is False for h in out["horizons"])
    assert all(h["has_prediction"] is False for h in out["horizons"])
    assert out["alignment"]["net_direction"] == "none"
    assert any("No model trained" in w for w in out["warnings"])


def test_trained_but_no_prediction_is_distinct_from_untrained(storage):
    _write_intraday(storage, "SPY", "15m", rows=[], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["has_model"] is True
    assert row["has_prediction"] is False
    assert "no prediction logged yet" in row["note"]


def test_horizons_are_returned_in_ascending_order(storage):
    out = build_consensus("SPY", include_daily=False, now=NOW)
    assert [h["horizon"] for h in out["horizons"]] == ["5m", "15m", "30m", "1h"]


# ── live signal ───────────────────────────────────────────────────────────────

def test_live_prediction_reports_direction_and_expiry(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")

    assert row["direction"] == "bullish"
    assert row["probability"] == pytest.approx(0.61)
    assert row["confidence"] == "high"
    # bar 10:30 closes 10:45, +75 min horizon -> 12:00
    assert row["expires_at"] == pd.Timestamp("2026-08-03 12:00").tz_localize(MARKET_TZ)
    assert row["is_expired"] is False
    assert row["minutes_remaining"] == pytest.approx(60.0)


def test_expired_prediction_is_flagged_and_excluded_from_the_vote(storage):
    # bar 08:00 -> closes 08:15 -> expires 09:30, well before NOW (11:00)
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 08:00")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")

    assert row["is_expired"] is True
    assert out["alignment"]["n_bullish"] == 0            # expired != evidence
    assert out["alignment"]["net_direction"] == "none"
    assert any("past its horizon" in w for w in out["warnings"])


def test_stale_prediction_is_flagged_while_still_live(storage):
    """Older than one bar interval but not yet past its horizon."""
    _write_intraday(storage, "SPY", "30m", rows=[
        _pred("2026-08-03 10:00", horizon_minutes=150),
    ], meta={**META, "horizon_minutes": 150})
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "30m")

    assert row["is_expired"] is False
    assert row["is_stale"] is True
    assert row["prediction_age_minutes"] == pytest.approx(60.0)
    assert any("older than one bar" in w for w in out["warnings"])


def test_fresh_prediction_is_not_stale(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:55")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["is_stale"] is False


def test_horizon_crossing_the_close_is_marked_never_gradeable(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 15:30")], meta=META)
    out = build_consensus(
        "SPY", include_daily=False,
        now=pd.Timestamp("2026-08-03 15:35").tz_localize(MARKET_TZ),
    )
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["crosses_session_close"] is True
    assert row["is_gradeable"] is False
    assert any("never be graded" in w for w in out["warnings"])


# ── alignment ─────────────────────────────────────────────────────────────────

def test_unanimous_agreement_is_reported(storage):
    for iv, hm in (("15m", 75), ("30m", 150)):
        _write_intraday(storage, "SPY", iv, rows=[
            _pred("2026-08-03 10:30", direction="bullish", horizon_minutes=hm),
        ], meta={**META, "horizon_minutes": hm})
    out = build_consensus("SPY", include_daily=False, now=NOW)
    a = out["alignment"]
    assert a["n_bullish"] == 2 and a["n_bearish"] == 0
    assert a["net_direction"] == "bullish"
    assert a["is_unanimous"] is True


def test_conflicting_horizons_report_mixed(storage):
    _write_intraday(storage, "SPY", "15m", rows=[
        _pred("2026-08-03 10:30", direction="bullish"),
    ], meta=META)
    _write_intraday(storage, "SPY", "30m", rows=[
        _pred("2026-08-03 10:30", direction="bearish", horizon_minutes=150),
    ], meta={**META, "horizon_minutes": 150})
    out = build_consensus("SPY", include_daily=False, now=NOW)
    a = out["alignment"]
    assert a["net_direction"] == "mixed"
    assert a["is_unanimous"] is False
    assert a["tightest_tradeable"] is None


def test_neutral_calls_are_counted_but_do_not_set_direction(storage):
    _write_intraday(storage, "SPY", "15m", rows=[
        _pred("2026-08-03 10:30", direction="neutral", prob=0.51, confidence="low"),
    ], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    a = out["alignment"]
    assert a["n_neutral"] == 1
    assert a["n_bullish"] == 0 and a["n_bearish"] == 0
    assert a["net_direction"] == "none"


def test_tightest_tradeable_picks_the_shortest_agreeing_economic_horizon(storage):
    for iv, hm in (("15m", 75), ("30m", 150)):
        _write_intraday(storage, "SPY", iv, rows=[
            _pred("2026-08-03 10:30", direction="bullish", horizon_minutes=hm),
        ], meta={**META, "horizon_minutes": hm})
    out = build_consensus("SPY", include_daily=False, now=NOW)
    assert out["alignment"]["tightest_tradeable"] == "15m"


def test_agreeing_but_uneconomic_horizon_is_named_not_recommended(storage):
    """
    The line the whole feature exists for: a horizon can point the right way and
    still be unable to pay for itself. It must not land in tightest_tradeable.
    """
    _write_intraday(storage, "SPY", "5m", rows=[
        _pred("2026-08-03 10:50", direction="bullish", horizon_minutes=15,
              tradeability={"avg_move_pct": 0.12, "net_edge_pct": -0.011, "is_tradeable": False}),
    ], meta={**META, "horizon_minutes": 15})
    _write_intraday(storage, "SPY", "30m", rows=[
        _pred("2026-08-03 10:30", direction="bullish", horizon_minutes=150),
    ], meta={**META, "horizon_minutes": 150})

    out = build_consensus("SPY", include_daily=False, now=NOW)
    a = out["alignment"]
    assert "5m" in a["agree_but_uneconomic"]
    assert a["tightest_tradeable"] == "30m"
    assert a["n_bullish"] == 2                     # still counted as agreement
    assert a["n_tradeable_bullish"] == 1           # but only one is economic


def test_single_live_signal_is_not_called_unanimous(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    assert out["alignment"]["is_unanimous"] is False


# ── cost model wiring ─────────────────────────────────────────────────────────

def test_without_quotes_the_shares_verdict_is_used_and_labelled(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["cost_model"] == "shares"
    assert row["net_edge_pct"] == pytest.approx(0.008)
    assert any("shares model" in w for w in out["warnings"])


def test_with_quotes_the_options_verdict_replaces_it(storage):
    """
    A 0DTE-only ladder must flip a 'tradeable' shares verdict to untradeable once
    theta is priced -- the correction Item 3 exists to make.
    """
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    quotes = {
        0: {"mid": 3.00, "bid": 2.99, "ask": 3.01, "delta": 0.50,
            "days_to_expiry": 300 / 1440, "expiry_date": "2026-08-03"},
    }
    out = build_consensus(
        "SPY", include_daily=False, ladder_quotes=quotes,
        underlying_price=600.0, now=NOW,
    )
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["cost_model"] == "options"
    assert row["is_tradeable"] is False
    assert row["dominant_cost"] == "theta"
    assert row["best_dte"] == 0


def test_options_verdict_picks_the_best_rung_on_a_full_ladder(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    quotes = {
        0: {"mid": 3.00, "bid": 2.99, "ask": 3.01, "delta": 0.50,
            "days_to_expiry": 300 / 1440},
        30: {"mid": 16.00, "bid": 15.97, "ask": 16.03, "delta": 0.50,
             "theta_per_day": -0.20, "days_to_expiry": 30.0},
    }
    out = build_consensus(
        "SPY", include_daily=False, ladder_quotes=quotes,
        underlying_price=600.0, now=NOW,
    )
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["best_dte"] == 30
    assert row["is_tradeable"] is True
    assert row["dominant_cost"] == "spread"


def test_missing_avg_move_pct_falls_back_rather_than_inventing_sigma(storage):
    """No stored avg_move_pct means no volatility anchor, so no options verdict."""
    _write_intraday(storage, "SPY", "15m", rows=[
        _pred("2026-08-03 10:30", tradeability={"net_edge_pct": 0.008, "is_tradeable": True}),
    ], meta=META)
    quotes = {30: {"mid": 16.0, "bid": 15.97, "ask": 16.03, "delta": 0.5,
                   "theta_per_day": -0.2, "days_to_expiry": 30.0}}
    out = build_consensus("SPY", include_daily=False, ladder_quotes=quotes,
                          underlying_price=600.0, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["cost_model"] == "shares"


# ── accuracy reporting ────────────────────────────────────────────────────────

def test_live_accuracy_is_paired_with_n_resolved(storage):
    rows = [
        _pred("2026-08-03 10:30", correct=True),
        _pred("2026-08-03 10:00", correct=True),
        _pred("2026-08-03 09:45", correct=False),
    ]
    _write_intraday(storage, "SPY", "15m", rows=rows, meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["n_resolved"] == 3
    assert row["live_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["trained_accuracy"] == pytest.approx(0.558)


def test_thin_sample_is_warned_about(storage):
    _write_intraday(storage, "SPY", "15m", rows=[
        _pred("2026-08-03 10:30", correct=True),
    ], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    assert any("Fewer than 20 resolved" in w for w in out["warnings"])


def test_unresolved_predictions_report_no_live_accuracy(storage):
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["n_resolved"] == 0
    assert row["live_accuracy"] is None          # not 0.0


# ── robustness ────────────────────────────────────────────────────────────────

def test_corrupt_prediction_row_does_not_take_down_the_view(storage):
    path = storage / "SPY_15m_predictions.jsonl"
    (storage / "SPY_15m_xgb.pkl").write_bytes(b"stub")
    (storage / "SPY_15m_rf.pkl").write_bytes(b"stub")
    (storage / "SPY_15m_accuracy.json").write_text(json.dumps(META))
    path.write_text("{not json at all\n" + json.dumps(_pred("2026-08-03 10:30")) + "\n")

    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["has_prediction"] is True         # the good line still read
    assert row["direction"] == "bullish"


def test_prediction_missing_bar_timestamp_degrades_to_no_expiry(storage):
    rec = _pred("2026-08-03 10:30")
    rec.pop("bar_timestamp")
    _write_intraday(storage, "SPY", "15m", rows=[rec], meta=META)
    out = build_consensus("SPY", include_daily=False, now=NOW)
    row = next(h for h in out["horizons"] if h["horizon"] == "15m")
    assert row["expires_at"] is None
    assert row["has_prediction"] is True


def test_build_consensus_writes_nothing_to_storage(storage):
    """
    The read-only guarantee. Any write here would corrupt the live win rate that
    Model Lab, the retrain triggers, and the scoreboard all read.
    """
    _write_intraday(storage, "SPY", "15m", rows=[_pred("2026-08-03 10:30")], meta=META)
    before = {p.name: p.stat().st_mtime_ns for p in storage.iterdir()}
    names_before = set(before)

    build_consensus("SPY", include_daily=False, now=NOW)

    after = {p.name: p.stat().st_mtime_ns for p in storage.iterdir()}
    assert set(after) == names_before          # no new files
    assert after == before                     # nothing rewritten
