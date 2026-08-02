"""
Tests for analysis/trade_attribution.py — Roadmap Item 12.

No network, no DB. The central assertions are the refusals: this module must not
report a comparison until both arms are big enough, because a comparison over a
handful of trades reads as self-knowledge while being noise.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.trade_attribution import (
    MIN_PER_ARM,
    MIN_ROUND_TRIPS_FOR_COMPARISON,
    compare_followed_vs_discretionary,
    parse_prediction_ref,
    split_by_attribution,
)


def _fill(i, *, ref=None, ticker="SPY", strike=600.0, opt="call", expiry="2026-08-21"):
    return {
        "ticker": ticker, "strike": strike, "option_type": opt,
        "expiry_date": expiry, "filled_at": f"2026-08-{3 + i % 20:02d}T10:30",
        "side": "buy", "qty": 1, "price": 3.0,
        "prediction_ref": ref,
    }


def _trip(i, *, win=True, pnl_pct=10.0, ticker="SPY", strike=600.0,
          opt="call", expiry="2026-08-21"):
    return {
        "ticker": ticker, "strike": strike, "option_type": opt,
        "expiry_date": expiry, "entry_time": f"2026-08-{3 + i % 20:02d}T10:30",
        "exit_time": f"2026-08-{4 + i % 20:02d}T10:30",
        "contract_key": f"{ticker}|{strike}|{opt}|{expiry}",
        "pnl_pct": pnl_pct, "pnl_dollars": pnl_pct * 3.0,
        "win": win, "hold_time_minutes": 1440, "hold_bucket": "1-2 days",
    }


def _paired(n, *, ref=None, win=True, pnl_pct=10.0, offset=0):
    """n matching (fill, trip) pairs with distinct timestamps."""
    fills, trips = [], []
    for i in range(offset, offset + n):
        strike = 600.0 + i          # distinct strike keeps keys unique
        fills.append(_fill(i, ref=ref, strike=strike))
        trips.append(_trip(i, win=win, pnl_pct=pnl_pct, strike=strike))
    return fills, trips


# ── ref parsing ───────────────────────────────────────────────────────────────

def test_parse_ref_splits_horizon_and_timestamp():
    out = parse_prediction_ref("15m|2026-08-03T10:30:00-04:00")
    assert out["horizon"] == "15m"
    assert out["predicted_at"] == "2026-08-03T10:30:00-04:00"


@pytest.mark.parametrize("bad", [None, "", "no-pipe-here", 42, "|"])
def test_malformed_ref_degrades_to_none_rather_than_raising(bad):
    out = parse_prediction_ref(bad)
    assert out["horizon"] is None
    assert out["predicted_at"] is None


# ── splitting ─────────────────────────────────────────────────────────────────

def test_split_attributes_by_the_entry_fill():
    fills, trips = _paired(3, ref="15m|2026-08-03T10:30:00-04:00")
    out = split_by_attribution(trips, fills)
    assert len(out["followed"]) == 3
    assert out["discretionary"] == []
    assert out["followed"][0]["horizon"] == "15m"


def test_untagged_fills_are_discretionary():
    fills, trips = _paired(3, ref=None)
    out = split_by_attribution(trips, fills)
    assert out["followed"] == []
    assert len(out["discretionary"]) == 3


def test_split_handles_a_mix():
    f1, t1 = _paired(2, ref="15m|2026-08-03T10:30:00-04:00", offset=0)
    f2, t2 = _paired(3, ref=None, offset=10)
    out = split_by_attribution(t1 + t2, f1 + f2)
    assert len(out["followed"]) == 2
    assert len(out["discretionary"]) == 3


def test_a_trip_with_no_matching_fill_is_discretionary_not_dropped():
    _, trips = _paired(2, ref="15m|x")
    out = split_by_attribution(trips, [])          # no fills at all
    assert len(out["discretionary"]) == 2
    assert out["followed"] == []


def test_split_of_empty_inputs_is_empty():
    out = split_by_attribution([], [])
    assert out == {"followed": [], "discretionary": []}


# ── the refusals ──────────────────────────────────────────────────────────────

def test_thin_total_sample_is_not_reportable():
    n = MIN_ROUND_TRIPS_FOR_COMPARISON - 1
    fills, trips = _paired(n, ref="15m|x")
    out = compare_followed_vs_discretionary(trips, fills)
    assert out["reportable"] is False
    assert "before this comparison means anything" in out["reason"]


def test_lopsided_split_is_not_reportable_even_when_the_total_clears():
    """
    A 60/2 split passes the total gate while telling you nothing about the 2.
    Each arm needs its own minimum.
    """
    big_f, big_t = _paired(MIN_ROUND_TRIPS_FOR_COMPARISON, ref="15m|x", offset=0)
    tiny_f, tiny_t = _paired(2, ref=None, offset=100)
    out = compare_followed_vs_discretionary(big_t + tiny_t, big_f + tiny_f)
    assert out["n_total"] >= MIN_ROUND_TRIPS_FOR_COMPARISON
    assert out["reportable"] is False
    assert "each side needs at least" in out["reason"]


def test_both_arms_clearing_their_minimum_becomes_reportable():
    f1, t1 = _paired(MIN_PER_ARM + 5, ref="15m|x", offset=0)
    f2, t2 = _paired(MIN_PER_ARM + 5, ref=None, offset=100)
    out = compare_followed_vs_discretionary(t1 + t2, f1 + f2)
    assert out["reportable"] is True
    assert out["reason"] is None


def test_empty_input_is_not_reportable():
    out = compare_followed_vs_discretionary([], [])
    assert out["reportable"] is False
    assert out["n_total"] == 0


# ── the comparison itself ─────────────────────────────────────────────────────

def _reportable_pair(followed_pnl, discretionary_pnl):
    n = MIN_PER_ARM + 5
    f1, t1 = _paired(n, ref="15m|x", win=followed_pnl > 0, pnl_pct=followed_pnl, offset=0)
    f2, t2 = _paired(n, ref=None, win=discretionary_pnl > 0,
                     pnl_pct=discretionary_pnl, offset=100)
    return compare_followed_vs_discretionary(t1 + t2, f1 + f2)


def test_following_the_model_shows_a_higher_average_when_it_did_better():
    out = _reportable_pair(followed_pnl=8.0, discretionary_pnl=-4.0)
    assert out["followed"]["avg_pnl_pct"] == pytest.approx(8.0)
    assert out["discretionary"]["avg_pnl_pct"] == pytest.approx(-4.0)
    assert out["followed"]["win_rate"] == pytest.approx(1.0)
    assert out["discretionary"]["win_rate"] == pytest.approx(0.0)


def test_the_unflattering_result_is_reportable_too():
    """Overriding the model beating it must be just as visible."""
    out = _reportable_pair(followed_pnl=-3.0, discretionary_pnl=6.0)
    assert out["followed"]["avg_pnl_pct"] < out["discretionary"]["avg_pnl_pct"]


def test_per_horizon_breakdown_is_keyed_by_horizon():
    n = MIN_PER_ARM + 5
    f15, t15 = _paired(n, ref="15m|x", offset=0)
    f30, t30 = _paired(n, ref="30m|x", offset=100)
    fd, td = _paired(n, ref=None, offset=200)
    out = compare_followed_vs_discretionary(t15 + t30 + td, f15 + f30 + fd)
    assert set(out["by_horizon"]) == {"15m", "30m"}
    assert out["by_horizon"]["15m"]["n"] == n


def test_empty_arm_reports_none_not_zero():
    """A zero win rate and 'no data' must not look identical."""
    out = compare_followed_vs_discretionary([], [])
    assert out["followed"]["win_rate"] is None
    assert out["followed"]["avg_pnl_pct"] is None
    assert out["followed"]["n"] == 0


def test_dollar_totals_are_summed_not_averaged():
    n = MIN_PER_ARM + 5
    f1, t1 = _paired(n, ref="15m|x", pnl_pct=10.0, offset=0)
    f2, t2 = _paired(n, ref=None, pnl_pct=10.0, offset=100)
    out = compare_followed_vs_discretionary(t1 + t2, f1 + f2)
    assert out["followed"]["total_pnl_dollars"] == pytest.approx(n * 30.0)


def test_median_is_reported_alongside_the_mean():
    n = MIN_PER_ARM + 5
    f1, t1 = _paired(n, ref="15m|x", pnl_pct=5.0, offset=0)
    f2, t2 = _paired(n, ref=None, pnl_pct=5.0, offset=100)
    out = compare_followed_vs_discretionary(t1 + t2, f1 + f2)
    assert out["followed"]["median_pnl_pct"] == pytest.approx(5.0)


def test_trips_missing_pnl_do_not_crash_the_summary():
    n = MIN_PER_ARM + 5
    f1, t1 = _paired(n, ref="15m|x", offset=0)
    for t in t1:
        t["pnl_pct"] = None
        t["pnl_dollars"] = None
    f2, t2 = _paired(n, ref=None, offset=100)
    out = compare_followed_vs_discretionary(t1 + t2, f1 + f2)
    assert out["followed"]["avg_pnl_pct"] is None
    assert out["followed"]["win_rate"] is not None      # win flags still present
