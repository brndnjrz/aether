"""
Tests for analysis/signal_attribution.py — Roadmap Item 10.

No network: the price fetcher is injected. Nothing here fabricates weights, and
the central assertion is the negative one — an under-sized history must produce
NO coefficients, because that is the trap this module exists to avoid.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.signal_attribution import (
    MIN_EVENTS_TO_FIT,
    SIGNAL_FIELDS,
    fit_signal_weights,
    parse_signal_events,
    resolve_signal_events,
    signal_contributions,
)


def _log_row(when, *, vwap="bull", momentum="bull", trend="bull",
             event_type="day_trading_analyze", ticker="SPY", interval="15m"):
    return {
        "event_type": event_type,
        "ticker": ticker,
        "logged_at": when,
        "detail_json": json.dumps({
            "interval": interval,
            "vwap_direction": vwap,
            "momentum_direction": momentum,
            "trend_direction": trend,
            "suggested_direction": "LONG",
        }),
    }


def _prices(n=120, start="2026-03-02", trend=0.001):
    """Deterministic daily closes; `trend` sets the per-bar drift."""
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    closes = 600.0 * np.cumprod(np.full(n, 1.0 + trend))
    return pd.DataFrame({"Close": closes}, index=idx)


def _fetcher(df):
    return lambda _ticker: df


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parse_filters_to_day_trading_events_only():
    rows = [
        _log_row("2026-03-02T14:30:00+00:00"),
        _log_row("2026-03-03T14:30:00+00:00", event_type="options_view"),
        _log_row("2026-03-04T14:30:00+00:00", event_type="prediction_generated"),
    ]
    out = parse_signal_events(rows)
    assert len(out) == 1


def test_parse_maps_directions_to_ordinals():
    rows = [_log_row("2026-03-02T14:30:00+00:00", vwap="bull", momentum="neutral", trend="bear")]
    out = parse_signal_events(rows)
    assert out.loc[0, "vwap_direction"] == 1.0
    assert out.loc[0, "momentum_direction"] == 0.0
    assert out.loc[0, "trend_direction"] == -1.0


def test_parse_treats_bullish_and_bull_as_the_same():
    a = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00", vwap="bull")])
    b = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00", vwap="bullish")])
    assert a.loc[0, "vwap_direction"] == b.loc[0, "vwap_direction"] == 1.0


def test_parse_skips_corrupt_detail_json_without_raising():
    rows = [
        {"event_type": "day_trading_analyze", "ticker": "SPY",
         "logged_at": "2026-03-02T14:30:00+00:00", "detail_json": "{not json"},
        _log_row("2026-03-03T14:30:00+00:00"),
    ]
    assert len(parse_signal_events(rows)) == 1


def test_parse_skips_rows_with_no_signal_fields():
    rows = [{
        "event_type": "day_trading_analyze", "ticker": "SPY",
        "logged_at": "2026-03-02T14:30:00+00:00",
        "detail_json": json.dumps({"interval": "15m"}),
    }]
    assert parse_signal_events(rows).empty


def test_parse_of_empty_input_returns_documented_columns():
    out = parse_signal_events([])
    assert out.empty
    for field in SIGNAL_FIELDS:
        assert field in out.columns


def test_parse_sorts_oldest_first():
    rows = [
        _log_row("2026-03-05T14:30:00+00:00"),
        _log_row("2026-03-02T14:30:00+00:00"),
    ]
    assert parse_signal_events(rows)["logged_at"].is_monotonic_increasing


# ── resolution ────────────────────────────────────────────────────────────────

def test_resolve_labels_an_uptrend_as_went_up():
    events = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00")])
    out = resolve_signal_events(events, _fetcher(_prices(trend=0.002)))
    assert len(out) == 1
    assert out.loc[0, "went_up"] == 1.0
    assert out.loc[0, "forward_return_pct"] > 0


def test_resolve_labels_a_downtrend_as_went_down():
    events = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00")])
    out = resolve_signal_events(events, _fetcher(_prices(trend=-0.002)))
    assert out.loc[0, "went_up"] == 0.0
    assert out.loc[0, "forward_return_pct"] < 0


def test_resolve_drops_events_whose_horizon_has_not_elapsed():
    """An event after the last available bar cannot be graded."""
    events = parse_signal_events([_log_row("2027-01-01T14:30:00+00:00")])
    assert resolve_signal_events(events, _fetcher(_prices())).empty


def test_resolve_honours_the_horizon_parameter():
    events = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00")])
    one = resolve_signal_events(events, _fetcher(_prices(trend=0.002)), horizon_days=1)
    five = resolve_signal_events(events, _fetcher(_prices(trend=0.002)), horizon_days=5)
    assert five.loc[0, "forward_return_pct"] > one.loc[0, "forward_return_pct"]


def test_resolve_uses_bars_strictly_after_the_event():
    """Matches resolve_predictions()'s convention; a same-bar entry would leak."""
    events = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00")])
    out = resolve_signal_events(events, _fetcher(_prices(trend=0.002)), horizon_days=1)
    assert out.loc[0, "forward_return_pct"] == pytest.approx(0.2, abs=1e-3)


def test_resolve_survives_a_price_fetch_failure():
    def boom(_t):
        raise RuntimeError("network down")
    events = parse_signal_events([_log_row("2026-03-02T14:30:00+00:00")])
    assert resolve_signal_events(events, boom).empty


def test_resolve_of_empty_events_returns_empty():
    assert resolve_signal_events(pd.DataFrame(), _fetcher(_prices())).empty


# ── fitting: the refusal path is the important one ────────────────────────────

def test_thin_history_produces_no_coefficients_at_all():
    """
    The whole point of the module. Under MIN_EVENTS_TO_FIT there must be nothing
    weight-shaped to render, so a caller cannot accidentally display noise.
    """
    rows = [_log_row(f"2026-03-{2 + i:02d}T14:30:00+00:00")
            for i in range(MIN_EVENTS_TO_FIT - 1)]
    resolved = resolve_signal_events(parse_signal_events(rows), _fetcher(_prices()))
    out = fit_signal_weights(resolved)
    assert out["fitted"] is False
    assert out["coefficients"] == {}
    assert "vote count" in out["reason"]


def test_single_class_label_is_refused_rather_than_fitted():
    """An all-up history has nothing to separate."""
    rows = [_log_row(f"2026-03-{2 + i:02d}T14:30:00+00:00")
            for i in range(MIN_EVENTS_TO_FIT + 10)]
    resolved = resolve_signal_events(parse_signal_events(rows), _fetcher(_prices(trend=0.002)))
    out = fit_signal_weights(resolved)
    assert out["fitted"] is False
    assert "same way" in out["reason"]


def test_empty_history_is_refused():
    out = fit_signal_weights(pd.DataFrame())
    assert out["fitted"] is False
    assert out["coefficients"] == {}


def _mixed_resolved(n=60):
    """
    Hand-built resolved frame where trend_direction perfectly predicts the
    outcome and vwap_direction is pure noise, so the fit has a known answer.
    """
    rows = []
    for i in range(n):
        up = i % 2 == 0
        rows.append({
            "logged_at": pd.Timestamp("2026-03-02", tz="UTC") + pd.Timedelta(days=i),
            "ticker": "SPY", "interval": "15m", "suggested_direction": "LONG",
            "vwap_direction": 1.0 if i % 3 == 0 else -1.0,      # noise
            "momentum_direction": 0.0,                           # no information
            "trend_direction": 1.0 if up else -1.0,              # perfect predictor
            "forward_return_pct": 0.5 if up else -0.5,
            "went_up": 1.0 if up else 0.0,
        })
    return pd.DataFrame(rows)


def test_fit_recovers_the_signal_that_actually_predicts():
    out = fit_signal_weights(_mixed_resolved())
    assert out["fitted"] is True
    coefs = out["coefficients"]
    assert coefs["trend_direction"] > abs(coefs["vwap_direction"])
    assert coefs["trend_direction"] > 0


def test_fit_gives_an_uninformative_signal_a_near_zero_weight():
    coefs = fit_signal_weights(_mixed_resolved())["coefficients"]
    assert abs(coefs["momentum_direction"]) < abs(coefs["trend_direction"])


def test_fit_reports_n_base_rate_and_in_sample_accuracy():
    out = fit_signal_weights(_mixed_resolved())
    assert out["n"] == 60
    assert out["base_rate"] == pytest.approx(0.5)
    assert out["train_accuracy"] == pytest.approx(1.0)      # trend is perfect here


def test_fit_returns_one_coefficient_per_signal_field():
    assert set(fit_signal_weights(_mixed_resolved())["coefficients"]) == set(SIGNAL_FIELDS)


# ── contributions ─────────────────────────────────────────────────────────────

def test_contributions_are_empty_when_weights_were_not_fitted():
    """No fit means no breakdown — fabricating one is the trap."""
    assert signal_contributions({"fitted": False}, {"trend_direction": "bull"}) == {}


def test_contributions_are_coefficient_times_signal_value():
    weights = fit_signal_weights(_mixed_resolved())
    out = signal_contributions(weights, {
        "vwap_direction": "bull", "momentum_direction": "neutral", "trend_direction": "bull",
    })
    assert out["contributions"]["trend_direction"] == pytest.approx(
        weights["coefficients"]["trend_direction"], abs=1e-6,
    )
    assert out["contributions"]["momentum_direction"] == 0.0


def test_a_bearish_reading_flips_its_contribution_sign():
    weights = fit_signal_weights(_mixed_resolved())
    bull = signal_contributions(weights, {"trend_direction": "bull"})
    bear = signal_contributions(weights, {"trend_direction": "bear"})
    assert bull["contributions"]["trend_direction"] == pytest.approx(
        -bear["contributions"]["trend_direction"], abs=1e-6,
    )


def _explicit_weights():
    """
    Hand-specified weights, so the contribution arithmetic is tested independently
    of whatever the fit happens to produce. (Fitting against _mixed_resolved()
    drives the noise coefficient to ~0, which is correct behaviour but makes it a
    poor probe for share arithmetic.)
    """
    return {
        "fitted": True,
        "coefficients": {
            "vwap_direction": 0.30,
            "momentum_direction": 0.10,
            "trend_direction": 0.60,
        },
        "intercept": 0.05,
    }


def test_shares_are_computed_over_absolute_contributions():
    """
    An opposing signal must show a real share rather than silently cancelling —
    otherwise "what drove this" hides the disagreement.
    """
    out = signal_contributions(_explicit_weights(), {
        "vwap_direction": "bear", "momentum_direction": "neutral", "trend_direction": "bull",
    })
    # contributions: vwap -0.30, momentum 0.00, trend +0.60; total_abs 0.90
    assert out["contributions"]["vwap_direction"] == pytest.approx(-0.30)
    assert out["contributions"]["trend_direction"] == pytest.approx(0.60)
    assert out["shares_pct"]["vwap_direction"] == pytest.approx(33.3, abs=0.1)
    assert out["shares_pct"]["trend_direction"] == pytest.approx(66.7, abs=0.1)
    assert sum(out["shares_pct"].values()) == pytest.approx(100.0, abs=0.5)


def test_opposing_signals_do_not_cancel_to_a_zero_share():
    """Equal and opposite readings must each show 50%, not 0%."""
    out = signal_contributions(
        {"fitted": True, "intercept": 0.0,
         "coefficients": {"vwap_direction": 0.5, "trend_direction": 0.5}},
        {"vwap_direction": "bull", "trend_direction": "bear"},
    )
    assert out["net_log_odds"] == pytest.approx(0.0)      # nets out...
    assert out["shares_pct"]["vwap_direction"] == pytest.approx(50.0)
    assert out["shares_pct"]["trend_direction"] == pytest.approx(50.0)


def test_net_log_odds_includes_the_intercept():
    out = signal_contributions(_explicit_weights(), {f: "neutral" for f in SIGNAL_FIELDS})
    assert out["net_log_odds"] == pytest.approx(0.05)


def test_shares_are_zero_when_every_signal_reads_neutral():
    out = signal_contributions(_explicit_weights(), {f: "neutral" for f in SIGNAL_FIELDS})
    assert out["total_abs"] == 0.0
    assert all(v == 0.0 for v in out["shares_pct"].values())
