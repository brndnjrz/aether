"""
Tests for analysis/horizon_clock.py — the expiry math behind "this signal is no
longer evidence."

No network. Every test injects `now` explicitly; nothing here may depend on when
the suite happens to run.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.horizon_clock import (
    MARKET_CLOSE,
    daily_expiry,
    describe_remaining,
    intraday_expiry,
)
from config.tz import MARKET_TZ


def _et(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_localize(MARKET_TZ)


# ── intraday ──────────────────────────────────────────────────────────────────

def test_intraday_expiry_accounts_for_start_labeled_bars():
    """
    A 15m bar labeled 10:00 closes at 10:15. A 5-bar (75 min) horizon exits when
    the bar labeled 11:15 closes, i.e. 11:30 -- not 11:15. Getting this wrong
    puts the expiry a full bar early.
    """
    clock = intraday_expiry(
        _et("2026-08-03 10:00"), horizon_minutes=75, interval_minutes=15,
        now=_et("2026-08-03 10:20"),
    )
    assert clock["expires_at"] == _et("2026-08-03 11:30")
    assert clock["is_expired"] is False
    assert clock["crosses_session_close"] is False
    assert clock["is_gradeable"] is True


def test_intraday_minutes_remaining_counts_down_and_goes_negative():
    before = intraday_expiry(
        _et("2026-08-03 10:00"), 75, 15, now=_et("2026-08-03 11:00"),
    )
    after = intraday_expiry(
        _et("2026-08-03 10:00"), 75, 15, now=_et("2026-08-03 12:00"),
    )
    assert before["minutes_remaining"] == pytest.approx(30.0)
    assert before["is_expired"] is False
    assert after["minutes_remaining"] == pytest.approx(-30.0)
    assert after["is_expired"] is True


def test_intraday_horizon_past_the_close_is_never_gradeable():
    """
    resolve_intraday_predictions() skips any prediction whose forward window
    spans the overnight gap, so a late-session signal stays unresolved forever.
    That is a dead state, not a pending one, and must be distinguishable.
    """
    clock = intraday_expiry(
        _et("2026-08-03 15:30"), horizon_minutes=75, interval_minutes=15,
        now=_et("2026-08-03 15:35"),
    )
    assert clock["crosses_session_close"] is True
    assert clock["is_gradeable"] is False
    assert "never" in clock["reason"]


def test_intraday_expiry_exactly_at_the_close_does_not_cross():
    """Boundary: exiting precisely at 16:00 is gradeable; 16:01 is not."""
    at_close = intraday_expiry(
        _et("2026-08-03 15:30"), horizon_minutes=15, interval_minutes=15,
        now=_et("2026-08-03 15:35"),
    )
    assert at_close["expires_at"] == _et("2026-08-03 16:00")
    assert at_close["crosses_session_close"] is False
    assert at_close["is_gradeable"] is True


def test_intraday_session_close_is_four_pm_on_the_signals_own_date():
    clock = intraday_expiry(_et("2026-08-03 10:00"), 75, 15, now=_et("2026-08-03 10:00"))
    assert clock["session_close"] == _et("2026-08-03 16:00")
    assert clock["session_close"].time() == MARKET_CLOSE


def test_intraday_naive_timestamp_is_read_as_market_local_not_utc():
    """
    Naive intraday timestamps are exchange-local ET. Localizing to UTC instead
    would shift 10:00 ET to 06:00 ET and break every session comparison -- the
    exact class of bug CLAUDE.md calls out.
    """
    naive = intraday_expiry("2026-08-03 10:00:00", 75, 15, now=_et("2026-08-03 10:20"))
    aware = intraday_expiry(_et("2026-08-03 10:00"), 75, 15, now=_et("2026-08-03 10:20"))
    assert naive["expires_at"] == aware["expires_at"]
    assert naive["expires_at"].hour == 11


@pytest.mark.parametrize("interval_minutes,horizon_bars", [(5, 3), (15, 5), (30, 5), (60, 3)])
def test_intraday_expiry_holds_across_every_supported_interval(interval_minutes, horizon_bars):
    horizon_minutes = interval_minutes * horizon_bars
    bar = _et("2026-08-03 10:00")
    clock = intraday_expiry(bar, horizon_minutes, interval_minutes, now=bar)
    expected = bar + pd.Timedelta(minutes=interval_minutes + horizon_minutes)
    assert clock["expires_at"] == expected


@pytest.mark.parametrize("bad", [None, "not-a-timestamp", float("nan")])
def test_intraday_bad_timestamp_degrades_without_raising(bad):
    clock = intraday_expiry(bad, 75, 15, now=_et("2026-08-03 10:00"))
    assert clock["expires_at"] is None
    assert clock["is_gradeable"] is False
    assert clock["reason"]


@pytest.mark.parametrize("horizon,interval", [(0, 15), (-5, 15), (75, 0), (75, -1)])
def test_intraday_nonpositive_inputs_degrade_without_raising(horizon, interval):
    clock = intraday_expiry(_et("2026-08-03 10:00"), horizon, interval, now=_et("2026-08-03 10:00"))
    assert clock["expires_at"] is None
    assert clock["reason"]


# ── daily ─────────────────────────────────────────────────────────────────────

def test_daily_expiry_matches_resolve_predictions_bar_walk():
    """
    resolve_predictions() does closes[dates > predicted_at].iloc[horizon - 1].
    Reproduce that on an explicit calendar and assert the same bar is chosen.
    """
    calendar = pd.DatetimeIndex([
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
        "2026-08-10", "2026-08-11",
    ]).tz_localize(MARKET_TZ)
    predicted_at = _et("2026-08-03 16:00")

    clock = daily_expiry(predicted_at, 5, trading_days=calendar, now=predicted_at)

    future = calendar[calendar > predicted_at]
    assert clock["expires_on"] == future[4]          # iloc[horizon - 1]
    assert clock["expires_on"] == _et("2026-08-10")
    assert clock["method"] == "trading_calendar"
    assert clock["is_gradeable"] is True


def test_daily_calendar_walk_skips_a_market_holiday():
    """
    A calendar with a gap (holiday) must push the expiry later than a naive
    business-day count would. This is exactly what the approximation gets wrong.
    """
    # 2026-09-07 (Labor Day) absent from the calendar.
    calendar = pd.DatetimeIndex([
        "2026-09-03", "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10",
    ]).tz_localize(MARKET_TZ)
    predicted_at = _et("2026-09-03 16:00")

    exact = daily_expiry(predicted_at, 3, trading_days=calendar, now=predicted_at)
    approx = daily_expiry(predicted_at, 3, now=predicted_at)

    assert exact["expires_on"] == _et("2026-09-09")
    assert approx["expires_on"] < exact["expires_on"]      # approximation runs early
    assert approx["method"] == "business_day_approximation"
    assert "holidays are not accounted for" in approx["reason"]


def test_daily_expiry_across_a_weekend_skips_saturday_and_sunday():
    predicted_at = _et("2026-08-07 16:00")             # a Friday
    clock = daily_expiry(predicted_at, 1, now=predicted_at)
    assert clock["expires_on"].strftime("%A") == "Monday"


def test_daily_horizon_not_yet_elapsed_is_not_gradeable():
    calendar = pd.DatetimeIndex(["2026-08-03", "2026-08-04"]).tz_localize(MARKET_TZ)
    clock = daily_expiry(_et("2026-08-03 16:00"), 5, trading_days=calendar,
                         now=_et("2026-08-04 16:00"))
    assert clock["is_gradeable"] is False
    assert clock["expires_on"] is None
    assert "needs 5" in clock["reason"]


def test_daily_is_expired_flips_once_the_date_passes():
    calendar = pd.DatetimeIndex([
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
    ]).tz_localize(MARKET_TZ)
    predicted_at = _et("2026-08-03 16:00")
    before = daily_expiry(predicted_at, 2, trading_days=calendar, now=_et("2026-08-04 12:00"))
    after = daily_expiry(predicted_at, 2, trading_days=calendar, now=_et("2026-08-06 12:00"))
    assert before["is_expired"] is False
    assert after["is_expired"] is True


@pytest.mark.parametrize("bad", [None, "nope"])
def test_daily_bad_timestamp_degrades_without_raising(bad):
    clock = daily_expiry(bad, 5, now=_et("2026-08-03 10:00"))
    assert clock["expires_on"] is None
    assert clock["reason"]


def test_daily_empty_calendar_falls_back_to_approximation():
    clock = daily_expiry(_et("2026-08-03 16:00"), 5, trading_days=[],
                         now=_et("2026-08-03 16:00"))
    assert clock["method"] == "business_day_approximation"
    assert clock["expires_on"] is not None


# ── formatting ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("minutes,expected", [
    (None, "—"),
    (45, "45m left"),
    (150, "2.5h left"),
    (2880, "2.0d left"),
    (-30, "expired 30m ago"),
    (-150, "expired 2.5h ago"),
    (-2880, "expired 2.0d ago"),
    (0, "expired 0m ago"),
])
def test_describe_remaining_wording(minutes, expected):
    assert describe_remaining(minutes) == expected
