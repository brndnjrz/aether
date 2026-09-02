"""
Horizon clock — when does a prediction stop being evidence?

Every prediction this app makes carries a horizon, and that horizon is a hard
contract: `resolve_predictions()` grades a 15m/5-bar call by comparing
`price_at_prediction` to the close exactly 5 bars later, and
`resolve_intraday_predictions()` does the same at the prediction's own interval.
The model has validated evidence about that window and none whatsoever about the
bar after it. Holding past the horizon is trading an expired signal.

Nothing in the UI said so before this module existed, which is why it exists.

Streamlit-free (see CLAUDE.md layer rules) — pure functions over timestamps.
Both entry points take an injectable `now` so callers and tests are never at the
mercy of wall-clock time.

Two things worth knowing before using this:

1. **Bars are labeled at their start.** yfinance labels an intraday bar with the
   time the bar opened, so the signal derived from bar `T` is only actually
   available at `T + interval`, and the exit bar's close lands at
   `T + (horizon_bars + 1) * interval`. The elapsed time between those two is
   exactly `horizon_minutes`, which is why `intraday_expiry()` needs
   `interval_minutes` as well as `horizon_minutes` — without it, the expiry
   lands one bar early.

2. **A horizon that crosses the session close is never gradeable.**
   `resolve_intraday_predictions()` deliberately skips any prediction whose
   forward window would span the overnight gap (matching how the labels were
   built), so such a prediction stays unresolved forever — it is not "pending,"
   it is dead. `crosses_session_close` flags that, because a late-session signal
   with a long horizon is not a tradeable signal at all.
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from config.tz import MARKET_TZ, now_et

logger = logging.getLogger(__name__)

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
MINUTES_PER_SESSION = 390          # 9:30 -> 16:00

_DISPLAY_FMT = "%Y-%m-%d %I:%M %p ET"
_DATE_FMT = "%Y-%m-%d"


def _to_market_ts(value: Any) -> Optional[pd.Timestamp]:
    """
    Coerce to a market-tz Timestamp. **Naive input is treated as market-local
    ET**, not UTC — the opposite of `config.tz.utc_iso_to_et_str`, and
    deliberately so: this function handles intraday *market data* timestamps,
    where yfinance returns exchange-local times. Localizing 09:30 to UTC would
    shift it to 05:30 ET and break every session-window comparison below. Same
    convention as `orbc_strategy.to_market_tz()` and
    `intraday_prediction._to_market_tz()`.
    """
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(MARKET_TZ) if ts.tz is None else ts.tz_convert(MARKET_TZ)


def _session_close(ts: pd.Timestamp) -> pd.Timestamp:
    """
    16:00 ET on `ts`'s own calendar date. Built by combining the naive date with
    MARKET_CLOSE and localizing, rather than adding a 16-hour Timedelta to
    midnight — Timedelta arithmetic on tz-aware timestamps is absolute, so it
    would drift by an hour across a DST boundary.
    """
    naive_date = ts.tz_localize(None).normalize().date()
    return pd.Timestamp(datetime.combine(naive_date, MARKET_CLOSE)).tz_localize(MARKET_TZ)


def _empty_clock(reason: str) -> Dict[str, Any]:
    return {
        "expires_at": None,
        "expires_at_str": None,
        "minutes_remaining": None,
        "is_expired": False,
        "crosses_session_close": False,
        "is_gradeable": False,
        "session_close": None,
        "reason": reason,
    }


def intraday_expiry(
    bar_timestamp: Any,
    horizon_minutes: float,
    interval_minutes: int,
    *,
    now: Any = None,
) -> Dict[str, Any]:
    """
    When does an intraday prediction stop being evidence?

    Parameters
    ----------
    bar_timestamp : the signal bar's label, as stored on the prediction record
        (`predict_intraday()` writes `market_df.index[-1].isoformat()`). Naive
        values are read as market-local ET.
    horizon_minutes : as stored — `horizon_bars * interval_minutes`.
    interval_minutes : the bar size, e.g. 15 for "15m". Needed because bars are
        labeled at their start; see the module docstring.
    now : injectable clock. Defaults to `now_et()`.

    Returns
    -------
    dict with:
        expires_at            tz-aware Timestamp (ET) of the exit bar's close
        expires_at_str        display string, or None
        minutes_remaining     float; negative once elapsed
        is_expired            bool
        crosses_session_close bool — horizon spans the overnight gap, so
                              resolve_intraday_predictions() will never grade it
        is_gradeable          bool — False when crosses_session_close
        session_close         16:00 ET on the signal's own date
        reason                populated only when the inputs were unusable
    """
    bar_ts = _to_market_ts(bar_timestamp)
    if bar_ts is None:
        return _empty_clock("Unparseable bar timestamp.")
    try:
        horizon_minutes = float(horizon_minutes)
        interval_minutes = int(interval_minutes)
    except (TypeError, ValueError):
        return _empty_clock("Non-numeric horizon or interval.")
    if horizon_minutes <= 0 or interval_minutes <= 0:
        return _empty_clock("Horizon and interval must both be positive.")

    now_ts = _to_market_ts(now) if now is not None else _to_market_ts(now_et())

    # Signal is available when the signal bar closes; the exit bar's close lands
    # horizon_minutes after that. See the module docstring on bar labeling.
    signal_available_at = bar_ts + pd.Timedelta(minutes=interval_minutes)
    expires_at = signal_available_at + pd.Timedelta(minutes=horizon_minutes)

    session_close = _session_close(bar_ts)
    crosses = expires_at > session_close

    minutes_remaining = (expires_at - now_ts).total_seconds() / 60.0

    return {
        "expires_at": expires_at,
        "expires_at_str": expires_at.strftime(_DISPLAY_FMT),
        "minutes_remaining": round(minutes_remaining, 1),
        "is_expired": bool(minutes_remaining <= 0),
        "crosses_session_close": bool(crosses),
        "is_gradeable": bool(not crosses),
        "session_close": session_close,
        "reason": (
            "Horizon extends past the 4:00 PM close — this prediction will never "
            "be graded, matching how the training labels were built."
            if crosses else None
        ),
    }


def daily_expiry(
    predicted_at: Any,
    horizon_days: int,
    *,
    trading_days: Optional[Sequence[Any]] = None,
    now: Any = None,
) -> Dict[str, Any]:
    """
    When does a daily prediction stop being evidence?

    Mirrors `resolve_predictions()` exactly when `trading_days` is supplied: that
    function takes `closes[dates > predicted_at]` and reads `.iloc[horizon - 1]`,
    i.e. the close of the horizon-th trading bar *strictly after* the prediction.
    Passing the price index reproduces that walk bar-for-bar, holidays included.

    Without `trading_days` there is no calendar to walk, so it falls back to a
    Mon-Fri business-day count and says so in `method`. That approximation runs
    early by one day per intervening market holiday — never present it as exact.

    Returns
    -------
    dict with:
        expires_on        tz-aware Timestamp (ET) of the exit bar
        expires_on_str    date string, or None
        days_remaining    float; negative once elapsed
        is_expired        bool
        is_gradeable      True once a real calendar confirmed the bar exists
        method            "trading_calendar" | "business_day_approximation"
        reason            populated when the horizon has not elapsed on the
                          supplied calendar, or on unusable input
    """
    start = _to_market_ts(predicted_at)
    if start is None:
        out = _empty_clock("Unparseable prediction timestamp.")
        out.update({"expires_on": None, "expires_on_str": None,
                    "days_remaining": None, "method": None})
        return out
    try:
        horizon_days = int(horizon_days)
    except (TypeError, ValueError):
        horizon_days = 0
    if horizon_days <= 0:
        out = _empty_clock("Horizon must be positive.")
        out.update({"expires_on": None, "expires_on_str": None,
                    "days_remaining": None, "method": None})
        return out

    now_ts = _to_market_ts(now) if now is not None else _to_market_ts(now_et())

    expires_on: Optional[pd.Timestamp] = None
    method = "business_day_approximation"
    gradeable = False
    reason: Optional[str] = None

    if trading_days is not None and len(trading_days) > 0:
        method = "trading_calendar"
        idx = pd.DatetimeIndex(pd.to_datetime(list(trading_days), utc=True))
        idx = idx.tz_convert(MARKET_TZ)
        future = idx[idx > start]
        if len(future) >= horizon_days:
            expires_on = future[horizon_days - 1]
            gradeable = True
        else:
            # Same condition resolve_predictions() uses to leave a record
            # pending: the calendar simply does not extend far enough yet.
            reason = (
                f"Only {len(future)} trading day(s) available after the prediction; "
                f"needs {horizon_days} before it can be graded."
            )
    else:
        expires_on = start + pd.offsets.BDay(horizon_days)
        reason = (
            "Expiry estimated from Mon-Fri business days — no trading calendar "
            "supplied, so market holidays are not accounted for."
        )

    days_remaining = (
        (expires_on - now_ts).total_seconds() / 86400.0 if expires_on is not None else None
    )

    return {
        "expires_on": expires_on,
        "expires_on_str": expires_on.strftime(_DATE_FMT) if expires_on is not None else None,
        "days_remaining": round(days_remaining, 2) if days_remaining is not None else None,
        "is_expired": bool(days_remaining is not None and days_remaining <= 0),
        "is_gradeable": gradeable,
        "crosses_session_close": False,   # not a concept for daily bars
        "session_close": None,
        "method": method,
        "reason": reason,
    }


def describe_remaining(minutes_remaining: Optional[float]) -> str:
    """
    Human phrasing for a countdown. Kept here rather than in a page so the
    cockpit and the Predictions tab cannot word the same state differently.
    """
    if minutes_remaining is None:
        return "—"
    if minutes_remaining <= 0:
        elapsed = abs(minutes_remaining)
        if elapsed < 60:
            return f"expired {elapsed:.0f}m ago"
        if elapsed < 1440:
            return f"expired {elapsed / 60:.1f}h ago"
        return f"expired {elapsed / 1440:.1f}d ago"
    if minutes_remaining < 60:
        return f"{minutes_remaining:.0f}m left"
    if minutes_remaining < 1440:
        return f"{minutes_remaining / 60:.1f}h left"
    return f"{minutes_remaining / 1440:.1f}d left"
