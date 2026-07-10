"""
US/Eastern time helpers — the user is in NC (Eastern), and the market itself
runs on Eastern hours, so all user-facing timestamps should be explicit ET
rather than naive server-local time or silently-UTC.
"""
from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(MARKET_TZ)


def now_et_iso() -> str:
    return now_et().isoformat()


def utc_iso_to_et_str(iso_str: str, fmt: str = "%Y-%m-%d %I:%M %p ET") -> str:
    """Render a stored ISO timestamp in ET. Naive input (no tzinfo) is assumed
    UTC, matching the existing datetime.utcnow().isoformat() storage format."""
    ts = pd.Timestamp(iso_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MARKET_TZ).strftime(fmt)
