"""
Candlestick pattern detectors — deterministic geometric rules on OHLCV bars.
Ported from docs/Identifying-Chart-Patterns.md (Fidelity/Kirkpatrick reference).
No ML, no external data — pure price-action geometry, same category as the
app's other rule-based signals (VWAP deviation, momentum count, trend alignment).
"""
import pandas as pd
from typing import Any, Dict, Optional

DOJI_BODY_RATIO = 0.1


def _is_doji(open_: float, high: float, low: float, close: float) -> bool:
    full_range = high - low
    if full_range <= 0:
        return False
    return abs(close - open_) / full_range <= DOJI_BODY_RATIO


def _is_bullish_engulfing(prev_open: float, prev_close: float, open_: float, close: float) -> bool:
    prev_bearish = prev_close < prev_open
    curr_bullish = close > open_
    engulfs = open_ <= prev_close and close >= prev_open
    return prev_bearish and curr_bullish and engulfs


def _is_bearish_engulfing(prev_open: float, prev_close: float, open_: float, close: float) -> bool:
    prev_bullish = prev_close > prev_open
    curr_bearish = close < open_
    engulfs = open_ >= prev_close and close <= prev_open
    return prev_bullish and curr_bearish and engulfs


def _is_inside_bar(prev_high: float, prev_low: float, high: float, low: float) -> bool:
    return high <= prev_high and low >= prev_low


def _is_nr4(ranges: list) -> bool:
    """ranges: last 4 bar ranges (High-Low), most recent last."""
    if len(ranges) < 4:
        return False
    return ranges[-1] < min(ranges[:-1])


def detect_candlestick_pattern(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect the most relevant candlestick pattern on the latest bar of `df`.
    Checked in priority order: Engulfing (most decisive), Doji, Inside Bar, NR4
    (both volatility-contraction setups). Returns None if no pattern matches,
    else {"name": str, "direction": "bull"|"bear"|"neutral", "note": str}.
    """
    if df is None or len(df) < 4:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    po, pc = float(prev["Open"]), float(prev["Close"])
    ph, pl = float(prev["High"]), float(prev["Low"])

    if _is_bullish_engulfing(po, pc, o, c):
        return {
            "name": "Bullish Engulfing", "direction": "bull",
            "note": "Prior bearish candle fully engulfed by today's bullish candle — buyers took control",
        }
    if _is_bearish_engulfing(po, pc, o, c):
        return {
            "name": "Bearish Engulfing", "direction": "bear",
            "note": "Prior bullish candle fully engulfed by today's bearish candle — sellers took control",
        }

    if _is_doji(o, h, l, c):
        return {
            "name": "Doji", "direction": "neutral",
            "note": "Open ≈ close — indecision, possible warning of a trend change",
        }

    if _is_inside_bar(ph, pl, h, l):
        return {
            "name": "Inside Bar", "direction": "neutral",
            "note": "Today's range is inside yesterday's — volatility contraction, watch for a breakout",
        }

    ranges = (df["High"] - df["Low"]).tail(4).tolist()
    if _is_nr4(ranges):
        return {
            "name": "NR4", "direction": "neutral",
            "note": "Narrowest range of the last 4 bars — coiling before a potential breakout",
        }

    return None
