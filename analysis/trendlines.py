"""
Trendline and swing-structure detection — deterministic price-action geometry,
same category as analysis/patterns.py (VWAP deviation, momentum count, trend
alignment). No ML, no external data.

Ported from two single-asset research scripts:
- Trendline fit + slope optimizer: originally fit on a rolling window with no
  out-of-sample projection. Here the fit window excludes the latest bar and
  the line is projected one bar forward, so "breakout" means the latest bar's
  actual close crossed a line that was fit without seeing it — not a line
  that was fit to include the point it's being compared against.
- ATR-thresholded zigzag swing detector: a pending high/low is confirmed as a
  swing point once price reverses by more than one ATR against it, the same
  "meaningful move" filter a manual chart-reader would apply, automated.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRENDLINE_LOOKBACK = 30
SWING_ATR_LOOKBACK = 14


# ── Trendline fitting ─────────────────────────────────────────────────────

def _check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    """Sum of squared error if `slope` through `pivot` stays on the valid side of all points, else -1."""
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y
    if support and diffs.max() > 1e-5:
        return -1.0
    if not support and diffs.min() < -1e-5:
        return -1.0
    return float((diffs ** 2.0).sum())


def _optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray) -> Tuple[float, float]:
    """Coordinate-descent search from the least-squares slope to the tightest valid support/resistance slope."""
    slope_unit = (y.max() - y.min()) / len(y)
    min_step = 0.0001
    curr_step = 1.0

    best_slope = init_slope
    best_err = _check_trend_line(support, pivot, init_slope, y)
    if best_err < 0.0:
        # Least-squares slope isn't even a valid one-sided line through the pivot — bail out flat.
        return (0.0, y[pivot])

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            test_err = _check_trend_line(support, pivot, best_slope + slope_unit * min_step, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                test_err = _check_trend_line(support, pivot, best_slope - slope_unit * min_step, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                break  # numerical derivative failed both directions — keep current best
            get_derivative = False

        test_slope = best_slope - slope_unit * curr_step if derivative > 0.0 else best_slope + slope_unit * curr_step
        test_err = _check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err, best_slope = test_err, test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Fit a resistance line through the highs and a support line through the lows
    of the given window. Returns {"support": (slope, intercept), "resistance": (slope, intercept)}
    in the same coordinate space as the input arrays (caller decides log-transform).
    """
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = int((high - line_points).argmax())
    lower_pivot = int((low - line_points).argmin())

    support = _optimize_slope(True, lower_pivot, coefs[0], low)
    resistance = _optimize_slope(False, upper_pivot, coefs[0], high)
    return {"support": support, "resistance": resistance}


def detect_recent_trendlines(df: pd.DataFrame, lookback: int = TRENDLINE_LOOKBACK) -> Optional[Dict[str, Any]]:
    """
    Fit support/resistance trendlines on the `lookback` bars before the latest
    bar, project both lines one bar forward, and compare against the latest
    bar's actual close. Fits in log-price space (stabilizes slope across price
    regimes) and returns lines back in price space, ready to overlay on a
    candlestick chart.

    Returns None if there isn't enough history. Returns a dict with:
      index            — timestamps for the full window (lookback bars + latest bar)
      support_line     — array of length lookback+1, last point is the projected value
      resist_line      — array of length lookback+1, last point is the projected value
      support_slope / resist_slope — fitted slopes (log-price space)
      breakout         — "up" | "down" | None, whether the latest close crossed
                         a line it was not part of fitting
    """
    if df is None or len(df) < lookback + 1:
        return None

    window = df.iloc[-(lookback + 1):-1]
    current_close = float(df["Close"].iloc[-1])

    try:
        lines = fit_trendlines(
            np.log(window["High"].to_numpy()),
            np.log(window["Low"].to_numpy()),
            np.log(window["Close"].to_numpy()),
        )
    except Exception as e:
        logger.warning(f"Trendline fit failed: {e}")
        return None

    x_full = np.arange(lookback + 1)  # lookback fitted points + 1 projected point
    s_slope, s_intercept = lines["support"]
    r_slope, r_intercept = lines["resistance"]
    support_line = np.exp(s_slope * x_full + s_intercept)
    resist_line = np.exp(r_slope * x_full + r_intercept)

    breakout = None
    if current_close > resist_line[-1]:
        breakout = "up"
    elif current_close < support_line[-1]:
        breakout = "down"

    return {
        "index": df.index[-(lookback + 1):],
        "support_line": support_line,
        "resist_line": resist_line,
        "support_slope": float(s_slope),
        "resist_slope": float(r_slope),
        "projected_support": float(support_line[-1]),
        "projected_resistance": float(resist_line[-1]),
        "current_close": current_close,
        "breakout": breakout,
        "lookback": lookback,
    }


# ── Swing structure ────────────────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, atr_lookback: int = SWING_ATR_LOOKBACK) -> List[Dict[str, Any]]:
    """
    ATR-thresholded zigzag swing detector. A pending high (or low) is confirmed
    as a swing point once price reverses by more than one ATR against it.
    Returns a list of {"type": "high"|"low", "index", "timestamp", "price",
    "confirmed_index", "confirmed_timestamp"} in chronological order.
    """
    if df is None or len(df) < atr_lookback + 2:
        return []

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    idx = df.index

    extremes: List[Dict[str, Any]] = []
    up_move = True
    pend_max = pend_min = np.nan
    pend_max_i = pend_min_i = 0
    atr_sum = np.nan

    for i in range(len(df)):
        if i < atr_lookback:
            continue
        if i == atr_lookback:
            h_window = high[i - atr_lookback + 1: i + 1]
            l_window = low[i - atr_lookback + 1: i + 1]
            c_window = close[i - atr_lookback: i]
            tr1 = h_window - l_window
            tr2 = np.abs(h_window - c_window)
            tr3 = np.abs(l_window - c_window)
            atr_sum = float(np.sum(np.max(np.stack([tr1, tr2, tr3]), axis=0)))
        else:
            tr_curr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            rm_i = i - atr_lookback
            tr_remove = max(high[rm_i] - low[rm_i], abs(high[rm_i] - close[rm_i - 1]), abs(low[rm_i] - close[rm_i - 1]))
            atr_sum += tr_curr - tr_remove

        atr = atr_sum / atr_lookback

        if np.isnan(pend_max):
            pend_max, pend_min = high[i], low[i]
            pend_max_i = pend_min_i = i

        if up_move:
            if high[i] > pend_max:
                pend_max, pend_max_i = high[i], i
            elif low[i] < pend_max - atr:
                extremes.append({
                    "type": "high", "index": pend_max_i, "price": float(high[pend_max_i]),
                    "timestamp": idx[pend_max_i],
                    "confirmed_index": i, "confirmed_timestamp": idx[i],
                })
                up_move = False
                pend_min, pend_min_i = low[i], i
        else:
            if low[i] < pend_min:
                pend_min, pend_min_i = low[i], i
            elif high[i] > pend_min + atr:
                extremes.append({
                    "type": "low", "index": pend_min_i, "price": float(low[pend_min_i]),
                    "timestamp": idx[pend_min_i],
                    "confirmed_index": i, "confirmed_timestamp": idx[i],
                })
                up_move = True
                pend_max, pend_max_i = high[i], i

    return extremes
