"""
Flag & Pennant continuation-pattern detection — pure price-action geometry,
same category as analysis/trendlines.py and analysis/patterns.py. No ML,
no scoring, no trade execution here; see analysis/flag_pennant_scoring.py
and analysis/flag_pennant_backtest.py for those concerns.

Pipeline: swing points (analysis.trendlines.detect_swing_points) -> pole
(a swing low->high or high->low leg) -> consolidation window immediately
after the pole tip -> trendline fit (analysis.trendlines.fit_trendlines,
log-price space) on that window -> breakout confirmation. A pattern is
only returned once its breakout has actually happened — never a forming,
unconfirmed pattern.

Ported from neurotrader888/TechnicalAnalysisAutomation's flags_pennants.py,
adapted to reuse this app's own ATR-zigzag swing detector (rather than the
reference repo's separate symmetric-window rw_extremes) so the app has a
single swing-point definition shared by every pattern module, and adapted
to operate on this app's OHLCV DataFrame convention instead of a raw 1D
price array.

Pattern dict schema (no dataclass — this codebase represents structured
records as plain dicts throughout, e.g. detect_candlestick_pattern,
detect_recent_trendlines):
    direction            "bull" | "bear"
    pennant              bool — False means parallel-channel Flag
    pole_base_index/_price, pole_tip_index/_price
    pole_width           bars from base to tip
    pole_height          |tip_price - base_price|
    pole_height_pct      pole_height / pole_tip_price
    flag_start_index/_end_index   consolidation window (inclusive)
    flag_width           bars in the consolidation window
    flag_height          high-low range of the consolidation window
    upper_slope/_intercept, lower_slope/_intercept   log-price space,
                         x=0 at flag_start_index
    breakout_index/_price, breakout_confirmed
    confidence_score, score_components   filled in by flag_pennant_scoring
    entry_price, stop_price, target_price, risk, reward, mfe, mae,
    return_pct, holding_period_bars      filled in by flag_pennant_backtest
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.trendlines import detect_swing_points, fit_trendlines
from analysis.flag_pennant_scoring import score_pattern, passes_trend_filter
from analysis.flag_pennant_backtest import evaluate_pattern_trade

logger = logging.getLogger(__name__)

MIN_POLE_PCT = 0.05
MAX_FLAG_WIDTH_RATIO = 0.5
MAX_FLAG_HEIGHT_RATIO = 0.6
MIN_FLAG_BARS = 3
DEFAULT_MIN_CONFIDENCE = 60.0
DEFAULT_ADX_MIN = 20.0
DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_MAX_HOLDING_BARS = 20
DEFAULT_MAX_SIGNAL_AGE_BARS = 10


# ── Pole detection ───────────────────────────────────────────────────────

def _detect_poles(swings: List[Dict[str, Any]], min_pole_pct: float) -> List[Dict[str, Any]]:
    """Pair consecutive swings (alternating high/low by construction) into
    candidate poles, filtered by minimum move size."""
    poles = []
    for base, tip in zip(swings, swings[1:]):
        if base["type"] == "low" and tip["type"] == "high":
            direction = "bull"
        elif base["type"] == "high" and tip["type"] == "low":
            direction = "bear"
        else:
            continue

        pole_height = abs(tip["price"] - base["price"])
        pole_height_pct = pole_height / tip["price"] if tip["price"] else 0.0
        if pole_height_pct < min_pole_pct:
            continue

        poles.append({
            "direction": direction,
            "pole_base_index": base["index"],
            "pole_base_price": float(base["price"]),
            "pole_tip_index": tip["index"],
            "pole_tip_price": float(tip["price"]),
            "pole_width": tip["index"] - base["index"],
            "pole_height": float(pole_height),
            "pole_height_pct": float(pole_height_pct),
        })
    return poles


# ── Consolidation + trendline fit ────────────────────────────────────────

def _fit_window_trendlines(df: pd.DataFrame, start: int, end: int) -> Dict[str, Tuple[float, float]]:
    """Fit support/resistance lines on df.iloc[start:end+1] in log-price
    space, x relative to `start` (0-based)."""
    window = df.iloc[start:end + 1]
    return fit_trendlines(
        np.log(window["High"].to_numpy()),
        np.log(window["Low"].to_numpy()),
        np.log(window["Close"].to_numpy()),
    )


def _classify_pennant(upper_slope: float, lower_slope: float) -> bool:
    """Converging (opposite-signed) slopes -> Pennant. Same-signed (roughly
    parallel) slopes -> Flag."""
    return bool((upper_slope * lower_slope) < 0)


def _measure_consolidation(
    df: pd.DataFrame,
    pole: Dict[str, Any],
    max_flag_width_ratio: float,
    max_flag_height_ratio: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Returns (pattern_candidate, rejection) — exactly one is non-None."""
    tip_index = pole["pole_tip_index"]
    bars_available = len(df) - 1 - tip_index
    max_width = min(bars_available, max(MIN_FLAG_BARS, int(round(pole["pole_width"] * max_flag_width_ratio))))

    if max_width < MIN_FLAG_BARS:
        return None, {"stage": "consolidation", "reason": "insufficient bars after pole tip", **pole}

    window_end = tip_index + max_width
    window = df.iloc[tip_index:window_end + 1]
    flag_height = float(window["High"].max() - window["Low"].min())
    flag_height_pct = flag_height / pole["pole_tip_price"] if pole["pole_tip_price"] else 0.0

    if flag_height_pct > max_flag_height_ratio * pole["pole_height_pct"]:
        return None, {"stage": "consolidation", "reason": "flag height exceeds threshold vs pole height", **pole}

    try:
        lines = _fit_window_trendlines(df, tip_index, window_end)
    except Exception as e:
        logger.debug(f"Trendline fit failed for pole at {tip_index}: {e}")
        return None, {"stage": "trendline_fit", "reason": str(e), **pole}

    s_slope, s_intercept = lines["support"]
    r_slope, r_intercept = lines["resistance"]
    upper_slope, upper_intercept = r_slope, r_intercept
    lower_slope, lower_intercept = s_slope, s_intercept

    pattern = {
        **pole,
        "pennant": _classify_pennant(upper_slope, lower_slope),
        "flag_start_index": tip_index,
        "flag_end_index": window_end,
        "flag_width": max_width,
        "flag_height": flag_height,
        "upper_slope": float(upper_slope),
        "upper_intercept": float(upper_intercept),
        "lower_slope": float(lower_slope),
        "lower_intercept": float(lower_intercept),
        "breakout_index": None,
        "breakout_price": None,
        "breakout_confirmed": False,
    }
    return pattern, None


# ── Breakout confirmation ────────────────────────────────────────────────

def _find_breakout(df: pd.DataFrame, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Search forward from the end of the consolidation window for the
    first bar whose close crosses the projected trendline in the pole's
    direction. Returns the updated pattern if found, else None (pattern
    stays unconfirmed and must not be surfaced — never signal before
    confirmation)."""
    tip_index = pattern["pole_tip_index"]
    window_end = pattern["flag_end_index"]
    lookahead_end = min(len(df) - 1, window_end + max(MIN_FLAG_BARS, pattern["flag_width"]))

    for idx in range(window_end + 1, lookahead_end + 1):
        x = idx - tip_index
        upper_val = np.exp(pattern["upper_slope"] * x + pattern["upper_intercept"])
        lower_val = np.exp(pattern["lower_slope"] * x + pattern["lower_intercept"])
        close = float(df["Close"].iloc[idx])

        if pattern["direction"] == "bull" and close > upper_val:
            pattern["breakout_index"] = idx
            pattern["breakout_price"] = close
            pattern["breakout_confirmed"] = True
            return pattern
        if pattern["direction"] == "bear" and close < lower_val:
            pattern["breakout_index"] = idx
            pattern["breakout_price"] = close
            pattern["breakout_confirmed"] = True
            return pattern

    return None


# ── Orchestration ─────────────────────────────────────────────────────────

def find_flags_and_pennants(
    df: pd.DataFrame,
    min_pole_pct: float = MIN_POLE_PCT,
    max_flag_width_ratio: float = MAX_FLAG_WIDTH_RATIO,
    max_flag_height_ratio: float = MAX_FLAG_HEIGHT_RATIO,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs the full swing -> pole -> consolidation -> breakout pipeline.
    Returns (confirmed_patterns, rejected_candidates) — every rejected
    candidate is logged with a stage/reason for later review or ML dataset
    generation, never silently dropped."""
    rejected: List[Dict[str, Any]] = []
    if df is None or len(df) < 20:
        return [], rejected

    swings = detect_swing_points(df)
    poles = _detect_poles(swings, min_pole_pct)

    confirmed: List[Dict[str, Any]] = []
    for pole in poles:
        pattern, rejection = _measure_consolidation(df, pole, max_flag_width_ratio, max_flag_height_ratio)
        if rejection is not None:
            rejected.append(rejection)
            continue

        result = _find_breakout(df, pattern)
        if result is None:
            rejected.append({"stage": "breakout", "reason": "no breakout within lookahead window", **pattern})
            continue

        confirmed.append(result)

    return confirmed, rejected


def detect_flag_pennant_patterns(
    df: pd.DataFrame,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    adx_min: float = DEFAULT_ADX_MIN,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
    max_signal_age_bars: int = DEFAULT_MAX_SIGNAL_AGE_BARS,
    **detection_kwargs: Any,
) -> Dict[str, Any]:
    """
    Top-level entry point for pages/trading.py. `df` must already be the
    output of calculate_indicators() (OHLCV + ATR/RSI/EMA/SMA/ADX/vol_ratio
    columns) — detection, scoring, and backtest all read from this single
    enriched frame, matching the rest of analysis/.

    Returns:
      signal          {"name","direction","note"} for the existing 5-card
                      grid, or None — only set from a pattern whose
                      breakout happened within the last `max_signal_age_bars`
      patterns        all confirmed patterns clearing min_confidence and
                      the trend filter (for chart overlay — can be several)
      rejected        every rejected candidate, all stages, for debug/ML use
      latest_pattern  the pattern behind `signal`, or None
    """
    if df is None or df.empty:
        return {"signal": None, "patterns": [], "rejected": [], "latest_pattern": None}

    patterns, rejected = find_flags_and_pennants(df, **detection_kwargs)

    accepted: List[Dict[str, Any]] = []
    for pattern in patterns:
        score_pattern(df, pattern)

        if not passes_trend_filter(df, pattern, adx_min=adx_min):
            rejected.append({"stage": "trend_filter", "reason": "against higher-timeframe trend", **pattern})
            continue
        if pattern["confidence_score"] < min_confidence:
            rejected.append({"stage": "confidence", "reason": f"below {min_confidence:.0f} threshold", **pattern})
            continue

        evaluate_pattern_trade(df, pattern, stop_atr_mult=stop_atr_mult, max_holding_bars=max_holding_bars)
        accepted.append(pattern)

    accepted.sort(key=lambda p: p["breakout_index"])

    last_index = len(df) - 1
    recent = [p for p in accepted if last_index - p["breakout_index"] <= max_signal_age_bars]
    latest = recent[-1] if recent else None

    signal = None
    if latest is not None:
        shape = "Pennant" if latest["pennant"] else "Flag"
        name = f"{'Bull' if latest['direction'] == 'bull' else 'Bear'} {shape}"
        signal = {
            "name": name,
            "direction": latest["direction"],
            "note": f"Confidence {latest['confidence_score']:.0f}/100 — breakout confirmed at ${latest['breakout_price']:.2f}",
        }

    return {"signal": signal, "patterns": accepted, "rejected": rejected, "latest_pattern": latest}
