"""
Flag & Pennant confidence scoring — a second pass over an already-detected
pattern from analysis/flag_pennant_detection.py. Kept separate from
detection so a future ML model can replace or augment this scoring layer
without touching the geometry pipeline.

Weighting (of 100), heaviest on what's observable at the trade decision
rather than cosmetic setup shape:
  Volume profile (pole/flag/breakout)   20
  Breakout momentum                     18
  Pole strength                         15
  ATR compression/expansion             15
  Trend alignment (EMA/SMA/ADX/RSI)     12
  Trendline/consolidation fit quality   10
  Flag symmetry                         10

Every sub-score is computed only from bars up to and including the
breakout bar — never from data that would not have been available at
signal time, since swing detection itself already lags real-time by
construction (a swing is only confirmed once price reverses by a full
ATR against it).
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEIGHTS = {
    "volume": 0.20,
    "breakout_momentum": 0.18,
    "pole_strength": 0.15,
    "atr_profile": 0.15,
    "trend_alignment": 0.12,
    "trendline_fit": 0.10,
    "flag_symmetry": 0.10,
}


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.5
    return float(max(0.0, min(1.0, x)))


def _col(df: pd.DataFrame, name: str, index: int, default: float = float("nan")) -> float:
    if name not in df.columns:
        return default
    val = df[name].iloc[index]
    return float(val) if pd.notna(val) else default


def _window_mean(df: pd.DataFrame, name: str, start: int, end: int) -> float:
    if name not in df.columns:
        return float("nan")
    vals = df[name].iloc[start:end + 1].dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _pole_strength(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    atr_pct_pole = _window_mean(df, "ATR_pct", pattern["pole_base_index"], pattern["pole_tip_index"])
    if atr_pct_pole != atr_pct_pole or atr_pct_pole <= 0:
        return 0.5
    return _clip01(pattern["pole_height_pct"] / (3 * atr_pct_pole / 100))


def _trendline_fit_quality(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    start, end = pattern["flag_start_index"], pattern["flag_end_index"]
    window = df.iloc[start:end + 1]
    x = np.arange(end - start + 1)
    upper_vals = np.exp(pattern["upper_slope"] * x + pattern["upper_intercept"])
    lower_vals = np.exp(pattern["lower_slope"] * x + pattern["lower_intercept"])
    upper_err = upper_vals - window["High"].to_numpy()
    lower_err = lower_vals - window["Low"].to_numpy()
    rmse = float(np.sqrt(np.mean(np.concatenate([upper_err, lower_err]) ** 2)))
    if pattern["flag_height"] <= 0:
        return 0.5
    return _clip01(1 - rmse / (0.5 * pattern["flag_height"]))


def _flag_symmetry(pattern: Dict[str, Any]) -> float:
    upper, lower = pattern["upper_slope"], pattern["lower_slope"]
    eps = 1e-9
    if pattern["pennant"]:
        return _clip01(1 - abs(abs(upper) - abs(lower)) / (max(abs(upper), abs(lower)) + eps))
    return _clip01(1 - abs(upper + lower) / (abs(upper) + abs(lower) + eps))


def _breakout_momentum(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    idx = pattern["breakout_index"]
    o, h, l, c = _col(df, "Open", idx), _col(df, "High", idx), _col(df, "Low", idx), _col(df, "Close", idx)
    atr = _col(df, "ATR", idx)
    if atr != atr or atr <= 0:
        return 0.5

    body = (c - o) if pattern["direction"] == "bull" else (o - c)
    momentum = _clip01(body / atr / 1.5)

    bar_range = h - l
    if bar_range > 0:
        clv = (c - l) / bar_range if pattern["direction"] == "bull" else (h - c) / bar_range
        if clv <= 0.6:
            momentum *= 0.5
    return momentum


def _volume_profile(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    pole_vol = _window_mean(df, "Volume", pattern["pole_base_index"], pattern["pole_tip_index"])
    pole_baseline = _window_mean(df, "vol_sma_20", pattern["pole_base_index"], pattern["pole_tip_index"])
    flag_vol = _window_mean(df, "Volume", pattern["flag_start_index"], pattern["flag_end_index"])
    breakout_vol = _col(df, "Volume", pattern["breakout_index"])

    vol_pole_score = _clip01(pole_vol / pole_baseline - 0.8) if pole_baseline > 0 else 0.5
    vol_flag_score = _clip01(1 - flag_vol / pole_vol) if pole_vol > 0 else 0.5
    vol_breakout_score = _clip01(breakout_vol / flag_vol / 2) if flag_vol > 0 else 0.5

    return 0.25 * vol_pole_score + 0.25 * vol_flag_score + 0.5 * vol_breakout_score


def _atr_profile(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    atr_pole = _window_mean(df, "ATR_pct", pattern["pole_base_index"], pattern["pole_tip_index"])
    atr_flag = _window_mean(df, "ATR_pct", pattern["flag_start_index"], pattern["flag_end_index"])
    atr_breakout = _col(df, "ATR_pct", pattern["breakout_index"])

    if atr_pole != atr_pole or atr_flag != atr_flag or atr_pole <= 0 or atr_flag <= 0:
        return 0.5

    compression = _clip01((atr_pole - atr_flag) / atr_pole / 0.5)
    expansion = _clip01(atr_breakout / atr_flag / 1.5) if atr_breakout == atr_breakout else 0.5
    return (compression + expansion) / 2


def _rsi_zone_score(rsi: float) -> float:
    if rsi != rsi:
        return 0.5
    if 40 <= rsi <= 60:
        return 1.0
    if rsi < 40:
        return _clip01((rsi - 30) / 10) if rsi > 30 else 0.0
    return _clip01((70 - rsi) / 10) if rsi < 70 else 0.0


def _trend_alignment(df: pd.DataFrame, pattern: Dict[str, Any]) -> float:
    idx = pattern["breakout_index"]
    ema50 = _col(df, "EMA_50", idx)
    sma200 = _col(df, "SMA_200", idx)
    adx = _col(df, "ADX", idx, default=0.0)
    rsi = _window_mean(df, "RSI", pattern["flag_start_index"], pattern["flag_end_index"])

    if ema50 != ema50 or sma200 != sma200:
        trend_component = 0.5
    else:
        trend_component = 1.0 if ((ema50 > sma200) == (pattern["direction"] == "bull")) else 0.0

    adx_component = _clip01(adx / 25)
    rsi_component = _rsi_zone_score(rsi)
    return 0.4 * trend_component + 0.3 * adx_component + 0.3 * rsi_component


def score_pattern(df: pd.DataFrame, pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Fills confidence_score (0-100) and score_components onto `pattern`
    in place, returns it. Requires df to already carry the indicator
    columns from calculate_indicators() (ATR/ATR_pct/RSI/EMA/SMA/ADX/
    vol_sma_20) — missing columns fall back to a neutral 0.5 sub-score
    rather than penalizing or crashing."""
    components = {
        "pole_strength": _pole_strength(df, pattern),
        "trendline_fit": _trendline_fit_quality(df, pattern),
        "flag_symmetry": _flag_symmetry(pattern),
        "breakout_momentum": _breakout_momentum(df, pattern),
        "volume": _volume_profile(df, pattern),
        "atr_profile": _atr_profile(df, pattern),
        "trend_alignment": _trend_alignment(df, pattern),
    }
    score = sum(WEIGHTS[key] * value for key, value in components.items()) * 100

    pattern["score_components"] = {k: round(v, 3) for k, v in components.items()}
    pattern["confidence_score"] = round(score, 1)
    logger.debug(f"score_pattern: confidence={pattern['confidence_score']:.1f} components={pattern['score_components']}")
    return pattern


def passes_trend_filter(df: pd.DataFrame, pattern: Dict[str, Any], adx_min: float = 20.0) -> bool:
    """Bull flags require EMA_50 > SMA_200 and ADX >= adx_min at the
    breakout bar; bear flags require the inverse. If EMA_50/SMA_200 aren't
    available yet (short history), the filter is skipped rather than
    rejecting the pattern outright."""
    idx = pattern["breakout_index"]
    ema50 = _col(df, "EMA_50", idx)
    sma200 = _col(df, "SMA_200", idx)
    adx = _col(df, "ADX", idx, default=0.0)

    if ema50 != ema50 or sma200 != sma200:
        logger.warning("passes_trend_filter: EMA_50/SMA_200 unavailable (short history) — skipping trend filter, passing by default.")
        return True

    trend_ok = (ema50 > sma200) if pattern["direction"] == "bull" else (ema50 < sma200)
    passed = bool(trend_ok and adx >= adx_min)
    logger.debug(f"passes_trend_filter: passed={passed} direction={pattern['direction']} ema50={ema50:.4f} sma200={sma200:.4f} adx={adx:.1f} adx_min={adx_min:.1f}")
    return passed
