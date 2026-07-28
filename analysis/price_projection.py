"""
Probabilistic 5-day price path — a Monte Carlo (geometric random walk)
simulation seeded with the stock's own recent volatility (hv_21, already
computed by calculate_indicators) and the direction model's own probability /
expected-move output.

This is NOT a new trained model and does not predict a specific price. It
simulates thousands of possible paths and reports the day-by-day median and
25th/75th percentile range, so the output is a probability band, not a point
forecast — consistent with the rest of the app's stance against overclaiming
precision (see docs/ML_PREDICTION.md).
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_N_SIMS = 3000
OVERNIGHT_GAP_VOL_FRACTION = 0.4  # overnight gaps are typically a fraction of a full day's vol


def simulate_price_path(
    df: pd.DataFrame,
    bull_probability: float,
    expected_move_pct: Optional[float],
    n_days: int = 5,
    n_sims: int = DEFAULT_N_SIMS,
) -> Dict[str, Any]:
    """
    Simulate n_days of daily open/close prices via a geometric random walk.

    Volatility comes from hv_21 (annualized, converted back to a daily
    figure); falls back to a direct recomputation from Close if hv_21 isn't
    present. Drift comes from the model's own expected_move_pct (median
    historical 5-day return for this setup), spread evenly across n_days; if
    that's unavailable, drift falls back to a small probability-scaled tilt.
    Each day's open is modeled as the prior day's simulated close plus an
    overnight gap term.
    """
    current_price = float(df["Close"].iloc[-1])

    hv_21 = df["hv_21"].iloc[-1] if "hv_21" in df.columns else None
    if hv_21 is None or pd.isna(hv_21) or hv_21 <= 0:
        logger.warning("simulate_price_path: hv_21 missing/invalid, falling back to recomputed Close volatility.")
        daily_sigma = float(df["Close"].pct_change().tail(21).std())
        if pd.isna(daily_sigma) or daily_sigma <= 0:
            logger.warning("simulate_price_path: recomputed daily_sigma is NaN/non-positive, falling back to default 0.015.")
            daily_sigma = 0.015
    else:
        daily_sigma = float(hv_21) / 100 / np.sqrt(TRADING_DAYS_PER_YEAR)

    if expected_move_pct is not None:
        total_drift = expected_move_pct / 100
    else:
        total_drift = float(np.clip((bull_probability - 0.5) * 0.04, -0.02, 0.02))
    daily_drift = total_drift / n_days

    seed = int(pd.Timestamp(df.index[-1]).timestamp()) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    daily_rets = rng.normal(loc=daily_drift, scale=daily_sigma, size=(n_sims, n_days))
    close_paths = current_price * np.cumprod(1 + daily_rets, axis=1)

    gap_rets = rng.normal(loc=0.0, scale=daily_sigma * OVERNIGHT_GAP_VOL_FRACTION, size=(n_sims, n_days))
    prev_close_paths = np.hstack([np.full((n_sims, 1), current_price), close_paths[:, :-1]])
    open_paths = prev_close_paths * (1 + gap_rets)

    days = []
    for d in range(n_days):
        opens, closes = open_paths[:, d], close_paths[:, d]
        days.append({
            "day": d + 1,
            "open_p25": round(float(np.percentile(opens, 25)), 2),
            "open_median": round(float(np.median(opens)), 2),
            "open_p75": round(float(np.percentile(opens, 75)), 2),
            "close_p25": round(float(np.percentile(closes, 25)), 2),
            "close_median": round(float(np.median(closes)), 2),
            "close_p75": round(float(np.percentile(closes, 75)), 2),
        })

    logger.debug(
        f"simulate_price_path: n_days={n_days} current_price={current_price:.2f} "
        f"daily_volatility_pct={daily_sigma * 100:.2f} day{n_days}_close_median={days[-1]['close_median']}"
    )
    return {
        "current_price": round(current_price, 2),
        "daily_volatility_pct": round(daily_sigma * 100, 2),
        "days": days,
        "n_sims": n_sims,
    }


def simulate_intraday_path(
    intraday_df: pd.DataFrame,
    bull_probability: float = 0.5,
    expected_move_pct: Optional[float] = None,
    n_sims: int = DEFAULT_N_SIMS,
) -> Dict[str, Any]:
    """
    Simulate the rest of TODAY's session as a geometric random walk in
    intraday-bar steps (matches whatever bar size intraday_df is, e.g. 1h).

    This is NOT the ML model gaining hourly resolution — ml_prediction.py
    only classifies multi-day direction from daily bars. Per-bar volatility
    here comes from this ticker's own recent intraday bars; the day's
    directional tilt is borrowed from the ML model's bull_probability /
    expected_move_pct (same inputs simulate_price_path uses for the 5-day
    path) and spread over the bars left before the 4pm ET close, so today's
    envelope is informed by the daily signal without pretending the
    classifier itself sees hourly bars.
    """
    if intraday_df is None or intraday_df.empty or len(intraday_df) < 2:
        logger.warning("simulate_intraday_path: not enough intraday bars for today's session.")
        return {"error": "Not enough intraday bars for today's session"}

    current_price = float(intraday_df["Close"].iloc[-1])
    bar_rets = intraday_df["Close"].pct_change().dropna()
    bar_sigma = float(bar_rets.tail(200).std())
    if pd.isna(bar_sigma) or bar_sigma <= 0:
        logger.warning("simulate_intraday_path: bar_sigma is NaN/non-positive, falling back to default 0.003.")
        bar_sigma = 0.003

    last_ts = intraday_df.index[-1]
    bar_duration = last_ts - intraday_df.index[-2]
    session_close = last_ts.normalize() + pd.Timedelta(hours=16)
    bars_remaining = max(int((session_close - last_ts) / bar_duration), 1)

    if expected_move_pct is not None:
        day_drift = expected_move_pct / 100
    else:
        day_drift = float(np.clip((bull_probability - 0.5) * 0.02, -0.01, 0.01))
    bar_drift = day_drift / bars_remaining

    seed = int(pd.Timestamp(last_ts).timestamp()) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    bar_rets_sim = rng.normal(loc=bar_drift, scale=bar_sigma, size=(n_sims, bars_remaining))
    paths = current_price * np.cumprod(1 + bar_rets_sim, axis=1)

    checkpoints = sorted(set(
        max(1, round(bars_remaining * frac)) for frac in (0.25, 0.5, 0.75, 1.0)
    ))
    steps = []
    for c in checkpoints:
        vals = paths[:, c - 1]
        steps.append({
            "bars_ahead": c,
            "p25": round(float(np.percentile(vals, 25)), 2),
            "median": round(float(np.median(vals)), 2),
            "p75": round(float(np.percentile(vals, 75)), 2),
        })

    logger.debug(
        f"simulate_intraday_path: bars_remaining={bars_remaining} current_price={current_price:.2f} "
        f"session_close_median={steps[-1]['median']}"
    )
    return {
        "current_price": round(current_price, 2),
        "bar_volatility_pct": round(bar_sigma * 100, 2),
        "bars_remaining": bars_remaining,
        "session_close_p25": steps[-1]["p25"],
        "session_close_median": steps[-1]["median"],
        "session_close_p75": steps[-1]["p75"],
        "steps": steps,
        "n_sims": n_sims,
    }
