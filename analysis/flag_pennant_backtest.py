"""
Flag & Pennant trade evaluation — fills in entry/stop/target/risk/reward/
MFE/MAE/return%/holding-period fields on an already-scored pattern from
analysis/flag_pennant_detection.py. Mirrors analysis/backtest.py's
close-to-close approximation (no intrabar fills) rather than introducing a
second simulation convention.

Stop placement: the flag's opposite boundary (flag low for bull, flag high
for bear) — violating that level invalidates the consolidation thesis
itself, which is a tighter and more pattern-derived level than the pole
midpoint. Floored at 1x ATR from entry so stops aren't unrealistically
tight in low-volatility consolidations.

Target placement: classic measured move — the pole's height projected
from the breakout price (Edwards & Magee convention for flag/pennant
continuation). A secondary fixed 2R target is also recorded so the two
conventions can be compared empirically once enough trades accumulate.
"""
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _col(df: pd.DataFrame, name: str, index: int, default: float = float("nan")) -> float:
    if name not in df.columns:
        logger.warning(f"_col: column '{name}' missing from df — falling back to default={default}.")
        return default
    val = df[name].iloc[index]
    return float(val) if pd.notna(val) else default


def evaluate_pattern_trade(
    df: pd.DataFrame,
    pattern: Dict[str, Any],
    stop_atr_mult: float = 1.5,
    max_holding_bars: int = 20,
) -> Dict[str, Any]:
    """Fills entry_price, stop_price, target_price, target_2r,
    risk, reward, mfe, mae, return_pct, holding_period_bars onto
    `pattern` in place, returns it. Walks forward from the bar after
    breakout_index, close-to-close, until the stop or target is hit or
    max_holding_bars elapses (whichever first)."""
    idx = pattern["breakout_index"]
    entry_price = pattern["breakout_price"]
    bullish = pattern["direction"] == "bull"

    atr = _col(df, "ATR", idx, default=0.0)
    min_stop_distance = stop_atr_mult * atr if atr > 0 else entry_price * 0.01

    if bullish:
        flag_boundary_stop = df["Low"].iloc[pattern["flag_start_index"]:pattern["flag_end_index"] + 1].min()
        stop_price = min(float(flag_boundary_stop), entry_price - min_stop_distance)
        target_price = entry_price + pattern["pole_height"]
    else:
        flag_boundary_stop = df["High"].iloc[pattern["flag_start_index"]:pattern["flag_end_index"] + 1].max()
        stop_price = max(float(flag_boundary_stop), entry_price + min_stop_distance)
        target_price = entry_price - pattern["pole_height"]

    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    target_2r = entry_price + 2 * risk if bullish else entry_price - 2 * risk

    logger.debug(
        f"evaluate_pattern_trade: entry_price={entry_price:.4f} stop_price={stop_price:.4f} "
        f"target_price={target_price:.4f} bullish={bullish}"
    )

    exit_price = None
    exit_reason = "open"
    holding_period_bars = 0
    mfe = mae = 0.0

    last_index = len(df) - 1
    lookahead_end = min(last_index, idx + max_holding_bars)

    for i in range(idx + 1, lookahead_end + 1):
        close = float(df["Close"].iloc[i])
        excursion = (close - entry_price) if bullish else (entry_price - close)
        mfe = max(mfe, excursion)
        mae = min(mae, excursion)
        holding_period_bars = i - idx

        hit_target = close >= target_price if bullish else close <= target_price
        hit_stop = close <= stop_price if bullish else close >= stop_price

        if hit_target:
            exit_price, exit_reason = target_price, "target"
            break
        if hit_stop:
            exit_price, exit_reason = stop_price, "stop"
            break
    else:
        if lookahead_end > idx:
            exit_price = float(df["Close"].iloc[lookahead_end])
            exit_reason = "time_exit"
            holding_period_bars = lookahead_end - idx

    return_pct = None
    if exit_price is not None:
        return_pct = ((exit_price - entry_price) / entry_price * 100) if bullish else ((entry_price - exit_price) / entry_price * 100)

    pattern.update({
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "target_2r": round(target_2r, 4),
        "risk": round(risk, 4),
        "reward": round(reward, 4),
        "mfe": round(mfe, 4),
        "mae": round(mae, 4),
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
        "holding_period_bars": holding_period_bars,
        "exit_reason": exit_reason,
    })
    logger.info(
        f"evaluate_pattern_trade: complete — exit_reason={exit_reason} "
        f"holding_period_bars={holding_period_bars} return_pct={pattern['return_pct']}"
    )
    return pattern
