"""
Opening Range Breakout Confirmation (ORBC) — intraday momentum strategy.

The premise: the first N minutes after the open establish a reference range,
and a genuine trend day tends to leave that range and stay out. Rather than
entering on the first candle that pokes outside (the classic false-breakout
trap), ORBC waits for a *second* consecutive close outside the range before
signalling.

Deterministic geometry over OHLCV bars only — same category of rule as
analysis/mtf_strategy.py and analysis/flag_pennant_detection.py. No ML, no
order-book data.

Confirmation semantics
----------------------
The signal fires on the Nth *consecutive* close outside the opening range
(`confirmation_closes`, default 2). A close back inside the range resets the
count and the tracked direction to zero — a breakout that round-trips has to
start over.

If the Nth close is blocked by a filter (volume/VWAP/ATR), the count keeps
advancing and later closes may still fire, up to `max_confirmation_closes`
(default 3). This is the "second OR third candle" rule from the strategy
spec: exactly one signal per breakout episode, but a filter miss on the
second bar doesn't kill a setup that confirms on the third.

Session handling
----------------
All timestamps are normalized to America/New_York before any time-of-day
comparison, so the 9:30 open, the opening-range window, and the entry cutoff
mean market clock time regardless of the machine's local zone or whether the
provider returned a naive index. Sessions are grouped by ET calendar date;
each session is evaluated independently, and no state carries across days.

Half-days (1:00 PM close) and holidays need no special handling: sessions are
derived from the bars actually present, so a short session simply has fewer
bars after the opening range.

Longs and shorts are both supported. `evaluate_orbc_trade()` is
direction-aware, unlike the long-only simulators elsewhere in this app.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.tz import MARKET_TZ

logger = logging.getLogger(__name__)

MARKET_OPEN = dtime(9, 30)

STOP_METHODS = ("opening_range", "atr", "percent")
TARGET_METHODS = ("risk_reward", "atr", "opening_range_extension")

# Confidence-score weights (sum to 1.0). Weighted toward what is actually
# observable at the signal bar rather than cosmetic range shape.
SCORE_WEIGHTS = {
    "confirmation_strength": 0.25,
    "volume_thrust": 0.25,
    "vwap_alignment": 0.20,
    "range_quality": 0.15,
    "trend_alignment": 0.15,
}


@dataclass
class ORBCConfig:
    """
    Every knob the strategy spec calls for. Defaults reproduce the spec's
    stated defaults: 15-minute opening range, confirm on the 2nd close
    (falling through to the 3rd), entries only between the end of the
    opening range and 11:00 AM ET, all three filters on, stop at the
    opposite side of the opening range, target at 2:1 risk/reward.
    """
    opening_range_minutes: int = 15

    # Fire on the Nth consecutive close outside the range; if a filter blocks
    # it, keep trying until max_confirmation_closes.
    confirmation_closes: int = 2
    max_confirmation_closes: int = 3

    entry_cutoff: dtime = dtime(11, 0)

    require_volume: bool = True
    volume_multiple: float = 1.0        # Volume > multiple * vol_sma_20
    require_vwap: bool = True
    require_atr: bool = True
    atr_multiple: float = 0.5           # opening range size > multiple * ATR

    allow_long: bool = True
    allow_short: bool = True

    stop_method: str = "opening_range"
    stop_atr_multiple: float = 1.0
    stop_percent: float = 0.5           # percent of entry price

    target_method: str = "risk_reward"
    target_rr: float = 2.0
    target_atr_multiple: float = 2.0

    # When True, the first confirmed signal ends the session's scan — including
    # a reversal. A day that breaks above the range, stops out, then breaks
    # below it yields one trade, not two. Set False to allow re-entry: after a
    # signal the count resets and a fresh breakout episode can fire again.
    one_signal_per_session: bool = True
    exit_at_session_end: bool = True    # intraday strategy — never hold overnight

    def __post_init__(self) -> None:
        if self.opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be positive")
        if self.confirmation_closes < 1:
            raise ValueError("confirmation_closes must be >= 1")
        if self.max_confirmation_closes < self.confirmation_closes:
            raise ValueError("max_confirmation_closes must be >= confirmation_closes")
        if self.stop_method not in STOP_METHODS:
            raise ValueError(f"stop_method must be one of {STOP_METHODS}")
        if self.target_method not in TARGET_METHODS:
            raise ValueError(f"target_method must be one of {TARGET_METHODS}")
        if not (self.allow_long or self.allow_short):
            raise ValueError("at least one of allow_long/allow_short must be True")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.5
    return float(max(0.0, min(1.0, x)))


def _col(df: pd.DataFrame, name: str, pos: int, default: float = float("nan")) -> float:
    """Positional column read that tolerates a missing column or NaN."""
    if name not in df.columns:
        return default
    val = df[name].iloc[pos]
    return float(val) if pd.notna(val) else default


def to_market_tz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return `df` with its DatetimeIndex expressed in America/New_York.

    A naive index is assumed to already be market-local (yfinance returns
    exchange-local timestamps for intraday intervals) and is localized rather
    than shifted, so the 9:30 open lines up either way.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(MARKET_TZ)
    else:
        idx = idx.tz_convert(MARKET_TZ)
    out = df.copy()
    out.index = idx
    return out


def _minutes_since_open(ts: pd.Timestamp) -> int:
    return (ts.hour - MARKET_OPEN.hour) * 60 + (ts.minute - MARKET_OPEN.minute)


def session_dates(df: pd.DataFrame) -> List[pd.Timestamp]:
    """Distinct ET calendar dates present in an already-tz-normalized frame."""
    return sorted(set(df.index.normalize()))


def compute_opening_range(
    session_df: pd.DataFrame,
    opening_range_minutes: int = 15,
) -> Optional[Dict[str, Any]]:
    """
    High/low of the bars inside [09:30, 09:30 + opening_range_minutes).

    Returns None when the session has no bars in that window at all (a late
    data start or a partial feed), since without a reference range there is
    nothing for the rest of the strategy to measure against.

    The window is half-open on the right so a 15-minute range over 5-minute
    bars captures exactly the 09:30/09:35/09:40 bars and stops before the
    09:45 bar, which is the first bar eligible to break out.
    """
    if session_df is None or session_df.empty:
        return None

    mins = np.array([_minutes_since_open(ts) for ts in session_df.index])
    mask = (mins >= 0) & (mins < opening_range_minutes)
    window = session_df[mask]
    if window.empty:
        return None

    high = float(window["High"].max())
    low = float(window["Low"].min())
    return {
        "opening_high": high,
        "opening_low": low,
        "range_size": high - low,
        "bar_count": int(len(window)),
        "range_start": window.index[0],
        "range_end": window.index[-1],
    }


def _stop_and_target(
    entry: float,
    direction: str,
    orange: Dict[str, Any],
    atr: float,
    config: ORBCConfig,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Stop/target for a confirmed signal, mirrored for shorts. Returns
    (None, None) when the requested method can't be computed (e.g. an ATR
    stop with no ATR available yet) so the caller can reject the signal
    instead of inventing a level.
    """
    sign = 1.0 if direction == "long" else -1.0
    has_atr = atr == atr and atr > 0

    if config.stop_method == "opening_range":
        stop = orange["opening_low"] if direction == "long" else orange["opening_high"]
    elif config.stop_method == "atr":
        if not has_atr:
            return None, None
        stop = entry - sign * config.stop_atr_multiple * atr
    else:  # percent
        stop = entry * (1 - sign * config.stop_percent / 100.0)

    risk = (entry - stop) * sign
    if risk <= 0:
        return None, None

    if config.target_method == "risk_reward":
        target = entry + sign * config.target_rr * risk
    elif config.target_method == "atr":
        if not has_atr:
            return None, None
        target = entry + sign * config.target_atr_multiple * atr
    else:  # opening_range_extension — project the range width from the break
        edge = orange["opening_high"] if direction == "long" else orange["opening_low"]
        target = edge + sign * orange["range_size"]

    if (target - entry) * sign <= 0:
        return None, None
    return float(stop), float(target)


# ── Filters ──────────────────────────────────────────────────────────────────

def _check_filters(
    session_df: pd.DataFrame,
    pos: int,
    direction: str,
    orange: Dict[str, Any],
    config: ORBCConfig,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Apply the volume/VWAP/ATR filters at the candidate signal bar.

    Returns (passed, list_of_failure_reasons, detail_dict). A filter whose
    input column is unavailable is skipped rather than failing the signal —
    same convention as flag_pennant_scoring.passes_trend_filter — and is
    recorded in the detail dict so the UI can say it was skipped.
    """
    failures: List[str] = []
    detail: Dict[str, Any] = {}

    close = _col(session_df, "Close", pos)
    volume = _col(session_df, "Volume", pos)
    vol_avg = _col(session_df, "vol_sma_20", pos)
    vwap = _col(session_df, "VWAP", pos)
    atr = _col(session_df, "ATR", pos)

    if config.require_volume:
        if vol_avg != vol_avg or vol_avg <= 0 or volume != volume:
            detail["volume"] = "skipped (no 20-bar volume average yet)"
        else:
            ratio = volume / vol_avg
            detail["volume"] = round(ratio, 2)
            if ratio <= config.volume_multiple:
                failures.append(
                    f"breakout volume {ratio:.2f}x the 20-bar average, needs > {config.volume_multiple:.2f}x"
                )

    if config.require_vwap:
        if vwap != vwap:
            detail["vwap"] = "skipped (no VWAP yet)"
        else:
            detail["vwap"] = round(vwap, 4)
            if direction == "long" and close <= vwap:
                failures.append(f"long signal but price {close:.2f} is not above VWAP {vwap:.2f}")
            elif direction == "short" and close >= vwap:
                failures.append(f"short signal but price {close:.2f} is not below VWAP {vwap:.2f}")

    if config.require_atr:
        if atr != atr or atr <= 0:
            detail["atr"] = "skipped (no ATR yet)"
        else:
            needed = config.atr_multiple * atr
            detail["atr"] = round(atr, 4)
            detail["range_vs_atr"] = round(orange["range_size"] / atr, 2) if atr else None
            if orange["range_size"] <= needed:
                failures.append(
                    f"opening range {orange['range_size']:.2f} is too tight — needs > "
                    f"{config.atr_multiple:.2f} x ATR ({needed:.2f})"
                )

    return (not failures), failures, detail


# ── Confidence scoring ───────────────────────────────────────────────────────

def _range_quality(range_size: float, atr: float) -> float:
    """
    Prefer an opening range roughly 1.5-3x ATR: tight enough that the stop
    isn't punishing, wide enough that the break means something. Falls off on
    both sides rather than rewarding ever-wider ranges.
    """
    if atr != atr or atr <= 0 or range_size <= 0:
        return 0.5
    ratio = range_size / atr
    if 1.5 <= ratio <= 3.0:
        return 1.0
    if ratio < 1.5:
        return _clip01(ratio / 1.5)
    return _clip01(1 - (ratio - 3.0) / 3.0)


def score_signal(
    session_df: pd.DataFrame,
    pos: int,
    direction: str,
    orange: Dict[str, Any],
    breach_distances: List[float],
) -> Dict[str, Any]:
    """
    0-100 confidence for a confirmed signal, from volume thrust, VWAP
    alignment, how decisively the confirming closes cleared the range, range
    quality vs ATR, and short-term trend agreement.

    Uses only bars up to and including the signal bar. Missing indicator
    columns fall back to a neutral 0.5 sub-score instead of penalizing.
    """
    atr = _col(session_df, "ATR", pos)
    close = _col(session_df, "Close", pos)
    vwap = _col(session_df, "VWAP", pos)
    volume = _col(session_df, "Volume", pos)
    vol_avg = _col(session_df, "vol_sma_20", pos)
    ema9 = _col(session_df, "EMA_9", pos)
    ema20 = _col(session_df, "EMA_20", pos)

    if atr == atr and atr > 0 and breach_distances:
        confirmation = _clip01(float(np.mean(breach_distances)) / atr / 0.75)
    else:
        confirmation = 0.5

    volume_thrust = _clip01((volume / vol_avg - 1.0) / 1.0) if (vol_avg == vol_avg and vol_avg > 0) else 0.5

    if vwap == vwap and atr == atr and atr > 0:
        signed = (close - vwap) if direction == "long" else (vwap - close)
        vwap_alignment = _clip01(signed / atr / 1.0)
    else:
        vwap_alignment = 0.5

    range_quality = _range_quality(orange["range_size"], atr)

    if ema9 == ema9 and ema20 == ema20:
        aligned = (ema9 > ema20) if direction == "long" else (ema9 < ema20)
        trend_alignment = 1.0 if aligned else 0.0
    else:
        trend_alignment = 0.5

    components = {
        "confirmation_strength": confirmation,
        "volume_thrust": volume_thrust,
        "vwap_alignment": vwap_alignment,
        "range_quality": range_quality,
        "trend_alignment": trend_alignment,
    }
    score = sum(SCORE_WEIGHTS[k] * v for k, v in components.items()) * 100
    return {
        "confidence_score": round(score, 1),
        "score_components": {k: round(v, 3) for k, v in components.items()},
    }


# ── Detection ────────────────────────────────────────────────────────────────

def detect_session_signals(
    session_df: pd.DataFrame,
    config: Optional[ORBCConfig] = None,
) -> Dict[str, Any]:
    """
    Run the ORBC state machine over a single session's bars.

    `session_df` must be one ET calendar day of intraday bars carrying the
    indicator columns from calculate_indicators() (ATR/VWAP/vol_sma_20/EMA),
    with its index already in market time — call to_market_tz() first.

    Returns a dict with the opening range, every signal that fired, and a
    `rejections` list recording candidate bars that reached the confirmation
    count but were turned away by a filter (useful for explaining "why didn't
    this fire?" in the UI).
    """
    config = config or ORBCConfig()

    result: Dict[str, Any] = {
        "opening_range": None,
        "signals": [],
        "rejections": [],
        "bars_scanned": 0,
    }

    orange = compute_opening_range(session_df, config.opening_range_minutes)
    if orange is None:
        result["skip_reason"] = "No bars inside the opening-range window for this session."
        return result
    result["opening_range"] = orange

    if orange["range_size"] <= 0:
        result["skip_reason"] = "Opening range has zero width — nothing to break out of."
        return result

    oh, ol = orange["opening_high"], orange["opening_low"]

    count = 0
    tracked_dir: Optional[str] = None
    breach_distances: List[float] = []

    for pos in range(len(session_df)):
        ts = session_df.index[pos]
        mins = _minutes_since_open(ts)

        # Bars inside the opening-range window define the range; they can't
        # break it. Anything before 9:30 (pre-market, if the feed includes it)
        # is likewise not part of the session's breakout scan.
        if mins < config.opening_range_minutes:
            continue
        if ts.time() > config.entry_cutoff:
            break

        result["bars_scanned"] += 1
        close = float(session_df["Close"].iloc[pos])

        if close > oh:
            direction = "long"
            breach = close - oh
        elif close < ol:
            direction = "short"
            breach = ol - close
        else:
            # Closed back inside the range — the episode is over, start fresh.
            count = 0
            tracked_dir = None
            breach_distances = []
            continue

        if direction != tracked_dir:
            tracked_dir = direction
            count = 1
            breach_distances = [breach]
        else:
            count += 1
            breach_distances.append(breach)

        if count < config.confirmation_closes or count > config.max_confirmation_closes:
            continue

        allowed = config.allow_long if direction == "long" else config.allow_short
        if not allowed:
            continue

        passed, failures, filter_detail = _check_filters(session_df, pos, direction, orange, config)
        if not passed:
            result["rejections"].append({
                "timestamp": ts,
                "direction": direction,
                "confirmation_count": count,
                "reasons": failures,
                "filters": filter_detail,
            })
            logger.debug(
                f"detect_session_signals: {ts} {direction} count={count} rejected by filters: {failures}"
            )
            continue

        entry = close
        atr = _col(session_df, "ATR", pos)
        stop, target = _stop_and_target(entry, direction, orange, atr, config)
        if stop is None or target is None:
            result["rejections"].append({
                "timestamp": ts,
                "direction": direction,
                "confirmation_count": count,
                "reasons": ["Could not compute a valid stop/target with the selected methods."],
                "filters": filter_detail,
            })
            continue

        sign = 1.0 if direction == "long" else -1.0
        risk = (entry - stop) * sign
        reward = (target - entry) * sign

        signal = {
            "timestamp": ts,
            "session_date": ts.normalize(),
            "direction": direction,
            "action": "BUY" if direction == "long" else "SELL",
            "confirmation_count": count,
            "entry_price": round(entry, 4),
            "stop_price": round(stop, 4),
            "target_price": round(target, 4),
            "risk": round(risk, 4),
            "reward": round(reward, 4),
            "rr_ratio": round(reward / risk, 2),
            "opening_high": round(oh, 4),
            "opening_low": round(ol, 4),
            "opening_range_size": round(orange["range_size"], 4),
            "bar_position": pos,
            "filters": filter_detail,
        }
        signal.update(score_signal(session_df, pos, direction, orange, breach_distances))
        result["signals"].append(signal)

        logger.info(
            f"detect_session_signals: {signal['action']} {ts} entry={signal['entry_price']} "
            f"stop={signal['stop_price']} target={signal['target_price']} "
            f"rr={signal['rr_ratio']} confidence={signal['confidence_score']}"
        )

        if config.one_signal_per_session:
            break

        # Signal taken; the episode is spent. Require a fresh break (a close
        # back inside the range, then a new run) before signalling again.
        count = 0
        tracked_dir = None
        breach_distances = []

    return result


def detect_orbc_signals(
    df: pd.DataFrame,
    config: Optional[ORBCConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run the state machine across every session in `df` and return a flat,
    chronologically-sorted list of signals. `df` needs the indicator columns
    from calculate_indicators(); its index is normalized to market time here.
    """
    config = config or ORBCConfig()
    if df is None or df.empty:
        return []

    market_df = to_market_tz(df)
    signals: List[Dict[str, Any]] = []
    for day in session_dates(market_df):
        session_df = market_df[market_df.index.normalize() == day]
        signals.extend(detect_session_signals(session_df, config)["signals"])
    signals.sort(key=lambda s: s["timestamp"])
    return signals


def latest_session_state(
    df: pd.DataFrame,
    config: Optional[ORBCConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate only the most recent session in `df` — what the live scanner
    needs. Returns the same shape as detect_session_signals() plus the
    session's bars and date, so the caller can chart the opening range and
    mark the breakout without re-slicing.
    """
    config = config or ORBCConfig()
    if df is None or df.empty:
        return {"opening_range": None, "signals": [], "rejections": [],
                "skip_reason": "No intraday data available.", "session_df": None, "session_date": None}

    market_df = to_market_tz(df)
    days = session_dates(market_df)
    if not days:
        return {"opening_range": None, "signals": [], "rejections": [],
                "skip_reason": "No sessions found in the data.", "session_df": None, "session_date": None}

    day = days[-1]
    session_df = market_df[market_df.index.normalize() == day]
    state = detect_session_signals(session_df, config)
    state["session_df"] = session_df
    state["session_date"] = day
    return state


# ── Trade simulation ─────────────────────────────────────────────────────────

def evaluate_orbc_trade(
    session_df: pd.DataFrame,
    signal: Dict[str, Any],
    config: Optional[ORBCConfig] = None,
    max_holding_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Direction-aware close-to-close simulation from the bar *after* the signal
    bar to the end of the session.

    Entry is the signal bar's close (the confirming close), and the walk
    starts at the next bar — so nothing from the signal bar's future is used
    to decide the entry, and no intrabar fills are assumed. Stop and target
    are both evaluated on closes, which understates whipsaw and overstates
    gap-throughs equally in both directions; it matches the convention in
    analysis/backtest.py and mtf_strategy.evaluate_setup_trade().

    With `exit_at_session_end` the trade is force-closed on the session's
    last bar and reported as exit_reason="session_end" — an intraday strategy
    should never carry risk overnight.
    """
    config = config or ORBCConfig()
    direction = signal["direction"]
    sign = 1.0 if direction == "long" else -1.0
    entry = signal["entry_price"]
    stop = signal["stop_price"]
    target = signal["target_price"]

    start = signal["bar_position"] + 1
    end = len(session_df) - 1
    if max_holding_bars is not None:
        end = min(end, signal["bar_position"] + max_holding_bars)

    result = dict(signal)
    exit_price: Optional[float] = None
    exit_reason = "open"
    exit_timestamp = None
    holding_period_bars = 0
    mfe = mae = 0.0

    for pos in range(start, end + 1):
        close = float(session_df["Close"].iloc[pos])
        excursion = (close - entry) * sign
        mfe = max(mfe, excursion)
        mae = min(mae, excursion)
        holding_period_bars = pos - signal["bar_position"]
        exit_timestamp = session_df.index[pos]

        if (close - target) * sign >= 0:
            exit_price, exit_reason = target, "target"
            break
        if (close - stop) * sign <= 0:
            exit_price, exit_reason = stop, "stop"
            break
    else:
        if end >= start:
            exit_price = float(session_df["Close"].iloc[end])
            exit_reason = "session_end" if config.exit_at_session_end else "time_exit"
            holding_period_bars = end - signal["bar_position"]
            exit_timestamp = session_df.index[end]

    return_pct = ((exit_price - entry) * sign / entry * 100) if exit_price is not None else None

    result.update({
        "exit_price": round(exit_price, 4) if exit_price is not None else None,
        "exit_reason": exit_reason,
        "exit_timestamp": exit_timestamp,
        "holding_period_bars": holding_period_bars,
        "mfe": round(mfe, 4),
        "mae": round(mae, 4),
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
        "win": bool(return_pct is not None and return_pct > 0),
    })
    return result


def backtest_orbc(
    ticker: str,
    config: Optional[ORBCConfig] = None,
    interval: str = "5m",
    period: str = "60d",
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Fetch intraday history for `ticker`, run ORBC session by session, and
    simulate every signal to its session-end exit.

    The data provider caps intraday history at roughly the trailing 60 days,
    and ORBC fires at most once or twice per session, so expect on the order
    of 40-60 trades — `num_sessions` is returned alongside `num_trades` so the
    caller can surface the sample size rather than presenting a win rate as
    if it were statistically settled.

    Pass `df` to backtest an already-fetched/indicator-enriched frame (used by
    the tests to run without network access).
    """
    config = config or ORBCConfig()

    if df is None:
        from data.price_data import get_price_history
        from analysis.indicators import calculate_indicators

        raw = get_price_history(ticker, period=period, interval=interval)
        if raw is None or raw.empty:
            logger.warning(f"backtest_orbc: no {interval} history available for {ticker}")
            return {"error": f"Not enough intraday history available for {ticker}."}
        df = calculate_indicators(raw)

    market_df = to_market_tz(df)
    days = session_dates(market_df)
    if not days:
        return {"error": f"No complete sessions found in the {interval} history for {ticker}."}

    trades: List[Dict[str, Any]] = []
    sessions_with_range = 0

    for day in days:
        session_df = market_df[market_df.index.normalize() == day]
        state = detect_session_signals(session_df, config)
        if state["opening_range"] is not None:
            sessions_with_range += 1
        for signal in state["signals"]:
            trades.append(evaluate_orbc_trade(session_df, signal, config))

    closed = [t for t in trades if t.get("return_pct") is not None]
    wins = [t for t in closed if t["return_pct"] > 0]
    win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
    avg_rr = round(float(np.mean([t["rr_ratio"] for t in trades])), 2) if trades else 0.0
    avg_return = round(float(np.mean([t["return_pct"] for t in closed])), 3) if closed else 0.0

    equity = [1.0]
    equity_dates = [days[0]]
    for t in sorted(closed, key=lambda x: x["timestamp"]):
        equity.append(equity[-1] * (1 + t["return_pct"] / 100))
        equity_dates.append(t["exit_timestamp"] or t["timestamp"])
    total_return_pct = round((equity[-1] - 1) * 100, 2)

    by_direction = {}
    for d in ("long", "short"):
        subset = [t for t in closed if t["direction"] == d]
        if subset:
            d_wins = [t for t in subset if t["return_pct"] > 0]
            by_direction[d] = {
                "trades": len(subset),
                "win_rate": round(len(d_wins) / len(subset) * 100, 1),
                "avg_return_pct": round(float(np.mean([t["return_pct"] for t in subset])), 3),
            }

    exit_breakdown: Dict[str, int] = {}
    for t in trades:
        exit_breakdown[t["exit_reason"]] = exit_breakdown.get(t["exit_reason"], 0) + 1

    logger.info(
        f"backtest_orbc: {ticker} complete — sessions={len(days)} trades={len(trades)} "
        f"win_rate={win_rate}% avg_rr={avg_rr} total_return_pct={total_return_pct}%"
    )

    return {
        "trades": trades,
        "num_trades": len(trades),
        "num_sessions": len(days),
        "sessions_with_range": sessions_with_range,
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "avg_return_pct": avg_return,
        "total_return_pct": total_return_pct,
        "equity_curve": pd.Series(equity, index=pd.DatetimeIndex(equity_dates), name="equity"),
        "by_direction": by_direction,
        "exit_breakdown": exit_breakdown,
        "window_start": market_df.index.min(),
        "window_end": market_df.index.max(),
        "interval": interval,
        "config": asdict(config),
    }
