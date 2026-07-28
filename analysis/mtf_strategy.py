"""
Multi-timeframe setup detector — 4H trend, 30-min pullback into a demand
zone, 5-min market-structure shift, and a price/volume "tape proxy" for the
absorption-then-buyers-in-control read. Modeled on the same category of
rule as analysis/patterns.py and analysis/trendlines.py: deterministic
geometry over OHLCV, no ML, no order-book data.

Tape reading proper (sellers absorbed, buyers holding above passive offers)
needs Level 2 / tick data this app doesn't have (yfinance is bars only).
tape_proxy_5m() is an explicit stand-in: a down-leg that stalls on shrinking
range and below-average volume (absorption), followed by a volume-backed
reversal bar that reclaims VWAP/short structure (buyers-in-control proxy).
It is not literal order-flow and is labeled as a proxy everywhere it's
surfaced.

The VAP entry target is likewise a proxy: analysis/volume_profile.py bins
bar volume by price (no true depth data), and the nearest high-volume node
above entry stands in for a real Volume-At-Price/Value-Area target.
"""
import logging
from typing import Any, Dict, Optional

import pandas as pd

from analysis.indicators import get_trend, detect_support_resistance
from analysis.trendlines import detect_recent_trendlines, detect_swing_points
from analysis.patterns import detect_candlestick_pattern
from analysis.volume_profile import build_volume_profile, nearest_hvn_above

logger = logging.getLogger(__name__)

MIN_BARS_4H = 20
MIN_BARS_30M = 30
MIN_BARS_5M = 20
ABSORPTION_LOOKBACK = 5


def resample_to_4h(df_1h: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Resample 1h OHLCV bars into 4h buckets (yfinance has no native 4h
    interval). Buckets are anchored to local midnight, not the market open,
    so they won't line up exactly with a broker's 4H chart — a reasonable
    approximation, not an exact match.
    """
    if df_1h is None or df_1h.empty:
        logger.warning("resample_to_4h: no 1h input data to resample.")
        return None
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df_1h.resample("4h", origin="start_day").agg(agg)
    out = out.dropna(subset=["Open", "Close"])
    out = out[out["Volume"] > 0]
    if out.empty:
        logger.warning(f"resample_to_4h: resampled {len(df_1h)} 1h bars into 0 usable 4h bars.")
        return None
    logger.debug(f"resample_to_4h: {len(df_1h)} 1h bars -> {len(out)} 4h bars.")
    return out


def _stage(ok: bool, reason: str, **extra: Any) -> Dict[str, Any]:
    return {"ok": bool(ok), "reason": reason, **extra}


def _trend_4h(df_4h: pd.DataFrame) -> Dict[str, Any]:
    if df_4h is None or len(df_4h) < MIN_BARS_4H or "above_50ma" not in df_4h.columns:
        return _stage(False, "Not enough 4H history to read a trend yet.")

    trend = get_trend(df_4h, window=20)
    above_50 = bool(df_4h["above_50ma"].iloc[-1])
    ok = trend == "uptrend" and above_50
    if ok:
        reason = "4H trend is up and price is holding above the 50-period average."
    elif trend != "uptrend":
        reason = f"4H trend reads '{trend}', not an uptrend — skip."
    else:
        reason = "4H trend is up but price has slipped below the 50-period average."
    logger.debug(f"_trend_4h: ok={ok} trend={trend} reason={reason}")
    return _stage(ok, reason, trend=trend)


def _pullback_30m(df_30m: pd.DataFrame) -> Dict[str, Any]:
    if df_30m is None or len(df_30m) < MIN_BARS_30M:
        return _stage(False, "Not enough 30-min history to locate a demand zone yet.")

    sr = detect_support_resistance(df_30m)
    trendlines = detect_recent_trendlines(df_30m)
    support_price = None
    if trendlines is not None:
        support_price = trendlines["projected_support"]
    elif sr and sr.get("support"):
        support_price = sr["support"][-1]

    if support_price is None:
        return _stage(False, "No clear support level found on the 30-min chart.")

    atr = float(df_30m["ATR"].iloc[-1]) if "ATR" in df_30m.columns and pd.notna(df_30m["ATR"].iloc[-1]) else support_price * 0.005
    zone_low = support_price - 0.5 * atr
    zone_high = support_price + 0.5 * atr

    recent_high = float(df_30m["High"].tail(40).max())
    had_runup = recent_high > support_price * 1.02

    latest_close = float(df_30m["Close"].iloc[-1])
    in_zone = zone_low <= latest_close <= zone_high

    ok = in_zone and had_runup
    if not had_runup:
        reason = "Price hasn't rallied meaningfully before this pullback — doesn't look like a demand zone yet."
    elif in_zone:
        reason = f"Price has pulled back into the {zone_low:.2f}-{zone_high:.2f} demand zone after an up-leg."
    else:
        reason = f"Price is not yet inside the {zone_low:.2f}-{zone_high:.2f} demand zone."

    logger.debug(f"_pullback_30m: ok={ok} zone=({zone_low:.2f}, {zone_high:.2f}) reason={reason}")
    return _stage(ok, reason, zone_low=zone_low, zone_high=zone_high)


def _structure_shift_5m(df_5m: pd.DataFrame) -> Dict[str, Any]:
    if df_5m is None or len(df_5m) < MIN_BARS_5M:
        return _stage(False, "Not enough 5-min history to read market structure yet.")

    swings = detect_swing_points(df_5m)
    lows = [s for s in swings if s["type"] == "low"]
    if not lows:
        return _stage(False, "No confirmed swing low yet on the 5-min chart.")

    swing_low = lows[-1]
    prior_highs = [s for s in swings if s["type"] == "high" and s["index"] < swing_low["index"]]
    if not prior_highs:
        return _stage(False, "No prior swing high to break for a structure shift.")

    prior_high = prior_highs[-1]
    latest_close = float(df_5m["Close"].iloc[-1])
    ok = latest_close > prior_high["price"]

    if ok:
        reason = f"Price broke back above the prior swing high ({prior_high['price']:.2f}) — bullish structure shift."
    else:
        reason = f"Price hasn't reclaimed the prior swing high ({prior_high['price']:.2f}) yet."

    logger.debug(f"_structure_shift_5m: ok={ok} reason={reason}")
    return _stage(ok, reason, swing_low=float(swing_low["price"]), swing_high=float(prior_high["price"]))


def tape_proxy_5m(df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Price/volume stand-in for tape reading: a down-leg that stalls
    (shrinking range + below-average volume = absorption), then a bar with
    above-average volume that closes back above VWAP or short structure,
    or a bullish-engulfing candle (buyers-in-control proxy).
    """
    if df_5m is None or len(df_5m) < ABSORPTION_LOOKBACK + 2 or "vol_ratio" not in df_5m.columns:
        return _stage(False, "Not enough 5-min bars to judge absorption/exhaustion yet.")

    absorption_window = df_5m.iloc[-(ABSORPTION_LOOKBACK + 1):-1]
    ranges = (absorption_window["High"] - absorption_window["Low"]).to_numpy()
    shrinking = bool((ranges[1:] <= ranges[:-1] * 1.05).all()) if len(ranges) >= 2 else False
    below_avg_volume = bool(absorption_window["vol_ratio"].mean() < 1.0)
    absorption_ok = shrinking and below_avg_volume

    latest = df_5m.iloc[-1]
    vwap = float(latest["VWAP"]) if "VWAP" in df_5m.columns and pd.notna(latest["VWAP"]) else None
    reversal_volume_ok = bool(latest.get("vol_ratio", 0) > 1.2) and (vwap is None or float(latest["Close"]) > vwap)

    pattern = detect_candlestick_pattern(df_5m)
    bullish_pattern = pattern is not None and pattern.get("direction") == "bull"

    ok = absorption_ok and (reversal_volume_ok or bullish_pattern)

    if not absorption_ok:
        reason = "No sign of selling stalling out (range/volume aren't contracting) yet."
    elif not (reversal_volume_ok or bullish_pattern):
        reason = "Selling looks exhausted but buyers haven't confirmed with volume yet."
    else:
        trigger = pattern["name"] if bullish_pattern else "a volume-backed reversal bar above VWAP"
        reason = f"Selling absorption followed by {trigger} — buyers appear in control (proxy read, not real tape)."

    logger.debug(f"tape_proxy_5m: ok={ok} reason={reason}")
    return _stage(ok, reason)


def evaluate_setup(df_4h: pd.DataFrame, df_30m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the full checklist against already-indicator-enriched DataFrames
    (i.e. each has already been through analysis.indicators.calculate_indicators).
    Returns the four stage dicts plus setup_valid and, only when valid,
    entry_price/target_price/stop_price/risk/reward/rr_ratio.
    """
    trend = _trend_4h(df_4h)
    pullback = _pullback_30m(df_30m)
    structure = _structure_shift_5m(df_5m)
    tape = tape_proxy_5m(df_5m)

    result: Dict[str, Any] = {
        "trend_4h": trend,
        "pullback_30m": pullback,
        "structure_shift_5m": structure,
        "tape_proxy_5m": tape,
        "setup_valid": False,
    }

    if not (trend["ok"] and pullback["ok"] and structure["ok"] and tape["ok"]):
        return result

    entry_price = float(df_5m["Close"].iloc[-1])
    stop_price = structure["swing_low"]

    profile = build_volume_profile(df_30m)
    target_price = nearest_hvn_above(profile, entry_price)
    target_is_fallback = target_price is None
    if target_is_fallback:
        target_price = float(df_5m["VWAP"].iloc[-1]) if "VWAP" in df_5m.columns and pd.notna(df_5m["VWAP"].iloc[-1]) else entry_price

    risk = entry_price - stop_price
    reward = target_price - entry_price

    if risk <= 0 or reward <= 0:
        result["setup_valid"] = False
        result["invalid_reason"] = "Computed stop/target don't form a valid risk/reward — skip this signal."
        logger.warning(
            f"evaluate_setup: invalid risk/reward — entry={entry_price:.4f} stop={stop_price:.4f} "
            f"target={target_price:.4f} risk={risk:.4f} reward={reward:.4f}"
        )
        return result

    result.update({
        "setup_valid": True,
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "target_is_vwap_fallback": target_is_fallback,
        "risk": round(risk, 4),
        "reward": round(reward, 4),
        "rr_ratio": round(reward / risk, 2),
    })
    logger.info(
        f"evaluate_setup: setup_valid entry={result['entry_price']} target={result['target_price']} "
        f"stop={result['stop_price']} rr_ratio={result['rr_ratio']}"
    )
    return result


def evaluate_setup_trade(df_5m_from_entry: pd.DataFrame, setup: Dict[str, Any], max_holding_bars: int = 60) -> Dict[str, Any]:
    """
    Walk-forward close-to-close simulation from the entry bar (index 0 of
    `df_5m_from_entry`) using the already-computed entry/stop/target from
    `evaluate_setup()`. Mirrors flag_pennant_backtest.evaluate_pattern_trade's
    loop structure and close-to-close convention (no intrabar fills).
    """
    entry_price = setup["entry_price"]
    stop_price = setup["stop_price"]
    target_price = setup["target_price"]

    result = dict(setup)
    n = len(df_5m_from_entry)
    lookahead_end = min(n - 1, max_holding_bars)

    exit_price = None
    exit_reason = "open"
    holding_period_bars = 0
    mfe = mae = 0.0

    for i in range(1, lookahead_end + 1):
        close = float(df_5m_from_entry["Close"].iloc[i])
        excursion = close - entry_price
        mfe = max(mfe, excursion)
        mae = min(mae, excursion)
        holding_period_bars = i

        if close >= target_price:
            exit_price, exit_reason = target_price, "target"
            break
        if close <= stop_price:
            exit_price, exit_reason = stop_price, "stop"
            break
    else:
        if lookahead_end > 0:
            exit_price = float(df_5m_from_entry["Close"].iloc[lookahead_end])
            exit_reason = "time_exit"
            holding_period_bars = lookahead_end

    return_pct = ((exit_price - entry_price) / entry_price * 100) if exit_price is not None else None

    result.update({
        "exit_price": round(exit_price, 4) if exit_price is not None else None,
        "exit_reason": exit_reason,
        "holding_period_bars": holding_period_bars,
        "mfe": round(mfe, 4),
        "mae": round(mae, 4),
        "return_pct": round(return_pct, 2) if return_pct is not None else None,
    })
    return result


def backtest_setup(ticker: str, max_holding_bars: int = 60) -> Dict[str, Any]:
    """
    Fetches 5m/30m/1h history for `ticker`, resamples 1h to 4h, and walks the
    5-min series bar by bar. At each bar, evaluates the setup using only data
    available as of that bar (no lookahead) across all three timeframes; on
    a valid signal, simulates the trade forward and skips past its holding
    period before resuming the scan. One trade at a time, same convention as
    analysis/backtest.py.

    yfinance caps intraday history to roughly the trailing 60 days for
    5m/30m bars, so the achievable backtest window is bounded by whichever
    of the three timeframes has the least history — `window_start`/
    `window_end` in the result reflect what was actually fetched, not a
    requested range, so the caller can display the real coverage instead of
    silently truncating it.
    """
    from data.price_data import get_price_history
    from analysis.indicators import calculate_indicators

    logger.info(f"backtest_setup: starting backtest for {ticker} (max_holding_bars={max_holding_bars}).")

    df_5m = get_price_history(ticker, period="60d", interval="5m")
    df_30m = get_price_history(ticker, period="60d", interval="30m")
    df_1h = get_price_history(ticker, period="60d", interval="1h")

    if df_5m is None or df_30m is None or df_1h is None or df_5m.empty or df_30m.empty or df_1h.empty:
        logger.warning(
            f"backtest_setup: insufficient intraday history for {ticker} — "
            f"5m={'missing' if df_5m is None else len(df_5m)} "
            f"30m={'missing' if df_30m is None else len(df_30m)} "
            f"1h={'missing' if df_1h is None else len(df_1h)}"
        )
        return {"error": f"Not enough intraday history available for {ticker}."}

    logger.debug(
        f"backtest_setup: fetched {ticker} data — 5m={df_5m.shape} 30m={df_30m.shape} 1h={df_1h.shape}"
    )

    df_5m = calculate_indicators(df_5m)
    df_30m = calculate_indicators(df_30m)
    df_4h = resample_to_4h(df_1h)
    if df_4h is None or len(df_4h) < MIN_BARS_4H:
        logger.warning(f"backtest_setup: insufficient resampled 4H history for {ticker}.")
        return {"error": f"Not enough 4H (resampled) history available for {ticker}."}
    df_4h = calculate_indicators(df_4h)

    window_start = max(df_5m.index.min(), df_30m.index.min(), df_4h.index.min())
    window_end = min(df_5m.index.max(), df_30m.index.max(), df_4h.index.max())
    logger.info(f"backtest_setup: {ticker} achieved window {window_start} to {window_end}.")

    trades = []
    equity = [1.0]
    equity_dates = [window_start]

    warmup = max(MIN_BARS_5M, ABSORPTION_LOOKBACK + 2)
    i = warmup
    n = len(df_5m)

    while i < n:
        ts = df_5m.index[i]
        if ts < window_start:
            i += 1
            continue

        # Bars are labeled at their *start*, so a bucket only becomes fully
        # known once its own duration has elapsed — filtering on start <= ts
        # alone would let the still-forming current bucket (aggregated over
        # its whole span) leak up to 4h/30m of future price action into ts.
        d4 = df_4h[df_4h.index <= ts - pd.Timedelta(hours=4)]
        d30 = df_30m[df_30m.index <= ts - pd.Timedelta(minutes=30)]
        d5 = df_5m.iloc[: i + 1]

        if len(d4) < MIN_BARS_4H or len(d30) < MIN_BARS_30M or len(d5) < warmup:
            i += 1
            continue

        setup = evaluate_setup(d4, d30, d5)
        if setup.get("setup_valid"):
            trade = evaluate_setup_trade(df_5m.iloc[i:], setup, max_holding_bars=max_holding_bars)
            trade["entry_date"] = ts
            trades.append(trade)

            if trade["return_pct"] is not None:
                equity.append(equity[-1] * (1 + trade["return_pct"] / 100))
                equity_dates.append(df_5m.index[min(i + trade["holding_period_bars"], n - 1)])

            if len(trades) % 10 == 0:
                logger.debug(f"backtest_setup: {ticker} scan progress — {len(trades)} trades found, at bar {i}/{n}.")

            i += max(trade.get("holding_period_bars", 1), 1) + 1
        else:
            i += 1

    wins = sum(1 for t in trades if t.get("return_pct") is not None and t["return_pct"] > 0)
    closed = [t for t in trades if t.get("return_pct") is not None]
    win_rate = round(wins / len(closed) * 100, 1) if closed else 0.0
    avg_rr = round(sum(t["rr_ratio"] for t in trades) / len(trades), 2) if trades else 0.0
    total_return_pct = round((equity[-1] - 1) * 100, 1)

    logger.info(
        f"backtest_setup: {ticker} complete — num_trades={len(trades)} win_rate={win_rate}% "
        f"avg_rr={avg_rr} total_return_pct={total_return_pct}%"
    )

    return {
        "trades": trades,
        "equity_curve": pd.Series(equity, index=pd.DatetimeIndex(equity_dates), name="equity"),
        "num_trades": len(trades),
        "wins": wins,
        "losses": len(closed) - wins,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "total_return_pct": total_return_pct,
        "window_start": window_start,
        "window_end": window_end,
    }
