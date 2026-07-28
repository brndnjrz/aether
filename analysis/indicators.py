"""
Technical indicators — ported and cleaned from ML-technical-analysis.
All calculations are self-contained using pandas/numpy (no pandas-ta dependency).
"""
import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def _is_intraday(index: pd.DatetimeIndex) -> bool:
    """True if bars are spaced by hours/minutes rather than calendar days."""
    if len(index) < 2:
        return False
    return (index[1] - index[0]) < pd.Timedelta(hours=20)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a price DataFrame.
    Input: OHLCV DataFrame from yfinance (columns: Open, High, Low, Close, Volume)
    Output: Same DataFrame with indicator columns appended.
    """
    if df is None or df.empty or len(df) < 14:
        logger.warning(f"calculate_indicators: insufficient input — {0 if df is None else len(df)} rows (need 14).")
        return df

    out = df.copy()

    # ── Returns & Historical Volatility ─────────────────────────────────
    out["returns"] = out["Close"].pct_change()
    out["hv_21"] = out["returns"].rolling(21).std() * np.sqrt(252) * 100
    out["hv_10"] = out["returns"].rolling(10).std() * np.sqrt(252) * 100
    out["hv_63"] = out["returns"].rolling(63).std() * np.sqrt(252) * 100

    # ── Moving Averages ──────────────────────────────────────────────────
    for w in [20, 50, 100, 200]:
        out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
    for span in [9, 20, 50]:
        out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean()

    # ── VWAP (session-anchored for intraday bars, cumulative for daily bars) ─
    if _is_intraday(out.index):
        session = out.index.normalize()
        pv = out["Close"] * out["Volume"]
        out["VWAP"] = pv.groupby(session).cumsum() / out["Volume"].groupby(session).cumsum()
    else:
        out["VWAP"] = (out["Close"] * out["Volume"]).cumsum() / out["Volume"].cumsum()

    # ── Bollinger Bands ──────────────────────────────────────────────────
    roll = out["Close"].rolling(20)
    out["BB_mid"] = roll.mean()
    out["BB_std"] = roll.std()
    out["BB_upper"] = out["BB_mid"] + 2 * out["BB_std"]
    out["BB_lower"] = out["BB_mid"] - 2 * out["BB_std"]
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / out["BB_mid"]
    out["BB_pct"] = (out["Close"] - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"])

    # ── RSI (14) ─────────────────────────────────────────────────────────
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # RSI divergence helper (previous high/low)
    out["RSI_prev_high"] = out["RSI"].rolling(14).max().shift(14)
    out["price_prev_high"] = out["Close"].rolling(14).max().shift(14)

    # ── Stochastic (14,3,3) ──────────────────────────────────────────────
    low14 = out["Low"].rolling(14).min()
    high14 = out["High"].rolling(14).max()
    out["STOCH_K"] = 100 * (out["Close"] - low14) / (high14 - low14).replace(0, np.nan)
    out["STOCH_D"] = out["STOCH_K"].rolling(3).mean()

    # ── MACD (12, 26, 9) ────────────────────────────────────────────────
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # ── ATR (14) ─────────────────────────────────────────────────────────
    hl = out["High"] - out["Low"]
    hc = (out["High"] - out["Close"].shift()).abs()
    lc = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    out["ATR_pct"] = out["ATR"] / out["Close"] * 100

    # ── ADX (14) ─────────────────────────────────────────────────────────
    try:
        high_diff = out["High"].diff()
        low_diff = out["Low"].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

        plus_dm_s = pd.Series(plus_dm, index=out.index).rolling(14).mean()
        minus_dm_s = pd.Series(minus_dm, index=out.index).rolling(14).mean()
        tr14 = tr.rolling(14).mean()

        plus_di = 100 * (plus_dm_s / tr14)
        minus_di = 100 * (minus_dm_s / tr14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        out["ADX"] = dx.rolling(14).mean()
        out["PLUS_DI"] = plus_di
        out["MINUS_DI"] = minus_di
    except Exception as e:
        logger.debug(f"ADX calculation: {e}")

    # ── OBV ──────────────────────────────────────────────────────────────
    direction = np.sign(out["Close"].diff().fillna(0))
    out["OBV"] = (out["Volume"] * direction).cumsum()

    # ── Support / Resistance (pivot-based, last 3 levels) ────────────────
    out["pivot"] = (out["High"] + out["Low"] + out["Close"]) / 3
    out["R1"] = 2 * out["pivot"] - out["Low"]
    out["S1"] = 2 * out["pivot"] - out["High"]
    out["R2"] = out["pivot"] + (out["High"] - out["Low"])
    out["S2"] = out["pivot"] - (out["High"] - out["Low"])

    # ── Volume analysis ──────────────────────────────────────────────────
    out["vol_sma_20"] = out["Volume"].rolling(20).mean()
    out["vol_ratio"] = out["Volume"] / out["vol_sma_20"]

    # ── Trend direction helper ────────────────────────────────────────────
    out["above_200ma"] = out["Close"] > out["SMA_200"]
    out["above_50ma"] = out["Close"] > out["SMA_50"]

    logger.debug(f"calculate_indicators: added indicators — {len(out)} rows, {len(out.columns) - len(df.columns)} new columns")

    return out


def get_trend(df: pd.DataFrame, window: int = 20) -> str:
    """Simple linear regression slope trend detection."""
    if df is None or len(df) < window:
        return "sideways"
    closes = df["Close"].tail(window)
    slope = np.polyfit(range(len(closes)), closes, 1)[0]
    normalized = slope / closes.mean()
    if normalized > 0.001:
        return "uptrend"
    elif normalized < -0.001:
        return "downtrend"
    return "sideways"


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> Optional[str]:
    """
    Detect bearish/bullish RSI divergence:
    - Bearish: price new high, RSI lower high
    - Bullish: price new low, RSI higher low
    """
    if "RSI" not in df.columns or len(df) < lookback * 2:
        return None
    try:
        recent = df.tail(lookback)
        prev = df.iloc[-(lookback * 2):-lookback]
        price_higher = recent["Close"].max() > prev["Close"].max()
        rsi_lower = recent["RSI"].max() < prev["RSI"].max()
        if price_higher and rsi_lower:
            logger.debug("detect_rsi_divergence: found bearish_divergence")
            return "bearish_divergence"
        price_lower = recent["Close"].min() < prev["Close"].min()
        rsi_higher = recent["RSI"].min() > prev["RSI"].min()
        if price_lower and rsi_higher:
            logger.debug("detect_rsi_divergence: found bullish_divergence")
            return "bullish_divergence"
    except Exception as e:
        logger.debug(f"detect_rsi_divergence: {e}")
    return None


def detect_support_resistance(df: pd.DataFrame, window: int = 20, tolerance: float = 0.015) -> dict:
    """Local minima/maxima support & resistance detection."""
    supports, resistances = [], []
    if df is None or len(df) < window * 2:
        logger.debug(f"detect_support_resistance: insufficient bars — {0 if df is None else len(df)} (need {window * 2}).")
        return {"support": [], "resistance": []}

    for i in range(window, len(df) - window):
        low_window = df["Low"].iloc[i - window: i + window]
        if df["Low"].iloc[i] == low_window.min():
            level = float(df["Low"].iloc[i])
            if not any(abs(level - s) / (s + 1e-9) < tolerance for s in supports):
                supports.append(level)

        high_window = df["High"].iloc[i - window: i + window]
        if df["High"].iloc[i] == high_window.max():
            level = float(df["High"].iloc[i])
            if not any(abs(level - r) / (r + 1e-9) < tolerance for r in resistances):
                resistances.append(level)

    return {
        "support": sorted(supports)[-3:],
        "resistance": sorted(resistances)[-3:],
    }


def get_signal_summary(df: pd.DataFrame) -> dict:
    """
    Return a concise signal dict for the latest row.
    Used by the AI brief and the scorecard.
    """
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    signals = {}

    # Trend
    signals["above_200ma"] = bool(last.get("above_200ma", False))
    signals["above_50ma"] = bool(last.get("above_50ma", False))
    signals["trend"] = get_trend(df)

    # RSI
    rsi = last.get("RSI")
    if rsi is not None:
        signals["rsi"] = round(float(rsi), 1)
        signals["rsi_zone"] = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")

    # MACD
    macd = last.get("MACD")
    macd_sig = last.get("MACD_signal")
    if macd is not None and macd_sig is not None:
        signals["macd_bullish"] = float(macd) > float(macd_sig)

    # Volume
    vol_ratio = last.get("vol_ratio")
    if vol_ratio is not None:
        signals["volume_surge"] = float(vol_ratio) > 1.5

    # Bollinger
    bb_pct = last.get("BB_pct")
    if bb_pct is not None:
        signals["bb_pct"] = round(float(bb_pct), 3)

    # ADX
    adx = last.get("ADX")
    if adx is not None:
        signals["adx"] = round(float(adx), 1)
        signals["strong_trend"] = float(adx) > 25

    # ATR %
    atr_pct = last.get("ATR_pct")
    if atr_pct is not None:
        signals["atr_pct"] = round(float(atr_pct), 2)

    # Divergence
    div = detect_rsi_divergence(df)
    if div:
        signals["rsi_divergence"] = div

    return signals
