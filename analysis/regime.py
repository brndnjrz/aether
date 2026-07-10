"""
Market regime detection — combines price action, volatility, and breadth signals.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from data.price_data import get_price_history
from data.macro_data import get_vix_data, get_sp500_regime

logger = logging.getLogger(__name__)


def detect_regime(df: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
    """
    Full market regime analysis for a single stock.
    Returns: regime label, trend, volatility regime, momentum, and raw values.
    """
    if df is None or df.empty or len(df) < 20:
        return {"regime": "Insufficient Data", "trend": "unknown"}

    last = df.iloc[-1]

    # ── Trend ────────────────────────────────────────────────────────────
    close = df["Close"]
    above_200 = bool(close.iloc[-1] > df["SMA_200"].iloc[-1]) if "SMA_200" in df.columns else None
    above_50 = bool(close.iloc[-1] > df["SMA_50"].iloc[-1]) if "SMA_50" in df.columns else None

    # 5-day and 21-day price change
    price_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) > 5 else 0
    price_21d = (close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100 if len(close) > 21 else 0
    price_63d = (close.iloc[-1] - close.iloc[-64]) / close.iloc[-64] * 100 if len(close) > 63 else 0

    # 200 MA slope
    if "SMA_200" in df.columns:
        ma200_slope = (df["SMA_200"].iloc[-1] - df["SMA_200"].iloc[-20]) / df["SMA_200"].iloc[-20] * 100 if len(df) > 20 else 0
    else:
        ma200_slope = 0

    # ── Volatility ───────────────────────────────────────────────────────
    hv_21 = last.get("hv_21") if "hv_21" in df.columns else None
    if hv_21 is None and "returns" in df.columns:
        hv_21 = float(df["returns"].rolling(21).std().iloc[-1] * np.sqrt(252) * 100)
    hv_series = df["hv_21"].dropna() if "hv_21" in df.columns else pd.Series()
    vol_pct = float(hv_series.rank(pct=True).iloc[-1] * 100) if len(hv_series) > 5 else 50
    vol_regime = "High" if vol_pct > 70 else ("Low" if vol_pct < 30 else "Normal")

    # ── Momentum ─────────────────────────────────────────────────────────
    rsi = float(last.get("RSI", 50)) if "RSI" in df.columns else 50
    adx = float(last.get("ADX", 20)) if "ADX" in df.columns else 20
    macd = float(last.get("MACD", 0)) if "MACD" in df.columns else 0
    macd_sig = float(last.get("MACD_signal", 0)) if "MACD_signal" in df.columns else 0

    # ── Composite regime label ────────────────────────────────────────────
    if above_200 and price_21d > 3 and adx > 20:
        trend = "Uptrend"
    elif not above_200 and price_21d < -3 and adx > 20:
        trend = "Downtrend"
    elif adx < 15 or abs(price_21d) < 2:
        trend = "Sideways"
    else:
        trend = "Choppy"

    if trend == "Uptrend" and vol_regime == "Low":
        regime = "Bullish Trend (Low Vol)"
    elif trend == "Uptrend" and vol_regime == "High":
        regime = "Bullish Trend (High Vol)"
    elif trend == "Downtrend" and vol_regime == "High":
        regime = "Bearish Trend (High Vol)"
    elif trend == "Downtrend":
        regime = "Bearish Trend"
    elif trend == "Sideways" and vol_regime == "Low":
        regime = "Range-Bound (Low Vol)"
    elif trend == "Sideways" and vol_regime == "High":
        regime = "Range-Bound (High Vol)"
    else:
        regime = "Choppy / Mixed"

    # ── Market context ───────────────────────────────────────────────────
    market_regime = get_sp500_regime()

    return {
        "regime": regime,
        "trend": trend,
        "vol_regime": vol_regime,
        "above_200ma": above_200,
        "above_50ma": above_50,
        "price_5d_pct": round(price_5d, 2),
        "price_21d_pct": round(price_21d, 2),
        "price_63d_pct": round(price_63d, 2),
        "ma200_slope": round(ma200_slope, 3),
        "vol_percentile": round(vol_pct, 1),
        "hv_21": round(float(hv_21), 2) if hv_21 is not None else None,
        "rsi": round(rsi, 1),
        "adx": round(adx, 1),
        "macd_bullish": macd > macd_sig,
        "market_regime": market_regime.get("regime", "Unknown"),
        "vix": market_regime.get("vix", 20),
    }
