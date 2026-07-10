"""
Market macro data — VIX, S&P 500 breadth, yield curve, market indices.
Used for regime detection and macro context overlay.
"""
import time
import logging
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
from data.price_data import get_price_history, get_ticker_info

logger = logging.getLogger(__name__)
_cache: Dict[str, Dict] = {}


def _fresh(entry: dict, ttl: int) -> bool:
    return (time.time() - entry["ts"]) < ttl


def get_vix_data(ttl: int = 300) -> Dict[str, Any]:
    key = "vix_data"
    if key in _cache and _fresh(_cache[key], ttl):
        return _cache[key]["data"]
    try:
        df = get_price_history("^VIX", period="1y", interval="1d")
        if df is None or df.empty:
            return {"current": 20.0, "status": "unavailable"}

        current = float(df["Close"].iloc[-1])
        week_ago = float(df["Close"].iloc[-6]) if len(df) > 5 else current
        month_ago = float(df["Close"].iloc[-22]) if len(df) > 21 else current
        year_high = float(df["Close"].max())
        year_low = float(df["Close"].min())

        regime = "Crisis" if current > 35 else ("Elevated Fear" if current > 25 else ("Normal" if current > 15 else "Complacency"))

        result = {
            "current": round(current, 2),
            "week_ago": round(week_ago, 2),
            "month_ago": round(month_ago, 2),
            "year_high": round(year_high, 2),
            "year_low": round(year_low, 2),
            "regime": regime,
            "trend": "Rising" if current > week_ago * 1.05 else ("Falling" if current < week_ago * 0.95 else "Stable"),
            "history": df["Close"].tail(63).tolist(),
        }
        _cache[key] = {"data": result, "ts": time.time()}
        return result
    except Exception as e:
        logger.error(f"VIX data error: {e}")
        return {"current": 20.0, "status": "error"}


def get_market_overview(ttl: int = 300) -> Dict[str, Any]:
    """Returns current prices and % changes for major indices."""
    key = "market_overview"
    if key in _cache and _fresh(_cache[key], ttl):
        return _cache[key]["data"]

    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
    }
    result = {}
    for name, ticker in indices.items():
        try:
            df = get_price_history(ticker, period="5d", interval="1d")
            if df is not None and len(df) >= 2:
                curr = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                result[name] = {
                    "price": round(curr, 2),
                    "change": round(curr - prev, 2),
                    "change_pct": round((curr - prev) / prev * 100, 2),
                }
        except Exception:
            continue
    _cache[key] = {"data": result, "ts": time.time()}
    return result


def get_sp500_regime(ttl: int = 600) -> Dict[str, Any]:
    """
    Detect market regime using S&P 500:
    - Price vs 200-day MA
    - 50-day MA slope
    - VIX level
    """
    key = "sp500_regime"
    if key in _cache and _fresh(_cache[key], ttl):
        return _cache[key]["data"]

    try:
        df = get_price_history("^GSPC", period="1y", interval="1d")
        if df is None or len(df) < 50:
            return {"regime": "Unknown"}

        price = float(df["Close"].iloc[-1])
        ma200 = float(df["Close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else float(df["Close"].rolling(len(df)).mean().iloc[-1])
        ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
        ma50_prev = float(df["Close"].rolling(50).mean().iloc[-20]) if len(df) >= 70 else ma50

        above_200 = price > ma200
        ma50_rising = ma50 > ma50_prev

        vix = get_vix_data().get("current", 20)

        if above_200 and ma50_rising and vix < 20:
            regime = "Bull Market"
            color = "green"
        elif above_200 and ma50_rising and vix < 25:
            regime = "Uptrend"
            color = "lightgreen"
        elif not above_200 and not ma50_rising and vix > 25:
            regime = "Bear Market"
            color = "red"
        elif not above_200 and vix > 20:
            regime = "Downtrend"
            color = "orange"
        else:
            regime = "Sideways / Choppy"
            color = "yellow"

        pct_from_200 = (price - ma200) / ma200 * 100

        result = {
            "regime": regime,
            "color": color,
            "price": round(price, 2),
            "ma200": round(ma200, 2),
            "ma50": round(ma50, 2),
            "above_200ma": above_200,
            "pct_from_200ma": round(pct_from_200, 2),
            "ma50_rising": ma50_rising,
            "vix": vix,
        }
        _cache[key] = {"data": result, "ts": time.time()}
        return result
    except Exception as e:
        logger.error(f"SP500 regime error: {e}")
        return {"regime": "Unknown", "status": "error"}


def get_sector_performance(ttl: int = 3600) -> Dict[str, Dict]:
    """Returns 1-month and 3-month performance for major sector ETFs."""
    key = "sector_perf"
    if key in _cache and _fresh(_cache[key], ttl):
        return _cache[key]["data"]

    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Disc.": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Utilities": "XLU",
        "Communication": "XLC",
    }
    result = {}
    for name, etf in sectors.items():
        try:
            df = get_price_history(etf, period="6mo", interval="1d")
            if df is not None and len(df) > 60:
                curr = float(df["Close"].iloc[-1])
                m1 = float(df["Close"].iloc[-22]) if len(df) >= 22 else curr
                m3 = float(df["Close"].iloc[-63]) if len(df) >= 63 else curr
                result[name] = {
                    "etf": etf,
                    "price": round(curr, 2),
                    "1m_pct": round((curr - m1) / m1 * 100, 2),
                    "3m_pct": round((curr - m3) / m3 * 100, 2),
                }
        except Exception:
            continue
    _cache[key] = {"data": result, "ts": time.time()}
    return result
