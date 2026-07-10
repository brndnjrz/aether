"""
Price data fetcher — wraps yfinance with a lightweight in-memory cache.
"""
import time
import logging
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_cache: Dict[str, Dict] = {}


def _is_fresh(entry: dict, ttl: int) -> bool:
    return (time.time() - entry["ts"]) < ttl


def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    ttl: int = 300,
) -> Optional[pd.DataFrame]:
    key = f"{ticker}_{period}_{interval}"
    if key in _cache and _is_fresh(_cache[key], ttl):
        return _cache[key]["data"]

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return None
        df.index = pd.to_datetime(df.index)
        _cache[key] = {"data": df, "ts": time.time()}
        return df
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker}: {e}")
        return None


def get_current_price(ticker: str) -> Optional[float]:
    df = get_price_history(ticker, period="5d", interval="1d")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


def get_multi_price_history(
    tickers: list,
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers efficiently using yfinance download."""
    key = f"multi_{'_'.join(sorted(tickers))}_{period}_{interval}"
    if key in _cache and _is_fresh(_cache[key], 300):
        return _cache[key]["data"]

    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )
        result: Dict[str, pd.DataFrame] = {}
        if len(tickers) == 1:
            result[tickers[0]] = raw
        else:
            for t in tickers:
                if t in raw.columns.get_level_values(0):
                    result[t] = raw[t].dropna(how="all")
        _cache[key] = {"data": result, "ts": time.time()}
        return result
    except Exception as e:
        logger.error(f"Error fetching multi-price data: {e}")
        return {}


def get_ticker_info(ticker: str, ttl: int = 3600) -> Dict[str, Any]:
    key = f"info_{ticker}"
    if key in _cache and _is_fresh(_cache[key], ttl):
        return _cache[key]["data"]

    try:
        info = yf.Ticker(ticker).info
        _cache[key] = {"data": info, "ts": time.time()}
        return info
    except Exception as e:
        logger.error(f"Error fetching ticker info for {ticker}: {e}")
        return {}
