"""
Real options chain data from yfinance.
Computes IVR, IV Percentile, IV vs RV spread, theta decay, P&L diagrams.
No mock data — all numbers come from live market data.
"""
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any, List
from data.price_data import get_price_history
from analysis.options_pricing import black_scholes_greeks, implied_volatility
from analysis.volatility_forecast import garch_forecast_vol
from config.settings import RISK_FREE_RATE
from config.tz import now_et

logger = logging.getLogger(__name__)
_cache: Dict[str, Dict] = {}


def _fresh(entry: dict, ttl: int) -> bool:
    return (time.time() - entry["ts"]) < ttl


def get_options_chain(ticker: str, expiry: Optional[str] = None, ttl: int = 600) -> Dict[str, Any]:
    """
    Fetch real options chain from yfinance.
    Returns calls df, puts df, expiration dates, and selected expiry.
    """
    key = f"chain_{ticker}_{expiry or 'nearest'}"
    if key in _cache and _fresh(_cache[key], ttl):
        logger.debug(f"get_options_chain cache hit for {ticker} ({expiry or 'nearest'})")
        return _cache[key]["data"]

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            logger.warning(f"No options chain available for {ticker}")
            return {"error": "No options available for this ticker", "ticker": ticker}

        target_expiry = expiry if expiry in expirations else expirations[0]
        chain = t.option_chain(target_expiry)
        current_price = get_price_history(ticker, period="5d", interval="1d")
        current_price = float(current_price["Close"].iloc[-1]) if current_price is not None else None

        result = {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "expirations": list(expirations),
            "selected_expiry": target_expiry,
            "calls": chain.calls,
            "puts": chain.puts,
        }
        _cache[key] = {"data": result, "ts": time.time()}
        logger.info(
            f"Fetched options chain for {ticker}: expiry={target_expiry} "
            f"calls={len(result['calls'])} puts={len(result['puts'])}"
        )
        return result
    except Exception as e:
        logger.error(f"Options chain error for {ticker}: {e}")
        return {"error": str(e), "ticker": ticker}


def calculate_iv_rank(ticker: str, ttl: int = 600) -> Dict[str, Any]:
    """
    Compute IV Rank and IV Percentile from real historical volatility.
    IVR = (current_HV - 52w_low_HV) / (52w_high_HV - 52w_low_HV) * 100
    Also computes IV vs Realized Volatility spread.
    """
    key = f"ivr_{ticker}"
    if key in _cache and _fresh(_cache[key], ttl):
        logger.debug(f"calculate_iv_rank cache hit for {ticker}")
        return _cache[key]["data"]

    try:
        df = get_price_history(ticker, period="1y", interval="1d")
        if df is None or len(df) < 30:
            logger.warning(f"IVR calculation: insufficient price history for {ticker}")
            return {"iv_rank": 50, "iv_percentile": 50, "hv_30": 0, "hv_10": 0, "status": "insufficient_data"}

        df["returns"] = df["Close"].pct_change()
        df["hv_10"] = df["returns"].rolling(10).std() * np.sqrt(252) * 100
        df["hv_21"] = df["returns"].rolling(21).std() * np.sqrt(252) * 100
        df["hv_63"] = df["returns"].rolling(63).std() * np.sqrt(252) * 100

        current_hv = df["hv_21"].iloc[-1]
        hv_10_val = df["hv_10"].iloc[-1]
        hv_63_val = df["hv_63"].iloc[-1]

        hv_series = df["hv_21"].dropna()
        min_hv = hv_series.min()
        max_hv = hv_series.max()
        hv_range = max_hv - min_hv

        iv_rank = ((current_hv - min_hv) / hv_range * 100) if hv_range > 0 else 50
        iv_rank = max(0, min(100, iv_rank))
        iv_percentile = float(hv_series.rank(pct=True).iloc[-1] * 100)

        # Volatility term structure
        vol_term_ratio = hv_10_val / hv_63_val if hv_63_val > 0 else 1.0
        term_structure = "Backwardation" if vol_term_ratio > 1.05 else ("Contango" if vol_term_ratio < 0.95 else "Flat")

        # Try to get real IV from nearest ATM option
        real_iv = None
        atm_iv = None
        days_to_expiry = None
        try:
            chain_data = get_options_chain(ticker, ttl=300)
            if "calls" in chain_data and not chain_data["calls"].empty and chain_data.get("current_price"):
                price = chain_data["current_price"]
                calls = chain_data["calls"]
                calls = calls[calls["impliedVolatility"] > 0]
                if not calls.empty:
                    idx = (calls["strike"] - price).abs().idxmin()
                    atm_iv = float(calls.loc[idx, "impliedVolatility"]) * 100
                    real_iv = atm_iv
                selected_expiry = chain_data.get("selected_expiry")
                if selected_expiry:
                    days_to_expiry = (pd.Timestamp(selected_expiry) - pd.Timestamp(now_et().date())).days
        except Exception:
            pass

        # GARCH(1,1) forward volatility forecast, horizon-matched to the
        # nearest expiry so it's comparable to atm_iv over the same window
        garch = garch_forecast_vol(df["returns"], horizon_days=days_to_expiry or 21)
        garch_vol = garch.get("garch_vol_horizon") if garch.get("status") == "ok" else None

        result = {
            "iv_rank": round(iv_rank, 1),
            "iv_percentile": round(iv_percentile, 1),
            "hv_10": round(hv_10_val, 2),
            "hv_21": round(current_hv, 2),
            "hv_63": round(hv_63_val, 2),
            "atm_iv": round(atm_iv, 2) if atm_iv else None,
            "iv_rv_spread": round((atm_iv - current_hv), 2) if atm_iv else None,
            "iv_rv_ratio": round(atm_iv / current_hv, 2) if atm_iv and current_hv > 0 else None,
            "vol_regime": "High" if iv_rank > 60 else ("Low" if iv_rank < 30 else "Medium"),
            "term_structure": term_structure,
            "vol_term_ratio": round(vol_term_ratio, 3),
            "garch_forecast_vol": round(garch_vol, 2) if garch_vol else None,
            "iv_vs_garch_spread": round(atm_iv - garch_vol, 2) if atm_iv and garch_vol else None,
            "iv_vs_garch_ratio": round(atm_iv / garch_vol, 2) if atm_iv and garch_vol else None,
        }
        _cache[key] = {"data": result, "ts": time.time()}
        logger.info(
            f"IVR computed for {ticker}: iv_rank={result['iv_rank']} "
            f"vol_regime={result['vol_regime']} atm_iv={result['atm_iv']}"
        )
        return result
    except Exception as e:
        logger.error(f"IVR calculation error for {ticker}: {e}")
        return {"iv_rank": 50, "iv_percentile": 50, "status": "error", "error": str(e)}


def get_atm_greeks(ticker: str, expiry: Optional[str] = None) -> Dict[str, Any]:
    """Return ATM call and put greeks for the selected expiry."""
    try:
        chain_data = get_options_chain(ticker, expiry)
        if "error" in chain_data:
            logger.warning(f"get_atm_greeks: options chain error for {ticker}: {chain_data['error']}")
            return {}
        price = chain_data.get("current_price")
        if not price:
            logger.warning(f"get_atm_greeks: no current price available for {ticker}")
            return {}
        calls = chain_data["calls"]
        puts = chain_data["puts"]
        if calls.empty or puts.empty:
            logger.warning(f"get_atm_greeks: empty calls/puts for {ticker}")
            return {}

        # ATM call
        atm_call_idx = (calls["strike"] - price).abs().idxmin()
        atm_call = calls.loc[atm_call_idx]

        # ATM put
        atm_put_idx = (puts["strike"] - price).abs().idxmin()
        atm_put = puts.loc[atm_put_idx]

        selected_expiry = chain_data["selected_expiry"]
        days_to_expiry = (pd.Timestamp(selected_expiry) - pd.Timestamp(now_et().date())).days
        T = max(days_to_expiry / 365, 1 / 365)

        def row_to_dict(row, option_type: str):
            strike = float(row.get("strike", 0))
            base = {
                "strike": strike,
                "bid": float(row.get("bid", 0)),
                "ask": float(row.get("ask", 0)),
                "iv": round(float(row.get("impliedVolatility", 0)) * 100, 2),
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
                "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                "open_interest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
            }

            try:
                sigma = row.get("impliedVolatility")
                sigma = float(sigma) if sigma is not None and pd.notna(sigma) else 0.0
                if sigma <= 0:
                    last_price = row.get("lastPrice")
                    last_price = float(last_price) if last_price is not None and pd.notna(last_price) else 0.0
                    if last_price <= 0:
                        return base
                    sigma = implied_volatility(
                        S=price, K=strike, T=T, r=RISK_FREE_RATE,
                        market_price=last_price, option_type=option_type,
                    )
                    if sigma is None:
                        return base

                greeks = black_scholes_greeks(
                    S=price, K=strike, T=T, r=RISK_FREE_RATE, sigma=sigma, option_type=option_type,
                )
                base["delta"] = greeks["delta"]
                base["gamma"] = greeks["gamma"]
                base["theta"] = greeks["theta"]
                base["vega"] = greeks["vega"]
                base["rho"] = greeks["rho"]
            except Exception as e:
                logger.warning(f"Greeks computation failed for {ticker} {option_type} strike {strike}: {e}")

            return base

        logger.debug(f"Computed ATM greeks for {ticker} @ expiry {selected_expiry}")
        return {
            "atm_call": row_to_dict(atm_call, "call"),
            "atm_put": row_to_dict(atm_put, "put"),
            "current_price": price,
            "expiry": selected_expiry,
        }
    except Exception as e:
        logger.error(f"ATM greeks error for {ticker}: {e}")
        return {}


def build_pnl_diagram(
    strategy: str,
    current_price: float,
    strikes: List[float],
    premiums: List[float],
    option_types: List[str],
    directions: List[int],   # +1 = long, -1 = short
) -> Dict[str, Any]:
    """
    Build expiration P&L diagram data for common strategies.
    Returns price range and P&L array.
    """
    price_range = np.linspace(current_price * 0.6, current_price * 1.4, 200)
    total_premium = sum(p * d for p, d in zip(premiums, directions))
    pnl = np.zeros(len(price_range))

    for strike, premium, opt_type, direction in zip(strikes, premiums, option_types, directions):
        if opt_type == "call":
            intrinsic = np.maximum(price_range - strike, 0)
        else:
            intrinsic = np.maximum(strike - price_range, 0)
        pnl += direction * (intrinsic - premium) * 100  # per contract (100 shares)

    max_profit = float(np.max(pnl))
    max_loss = float(np.min(pnl))
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] <= 0:
            be = price_range[i] + (price_range[i + 1] - price_range[i]) * abs(pnl[i]) / (abs(pnl[i]) + abs(pnl[i + 1]))
            breakevens.append(round(float(be), 2))

    return {
        "strategy": strategy,
        "price_range": price_range.tolist(),
        "pnl": pnl.tolist(),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": breakevens,
        "net_premium": total_premium * 100,
    }
