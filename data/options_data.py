"""
Real options chain data from yfinance.
Computes IVR, IV Percentile, IV vs RV spread, theta decay, P&L diagrams.
No mock data — all numbers come from live market data.
"""
import time
import logging
from datetime import datetime, time as dtime
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any, List
from data.price_data import get_price_history
from analysis.options_pricing import black_scholes_greeks, implied_volatility
from analysis.volatility_forecast import garch_forecast_vol
from config.settings import OPTIONS_EXPIRY_LADDER_DTE, RISK_FREE_RATE
from config.tz import MARKET_TZ, now_et

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
                # Black-Scholes theoretical price. Used as a mid-price fallback by
                # get_expiry_ladder_quotes() when the book is empty (after hours,
                # where bid/ask both come back 0) so the options cost model can
                # still produce a labelled estimate instead of nothing.
                "model_price": None,
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
                base["model_price"] = greeks["price"]
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


def _fractional_dte(expiry_str: str, now: Optional[Any] = None) -> Optional[float]:
    """
    Days until the contract actually expires, as a float, measured to the 4:00 PM
    ET close on the expiry date.

    `get_atm_greeks` uses `(expiry - today).days`, which is 0 for a same-day
    expiry — fine for a Black-Scholes T, useless for the options cost model,
    which needs to know whether 4 hours or 4 minutes remain. Returns None on an
    unparseable date, 0.0 for an expiry already past.
    """
    try:
        expiry_date = pd.Timestamp(expiry_str).date()
    except (ValueError, TypeError):
        logger.debug(f"_fractional_dte: unparseable expiry {expiry_str!r}")
        return None
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp(now_et())
    if now_ts.tz is None:
        now_ts = now_ts.tz_localize(MARKET_TZ)
    else:
        now_ts = now_ts.tz_convert(MARKET_TZ)
    close = pd.Timestamp(
        datetime.combine(expiry_date, dtime(16, 0))
    ).tz_localize(MARKET_TZ)
    return max((close - now_ts).total_seconds() / 86400.0, 0.0)


def get_expiry_ladder_quotes(
    ticker: str,
    ladder_dte: Optional[List[float]] = None,
    *,
    option_type: str = "call",
    now: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Assemble ATM quotes across an expiry ladder, shaped for
    `analysis.options_pricing.sweep_expiries`.

    Each requested rung is snapped to the **nearest actually-listed** expiry —
    there is no guarantee a contract exists at exactly 2 or 7 days out — and the
    date used is reported back so the UI can show it rather than the target.
    When two rungs snap to the same listed expiry the longer one is dropped
    (recorded in `skipped`) so the grid never shows two identical columns.

    Cost: one `get_options_chain` fetch per distinct snapped expiry, each cached
    for `OPTIONS_CACHE_TTL`. Four rungs is four fetches that then serve every
    signal horizon — call this once per render, not once per horizon.

    Returns
    -------
    {
      "ticker": str,
      "underlying_price": float | None,
      "quotes": {requested_dte: {"mid", "bid", "ask", "delta", "theta_per_day",
                                 "days_to_expiry", "expiry_date", "quote_source",
                                 "requested_dte", "iv"}},
      "skipped": [{"requested_dte", "reason"}],
      "error": str | None,
    }

    `quote_source` is one of "live" (a real two-sided book), "model_price"
    (empty book — Black-Scholes theoretical used as the mid, which happens after
    hours), or "unavailable". Never presented as live when it isn't.
    """
    ticker = ticker.upper().strip()
    if ladder_dte is None:
        ladder_dte = list(OPTIONS_EXPIRY_LADDER_DTE)

    out: Dict[str, Any] = {
        "ticker": ticker, "underlying_price": None,
        "quotes": {}, "skipped": [], "error": None,
    }

    chain = get_options_chain(ticker)
    if "error" in chain:
        out["error"] = chain["error"]
        logger.warning(f"get_expiry_ladder_quotes: no chain for {ticker}: {chain['error']}")
        return out

    expirations = chain.get("expirations") or []
    if not expirations:
        out["error"] = "No listed expirations."
        return out
    out["underlying_price"] = chain.get("current_price")

    listed = [(e, _fractional_dte(e, now=now)) for e in expirations]
    listed = [(e, d) for e, d in listed if d is not None]
    if not listed:
        out["error"] = "No parseable expirations."
        return out

    used_expiries: Dict[str, float] = {}
    for target in sorted(ladder_dte):
        expiry_str, actual_dte = min(listed, key=lambda ed: abs(ed[1] - float(target)))
        if expiry_str in used_expiries:
            out["skipped"].append({
                "requested_dte": target,
                "reason": (
                    f"Nearest listed expiry ({expiry_str}) was already used by the "
                    f"{used_expiries[expiry_str]:g}-day rung."
                ),
            })
            continue
        used_expiries[expiry_str] = target

        greeks = get_atm_greeks(ticker, expiry_str)
        leg = (greeks or {}).get("atm_call" if option_type == "call" else "atm_put") or {}
        if not leg:
            out["skipped"].append({
                "requested_dte": target,
                "reason": f"No ATM {option_type} data for {expiry_str}.",
            })
            continue

        bid, ask = leg.get("bid") or 0.0, leg.get("ask") or 0.0
        if bid > 0 and ask > 0:
            mid, quote_source = (bid + ask) / 2.0, "live"
        elif leg.get("model_price"):
            mid, quote_source = float(leg["model_price"]), "model_price"
        else:
            out["skipped"].append({
                "requested_dte": target,
                "reason": f"No usable price for {expiry_str} (empty book, no model price).",
            })
            continue

        out["quotes"][target] = {
            "mid": mid,
            "bid": bid if bid > 0 else None,
            "ask": ask if ask > 0 else None,
            "delta": leg.get("delta"),
            "theta_per_day": leg.get("theta"),
            "vega": leg.get("vega"),
            "iv": leg.get("iv"),
            "days_to_expiry": actual_dte,
            "expiry_date": expiry_str,
            "quote_source": quote_source,
            "requested_dte": target,
        }

    logger.info(
        f"get_expiry_ladder_quotes: {ticker} built {len(out['quotes'])} rung(s) "
        f"from {len(ladder_dte)} requested ({len(out['skipped'])} skipped)"
    )
    return out


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
