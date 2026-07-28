"""
Black-Scholes option pricing, Greeks, and implied volatility solver.
Closed-form / numerical math only — no fitting, no lookahead.
"""
import logging
from typing import Optional, Dict

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

MIN_SIGMA = 1e-6
MIN_T = 1e-6


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    sigma = max(sigma, MIN_SIGMA)
    T = max(T, MIN_T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if sigma <= 0 or T <= 0:
        logger.warning(f"black_scholes_price: sigma={sigma} T={T} below minimum — clamping to MIN_SIGMA/MIN_T.")
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    logger.debug(f"black_scholes_price: S={S} K={K} T={T} r={r} sigma={sigma} option_type={option_type} price={price:.4f}")
    return price


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
    T_eff = max(T, MIN_T)
    sigma_eff = max(sigma, MIN_SIGMA)
    if T_eff != T or sigma_eff != sigma:
        logger.warning(f"black_scholes_greeks: clamped input T={T}->{T_eff} sigma={sigma}->{sigma_eff} (expired option or non-positive volatility).")
    d1, d2 = _d1_d2(S, K, T_eff, r, sigma_eff)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T_eff)

    price = black_scholes_price(S, K, T_eff, r, sigma_eff, option_type)
    gamma = pdf_d1 / (S * sigma_eff * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma_eff) / (2 * sqrt_T)
            - r * K * np.exp(-r * T_eff) * norm.cdf(d2)
        )
        rho = K * T_eff * np.exp(-r * T_eff) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta_annual = (
            -(S * pdf_d1 * sigma_eff) / (2 * sqrt_T)
            + r * K * np.exp(-r * T_eff) * norm.cdf(-d2)
        )
        rho = -K * T_eff * np.exp(-r * T_eff) * norm.cdf(-d2) / 100

    greeks = {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_annual / 365),
        "vega": float(vega),
        "rho": float(rho),
        "price": float(price),
    }
    logger.debug(
        f"black_scholes_greeks: S={S} K={K} T={T} option_type={option_type} "
        f"price={greeks['price']:.4f} delta={greeks['delta']:.4f} gamma={greeks['gamma']:.4f} "
        f"theta={greeks['theta']:.4f} vega={greeks['vega']:.4f} rho={greeks['rho']:.4f}"
    )
    return greeks


def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: str,
    max_iterations: int = 50,
    tol: float = 1e-6,
) -> Optional[float]:
    if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
        logger.warning(f"implied_volatility: invalid input S={S} K={K} T={T} market_price={market_price} — returning None.")
        return None

    sigma = 0.3
    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            logger.debug(f"implied_volatility: Newton converged sigma={sigma:.6f} option_type={option_type} market_price={market_price}.")
            return float(sigma)
        vega_per_unit = black_scholes_greeks(S, K, T, r, sigma, option_type)["vega"] * 100
        if vega_per_unit < 1e-8:
            logger.warning(f"implied_volatility: Newton vega near zero (vega={vega_per_unit}) at sigma={sigma:.6f} — falling back to bisection.")
            break
        sigma -= diff / vega_per_unit
        if sigma <= 0 or sigma > 5.0 or not np.isfinite(sigma):
            logger.warning(f"implied_volatility: Newton sigma out of bounds (sigma={sigma}) — falling back to bisection.")
            break
    else:
        # Loop exhausted max_iterations without converging (no break fired) —
        # not evidence of a solution, so fall through to bisection instead of
        # returning the still-off sigma as if it were a clean solve.
        logger.warning(f"implied_volatility: Newton failed to converge within {max_iterations} iterations at sigma={sigma:.6f} — falling back to bisection.")

    lo, hi = 0.001, 5.0
    price_lo = black_scholes_price(S, K, T, r, lo, option_type) - market_price
    price_hi = black_scholes_price(S, K, T, r, hi, option_type) - market_price
    if price_lo * price_hi > 0:
        logger.warning(f"implied_volatility: bisection cannot bracket a root for market_price={market_price} (price_lo={price_lo:.4f}, price_hi={price_hi:.4f}) — returning None.")
        return None

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        price_mid = black_scholes_price(S, K, T, r, mid, option_type) - market_price
        if abs(price_mid) < tol:
            logger.debug(f"implied_volatility: bisection converged sigma={mid:.6f} option_type={option_type} market_price={market_price}.")
            return float(mid)
        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo = mid
            price_lo = price_mid

    result = (lo + hi) / 2
    logger.debug(f"implied_volatility: bisection exhausted max_iterations={max_iterations}, returning midpoint sigma={result:.6f}.")
    return float(result)
