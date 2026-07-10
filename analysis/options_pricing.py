"""
Black-Scholes option pricing, Greeks, and implied volatility solver.
Closed-form / numerical math only — no fitting, no lookahead.
"""
import numpy as np
from scipy.stats import norm
from typing import Optional, Dict

MIN_SIGMA = 1e-6
MIN_T = 1e-6


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    sigma = max(sigma, MIN_SIGMA)
    T = max(T, MIN_T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
    T_eff = max(T, MIN_T)
    sigma_eff = max(sigma, MIN_SIGMA)
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

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_annual / 365),
        "vega": float(vega),
        "rho": float(rho),
        "price": float(price),
    }


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
        return None

    sigma = 0.3
    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)
        vega_per_unit = black_scholes_greeks(S, K, T, r, sigma, option_type)["vega"] * 100
        if vega_per_unit < 1e-8:
            break
        sigma -= diff / vega_per_unit
        if sigma <= 0 or sigma > 5.0 or not np.isfinite(sigma):
            break
    else:
        return float(sigma)

    lo, hi = 0.001, 5.0
    price_lo = black_scholes_price(S, K, T, r, lo, option_type) - market_price
    price_hi = black_scholes_price(S, K, T, r, hi, option_type) - market_price
    if price_lo * price_hi > 0:
        return None

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        price_mid = black_scholes_price(S, K, T, r, mid, option_type) - market_price
        if abs(price_mid) < tol:
            return float(mid)
        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo = mid
            price_lo = price_mid

    return float((lo + hi) / 2)
