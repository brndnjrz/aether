"""
Black-Scholes option pricing, Greeks, and implied volatility solver.
Closed-form / numerical math only — no fitting, no lookahead.

Also hosts the options cost model (`assess_options_tradeability`,
`sweep_expiries`) — the contract-aware counterpart to
`intraday_prediction.assess_tradeability`, which prices costs in *underlying*
percentage points. Deliberately kept dependency-pure (numpy/scipy only, no
yfinance): callers assemble live quotes and pass them in. `data/options_data.py`
owns the fetching.
"""
import logging
from typing import Any, Optional, Dict, List

import numpy as np
from scipy.stats import norm

from config.settings import (
    OPTIONS_MIN_DTE_FOR_BS_THETA,
    OPTIONS_THETA_BASIS,
)

logger = logging.getLogger(__name__)

MIN_SIGMA = 1e-6
MIN_T = 1e-6

MINUTES_PER_TRADING_DAY = 390
MINUTES_PER_CALENDAR_DAY = 1440


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


# ── Options cost model (Roadmap Item 3) ──────────────────────────────────────
#
# intraday_prediction.assess_tradeability() models the edge as
#     (2 * accuracy - 1) * avg_underlying_move  -  round_trip_cost
# with every term in UNDERLYING percentage points and a 2 bps cost. That is
# correct for trading shares and wrong in three ways for trading contracts:
#
#   1. The payoff is leveraged by elasticity  L = |delta| * S / P  (roughly 100x
#      for an ATM 0DTE SPY contract, ~19x at 30 days) -- the gross edge is
#      understated by one to two orders of magnitude.
#   2. A 1-2 cent spread on a $2 contract is a 0.5-1.0% round trip, not 0.02%.
#   3. Time is a cost, and there was no theta term at all.
#
# (1) works in your favour and (2)+(3) against, so the sign of the answer is not
# obvious from raising a constant -- it has to be modelled. Because elasticity
# and theta BOTH scale inversely with time to expiry they partly cancel, which
# is why sweep_expiries() exists: the verdict genuinely flips across the ladder.

def _theta_drag_pct(
    *,
    option_mid: float,
    horizon_minutes: float,
    days_to_expiry: float,
    theta_per_day: Optional[float],
    theta_basis: str,
) -> Dict[str, Any]:
    """
    Cost of holding through the horizon, as a percent of the contract's mid.

    Two methods, chosen by time to expiry:

    - **black_scholes** — pro-rates the closed-form per-day theta over the
      horizon. `theta_basis="trading"` spreads a day's decay across the
      390-minute session (most decay is realized while the market is open);
      `"calendar"` spreads it across 1440 minutes and is the gentler read.

    - **sqrt_extrinsic** — used below `OPTIONS_MIN_DTE_FOR_BS_THETA`, because
      Black-Scholes theta diverges as `T -> 0` and a near-dated contract cannot
      be costed from it. An ATM option's extrinsic value scales as `sqrt(T)`, so
      the fraction retained over the horizon is `sqrt(T_after / T_now)` and the
      drag is the remainder. This captures the acceleration a linear pro-rata
      would miss, which for 0DTE is the difference between a few percent and
      tens of percent.
    """
    if days_to_expiry <= OPTIONS_MIN_DTE_FOR_BS_THETA or theta_per_day is None:
        minutes_to_expiry = days_to_expiry * MINUTES_PER_CALENDAR_DAY
        if minutes_to_expiry <= 0:
            # Already expired, or expiring inside this bar: all extrinsic goes.
            return {"theta_drag_pct": 100.0, "theta_method": "sqrt_extrinsic"}
        after = max(minutes_to_expiry - horizon_minutes, 0.0)
        retained = float(np.sqrt(after / minutes_to_expiry))
        return {
            "theta_drag_pct": round((1.0 - retained) * 100.0, 4),
            "theta_method": "sqrt_extrinsic",
        }

    minutes_per_day = (
        MINUTES_PER_TRADING_DAY if theta_basis == "trading" else MINUTES_PER_CALENDAR_DAY
    )
    day_fraction = horizon_minutes / minutes_per_day
    drag = abs(float(theta_per_day)) * day_fraction / option_mid * 100.0
    return {"theta_drag_pct": round(drag, 4), "theta_method": "black_scholes"}


def _degraded_options_verdict(reason: str, **passthrough: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "cost_model": "options",
        "avg_move_pct": None,
        "elasticity": None,
        "gross_edge_pct": None,
        "spread_cost_pct": None,
        "spread_source": None,
        "theta_drag_pct": None,
        "theta_method": None,
        "net_edge_pct": None,
        "breakeven_accuracy": None,
        "dominant_cost": None,
        "is_tradeable": None,
        "days_to_expiry": None,
        "expiry_date": None,
        "quote_source": None,
        "reason": reason,
    }
    out.update(passthrough)
    return out


def assess_options_tradeability(
    mean_accuracy: float,
    horizon_sigma: float,
    *,
    underlying_price: float,
    option_mid: float,
    delta: float,
    horizon_minutes: float,
    days_to_expiry: float,
    option_bid: Optional[float] = None,
    option_ask: Optional[float] = None,
    theta_per_day: Optional[float] = None,
    theta_basis: Optional[str] = None,
    fallback_spread_pct: Optional[float] = None,
    expiry_date: Optional[str] = None,
    quote_source: str = "live",
) -> Dict[str, Any]:
    """
    Is a directional model's edge economic once it is expressed as an OPTION
    return rather than an underlying return?

        elasticity      L = |delta| * underlying_price / option_mid
        avg_move_pct      = horizon_sigma * 100 * 0.8      (mean |move| ~ 0.8 sigma,
                                                            same as assess_tradeability)
        gross_edge_pct    = (2 * accuracy - 1) * L * avg_move_pct
        spread_cost_pct   = (ask - bid) / option_mid * 100
        theta_drag_pct     see _theta_drag_pct()
        net_edge_pct      = gross_edge_pct - spread_cost_pct - theta_drag_pct

        breakeven_accuracy = ((spread + theta) / (L * avg_move_pct) + 1) / 2

    Every term is returned separately and `dominant_cost` names the bigger of
    the two costs, because that is the actionable part: a spread-dominated
    failure is fixable with better fills or a higher-priced contract, while a
    theta-dominated failure means the *expiry is wrong for the horizon* — not
    that the model is bad.

    Known omissions, stated rather than papered over:

    - **Gamma ignored.** Delta is held constant across the horizon. Fine for
      small moves; optimistic on large ones (in your favour on a winner,
      against you on a loser).
    - **Vega ignored, and this is the big one.** An IV crush after a correct
      directional call can erase the gain entirely. A "tradeable" verdict here
      assumes IV holds. See Roadmap Item 11.
    - **Single ATM contract.** No spreads, no multi-leg structures; selling
      premium has an entirely different cost profile.

    Returns a dict; never raises. On unusable input every numeric field is None
    and `reason` explains why — `is_tradeable` is None, never False, so
    "could not compute" can't be misread as "computed, and it's bad."
    """
    theta_basis = theta_basis or OPTIONS_THETA_BASIS

    try:
        mean_accuracy = float(mean_accuracy)
        horizon_sigma = float(horizon_sigma)
        underlying_price = float(underlying_price)
        option_mid = float(option_mid)
        delta = float(delta)
        horizon_minutes = float(horizon_minutes)
        days_to_expiry = float(days_to_expiry)
    except (TypeError, ValueError):
        return _degraded_options_verdict("Non-numeric input.")

    common = {
        "days_to_expiry": round(days_to_expiry, 3),
        "expiry_date": expiry_date,
        "quote_source": quote_source,
    }

    if not np.isfinite(option_mid) or option_mid <= 0:
        return _degraded_options_verdict("Contract mid price is zero or unavailable.", **common)
    if underlying_price <= 0:
        return _degraded_options_verdict("Underlying price unavailable.", **common)
    if horizon_minutes <= 0:
        return _degraded_options_verdict("Horizon must be positive.", **common)
    if abs(delta) < 1e-9:
        return _degraded_options_verdict(
            "Delta is zero — this contract has no directional exposure to model.", **common,
        )

    avg_move_pct = horizon_sigma * 100.0 * 0.8
    if avg_move_pct <= 0:
        return _degraded_options_verdict(
            "Horizon volatility is zero — no move to trade.", **common,
        )

    elasticity = abs(delta) * underlying_price / option_mid
    gross_edge_pct = (2.0 * mean_accuracy - 1.0) * elasticity * avg_move_pct

    # Spread from live quotes when they are sane; otherwise the pessimistic
    # settings fallback. A zero bid, a crossed book, or a zero-width spread all
    # occur in real chain data — never divide through them.
    spread_cost_pct: float
    spread_source: str
    quotes_usable = (
        option_bid is not None and option_ask is not None
        and np.isfinite(float(option_bid)) and np.isfinite(float(option_ask))
        and float(option_ask) > float(option_bid) > 0
    )
    if quotes_usable:
        spread_cost_pct = (float(option_ask) - float(option_bid)) / option_mid * 100.0
        spread_source = "live_quotes"
    else:
        from config.settings import OPTIONS_FALLBACK_SPREAD_PCT
        pct = OPTIONS_FALLBACK_SPREAD_PCT if fallback_spread_pct is None else fallback_spread_pct
        spread_cost_pct = float(pct) * 100.0
        spread_source = "fallback_assumption"

    theta = _theta_drag_pct(
        option_mid=option_mid,
        horizon_minutes=horizon_minutes,
        days_to_expiry=days_to_expiry,
        theta_per_day=theta_per_day,
        theta_basis=theta_basis,
    )
    theta_drag_pct = theta["theta_drag_pct"]

    total_cost_pct = spread_cost_pct + theta_drag_pct
    net_edge_pct = gross_edge_pct - total_cost_pct

    denom = elasticity * avg_move_pct
    breakeven_accuracy = (total_cost_pct / denom + 1.0) / 2.0 if denom > 0 else None
    if breakeven_accuracy is not None:
        # An accuracy above 1.0 is not achievable — report it as such rather
        # than clamping silently to 1.0, which would read as "just barely
        # possible" when the honest answer is "impossible at this expiry."
        breakeven_accuracy = round(min(max(breakeven_accuracy, 0.0), 2.0), 4)

    return {
        "cost_model": "options",
        "avg_move_pct": round(avg_move_pct, 4),
        "elasticity": round(elasticity, 3),
        "gross_edge_pct": round(gross_edge_pct, 4),
        "spread_cost_pct": round(spread_cost_pct, 4),
        "spread_source": spread_source,
        "theta_drag_pct": theta_drag_pct,
        "theta_method": theta["theta_method"],
        "net_edge_pct": round(net_edge_pct, 4),
        "breakeven_accuracy": breakeven_accuracy,
        "dominant_cost": "theta" if theta_drag_pct >= spread_cost_pct else "spread",
        "is_tradeable": bool(net_edge_pct > 0),
        "theta_basis": theta_basis,
        "reason": None,
        **common,
    }


def sweep_expiries(
    mean_accuracy: float,
    horizon_sigma: float,
    *,
    underlying_price: float,
    horizon_minutes: float,
    quotes: Dict[Any, Dict[str, Any]],
    theta_basis: Optional[str] = None,
    fallback_spread_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run `assess_options_tradeability` across an expiry ladder for one signal
    horizon.

    Because elasticity and theta scale in opposite directions with time to
    expiry, one verdict per horizon is not enough — each signal horizon has a
    minimum viable expiry, and finding it is the point.

    Parameters
    ----------
    quotes : {days_to_expiry: {"mid", "delta", "bid", "ask", "theta_per_day",
              "expiry_date", "quote_source"}} — assembled by the caller
              (`data/options_data.get_expiry_ladder_quotes`). Keeping the fetch
              out of this module is what lets it stay numpy/scipy-only.

    Returns
    -------
    {"by_dte": {dte: verdict}, "best_dte": float|None, "best": verdict|None,
     "any_tradeable": bool, "ladder": [dte, ...]}

    `best` is the highest `net_edge_pct` among computable rungs. **Ties break
    toward the shorter expiry** — less capital committed for the same modelled
    edge.
    """
    by_dte: Dict[Any, Dict[str, Any]] = {}
    for dte, q in sorted(quotes.items(), key=lambda kv: float(kv[0])):
        by_dte[dte] = assess_options_tradeability(
            mean_accuracy, horizon_sigma,
            underlying_price=underlying_price,
            option_mid=q.get("mid"),
            option_bid=q.get("bid"),
            option_ask=q.get("ask"),
            delta=q.get("delta"),
            theta_per_day=q.get("theta_per_day"),
            horizon_minutes=horizon_minutes,
            days_to_expiry=q.get("days_to_expiry", dte),
            theta_basis=theta_basis,
            fallback_spread_pct=fallback_spread_pct,
            expiry_date=q.get("expiry_date"),
            quote_source=q.get("quote_source", "live"),
        )

    computable = [
        (dte, v) for dte, v in by_dte.items() if v.get("net_edge_pct") is not None
    ]
    best_dte = None
    best = None
    if computable:
        # sorted() is stable and by_dte was built in ascending-DTE order, so
        # max() returns the first (shortest) rung on a tie.
        best_dte, best = max(computable, key=lambda kv: kv[1]["net_edge_pct"])

    return {
        "by_dte": by_dte,
        "best_dte": best_dte,
        "best": best,
        "any_tradeable": any(v.get("is_tradeable") for v in by_dte.values()),
        "ladder": [dte for dte in by_dte],
    }
