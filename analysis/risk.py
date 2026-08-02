"""
Position sizing and risk — implements the Financial Analyst's framework:
- Half-Kelly position sizing
- Stop-loss based risk calculation (1R = 1% of portfolio)
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def position_size_from_stop(
    portfolio_value: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.01,
) -> Dict[str, Any]:
    """
    Calculate position size so that hitting the stop = losing risk_pct of portfolio.
    This is the correct approach: define the stop first, let size follow from risk.
    """
    if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        logger.warning(
            f"position_size_from_stop: invalid price inputs — entry={entry_price} stop={stop_price}"
        )
        return {"error": "Invalid price inputs — stop must be below entry"}

    dollar_risk = portfolio_value * risk_pct
    risk_per_share = entry_price - stop_price
    shares = dollar_risk / risk_per_share
    position_value = shares * entry_price
    position_pct = position_value / portfolio_value

    logger.info(
        f"position_size_from_stop: shares={round(shares)} position_pct={round(position_pct * 100, 2)}% "
        f"dollar_risk={round(dollar_risk, 2)}"
    )
    return {
        "shares": round(shares),
        "position_value": round(position_value, 2),
        "position_pct": round(position_pct * 100, 2),
        "dollar_risk": round(dollar_risk, 2),
        "risk_per_share": round(risk_per_share, 2),
        "risk_pct_of_portfolio": round(risk_pct * 100, 2),
        "risk_reward_3to1_target": round(entry_price + risk_per_share * 3, 2),
        "risk_reward_2to1_target": round(entry_price + risk_per_share * 2, 2),
    }


def regime_kelly_multiplier(signal: float, confidence: float) -> float:
    """
    Position-size multiplier derived from a Markov regime signal
    (analysis.regime_markov.analyze_regime_markov), in [0.5, 1.5].
    Scales toward 1.5x when the regime signal agrees with going long and
    the sample size behind it is reliable, toward 0.5x when it disagrees,
    and stays near 1.0x when the signal is weak or unreliable (low
    confidence) — confidence gates how far the multiplier can move from
    1.0 rather than being applied as a separate discount.
    """
    return round(1.0 + 0.5 * signal * confidence, 4)


def horizon_stop(
    price: float,
    avg_move_pct: float,
    *,
    direction: str = "long",
    k: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Stop distance scaled to a signal's own horizon (Roadmap Item 2B).

    The Day Trading card's stop is `1.5 x daily ATR`. Against a 15-minute signal
    with a 75-minute horizon that stop is enormous — it will essentially never be
    touched inside the window, so any R:R computed from it pairs a 75-minute
    target with a multi-day stop and means nothing.

    `avg_move_pct` is the mean absolute move over that horizon, which both cost
    models already compute (`assess_tradeability` /
    `assess_options_tradeability` -> `avg_move_pct`). Using it keeps the stop, the
    target, and the cost verdict anchored to one volatility estimate.

    Returns {} on unusable input rather than a stop that would mislead.
    """
    from config.settings import HORIZON_STOP_ATR_MULTIPLE

    k = HORIZON_STOP_ATR_MULTIPLE if k is None else k
    try:
        price = float(price)
        avg_move_pct = float(avg_move_pct)
        k = float(k)
    except (TypeError, ValueError):
        return {}
    if price <= 0 or avg_move_pct <= 0 or k <= 0:
        logger.debug(
            "horizon_stop: unusable input price=%s avg_move_pct=%s k=%s",
            price, avg_move_pct, k,
        )
        return {}

    distance = k * (avg_move_pct / 100.0) * price
    is_long = direction.lower() in ("long", "bull", "bullish")
    return {
        "distance": round(distance, 4),
        "stop_price": round(price - distance if is_long else price + distance, 2),
        "stop_pct": round(distance / price * 100, 3),
        "k": k,
        "avg_move_pct": avg_move_pct,
        "basis": f"{k:g} x the average {avg_move_pct:.3f}% move over this signal's horizon",
    }


def contract_loss_at_stop(
    stop_distance: float,
    *,
    delta: float,
    option_mid: float,
) -> Dict[str, Any]:
    """
    What a stop on the underlying actually costs at the contract level.

    The stop is expressed on the underlying because that is what you watch, but
    the loss is leveraged by delta: a $2 adverse move against a $3.00 contract
    with 0.50 delta is a $1.00 hit, i.e. a third of the premium. Reporting only
    the underlying distance understates the risk by roughly the elasticity
    factor, which for a near-dated contract is an order of magnitude.

    Returns {} when delta or premium are unusable.
    """
    try:
        stop_distance = float(stop_distance)
        delta = abs(float(delta))
        option_mid = float(option_mid)
    except (TypeError, ValueError):
        return {}
    if stop_distance <= 0 or delta <= 0 or option_mid <= 0:
        return {}

    premium_loss = delta * stop_distance
    return {
        "premium_loss": round(premium_loss, 4),
        "premium_loss_pct": round(premium_loss / option_mid * 100, 2),
        # Capped at the premium: an option cannot lose more than it cost, and a
        # linear delta approximation happily projects past that.
        "is_total_loss": bool(premium_loss >= option_mid),
    }
