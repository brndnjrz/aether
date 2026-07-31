"""
Position sizing and risk — implements the Financial Analyst's framework:
- Half-Kelly position sizing
- Stop-loss based risk calculation (1R = 1% of portfolio)
"""
import logging
from typing import Dict, Any

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
