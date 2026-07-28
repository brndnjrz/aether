"""
Position sizing and portfolio risk — implements the Financial Analyst's framework:
- Half-Kelly position sizing
- Stop-loss based risk calculation (1R = 1% of portfolio)
- Portfolio-level: correlation, drawdown, Sharpe, stress tests
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

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


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
    """
    Half-Kelly position sizing.
    fraction=0.5 is the standard safer variant.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        logger.warning(
            f"kelly_fraction: invalid inputs — win_rate={win_rate} avg_loss={avg_loss}, returning 0.0"
        )
        return 0.0
    odds = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / odds
    result = max(0.0, round(kelly * fraction, 4))
    logger.debug(f"kelly_fraction: win_rate={win_rate} odds={round(odds, 4)} kelly={result}")
    return result


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


def kelly_position_size(
    portfolio_value: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    regime_signal: float = 0.0,
    regime_confidence: float = 0.0,
    fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Half-Kelly position size in dollars, adjusted by a regime-derived
    multiplier so sizing leans in with a favorable regime and pulls back
    against an unfavorable one, rather than sizing purely off historical
    win/loss stats.
    """
    base_kelly = kelly_fraction(win_rate, avg_win, avg_loss, fraction)
    multiplier = regime_kelly_multiplier(regime_signal, regime_confidence)
    adjusted_kelly = round(base_kelly * multiplier, 4)
    logger.info(
        f"kelly_position_size: base_kelly_pct={round(base_kelly * 100, 2)}% "
        f"regime_multiplier={multiplier} adjusted_kelly_pct={round(adjusted_kelly * 100, 2)}%"
    )
    return {
        "base_kelly_pct": round(base_kelly * 100, 2),
        "regime_multiplier": multiplier,
        "adjusted_kelly_pct": round(adjusted_kelly * 100, 2),
        "position_value": round(portfolio_value * adjusted_kelly, 2),
    }


def calculate_portfolio_metrics(
    returns,
    risk_free_rate: float = 0.045,
) -> Dict[str, Any]:
    """
    Compute Sharpe, Sortino, max drawdown, CAGR, and volatility
    from a series of periodic returns. `returns` may be a plain list
    (no date attribution) or a pd.Series with a DatetimeIndex (enables
    max_drawdown_date and a plottable wealth index / drawdown series).
    Ported from MonarchAI result_tools.py and enhanced.
    """
    r_series = returns if isinstance(returns, pd.Series) else pd.Series(returns)

    if len(r_series) < 5:
        logger.warning(
            f"calculate_portfolio_metrics: insufficient return history — n={len(r_series)} (need >= 5)"
        )
        return {
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "max_drawdown_date": None,
            "annualized_return": 0.0, "volatility": 0.0,
            "win_rate": 0.0, "profit_loss_ratio": 0.0,
            "wealth_index": pd.Series(dtype=float), "drawdown_series": pd.Series(dtype=float),
        }

    r = r_series.to_numpy()
    n = len(r)

    # True CAGR: compound the full-period return, then annualize with a
    # compounding exponent — NOT mean(r) * 252, which overstates growth
    # under volatility (variance drag).
    compound_return = float(np.prod(1 + r) - 1)
    ann_return = float((1 + compound_return) ** (252 / n) - 1)

    vol = float(np.std(r, ddof=1) * np.sqrt(252))
    sharpe = (ann_return - risk_free_rate) / vol if vol > 0 else 0.0

    downside = r[r < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else vol
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    # Wealth index (growth of $1), previous peaks, and drawdown series —
    # exposed (not just reduced to a single number) so the caller can plot
    # the equity curve and drawdown chart, and attribute the trough to a date.
    wealth_index = (1 + r_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown_series = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = float(drawdown_series.min())
    max_drawdown_date = drawdown_series.idxmin()
    if not isinstance(max_drawdown_date, pd.Timestamp):
        max_drawdown_date = None

    win_rate = float(np.mean(r > 0))
    pos_r = r[r > 0]
    neg_r = r[r < 0]
    pl_ratio = (float(np.mean(pos_r)) / abs(float(np.mean(neg_r)))) if (len(pos_r) > 0 and len(neg_r) > 0) else 0.0

    logger.info(
        f"calculate_portfolio_metrics: n={n} sharpe={round(sharpe, 3)} sortino={round(sortino, 3)} "
        f"max_drawdown={round(max_drawdown * 100, 2)}% annualized_return={round(ann_return * 100, 2)}%"
    )
    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "max_drawdown_date": max_drawdown_date,
        "annualized_return": round(ann_return * 100, 2),
        "volatility": round(vol * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "profit_loss_ratio": round(pl_ratio, 3),
        "wealth_index": wealth_index,
        "drawdown_series": drawdown_series,
    }


def portfolio_correlation_matrix(price_history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build correlation matrix from a dict of {ticker: price_df}.
    Returns a DataFrame correlation matrix.
    """
    closes = {}
    for ticker, df in price_history.items():
        if df is not None and not df.empty and "Close" in df.columns:
            closes[ticker] = df["Close"].pct_change().dropna()

    if len(closes) < 2:
        logger.warning(
            f"portfolio_correlation_matrix: fewer than 2 tickers with usable price data ({len(closes)}) — cannot correlate."
        )
        return pd.DataFrame()

    combined = pd.DataFrame(closes).dropna()
    logger.debug(f"portfolio_correlation_matrix: built {len(combined.columns)}x{len(combined.columns)} matrix from {len(combined)} rows.")
    return combined.corr()


def stress_test_portfolio(
    weights: Dict[str, float],
    scenarios: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Apply historical stress scenarios to a weighted portfolio.
    weights: {ticker: weight_fraction}
    scenarios: {scenario_name: {ticker: return_assumption}}
    Returns estimated portfolio return per scenario.
    """
    if scenarios is None:
        # Historical benchmark declines mapped to sectors/beta
        scenarios = {
            "2008 Financial Crisis (-55% S&P)": -0.55,
            "2020 COVID Crash (-34% in 33 days)": -0.34,
            "2022 Rate Hike (-25% S&P)": -0.25,
            "2022 Growth Selloff (-50% avg growth)": -0.50,
            "+200bps Rate Shock (est -15% P/E compression)": -0.15,
        }
        total_weight = sum(weights.values())
        results = {}
        for scenario, mkt_return in scenarios.items():
            # Simple beta-adjusted estimate (assume beta=1 for all if unknown)
            est = sum(w * mkt_return for w in weights.values()) / total_weight if total_weight > 0 else mkt_return
            results[scenario] = round(est * 100, 1)
        logger.info(f"stress_test_portfolio: ran {len(results)} default scenarios for {len(weights)} tickers.")
        return results

    results = {}
    for scenario_name, ticker_returns in scenarios.items():
        port_return = sum(weights.get(t, 0) * r for t, r in ticker_returns.items())
        results[scenario_name] = round(port_return * 100, 1)
    logger.info(f"stress_test_portfolio: ran {len(results)} custom scenarios for {len(weights)} tickers.")
    return results
