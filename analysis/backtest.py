"""
Generic long-only backtest engine — simulate any boolean entry signal against
historical price data with a fixed stop-loss/take-profit, one trade at a time.
Reports win rate and an equity curve so any rule-based signal in the app can be
checked against its own history instead of trusted on assumption alone.
"""
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def macd_bullish_cross_signal(df: pd.DataFrame) -> pd.Series:
    """
    Boolean entry signal: MACD histogram crosses from negative to positive while
    MACD and Signal are both still below zero (catching the momentum turn early,
    not late) and price is above the 200-day SMA (long-term uptrend filter).
    Requires MACD/MACD_signal/MACD_hist/above_200ma columns from calculate_indicators().
    """
    hist_prev = df["MACD_hist"].shift(1)
    cross_up = (hist_prev < 0) & (df["MACD_hist"] > 0)
    return (cross_up & (df["MACD"] < 0) & (df["MACD_signal"] < 0) & df["above_200ma"]).fillna(False)


def simulate_trades(
    df: pd.DataFrame,
    signal: pd.Series,
    price_col: str = "Close",
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03,
    initial_cash: float = 1000.0,
) -> Dict[str, Any]:
    """
    Long-only, one-trade-at-a-time simulation. On a signal bar, enters at that
    bar's price_col; the earliest later bar where price closes at/above the
    take-profit or at/below the stop-loss exits the trade (close-to-close
    comparison — no intrabar fills, so this is an approximation, not a live-fill
    simulation). Returns trades, an equity curve, and summary stats.
    """
    prices = df[price_col].reset_index(drop=True)
    dates = pd.Series(df.index).reset_index(drop=True)
    sig = signal.reset_index(drop=True)

    cash = initial_cash
    in_trade = False
    entry_price = stop_price = target_price = 0.0
    entry_date = None
    trades = []
    equity = []

    for i in range(len(prices)):
        price = float(prices.iloc[i])
        date = dates.iloc[i]

        if not in_trade:
            if bool(sig.iloc[i]):
                entry_price = price
                stop_price = entry_price * (1 - stop_loss_pct)
                target_price = entry_price * (1 + take_profit_pct)
                entry_date = date
                in_trade = True
        else:
            if price >= target_price:
                cash *= target_price / entry_price
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry_price": round(entry_price, 2), "exit_price": round(target_price, 2),
                    "result": "win",
                })
                in_trade = False
                if len(trades) % 10 == 0:
                    logger.debug(f"simulate_trades: progress — {len(trades)} trades closed, at bar {i}/{len(prices)}.")
            elif price <= stop_price:
                cash *= stop_price / entry_price
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry_price": round(entry_price, 2), "exit_price": round(stop_price, 2),
                    "result": "loss",
                })
                in_trade = False
                if len(trades) % 10 == 0:
                    logger.debug(f"simulate_trades: progress — {len(trades)} trades closed, at bar {i}/{len(prices)}.")

        equity.append(cash)

    equity_curve = pd.Series(equity, index=dates.values, name="equity")

    n = len(trades)
    wins = sum(1 for t in trades if t["result"] == "win")
    win_rate = (wins / n * 100) if n else 0.0
    total_return_pct = (cash / initial_cash - 1) * 100

    if n == 0:
        logger.warning("simulate_trades: signal produced no trades over the given history.")

    logger.info(
        f"simulate_trades: complete — num_trades={n} win_rate={round(win_rate, 1)}% "
        f"total_return_pct={round(total_return_pct, 1)}%"
    )

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "num_trades": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(win_rate, 1),
        "final_value": round(cash, 2),
        "total_return_pct": round(total_return_pct, 1),
    }
