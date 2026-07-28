"""
Investment journal — SQLite-backed position tracking (read-only from the UI).
"""
import logging
from typing import List, Dict, Any
from portfolio.db import get_conn, init_db

logger = logging.getLogger(__name__)


def get_open_positions() -> List[Dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM positions WHERE status='open' ORDER BY entry_date DESC").fetchall()
        logger.debug(f"get_open_positions: {len(rows)} open positions")
        return [dict(r) for r in rows]


def get_closed_performance() -> Dict[str, Any]:
    """Compute win rate, avg gain, avg loss from closed positions."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT entry_price, exit_price, entry_date, exit_date, ticker FROM positions WHERE status='closed' AND exit_price IS NOT NULL"
        ).fetchall()

    if not rows:
        logger.info("get_closed_performance: no closed positions with exit price found")
        return {"trades": 0}

    returns = []
    for r in rows:
        if r["entry_price"] and r["exit_price"] and r["entry_price"] > 0:
            ret = (r["exit_price"] - r["entry_price"]) / r["entry_price"]
            returns.append(ret)

    if not returns:
        logger.warning(f"get_closed_performance: {len(rows)} closed rows but zero valid returns computed")
        return {"trades": len(rows)}

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    result = {
        "trades": len(returns),
        "win_rate": round(len(wins) / len(returns) * 100, 1),
        "avg_win": round(sum(wins) / len(wins) * 100, 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses) * 100, 2) if losses else 0,
        "avg_return": round(sum(returns) / len(returns) * 100, 2),
        "best_trade": round(max(returns) * 100, 2),
        "worst_trade": round(min(returns) * 100, 2),
        "expectancy": round((sum(wins) / len(wins) if wins else 0) * (len(wins) / len(returns)) +
                            (sum(losses) / len(losses) if losses else 0) * (len(losses) / len(returns)), 4),
    }
    logger.info(f"get_closed_performance: {result['trades']} trades, win_rate={result['win_rate']}%")
    return result
