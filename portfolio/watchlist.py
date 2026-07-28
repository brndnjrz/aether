"""
Weekly watchlist — SQLite-backed shortlist of tickers with a trade plan/thesis
note, so a weekly plan persists across sessions instead of living only in a
Screener run you'd have to redo every morning.
"""
import logging
from typing import List, Dict, Optional
from portfolio.db import get_conn, init_db

logger = logging.getLogger(__name__)


def add_to_watchlist(ticker: str, plan_notes: str = "", target_price: Optional[float] = None,
                      stop_price: Optional[float] = None):
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO watchlist (ticker, plan_notes, target_price, stop_price) VALUES (?, ?, ?, ?)",
            (ticker.upper().strip(), plan_notes.strip(), target_price, stop_price),
        )
        conn.commit()
        logger.info(f"Added to watchlist id={cur.lastrowid} ticker={ticker.upper().strip()}")


def get_watchlist() -> List[Dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY added_date DESC").fetchall()
        logger.debug(f"get_watchlist: {len(rows)} rows")
        return [dict(r) for r in rows]


def remove_from_watchlist(item_id: int):
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE id = ?", (item_id,))
        conn.commit()
        logger.info(f"Removed from watchlist id={item_id}")
