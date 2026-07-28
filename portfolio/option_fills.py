"""
Options trade log — SQLite-backed ledger of individual option fills (one row
per buy or sell), since brokers report fills, not round trips. Round-trip
P&L is derived separately in portfolio/round_trips.py.
"""
import logging
from typing import List, Dict, Optional
from portfolio.db import get_conn, init_db

logger = logging.getLogger(__name__)


def add_fill(ticker: str, strike: float, option_type: str, expiry_date: str,
             side: str, qty: int, price: float, filled_at: str, notes: str = ""):
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO option_fills
               (ticker, strike, option_type, expiry_date, side, qty, price, filled_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker.upper().strip(), strike, option_type.lower(), expiry_date,
             side.lower(), qty, price, filled_at, notes.strip()),
        )
        conn.commit()
        logger.info(
            f"Added option fill id={cur.lastrowid} {ticker.upper().strip()} "
            f"${strike:g} {option_type.lower()} {side.lower()} x{qty} @ {price}"
        )


def get_fills(ticker: Optional[str] = None) -> List[Dict]:
    init_db()
    with get_conn() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM option_fills WHERE ticker = ? ORDER BY filled_at ASC",
                (ticker.upper().strip(),),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM option_fills ORDER BY filled_at ASC").fetchall()
        logger.debug(f"get_fills(ticker={ticker}): {len(rows)} rows")
        return [dict(r) for r in rows]


def remove_fill(fill_id: int):
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM option_fills WHERE id = ?", (fill_id,))
        conn.commit()
        logger.info(f"Removed option fill id={fill_id}")


def update_fill(fill_id: int, ticker: str, strike: float, option_type: str, expiry_date: str,
                 side: str, qty: int, price: float, filled_at: str, notes: str = ""):
    init_db()
    with get_conn() as conn:
        conn.execute(
            """UPDATE option_fills
               SET ticker = ?, strike = ?, option_type = ?, expiry_date = ?,
                   side = ?, qty = ?, price = ?, filled_at = ?, notes = ?
               WHERE id = ?""",
            (ticker.upper().strip(), strike, option_type.lower(), expiry_date,
             side.lower(), qty, price, filled_at, notes.strip(), fill_id),
        )
        conn.commit()
        logger.info(f"Updated option fill id={fill_id} ticker={ticker.upper().strip()}")
