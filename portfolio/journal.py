"""
Investment journal — SQLite-backed position tracking (read-only from the UI).
"""
import logging
from typing import List, Dict
from portfolio.db import get_conn, init_db

logger = logging.getLogger(__name__)


def get_open_positions() -> List[Dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM positions WHERE status='open' ORDER BY entry_date DESC").fetchall()
        logger.debug(f"get_open_positions: {len(rows)} open positions")
        return [dict(r) for r in rows]
