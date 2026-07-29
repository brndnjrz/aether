"""
Activity log — SQLite-backed record of meaningful signal views (Day Trading
analyze, Options chain view, Prediction generated), so a trade entry/exit
can be checked against what was actually on screen at the time.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from portfolio.db import get_conn, init_db
from config.tz import now_et_iso

logger = logging.getLogger(__name__)

# Single source of truth for turning a raw event_type into something
# readable — shared by the Dashboard's Recent Activity feed and the Options
# Log's "what were you looking at" lookup, so the two can't drift apart.
EVENT_LABELS = {
    "day_trading_analyze": "Day Trading — Analyze",
    "options_view": "Options — Chain viewed",
    "options_expiry_view": "Options — Expiry picked",
    "prediction_generated": "Predictions — Daily signal",
    "intraday_prediction_generated": "Predictions — Intraday signal",
    "strategy_lab_setup": "Strategy Lab — MTF setup logged",
    "orbc_signal": "Strategy Lab — ORBC signal logged",
}


def log_activity(event_type: str, ticker: str, detail: Dict[str, Any]):
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO activity_log (event_type, ticker, detail_json, logged_at) VALUES (?, ?, ?, ?)",
            (event_type, ticker.upper().strip(), json.dumps(detail, default=str), now_et_iso()),
        )
        conn.commit()
        logger.info(f"Logged activity id={cur.lastrowid} event_type={event_type} ticker={ticker.upper().strip()}")


def get_activity_in_window(start_iso: str, end_iso: str, ticker: Optional[str] = None) -> List[Dict]:
    init_db()
    with get_conn() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM activity_log WHERE logged_at BETWEEN ? AND ? AND ticker = ? ORDER BY logged_at",
                (start_iso, end_iso, ticker.upper().strip()),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM activity_log WHERE logged_at BETWEEN ? AND ? ORDER BY logged_at",
                (start_iso, end_iso),
            ).fetchall()
        logger.debug(f"get_activity_in_window({start_iso} to {end_iso}, ticker={ticker}): {len(rows)} rows")
        return [dict(r) for r in rows]


def get_recent_activity(limit: int = 10) -> List[Dict]:
    """Most recent activity rows across all tickers/event types, newest first."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_log ORDER BY logged_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        logger.debug(f"get_recent_activity(limit={limit}): {len(rows)} rows")
        return [dict(r) for r in rows]
