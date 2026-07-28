"""
Shared SQLite connection and schema for portfolio/journal.db — journal,
activity_log, and option_fills all persist to this one file, so the
connection helper and schema setup live here instead of being copied
into each module.
"""
import sqlite3
import os
import logging
from contextlib import contextmanager
from config.settings import STORAGE_DIR

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(STORAGE_DIR, "journal.db")

_initialized = False


@contextmanager
def get_conn():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    logger.debug(f"Opened SQLite connection to {DB_PATH}")
    try:
        yield conn
    except Exception as e:
        logger.error(f"SQLite error on connection to {DB_PATH}: {e}")
        raise
    finally:
        conn.close()


def init_db():
    global _initialized
    if _initialized:
        logger.debug("init_db: schema already initialized, skipping")
        return
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                name TEXT,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                shares REAL,
                position_value REAL,
                portfolio_pct REAL,
                thesis TEXT,
                key_assumptions TEXT,
                thesis_breakers TEXT,
                conviction INTEGER DEFAULT 3,
                expected_horizon TEXT,
                target_price REAL,
                stop_price REAL,
                bear_case TEXT,
                status TEXT DEFAULT 'open',
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                outcome_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                detail_json TEXT,
                logged_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS option_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                expiry_date TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                price REAL NOT NULL,
                filled_at TEXT NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                plan_notes TEXT,
                target_price REAL,
                stop_price REAL,
                added_date TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    _initialized = True
    logger.info(f"Initialized portfolio DB schema at {DB_PATH}")
