"""
Round-trip P&L derivation from a raw fill ledger. Brokers report individual
buy/sell fills, not matched trades — a contract can be opened and closed
multiple times, and a single fill can partially close one lot and open
another, so matching has to be done generically rather than assuming
buy/sell pairs line up 1:1.
"""
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import List, Dict

logger = logging.getLogger(__name__)


def _parse_dt(value: str) -> datetime:
    # Naive timestamps are legacy datetime.utcnow().isoformat() writes and are
    # UTC (matching config/tz.py's utc_iso_to_et_str assumption); localize them
    # so a naive/aware mix in the fill ledger can't raise on subtraction/sort.
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hold_bucket(minutes: float) -> str:
    if minutes < 30:
        return "<30min"
    if minutes < 120:
        return "30min-2h"
    if minutes < 1440:
        return "2h-24h (overnight)"
    return ">24h"


def compute_round_trips(fills: List[Dict]) -> List[Dict]:
    """FIFO-match buy/sell fills within each (ticker, strike, option_type,
    expiry_date) group, sorted by filled_at, and emit one record per matched
    quantity chunk. Unmatched quantity remains an open lot and produces no
    round trip."""
    if not fills:
        logger.warning("compute_round_trips: called with empty fills list")
        return []

    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for f in fills:
        key = (f["ticker"], f["strike"], f["option_type"], f["expiry_date"])
        groups[key].append(f)

    logger.debug(f"compute_round_trips: {len(fills)} fills grouped into {len(groups)} contract keys")

    round_trips = []

    for key, group_fills in groups.items():
        ticker, strike, option_type, expiry_date = key
        group_fills = sorted(group_fills, key=lambda f: _parse_dt(f["filled_at"]))

        open_buys = deque()
        open_sells = deque()

        for fill in group_fills:
            side = fill["side"]
            remaining = fill["qty"]
            filled_at = fill["filled_at"]
            price = fill["price"]

            opposite = open_sells if side == "buy" else open_buys

            while remaining > 0 and opposite:
                lot = opposite[0]
                matched = min(remaining, lot["qty"])

                entry_time, exit_time = lot["filled_at"], filled_at
                entry_price, exit_price = lot["price"], price

                # side == "sell" closes a long (open_buys): profit when exit > entry.
                # side == "buy" closes a short (open_sells): profit when exit < entry.
                if side == "buy":
                    pnl_dollars = (entry_price - exit_price) * matched * 100
                else:
                    pnl_dollars = (exit_price - entry_price) * matched * 100

                entry_dt = _parse_dt(entry_time)
                exit_dt = _parse_dt(exit_time)
                hold_minutes = (exit_dt - entry_dt).total_seconds() / 60

                round_trips.append({
                    "contract_key": f"{ticker} ${strike:g} {option_type} {expiry_date}",
                    "ticker": ticker,
                    "strike": strike,
                    "option_type": option_type,
                    "expiry_date": expiry_date,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "hold_time_minutes": round(hold_minutes, 1),
                    "hold_bucket": _hold_bucket(hold_minutes),
                    "qty": matched,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_dollars": round(pnl_dollars, 2),
                    "pnl_pct": round(pnl_dollars / (entry_price * matched * 100) * 100, 2) if entry_price else None,
                    "win": pnl_dollars >= 0,  # breakeven counts as a win, not a loss, so it doesn't deflate win-rate stats
                })

                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] == 0:
                    opposite.popleft()

            if remaining > 0:
                same_side = open_buys if side == "buy" else open_sells
                same_side.append({"qty": remaining, "price": price, "filled_at": filled_at})

    round_trips.sort(key=lambda r: _parse_dt(r["exit_time"]))
    logger.info(f"compute_round_trips: matched {len(round_trips)} round trips from {len(fills)} fills")
    return round_trips
