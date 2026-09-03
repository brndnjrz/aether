#!/usr/bin/env python3
"""
Backfill prediction_ref on option fills logged before tagging existed. NOT
imported by the app — run directly:

    python3 scripts/backfill_fill_links.py                 # dry run
    python3 scripts/backfill_fill_links.py --apply
    python3 scripts/backfill_fill_links.py --apply --window 30

What this can and cannot establish
----------------------------------
It matches each *entry* fill to the nearest prediction for the same ticker that
was logged at or before it, within `--window` minutes. Attribution keys off the
entry fill because what matters is whether the decision to open was model-driven.

That match is a **guess**, and the guess is the whole difficulty. A signal
existing shortly before a trade does not mean it caused the trade — you may have
opened on your own read while a model happened to have fired. So every link
written here is stamped prediction_ref_source='inferred', kept apart from the
'user' tags the fill form writes.

The separation is not bookkeeping fussiness. The question these links exist to
answer is "does following the model make money?", and inferring the links from
the very signals under test makes that answer partly circular: trades near
signals get attributed to signals, so the model's arm fills with trades it may
have had nothing to do with. Treat inferred results as a hypothesis to check
against user-tagged fills once enough of those accumulate, never as the finding.

Sell fills are left alone. A round trip is classified by how it was opened.

Reversible: --undo clears only inferred links, never user tags.
"""
import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_fill_links")

DEFAULT_WINDOW_MINUTES = 60


def _predictions_for(ticker: str):
    """Every logged prediction for `ticker` as (when, ref, direction, horizon)."""
    from config.tz import MARKET_TZ
    out = []

    try:
        from analysis.intraday_prediction import (
            INTERVAL_SPECS, get_intraday_prediction_history,
        )
        for interval in INTERVAL_SPECS:
            hist = get_intraday_prediction_history(ticker, interval, resolve=False)
            for _, row in hist.iterrows():
                when = row.get("date")
                if pd.isna(when):
                    continue
                et = when.tz_convert(MARKET_TZ)
                out.append((et, f"{interval}|{et.isoformat()}",
                            (row.get("direction") or "—"), interval))
    except Exception as exc:
        logger.debug("intraday predictions unavailable for %s: %s", ticker, exc)

    try:
        from analysis.ml_prediction import get_prediction_history
        hist = get_prediction_history(ticker, resolve=False)
        for _, row in hist.iterrows():
            when = row.get("date")
            if pd.isna(when):
                continue
            et = pd.Timestamp(when).tz_convert(MARKET_TZ)
            out.append((et, f"daily|{et.isoformat()}",
                        (row.get("direction") or "—"), "daily"))
    except Exception as exc:
        logger.debug("daily predictions unavailable for %s: %s", ticker, exc)

    return sorted(out, key=lambda r: r[0])


def _trade_direction(fill) -> str:
    """
    The directional bet a fill expresses. Only entries (buys) are considered, so
    buying a call is bullish and buying a put is bearish.
    """
    return "bullish" if (fill.get("option_type") or "").lower() == "call" else "bearish"


def _match(fill_time, predictions, window_minutes: int, trade_direction: str):
    """
    Nearest prediction at or before `fill_time`, within the window, that points
    the same way the trade does.

    Backward only. A prediction logged after the fill cannot have motivated it,
    and counting it would be the same look-ahead the agreement backtest had.

    Direction consistency is required, and it is what makes this inference worth
    anything. Matching on time alone, the first run over the real ledger paired a
    put with a BULLISH signal, another put with BULLISH, and several trades with
    NEUTRAL — only a third of matches pointed the same way as the trade. Nobody
    buys puts because a model said bullish, so those were coincidence: a signal
    happened to exist nearby. Writing them would have loaded the "followed the
    model" arm with trades the model had no part in.

    Neutral predictions never match. A neutral call is not a reason to open
    anything, so a trade next to one is discretionary by definition.
    """
    cutoff = fill_time - timedelta(minutes=window_minutes)
    candidates = [
        p for p in predictions
        if cutoff <= p[0] <= fill_time
        and str(p[2]).lower() == trade_direction
    ]
    return max(candidates, key=lambda p: p[0]) if candidates else None


def _undo(apply: bool) -> int:
    from portfolio.db import init_db, get_conn
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM option_fills WHERE prediction_ref_source = 'inferred'"
        ).fetchone()
        n = rows["n"] if rows else 0
        if not apply:
            logger.info("Would clear %d inferred link(s). User tags untouched.", n)
            return 0
        conn.execute(
            "UPDATE option_fills SET prediction_ref = NULL, prediction_ref_source = NULL "
            "WHERE prediction_ref_source = 'inferred'"
        )
        conn.commit()
    logger.info("Cleared %d inferred link(s). User tags untouched.", n)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true", help="Write the links.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_MINUTES,
                        help=f"Minutes before a fill to look back (default {DEFAULT_WINDOW_MINUTES}).")
    parser.add_argument("--undo", action="store_true",
                        help="Clear inferred links only, leaving user tags in place.")
    args = parser.parse_args()

    from portfolio.db import init_db, get_conn
    from portfolio.option_fills import get_fills
    from config.tz import MARKET_TZ

    init_db()
    if args.undo:
        return _undo(args.apply)

    fills = get_fills()
    if not fills:
        logger.error("No fills found.")
        return 1

    entries = [f for f in fills if (f.get("side") or "").lower() == "buy"]
    logger.info("%d fills, %d of them entries (buys)", len(fills), len(entries))

    preds = {}
    matched, already, unmatched = [], [], []

    for f in entries:
        if f.get("prediction_ref"):
            already.append(f)
            continue
        ticker = (f.get("ticker") or "").upper()
        if ticker not in preds:
            preds[ticker] = _predictions_for(ticker)
        try:
            ft = pd.Timestamp(f["filled_at"])
            ft = ft.tz_localize(MARKET_TZ) if ft.tz is None else ft.tz_convert(MARKET_TZ)
        except Exception:
            unmatched.append((f, "unparseable filled_at"))
            continue
        want = _trade_direction(f)
        hit = _match(ft, preds[ticker], args.window, want)
        if hit:
            matched.append((f, ft, hit))
        else:
            unmatched.append((
                f, f"no {want} prediction within {args.window} min before",
            ))

    print(f"\n{'fill':<34}{'signal matched':<34}{'lag'}")
    print("-" * 78)
    for f, ft, (when, ref, direction, horizon) in matched:
        lag = (ft - when).total_seconds() / 60
        desc = (f"{f['ticker']} {f['strike']:g}{(f['option_type'] or '')[:1].upper()} "
                f"{ft.strftime('%m/%d %H:%M')} ({_trade_direction(f)[:4]})")
        print(f"{desc:<34}{horizon + ' ' + str(direction).upper():<34}{lag:>4.0f}m")
    for f, why in unmatched:
        desc = f"{f['ticker']} {f['strike']:g}{(f['option_type'] or '')[:1].upper()}"
        print(f"{desc:<34}{'— ' + why}")

    print(f"\n{len(matched)} matchable, {len(unmatched)} not, {len(already)} already tagged")

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply.")
        return 0
    if not matched:
        print("\nNothing to write.")
        return 0

    with get_conn() as conn:
        for f, _ft, (_when, ref, _d, _h) in matched:
            conn.execute(
                "UPDATE option_fills SET prediction_ref = ?, prediction_ref_source = 'inferred' "
                "WHERE id = ?",
                (ref, f["id"]),
            )
        conn.commit()

    print(f"\nWrote {len(matched)} inferred link(s), stamped source='inferred'.")
    print("These are guesses: a signal near a trade is not proof it caused the trade.")
    print("Trade attribution reports them separately from fills you tag yourself,")
    print("and 'python3 scripts/backfill_fill_links.py --undo --apply' clears them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
