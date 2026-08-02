#!/usr/bin/env python3
"""
Alert sweep — standalone CLI, NOT imported by the app (Roadmap Item 14).

Streamlit has no background loop: a page computes only while someone is looking
at it, so in-app alerting is not possible. This runs from cron/launchd, evaluates
alert conditions, and appends to storage/alerts.jsonl. Same shape as
scripts/scheduled_retrain.py — discovers tickers from storage/ filenames, never
wired into a page.

Critically, every prediction here is generated with **persist=False**. An alert
loop polling every few minutes would otherwise append to
storage/*_predictions.jsonl on every poll, inflating N and corrupting the live
win rate that Model Lab, the retrain triggers, and the horizon scoreboard all
read off it. Alerts observe; they must not vote.

Usage
-----
    python3 scripts/alert_sweep.py --dry-run
    python3 scripts/alert_sweep.py --tickers SPY
    python3 scripts/alert_sweep.py --tickers SPY --quiet

Cron example (every 15 minutes during market hours, Mon-Fri):
    */15 9-16 * * 1-5 cd /path/to/aether && python3 scripts/alert_sweep.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.tz import now_et_iso                                     # noqa: E402
from config.settings import IVR_HIGH, IVR_LOW                        # noqa: E402

logger = logging.getLogger("alert_sweep")

STORAGE_DIR = _PROJECT_ROOT / "storage"
ALERTS_PATH = STORAGE_DIR / "alerts.jsonl"

# Confidence at or above which a directional call is worth surfacing unprompted.
ALERT_CONFIDENCE_LEVELS = {"high"}


def discover_tickers() -> List[str]:
    """
    Tickers with at least one trained model, from storage/ filenames.

    Same discovery approach as scripts/scheduled_retrain.py: the filesystem is
    the registry, so there is no config file to drift out of sync.
    """
    found = set()
    for path in STORAGE_DIR.glob("*_accuracy.json"):
        stem = path.stem.replace("_accuracy", "")
        # Strip an interval suffix if present: SPY_15m -> SPY
        parts = stem.split("_")
        found.add(parts[0].upper())
    return sorted(found)


def check_consensus_alerts(ticker: str) -> List[Dict[str, Any]]:
    """
    Alerts from the cross-horizon consensus. Reads saved predictions only —
    build_consensus() never generates or writes.
    """
    from analysis.interval_consensus import build_consensus

    alerts: List[Dict[str, Any]] = []
    consensus = build_consensus(ticker)
    a = consensus["alignment"]

    if a["is_unanimous"] and a["net_direction"] in ("bullish", "bearish"):
        alerts.append({
            "kind": "consensus_unanimous",
            "ticker": ticker,
            "detail": (
                f"All live horizons agree {a['net_direction']} "
                f"({a['n_bullish']} bull / {a['n_bearish']} bear)."
            ),
            "tightest_tradeable": a["tightest_tradeable"],
        })

    if a["agree_but_uneconomic"]:
        alerts.append({
            "kind": "agree_but_uneconomic",
            "ticker": ticker,
            "detail": (
                f"Horizons agreeing but failing costs: "
                f"{', '.join(a['agree_but_uneconomic'])}."
            ),
        })

    for h in consensus["horizons"]:
        if h["crosses_session_close"]:
            alerts.append({
                "kind": "signal_never_gradeable",
                "ticker": ticker,
                "detail": (
                    f"{h['horizon']} signal's horizon runs past the close — it can "
                    f"never be graded."
                ),
            })
    return alerts


def check_retrain_alerts(ticker: str, intervals: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Retrain triggers, reusing the same checks Model Lab and Trading Desk show."""
    from analysis.retrain_triggers import check_all_retrain_triggers

    alerts: List[Dict[str, Any]] = []
    targets: List[Optional[str]] = [None] + list(intervals or [])
    for interval in targets:
        try:
            check = check_all_retrain_triggers(ticker, interval=interval)
        except Exception as exc:
            logger.debug("retrain check failed for %s %s: %s", ticker, interval, exc)
            continue
        if not check["should_retrain"]:
            continue
        fired = [n for n, t in check["triggers"].items() if t["triggered"]]
        alerts.append({
            "kind": "retrain_due",
            "ticker": ticker,
            "interval": interval or "daily",
            "detail": f"Triggers fired: {', '.join(fired)}.",
        })
    return alerts


def check_iv_alerts(ticker: str) -> List[Dict[str, Any]]:
    """IV Rank crossing its configured bands — the premium buy/sell framing."""
    from data.options_data import calculate_iv_rank

    alerts: List[Dict[str, Any]] = []
    try:
        iv = calculate_iv_rank(ticker)
    except Exception as exc:
        logger.debug("IV lookup failed for %s: %s", ticker, exc)
        return alerts

    rank = iv.get("iv_rank")
    if rank is None:
        return alerts
    if rank > IVR_HIGH:
        alerts.append({
            "kind": "iv_rank_high", "ticker": ticker,
            "detail": f"IV Rank {rank:.0f} (> {IVR_HIGH}) — premium is rich vs its own year.",
        })
    elif rank < IVR_LOW:
        alerts.append({
            "kind": "iv_rank_low", "ticker": ticker,
            "detail": f"IV Rank {rank:.0f} (< {IVR_LOW}) — premium is cheap vs its own year.",
        })
    return alerts


def check_fresh_signal_alerts(ticker: str, intervals: List[str]) -> List[Dict[str, Any]]:
    """
    Generate a fresh prediction per interval and alert on high-confidence
    directional calls.

    **persist=False on every call.** This is the one place in the codebase that
    generates predictions without a human asking, so it is also the one place
    where writing them would corrupt the track record. See the module docstring.
    """
    from analysis.intraday_prediction import predict_intraday

    alerts: List[Dict[str, Any]] = []
    for interval in intervals:
        try:
            result = predict_intraday(ticker, interval, auto_train=False, persist=False)
        except Exception as exc:
            logger.debug("predict failed for %s %s: %s", ticker, interval, exc)
            continue
        if result.get("error"):
            continue
        if (result.get("confidence") or "").lower() not in ALERT_CONFIDENCE_LEVELS:
            continue
        if result.get("direction") not in ("bullish", "bearish"):
            continue
        trade = result.get("tradeability") or {}
        alerts.append({
            "kind": "high_confidence_signal",
            "ticker": ticker,
            "interval": interval,
            "detail": (
                f"{result['direction'].upper()} at {result['probability'] * 100:.0f}% "
                f"({result['confidence']}), "
                f"{'clears' if trade.get('is_tradeable') else 'FAILS'} the shares cost check."
            ),
            "is_tradeable": trade.get("is_tradeable"),
        })
    return alerts


def write_alerts(alerts: List[Dict[str, Any]]) -> int:
    """Append to storage/alerts.jsonl with a timestamp. Returns rows written."""
    if not alerts:
        return 0
    STORAGE_DIR.mkdir(exist_ok=True)
    stamped_at = now_et_iso()
    with open(ALERTS_PATH, "a") as f:
        for alert in alerts:
            f.write(json.dumps({"at": stamped_at, **alert}) + "\n")
    return len(alerts)


def run_sweep(
    tickers: List[str], *, intervals: List[str], dry_run: bool = False,
    include_fresh: bool = False,
) -> Dict[str, Any]:
    """
    One pass. Each check is isolated so a single failing ticker or a network
    hiccup cannot abort the sweep — same failure-isolation contract as
    scripts/scheduled_retrain.py.
    """
    all_alerts: List[Dict[str, Any]] = []
    per_ticker: Dict[str, int] = {}

    for ticker in tickers:
        found: List[Dict[str, Any]] = []
        for check, label in (
            (lambda t: check_consensus_alerts(t), "consensus"),
            (lambda t: check_retrain_alerts(t, intervals), "retrain"),
            (lambda t: check_iv_alerts(t), "iv"),
        ):
            try:
                found.extend(check(ticker))
            except Exception as exc:
                logger.warning("%s check failed for %s: %s", label, ticker, exc)
        if include_fresh:
            try:
                found.extend(check_fresh_signal_alerts(ticker, intervals))
            except Exception as exc:
                logger.warning("fresh-signal check failed for %s: %s", ticker, exc)

        per_ticker[ticker] = len(found)
        all_alerts.extend(found)

    written = 0 if dry_run else write_alerts(all_alerts)
    return {
        "tickers": tickers,
        "alerts": all_alerts,
        "per_ticker": per_ticker,
        "written": written,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate alert conditions and log them.")
    parser.add_argument("--tickers", nargs="*", help="Defaults to every ticker with a trained model.")
    parser.add_argument(
        "--intervals", nargs="*", default=["5m", "15m", "30m", "1h"],
        help="Intraday intervals to check.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing alerts.jsonl.")
    parser.add_argument(
        "--include-fresh", action="store_true",
        help="Also generate fresh predictions (persist=False) and alert on high-confidence calls.",
    )
    parser.add_argument("--quiet", action="store_true", help="Warnings and errors only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tickers = [t.upper() for t in (args.tickers or discover_tickers())]
    if not tickers:
        logger.warning("No tickers found — train a model first, or pass --tickers.")
        return 1

    result = run_sweep(
        tickers, intervals=args.intervals, dry_run=args.dry_run,
        include_fresh=args.include_fresh,
    )

    for alert in result["alerts"]:
        scope = f"{alert['ticker']}"
        if alert.get("interval"):
            scope += f" {alert['interval']}"
        logger.info("[%s] %s — %s", alert["kind"], scope, alert["detail"])

    if result["dry_run"]:
        logger.info("Dry run — %d alert(s) found, nothing written.", len(result["alerts"]))
    else:
        logger.info("%d alert(s) written to %s", result["written"], ALERTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
