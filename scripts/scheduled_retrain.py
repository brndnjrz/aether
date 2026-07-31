#!/usr/bin/env python3
"""
Standalone, OS-cron-triggered retraining sweep (Prediction Improvement
Engine, Phase 8). NOT imported by the app — run directly:

    python3 scripts/scheduled_retrain.py [--tickers AAPL,SPY] [--dry-run] [--force]

Without --tickers, discovers every ticker that already has a daily model
(storage/*_accuracy.json) and every (ticker, interval) pair that already
has an intraday model, and checks each for a retrain trigger
(analysis.retrain_triggers.check_all_retrain_triggers). Only tickers that
already have a model are ever considered — this script never trains a
brand-new ticker, it only maintains ones already in use.

This script does not register its own schedule. Wire it to cron/launchd
outside this repo, e.g.:

    0 6 * * 1-5 cd /path/to/aether && python3 scripts/scheduled_retrain.py >> storage/retrain_cron.log 2>&1

Logs one line per (ticker, interval) to storage/retrain_log.jsonl
(append-only). Never lets one ticker's failure abort the sweep.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scheduled_retrain")

_DAILY_ACC_RE = re.compile(r"^([A-Z0-9.\-]+)_accuracy\.json$")
_INTRADAY_ACC_RE = re.compile(r"^([A-Z0-9.\-]+)_(5m|15m|30m|1h)_accuracy\.json$")


def _discover_daily_tickers(storage_dir: Path):
    tickers = set()
    for p in storage_dir.glob("*_accuracy.json"):
        m = _DAILY_ACC_RE.match(p.name)
        if m:
            tickers.add(m.group(1))
    return sorted(tickers)


def _discover_intraday_pairs(storage_dir: Path):
    pairs = set()
    for p in storage_dir.glob("*_accuracy.json"):
        m = _INTRADAY_ACC_RE.match(p.name)
        if m:
            pairs.add((m.group(1), m.group(2)))
    return sorted(pairs)


def _log_retrain_event(storage_dir: Path, record: dict):
    path = storage_dir / "retrain_log.jsonl"
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Could not write retrain log: %s", exc)


def _process_daily(ticker: str, force: bool, dry_run: bool, storage_dir: Path) -> dict:
    from analysis.retrain_triggers import check_all_retrain_triggers
    from analysis.ml_prediction import train_model
    from config.tz import now_et_iso

    check = check_all_retrain_triggers(ticker)
    should_retrain = force or check["should_retrain"]
    record = {
        "ticker": ticker, "interval": None, "ran_at": now_et_iso(),
        "should_retrain": should_retrain, "triggers": check["triggers"],
        "retrained": False, "result_summary": None,
    }

    if not should_retrain:
        logger.info("%s (daily): no trigger fired, skipping", ticker)
        _log_retrain_event(storage_dir, record)
        return record

    if dry_run:
        logger.info("%s (daily): would retrain (dry-run)", ticker)
        _log_retrain_event(storage_dir, record)
        return record

    try:
        result = train_model(ticker)
        record["retrained"] = result.get("error") is None
        record["result_summary"] = (
            f"accuracy={result.get('directional_accuracy')} reliable={result.get('is_reliable')}"
            if result.get("error") is None else result["error"]
        )
    except Exception as exc:
        logger.error("%s (daily): train_model raised: %s", ticker, exc)
        record["result_summary"] = f"exception: {exc}"

    _log_retrain_event(storage_dir, record)
    return record


def _process_intraday(ticker: str, interval: str, force: bool, dry_run: bool, storage_dir: Path) -> dict:
    from analysis.retrain_triggers import check_all_retrain_triggers
    from analysis.intraday_prediction import train_intraday_model
    from config.tz import now_et_iso

    check = check_all_retrain_triggers(ticker, interval=interval)
    should_retrain = force or check["should_retrain"]
    record = {
        "ticker": ticker, "interval": interval, "ran_at": now_et_iso(),
        "should_retrain": should_retrain, "triggers": check["triggers"],
        "retrained": False, "result_summary": None,
    }

    if not should_retrain:
        logger.info("%s %s: no trigger fired, skipping", ticker, interval)
        _log_retrain_event(storage_dir, record)
        return record

    if dry_run:
        logger.info("%s %s: would retrain (dry-run)", ticker, interval)
        _log_retrain_event(storage_dir, record)
        return record

    try:
        result = train_intraday_model(ticker, interval)
        record["retrained"] = result.get("error") is None
        record["result_summary"] = (
            f"accuracy={result.get('directional_accuracy')} reliable={result.get('is_reliable')}"
            if result.get("error") is None else result["error"]
        )
    except Exception as exc:
        logger.error("%s %s: train_intraday_model raised: %s", ticker, interval, exc)
        record["result_summary"] = f"exception: {exc}"

    _log_retrain_event(storage_dir, record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check retrain triggers and retrain models that need it.")
    parser.add_argument(
        "--tickers",
        help="Comma-separated tickers to restrict to (default: every ticker with an existing model).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would retrain; retrain nothing.")
    parser.add_argument("--force", action="store_true", help="Retrain regardless of trigger state.")
    args = parser.parse_args(argv)

    from analysis.ml_prediction import _STORAGE_DIR as daily_storage_dir
    from analysis.intraday_prediction import _STORAGE_DIR as intraday_storage_dir

    ticker_filter = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None

    daily_tickers = _discover_daily_tickers(daily_storage_dir)
    intraday_pairs = _discover_intraday_pairs(intraday_storage_dir)
    if ticker_filter:
        daily_tickers = [t for t in daily_tickers if t in ticker_filter]
        intraday_pairs = [(t, i) for t, i in intraday_pairs if t in ticker_filter]

    logger.info(
        "Sweeping %d daily model(s) and %d intraday model(s)%s",
        len(daily_tickers), len(intraday_pairs), " (dry-run)" if args.dry_run else "",
    )

    results = []
    for ticker in daily_tickers:
        try:
            results.append(_process_daily(ticker, args.force, args.dry_run, daily_storage_dir))
        except Exception as exc:
            logger.error("%s (daily): unexpected failure, continuing sweep: %s", ticker, exc)
            results.append({
                "ticker": ticker, "interval": None, "should_retrain": True,
                "retrained": False, "result_summary": f"unexpected failure: {exc}",
            })

    for ticker, interval in intraday_pairs:
        try:
            results.append(_process_intraday(ticker, interval, args.force, args.dry_run, intraday_storage_dir))
        except Exception as exc:
            logger.error("%s %s: unexpected failure, continuing sweep: %s", ticker, interval, exc)
            results.append({
                "ticker": ticker, "interval": interval, "should_retrain": True,
                "retrained": False, "result_summary": f"unexpected failure: {exc}",
            })

    n_retrained = sum(1 for r in results if r["retrained"])
    n_should = sum(1 for r in results if r["should_retrain"])
    n_skipped = len(results) - n_should
    n_failed = n_should - n_retrained if not args.dry_run else 0

    print(f"Sweep complete: {n_retrained} retrained, {n_skipped} skipped (no trigger), {n_failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
