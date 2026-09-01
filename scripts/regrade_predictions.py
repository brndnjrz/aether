#!/usr/bin/env python3
"""
One-off re-grader for existing prediction logs. NOT imported by the app — run
directly:

    python3 scripts/regrade_predictions.py                  # dry run, changes nothing
    python3 scripts/regrade_predictions.py --apply          # rewrite the logs
    python3 scripts/regrade_predictions.py --apply --tickers SPY

Why this exists
---------------
Every `correct` field written before the resolver fixes was computed by code with
three defects:

  1. The sign was taken after rounding, so a move under 0.005% became exactly 0.0,
     satisfied neither comparison, and was stored incorrect regardless of which
     way price actually went.
  2. Outcomes inside the model's neutral band were graded on a bare sign test.
     Training excludes those rows entirely, so trained and live accuracy measured
     different populations and their gap drove a retrain that could never close it.
  3. Intraday predictions whose bar had aged out of the refetch window were snapped
     to the nearest surviving bar — in either direction — and graded against
     unrelated prices.

resolve_predictions() only grades rows where `correct is None`, so fixing the code
does nothing for already-graded history. This script clears the grading fields and
lets the corrected resolver recompute them.

What it does NOT fix
-------------------
`neutral_threshold` / `threshold_pct` was not persisted on predictions until the
same round of fixes, so for older rows the band has to be read from the model's
*current* metadata. If a model has been retrained to a different label scheme
since a prediction was made, that band is an approximation rather than the one
the prediction was actually made under. Rows carrying their own threshold are
graded exactly; the rest are flagged in the report.

Prices come from the live provider, so this needs network. Rows whose bars are no
longer inside the provider's window (60 days for most intraday intervals) will
correctly go back to unresolved rather than being graded against the wrong bar —
expect the resolved count to *fall* for older intraday logs. That is the point.
"""
import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regrade_predictions")

_DAILY_LOG_RE = re.compile(r"^([A-Z0-9.\-]+)_predictions\.jsonl$")
_INTRADAY_LOG_RE = re.compile(r"^([A-Z0-9.\-]+)_(5m|15m|30m|1h)_predictions\.jsonl$")

_GRADING_FIELDS = ("correct", "actual_outcome", "neutral_outcome")


def _read_log(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s: skipping unparseable line", path.name)
    return records


def _summarize(records):
    resolved = [r for r in records if r.get("correct") is not None]
    correct = [r for r in resolved if r.get("correct")]
    neutral = [r for r in records if r.get("neutral_outcome")]
    return {
        "logged": len(records),
        "resolved": len(resolved),
        "correct": len(correct),
        "neutral_excluded": len(neutral),
        "accuracy": (len(correct) / len(resolved)) if resolved else None,
    }


def _discover(storage_dir: Path, tickers):
    """Returns [(path, ticker, interval_or_None)] for every prediction log."""
    found = []
    for p in sorted(storage_dir.glob("*_predictions.jsonl")):
        m = _INTRADAY_LOG_RE.match(p.name)
        if m:
            ticker, interval = m.group(1), m.group(2)
        else:
            m = _DAILY_LOG_RE.match(p.name)
            if not m:
                logger.warning("%s: unrecognized log name, skipping", p.name)
                continue
            ticker, interval = m.group(1), None
        if tickers and ticker not in tickers:
            continue
        found.append((p, ticker, interval))
    return found


def _backup(path: Path, stamp: str) -> Path:
    """storage/ is gitignored and has no backup, so never rewrite without one."""
    backup_dir = path.parent / "regrade_backups"
    backup_dir.mkdir(exist_ok=True)
    dest = backup_dir / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, dest)
    return dest


def _clear_grading(path: Path, records):
    from analysis.ml_prediction import _rewrite_jsonl_atomic, _jsonl_lock

    cleared = []
    for r in records:
        c = dict(r)
        for field in _GRADING_FIELDS:
            c.pop(field, None)
        c["actual_outcome"] = None
        c["correct"] = None
        cleared.append(c)
    with _jsonl_lock(path):
        _rewrite_jsonl_atomic(path, cleared)
    return cleared


def _threshold_coverage(records, interval):
    """How many rows carry the band they were actually predicted under."""
    key = "threshold_pct" if interval else "neutral_threshold"
    have = sum(1 for r in records if r.get(key) is not None)
    return have, len(records)


def _prices_reachable(ticker: str, interval) -> bool:
    """
    Pre-flight check. The resolvers swallow fetch failures and return 0 rather
    than raising, so without this a network outage would clear every grading
    field and then quietly fail to recompute any of them — the log would read as
    wiped. Verify prices are actually obtainable before touching anything.
    """
    from data.price_data import get_price_history

    try:
        if interval:
            from analysis.intraday_prediction import INTERVAL_SPECS
            spec = INTERVAL_SPECS[interval]
            df = get_price_history(ticker, period=spec["max_period"], interval=interval)
        else:
            df = get_price_history(ticker, period="2y")
    except Exception as exc:
        logger.error("%s %s: price fetch raised (%s)", ticker, interval or "daily", exc)
        return False
    if df is None or df.empty:
        logger.error("%s %s: price fetch returned nothing", ticker, interval or "daily")
        return False
    return True


def regrade_one(path: Path, ticker: str, interval, stamp: str, apply: bool):
    from analysis.ml_prediction import resolve_predictions
    from analysis.intraday_prediction import resolve_intraday_predictions

    label = f"{ticker} {interval or 'daily'}"
    before_records = _read_log(path)
    before = _summarize(before_records)
    have_band, total = _threshold_coverage(before_records, interval)

    record = {
        "log": path.name, "ticker": ticker, "interval": interval,
        "before": before, "own_band_rows": have_band, "total_rows": total,
        "after": None, "error": None,
    }

    if not apply:
        logger.info(
            "%s: DRY RUN — %d logged, %d resolved (%s), %d/%d rows carry their own band",
            label, before["logged"], before["resolved"], _acc_str(before),
            have_band, total,
        )
        return record

    if before["resolved"] and not _prices_reachable(ticker, interval):
        logger.error("%s: skipped — cannot reach prices, refusing to clear gradings", label)
        record["error"] = "prices unreachable"
        return record

    backup = _backup(path, stamp)
    logger.info("%s: backed up to %s", label, backup.name)

    try:
        _clear_grading(path, before_records)
        if interval:
            resolve_intraday_predictions(ticker, interval)
        else:
            resolve_predictions(ticker)
    except Exception as exc:
        logger.error("%s: re-grade failed (%s) — restoring from backup", label, exc)
        shutil.copy2(backup, path)
        record["error"] = str(exc)
        return record

    after = _summarize(_read_log(path))

    # Collapse guard. Some fall in resolved count is expected and correct — rows
    # whose bars aged out of the provider window, and rows now excluded as neutral.
    # Going from "some resolved" to "none resolved" is not that; it means the
    # recompute silently did nothing.
    if before["resolved"] and after["resolved"] == 0 and after["neutral_excluded"] == 0:
        logger.error(
            "%s: resolved count collapsed %d -> 0 with nothing excluded — "
            "restoring from backup", label, before["resolved"],
        )
        shutil.copy2(backup, path)
        record["error"] = "resolved count collapsed to zero"
        return record

    record["after"] = after
    logger.info(
        "%s: %d resolved (%s) -> %d resolved (%s), %d excluded as neutral",
        label, before["resolved"], _acc_str(before),
        after["resolved"], _acc_str(after), after["neutral_excluded"],
    )
    return record


def _acc_str(summary):
    acc = summary["accuracy"]
    return f"{acc * 100:.1f}%" if acc is not None else "n/a"


def _print_report(results, apply: bool):
    print()
    print(f"{'log':<32}{'logged':>7}{'resolved':>19}{'accuracy':>19}{'neutral':>9}")
    print("-" * 86)
    for r in results:
        before, after = r["before"], r["after"]
        if r["error"]:
            print(f"{r['log']:<32}  FAILED — restored from backup")
            continue
        if after is None:
            print(
                f"{r['log']:<32}{before['logged']:>7}{before['resolved']:>19}"
                f"{_acc_str(before):>19}{before['neutral_excluded']:>9}"
            )
            continue
        resolved_delta = f"{before['resolved']} -> {after['resolved']}"
        acc_delta = f"{_acc_str(before)} -> {_acc_str(after)}"
        print(
            f"{r['log']:<32}{after['logged']:>7}{resolved_delta:>19}"
            f"{acc_delta:>19}{after['neutral_excluded']:>9}"
        )
    print()

    partial = [r for r in results if r["own_band_rows"] < r["total_rows"]]
    if partial:
        print("Rows without their own recorded band (graded against the model's")
        print("current metadata, so approximate):")
        for r in partial:
            missing = r["total_rows"] - r["own_band_rows"]
            print(f"  {r['log']:<32}{missing} of {r['total_rows']}")
        print()

    if not apply:
        print("Dry run — nothing was written. Re-run with --apply to rewrite the logs.")
    else:
        print("Backups are in storage/regrade_backups/. To undo, copy them back over")
        print("the originals; nothing else in the app reads that directory.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite the logs. Without this, reports only.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers to limit to, e.g. SPY,AAPL")
    args = parser.parse_args()

    from config.settings import STORAGE_DIR
    from config.tz import now_et

    storage_dir = Path(STORAGE_DIR).resolve()
    tickers = (
        {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        if args.tickers else None
    )

    logs = _discover(storage_dir, tickers)
    if not logs:
        logger.error("No prediction logs found in %s", storage_dir)
        return 1

    stamp = now_et().strftime("%Y%m%dT%H%M%S")
    logger.info(
        "Re-grading %d log(s) in %s%s",
        len(logs), storage_dir, "" if args.apply else " (dry run)",
    )

    results = [regrade_one(p, t, i, stamp, args.apply) for p, t, i in logs]
    _print_report(results, args.apply)

    n_failed = sum(1 for r in results if r["error"])
    if n_failed:
        logger.error("%d log(s) failed and were restored", n_failed)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
