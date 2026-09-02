"""
Classification-style performance metrics for logged model predictions —
precision/recall/F1, false-positive/false-negative rate, profit-per-signal,
holding time, and confidence calibration, on top of the accuracy/win-rate
already computed by ml_prediction.evaluate_model().

Streamlit-free (see CLAUDE.md layer rules) — pure functions over the
DataFrames analysis.ml_prediction.get_prediction_history() and
analysis.intraday_prediction.get_intraday_prediction_history() already
return. Ground truth for "was this call right" is never re-derived here —
it's read straight from the `correct` column those functions already
compute (resolve_predictions()'s strict >/< comparison against actual
price), so this module can never silently disagree with the accuracy
numbers already shown elsewhere in the app.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONFIDENCE_LEVELS = ["high", "medium", "low"]
REQUIRED_COLUMNS = ["direction", "probability", "confidence", "actual_outcome", "correct"]


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    """None (not 0.0) on a zero denominator — 0/0 is undefined, not zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro(*values: Optional[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _round(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 4)


def _horizon_unit(horizon_col: str) -> Optional[str]:
    return {"horizon_days": "days", "horizon_minutes": "minutes"}.get(horizon_col)


def _empty_result(horizon_col: str, warnings: List[str], n_total: int = 0) -> Dict[str, Any]:
    return {
        "ticker": None,
        "interval": None,
        "n_total": n_total,
        "n_resolved": 0,
        "n_unresolved": n_total,
        "accuracy": None,
        "win_rate": None,
        "win_rate_by_confidence": {level: None for level in CONFIDENCE_LEVELS},
        "precision": {"bullish": None, "bearish": None, "macro": None},
        "recall": {"bullish": None, "bearish": None, "macro": None},
        "f1": {"bullish": None, "bearish": None, "macro": None},
        "false_positive_rate": {"bullish": None, "bearish": None, "macro": None},
        "false_negative_rate": {"bullish": None, "bearish": None, "macro": None},
        "confusion_counts": {
            "bullish": {"correct": 0, "incorrect": 0},
            "bearish": {"correct": 0, "incorrect": 0},
        },
        "avg_profit_per_signal": None,
        "avg_profit_per_signal_by_direction": {"bullish": None, "bearish": None},
        "holding_time": {
            "mean": None, "min": None, "max": None,
            "unit": _horizon_unit(horizon_col),
            "distinct_horizons": [],
        },
        "calibration": {
            "by_confidence": {
                level: {"n": 0, "avg_predicted_probability": None, "realized_accuracy": None}
                for level in CONFIDENCE_LEVELS
            },
            "by_probability_decile": [],
            "note": None,
        },
        "warnings": warnings,
    }


def compute_prediction_metrics(
    history: pd.DataFrame,
    *,
    horizon_col: str = "horizon_days",
    min_decile_sample: int = 10,
) -> Dict[str, Any]:
    """
    Compute the full performance-metric set for one ticker's (or one
    ticker+interval's) logged prediction history.

    Parameters
    ----------
    history : pd.DataFrame
        The exact shape get_prediction_history()/get_intraday_prediction_history()
        return — must have direction/probability/confidence/actual_outcome/correct
        columns. `horizon_col` is looked up if present; missing entirely is
        tolerated (holding_time comes back None with a warning) since a
        caller may pass a trimmed frame.
    horizon_col : str
        "horizon_days" for the daily model, "horizon_minutes" for intraday.
    min_decile_sample : int
        Minimum resolved rows required before the per-probability-decile
        calibration curve is computed — below this, only the coarser
        3-bucket by_confidence calibration runs.

    Returns
    -------
    dict — see module tests for the exact shape. Every rate/ratio is a
    float in [0, 1] or None (never 0.0 masquerading as "computed but zero"
    when the underlying count is a 0/0 division).

    Rows where `correct` is None (unresolved: still pending, permanently
    unresolvable neutral predictions, or legacy rows with no baseline
    price) are excluded from every metric here — same convention as
    ml_prediction.evaluate_model()'s `history.dropna(subset=["correct"])`.
    """
    warnings: List[str] = []
    n_total = len(history)

    missing = [c for c in REQUIRED_COLUMNS if c not in history.columns]
    if missing:
        warnings.append(f"history is missing required column(s): {missing}")
        return _empty_result(horizon_col, warnings, n_total)

    resolved = history[history["correct"].notna()].copy()
    n_resolved = len(resolved)
    n_unresolved = n_total - n_resolved

    if n_resolved == 0:
        warnings.append(
            "No resolved predictions — every metric is None until at least "
            "one prediction's horizon has elapsed."
        )
        result = _empty_result(horizon_col, warnings, n_total)
        result["n_unresolved"] = n_unresolved
        return result

    resolved["correct"] = resolved["correct"].astype(bool)
    is_bull = resolved["direction"] == "bullish"
    is_bear = resolved["direction"] == "bearish"
    correct = resolved["correct"]

    n_bull_correct = int((is_bull & correct).sum())
    n_bull_wrong = int((is_bull & ~correct).sum())
    n_bear_correct = int((is_bear & correct).sum())
    n_bear_wrong = int((is_bear & ~correct).sum())

    accuracy = _safe_div(n_bull_correct + n_bear_correct, n_resolved)

    precision_bull = _safe_div(n_bull_correct, n_bull_correct + n_bull_wrong)
    recall_bull = _safe_div(n_bull_correct, n_bull_correct + n_bear_wrong)
    fpr_bull = _safe_div(n_bull_wrong, n_bull_wrong + n_bear_correct)
    fnr_bull = _safe_div(n_bear_wrong, n_bear_wrong + n_bull_correct)
    f1_bull = _f1(precision_bull, recall_bull)

    precision_bear = _safe_div(n_bear_correct, n_bear_correct + n_bear_wrong)
    recall_bear = _safe_div(n_bear_correct, n_bear_correct + n_bull_wrong)
    fpr_bear = _safe_div(n_bear_wrong, n_bear_wrong + n_bull_correct)
    fnr_bear = _safe_div(n_bull_wrong, n_bull_wrong + n_bear_correct)
    f1_bear = _f1(precision_bear, recall_bear)

    out: Dict[str, Any] = _empty_result(horizon_col, warnings, n_total)
    out["n_resolved"] = n_resolved
    out["n_unresolved"] = n_unresolved
    out["accuracy"] = _round(accuracy)
    out["win_rate"] = _round(accuracy)
    out["confusion_counts"] = {
        "bullish": {"correct": n_bull_correct, "incorrect": n_bull_wrong},
        "bearish": {"correct": n_bear_correct, "incorrect": n_bear_wrong},
    }
    out["precision"] = {
        "bullish": _round(precision_bull), "bearish": _round(precision_bear),
        "macro": _round(_macro(precision_bull, precision_bear)),
    }
    out["recall"] = {
        "bullish": _round(recall_bull), "bearish": _round(recall_bear),
        "macro": _round(_macro(recall_bull, recall_bear)),
    }
    out["f1"] = {
        "bullish": _round(f1_bull), "bearish": _round(f1_bear),
        "macro": _round(_macro(f1_bull, f1_bear)),
    }
    out["false_positive_rate"] = {
        "bullish": _round(fpr_bull), "bearish": _round(fpr_bear),
        "macro": _round(_macro(fpr_bull, fpr_bear)),
    }
    out["false_negative_rate"] = {
        "bullish": _round(fnr_bull), "bearish": _round(fnr_bear),
        "macro": _round(_macro(fnr_bull, fnr_bear)),
    }

    # Win rate by confidence — same bucket/mask convention as
    # ml_prediction.evaluate_model()'s win_rate_by_confidence.
    by_conf_win_rate = {}
    for level in CONFIDENCE_LEVELS:
        subset = resolved[resolved["confidence"] == level]
        by_conf_win_rate[level] = _round(subset["correct"].mean()) if len(subset) else None
    out["win_rate_by_confidence"] = by_conf_win_rate

    # Profit per signal — sign-flipped for bearish, since a bearish call
    # implies a short: a falling price is the profit for that call.
    outcome = pd.to_numeric(resolved["actual_outcome"], errors="coerce")
    adjusted = outcome.where(is_bull, -outcome)
    adjusted_valid = adjusted.dropna()
    if len(adjusted_valid):
        bull_adj = adjusted[is_bull].dropna()
        bear_adj = adjusted[is_bear].dropna()
        out["avg_profit_per_signal"] = _round(adjusted_valid.mean())
        out["avg_profit_per_signal_by_direction"] = {
            "bullish": _round(bull_adj.mean()) if len(bull_adj) else None,
            "bearish": _round(bear_adj.mean()) if len(bear_adj) else None,
        }

    # Holding time — horizon is a per-model-version config value, so
    # distinct_horizons flags a mid-log retrain that changed it.
    if horizon_col in resolved.columns:
        horizon_values = pd.to_numeric(resolved[horizon_col], errors="coerce").dropna()
        if len(horizon_values):
            out["holding_time"] = {
                "mean": _round(horizon_values.mean()),
                "min": _round(horizon_values.min()),
                "max": _round(horizon_values.max()),
                "unit": _horizon_unit(horizon_col),
                "distinct_horizons": sorted(float(v) for v in horizon_values.unique()),
            }
    else:
        warnings.append(f"'{horizon_col}' column missing — cannot compute average holding time.")

    # Confidence calibration. The logged `probability` is P(bullish), so a
    # bearish call at probability=0.1 and a bullish call at probability=0.9
    # represent the SAME 90% confidence in the direction actually called.
    # Bucketing raw probability would scatter economically-identical calls
    # to opposite ends of the range and invert the calibration curve —
    # directional_probability corrects for that.
    probability = pd.to_numeric(resolved["probability"], errors="coerce")
    directional_probability = probability.where(is_bull, 1 - probability)

    by_confidence = {}
    for level in CONFIDENCE_LEVELS:
        subset_mask = resolved["confidence"] == level
        n = int(subset_mask.sum())
        if n == 0:
            by_confidence[level] = {"n": 0, "avg_predicted_probability": None, "realized_accuracy": None}
            continue
        by_confidence[level] = {
            "n": n,
            "avg_predicted_probability": _round(directional_probability[subset_mask].mean()),
            "realized_accuracy": _round(correct[subset_mask].mean()),
        }
    out["calibration"]["by_confidence"] = by_confidence

    if n_resolved < min_decile_sample:
        out["calibration"]["note"] = (
            f"Fewer than {min_decile_sample} resolved predictions ({n_resolved}) — "
            "skipping per-decile calibration."
        )
    else:
        # Quantile bins, not fixed edges — intraday's neutral-band threshold
        # is volatility-scaled per ticker, so there's no fixed probability
        # constant to anchor fixed bin edges to.
        q = max(2, min(5, n_resolved // 5))
        buckets = None
        try:
            buckets = pd.qcut(directional_probability, q=q, duplicates="drop")
        except ValueError as exc:
            out["calibration"]["note"] = f"Could not form probability buckets: {exc}"

        if buckets is not None and buckets.notna().sum() == 0:
            # Too few distinct directional_probability values for `q` bins —
            # qcut with duplicates="drop" degenerates to zero bins rather
            # than one. Fall back to a single bucket over everyone instead
            # of silently reporting no calibration data.
            lo = float(directional_probability.min())
            hi = float(directional_probability.max())
            buckets = pd.Series(f"{lo:.2f}-{hi:.2f}", index=directional_probability.index)

        if buckets is not None:
            grouped = pd.DataFrame({
                "bucket": buckets,
                "probability": directional_probability,
                "correct": correct,
            }).groupby("bucket", observed=True, sort=True)
            out["calibration"]["by_probability_decile"] = [
                {
                    "bucket": str(bucket_label),
                    "n": int(len(group)),
                    "avg_predicted_probability": _round(group["probability"].mean()),
                    "realized_accuracy": _round(group["correct"].mean()),
                }
                for bucket_label, group in grouped
            ]

    out["warnings"] = warnings
    return out


def compute_daily_prediction_metrics(
    ticker: str, history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Thin wrapper — auto-fetches via ml_prediction.get_prediction_history() if history is None."""
    ticker = ticker.upper().strip()
    if history is None:
        from analysis.ml_prediction import get_prediction_history
        history = get_prediction_history(ticker)
    metrics = compute_prediction_metrics(history, horizon_col="horizon_days")
    metrics["ticker"] = ticker
    return metrics


def compute_intraday_prediction_metrics(
    ticker: str, interval: str = "15m", history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Thin wrapper — auto-fetches via intraday_prediction.get_intraday_prediction_history() if history is None."""
    ticker = ticker.upper().strip()
    if history is None:
        from analysis.intraday_prediction import get_intraday_prediction_history
        history = get_intraday_prediction_history(ticker, interval)
    metrics = compute_prediction_metrics(history, horizon_col="horizon_minutes")
    metrics["ticker"] = ticker
    metrics["interval"] = interval
    return metrics


# ── Cross-horizon comparison (Roadmap Item 4) ────────────────────────────────
#
# Model Lab answers "how is the 15m model doing" only after you pick 15m. It
# never answered "WHICH horizon should I trade," which is the actual decision.
# compare_horizons() puts all five side by side and, when options quotes are
# supplied, sweeps an expiry ladder per horizon so the grid shows both halves of
# the question at once: which horizon, and which expiry to trade it with.
#
# Deliberately distinct from analysis.interval_consensus.build_consensus():
# that reports what is LIVE RIGHT NOW (the latest logged prediction, its expiry,
# whether it is stale). This reports the TRACK RECORD, and needs no current
# prediction at all — the cost sweep is anchored to the model's trained-in
# accuracy, not to whatever it happened to say most recently.

DAILY_HORIZON_KEY = "daily"

# Reliability gate used by both trainers (ml_prediction: mean_acc >= 0.52).
# A horizon below it whose costs also fail is hopeless; above it, the model has
# a real edge that costs are eating — a materially different diagnosis.
MIN_USEFUL_ACCURACY = 0.52

# A calibration verdict needs enough rows in BOTH the high and low buckets;
# below this the comparison is noise and is reported as unknown rather than
# guessed at.
MIN_N_PER_CONFIDENCE_BUCKET = 10


def _calibration_verdict(metrics: Dict[str, Any]) -> str:
    """
    Does HIGH confidence actually land more often than LOW for this horizon?

    This is the column that decides whether the confidence badge is safe to size
    on at all — the app's confidence tiers are distance from the neutral band, a
    spread measure, not a probability of being right.
    """
    by_conf = (metrics.get("calibration") or {}).get("by_confidence") or {}
    high, low = by_conf.get("high") or {}, by_conf.get("low") or {}
    n_high, n_low = high.get("n") or 0, low.get("n") or 0
    acc_high, acc_low = high.get("realized_accuracy"), low.get("realized_accuracy")

    if n_high < MIN_N_PER_CONFIDENCE_BUCKET or n_low < MIN_N_PER_CONFIDENCE_BUCKET:
        return f"unknown ({min(n_high, n_low)})"
    if acc_high is None or acc_low is None:
        return "unknown (—)"
    return "yes" if acc_high > acc_low else "no"


def _horizon_sigma_from_metadata(meta: Dict[str, Any]) -> Optional[float]:
    """
    Recover the volatility anchor the shares cost model was built on, so the
    options model prices the same move rather than re-deriving one from price
    data (which this module is not allowed to fetch).

        avg_move_pct = horizon_sigma * 100 * 0.8   =>   sigma = avg / 80
    """
    trade = meta.get("tradeability") or {}
    avg_move_pct = trade.get("avg_move_pct")
    if not avg_move_pct or float(avg_move_pct) <= 0:
        return None
    return float(avg_move_pct) / 100.0 / 0.8


def _stage_verdict(row: Dict[str, Any]) -> str:
    """
    Explicit rule chain, evaluated in order — never a score. Returns the sentinel
    "candidate" for rows that survive every gate; those get ranked by net edge
    afterwards, which cannot be decided per-row.
    """
    from config.settings import (
        RETRAIN_ACCURACY_DROP_THRESHOLD,
        RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK,
    )

    trained = row.get("trained_accuracy")
    live = row.get("live_accuracy")
    net = row.get("net_edge_pct")
    n = row.get("n_resolved") or 0

    if trained is None:
        return "Not trained"
    if n < RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK:
        return "Insufficient data"
    if net is not None and net <= 0:
        # The distinction that matters most for an options trader: a real edge
        # that costs eat is a different problem from no edge at all, and has a
        # different fix (buy more time, or trade the underlying).
        return "Do not trade" if trained <= MIN_USEFUL_ACCURACY else "Uneconomic"
    if live is not None and (trained - live) >= RETRAIN_ACCURACY_DROP_THRESHOLD:
        return "Degraded, retrain"
    return "candidate"


def compare_horizons(
    ticker: str,
    *,
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]] = None,
    underlying_price: Optional[float] = None,
    include_daily: bool = True,
) -> Dict[str, Any]:
    """
    Score every horizon against every other, one row each.

    Parameters
    ----------
    ladder_quotes : optional expiry-ladder quotes from
        `data.options_data.get_expiry_ladder_quotes(...)["quotes"]`. Supplied ->
        each horizon gets an options cost verdict and the `grid` is populated.
        Omitted -> `net_edge_pct` falls back to the stored shares verdict and
        `grid` is empty, with `cost_model` saying which was used.
    underlying_price : required alongside `ladder_quotes` for elasticity.

    Returns
    -------
    {
      "ticker": str,
      "rows": [{horizon, trained_accuracy, accuracy_std, live_accuracy,
                n_resolved, best_dte, net_edge_pct, dominant_cost, cost_model,
                calibrated, verdict, expiry_date}],
      "grid": {horizon: {dte: net_edge_pct | None}},
      "ladder": [dte, ...],
      "warnings": [str],
    }

    Never raises — a horizon that cannot be read becomes a row with a
    "Not trained" verdict and a warning.
    """
    ticker = ticker.upper().strip()
    rows: List[Dict[str, Any]] = []
    grid: Dict[str, Dict[Any, Optional[float]]] = {}
    warnings: List[str] = []

    from analysis.intraday_prediction import INTERVAL_SPECS, load_metadata, model_exists

    def _blank(label: str) -> Dict[str, Any]:
        return {
            "horizon": label, "trained_accuracy": None, "accuracy_std": None,
            "live_accuracy": None, "n_resolved": 0, "best_dte": None,
            "net_edge_pct": None, "dominant_cost": None, "cost_model": None,
            "calibrated": "unknown (0)", "verdict": "Not trained",
            "expiry_date": None, "horizon_minutes": None,
            "iv_points_to_erase_edge": None,
        }

    for interval in INTERVAL_SPECS:
        try:
            if not model_exists(ticker, interval):
                rows.append(_blank(interval))
                continue
            meta = load_metadata(ticker, interval)
            metrics = compute_intraday_prediction_metrics(ticker, interval)
            stored = meta.get("tradeability") or {}
            row = {
                "horizon": interval,
                "trained_accuracy": meta.get("directional_accuracy"),
                "accuracy_std": meta.get("accuracy_std"),
                "live_accuracy": metrics.get("win_rate"),
                "n_resolved": metrics.get("n_resolved", 0),
                "horizon_minutes": meta.get("horizon_minutes"),
                "net_edge_pct": stored.get("net_edge_pct"),
                "dominant_cost": None,
                "cost_model": "shares" if stored else None,
                "best_dte": None,
                "expiry_date": None,
                "iv_points_to_erase_edge": None,
                "calibrated": _calibration_verdict(metrics),
            }
            _apply_options_sweep(
                row, meta=meta, ladder_quotes=ladder_quotes,
                underlying_price=underlying_price, grid=grid,
            )
            rows.append(row)
        except Exception as exc:
            logger.warning("compare_horizons: %s %s failed: %s", ticker, interval, exc)
            rows.append(_blank(interval))
            warnings.append(f"{interval}: could not read ({exc}).")

    if include_daily:
        try:
            from analysis.ml_prediction import _load_model_metadata, _rf_path, _xgb_path

            if not (_xgb_path(ticker).exists() and _rf_path(ticker).exists()):
                rows.append(_blank(DAILY_HORIZON_KEY))
            else:
                meta = _load_model_metadata(ticker)
                metrics = compute_daily_prediction_metrics(ticker)
                horizon_days = meta.get("horizon_days") or 5
                row = {
                    "horizon": DAILY_HORIZON_KEY,
                    "trained_accuracy": meta.get("directional_accuracy"),
                    "accuracy_std": meta.get("accuracy_std"),
                    "live_accuracy": metrics.get("win_rate"),
                    "n_resolved": metrics.get("n_resolved", 0),
                    # 1440 min/day: the daily model's horizon in the same units
                    # the options theta model consumes.
                    "horizon_minutes": float(horizon_days) * 1440,
                    # The daily trainer persists no tradeability record, so there
                    # is no shares fallback to inherit -- only a live sweep can
                    # price it. None here means "unknown", never "zero".
                    "net_edge_pct": None,
                    "dominant_cost": None,
                    "cost_model": None,
                    "best_dte": None,
                    "expiry_date": None,
                    "iv_points_to_erase_edge": None,
                "calibrated": _calibration_verdict(metrics),
                }
                _apply_options_sweep(
                    row,
                    meta={
                        **meta,
                        "tradeability": {
                            # Daily sigma scaled to the horizon by sqrt(t); the
                            # neutral threshold is the model's own per-bar band.
                            "avg_move_pct": (
                                float(meta["neutral_threshold"]) * 100 * 0.8
                                * (float(horizon_days) ** 0.5)
                                if meta.get("neutral_threshold") else None
                            ),
                        },
                    },
                    ladder_quotes=ladder_quotes,
                    underlying_price=underlying_price,
                    grid=grid,
                )
                rows.append(row)
        except Exception as exc:
            logger.warning("compare_horizons: %s daily failed: %s", ticker, exc)
            rows.append(_blank(DAILY_HORIZON_KEY))
            warnings.append(f"daily: could not read ({exc}).")

    # Stage every verdict, then rank the survivors — Primary/Secondary is a
    # relative call that cannot be made one row at a time.
    for row in rows:
        row["verdict"] = _stage_verdict(row)

    candidates = [r for r in rows if r["verdict"] == "candidate"]
    intraday_candidates = [r for r in candidates if r["horizon"] != DAILY_HORIZON_KEY]
    intraday_candidates.sort(
        key=lambda r: (r["net_edge_pct"] is None, -(r["net_edge_pct"] or 0.0))
    )
    for i, row in enumerate(intraday_candidates):
        row["verdict"] = "Primary" if i == 0 else "Secondary"
    for row in candidates:
        if row["horizon"] == DAILY_HORIZON_KEY:
            # A multi-day model is context for an intraday session, not a
            # competitor to it — ranking them against each other would invite
            # comparing a 5-day call to a 75-minute one.
            row["verdict"] = "Swing context"

    if ladder_quotes is None:
        warnings.append(
            "No options quotes supplied — net edge falls back to the shares model "
            "(2 bps round trip), which ignores delta leverage and theta entirely."
        )
    thin = [r["horizon"] for r in rows if r["verdict"] == "Insufficient data"]
    if thin:
        warnings.append(f"Too few resolved predictions to judge: {', '.join(thin)}.")

    return {
        "ticker": ticker,
        "rows": rows,
        "grid": grid,
        "ladder": sorted(ladder_quotes) if ladder_quotes else [],
        "warnings": warnings,
    }


DEFAULT_ROLLING_WINDOW = 20


def rolling_accuracy(
    history: pd.DataFrame, *, window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Rolling hit rate over resolved predictions, oldest-first (Roadmap Item 8).

    A single current accuracy number cannot distinguish "steady at 55%" from
    "was 62%, now 48%" — and those call for different actions. Plotting this
    against retrain dates also makes each retrain's effect visible instead of
    inferred.

    Only resolved, directional rows count, matching every other metric here.
    Returns columns `date`, `accuracy`, `n_window`, and an empty frame (with
    those columns) when there is nothing to roll over — never raises.
    """
    cols = ["date", "accuracy", "n_window"]
    if history is None or history.empty or "correct" not in history.columns:
        return pd.DataFrame(columns=cols)

    resolved = history[history["correct"].notna()].copy()
    if "direction" in resolved.columns:
        resolved = resolved[resolved["direction"].isin(["bullish", "bearish"])]
    if resolved.empty or "date" not in resolved.columns:
        return pd.DataFrame(columns=cols)

    resolved = resolved[resolved["date"].notna()].sort_values("date")
    if resolved.empty:
        return pd.DataFrame(columns=cols)

    correct = resolved["correct"].astype(bool).astype(float)
    # min_periods=1 so the line starts immediately rather than after `window`
    # rows; n_window is returned alongside so a 3-sample point is visibly thin
    # rather than looking as solid as a 20-sample one.
    effective = min(int(window), len(resolved))
    return pd.DataFrame({
        "date": resolved["date"].values,
        "accuracy": correct.rolling(effective, min_periods=1).mean().values,
        "n_window": correct.rolling(effective, min_periods=1).count().values,
    })


def _apply_options_sweep(
    row: Dict[str, Any],
    *,
    meta: Dict[str, Any],
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]],
    underlying_price: Optional[float],
    grid: Dict[str, Dict[Any, Optional[float]]],
) -> None:
    """
    Overwrite `row`'s cost fields with an options verdict and fill this horizon's
    row of `grid`. No-op when quotes, price, accuracy, or the volatility anchor
    are missing — the shares fallback already in `row` then stands, labelled.
    """
    if not ladder_quotes or not underlying_price:
        return
    accuracy = meta.get("directional_accuracy")
    horizon_minutes = row.get("horizon_minutes")
    sigma = _horizon_sigma_from_metadata(meta)
    if accuracy is None or not horizon_minutes or sigma is None:
        return

    from analysis.options_pricing import sweep_expiries

    sweep = sweep_expiries(
        float(accuracy), sigma,
        underlying_price=float(underlying_price),
        horizon_minutes=float(horizon_minutes),
        quotes=ladder_quotes,
    )
    grid[row["horizon"]] = {
        dte: v.get("net_edge_pct") for dte, v in sweep["by_dte"].items()
    }
    best = sweep.get("best") or {}
    if best:
        row.update({
            "net_edge_pct": best.get("net_edge_pct"),
            "dominant_cost": best.get("dominant_cost"),
            "cost_model": "options",
            "best_dte": sweep.get("best_dte"),
            "expiry_date": best.get("expiry_date"),
            # How much of an IV crush would erase the edge. Arithmetic, not a
            # forecast — see options_pricing._iv_points_to_erase_edge.
            "iv_points_to_erase_edge": best.get("iv_points_to_erase_edge"),
        })
