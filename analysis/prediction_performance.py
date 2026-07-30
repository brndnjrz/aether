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
