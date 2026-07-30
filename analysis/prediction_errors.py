"""
Failure categorization for incorrect predictions — rule-based, grounded only
in indicator vocabulary that already exists (analysis/indicators.py's
get_signal_summary()) plus VIX regime and historical earnings dates. No new
data sources, no ML — this is a diagnostic layer that explains *why* a
resolved-incorrect prediction likely missed, not a corrective model.

Streamlit-free (see CLAUDE.md layer rules). Never re-derives "correct" —
callers must pass history from get_prediction_history()/
get_intraday_prediction_history(), and this module filters on the exact
`correct == False` predicate those already compute (same convention as
analysis/prediction_performance.py).
"""
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analysis.indicators import calculate_indicators, get_signal_summary

logger = logging.getLogger(__name__)

CATEGORIES = [
    "counter_trend",
    "low_conviction_zone",
    "volume_anomaly",
    "choppy_market",
    "rsi_divergence_present",
    "elevated_vol_regime",
    "earnings_window",
    "uncategorized",
]

DEFAULT_EARNINGS_WINDOW_DAYS = 3


def build_indicator_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compact point-in-time context for the latest bar in `df` — wraps
    get_signal_summary() and adds vix_regime + day_of_week. Returns {} if df
    is empty/None. Never raises — a snapshot failure must never block saving
    the prediction itself (callers in ml_prediction.py/intraday_prediction.py
    call this right before save_prediction()/save_intraday_prediction()).
    """
    if df is None or df.empty:
        return {}
    snapshot = dict(get_signal_summary(df))

    try:
        from data.macro_data import get_vix_data
        snapshot["vix_regime"] = get_vix_data().get("regime")
    except Exception as exc:
        logger.debug(f"build_indicator_snapshot: vix regime lookup failed: {exc}")

    try:
        snapshot["day_of_week"] = df.index[-1].strftime("%A")
    except Exception as exc:
        logger.debug(f"build_indicator_snapshot: day_of_week lookup failed: {exc}")

    return snapshot


def categorize_failure(snapshot: Dict[str, Any], direction: str) -> List[str]:
    """
    Indicator-based failure categories only (no earnings check here — that
    needs the prediction's own date + a ticker's earnings calendar, handled
    by categorize_incorrect_predictions). May return an empty list; the
    caller is responsible for the "uncategorized" fallback once the
    earnings check has also had a chance to fire.
    """
    if direction not in ("bullish", "bearish"):
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")
    if not snapshot:
        return []

    categories: List[str] = []

    above_200 = snapshot.get("above_200ma")
    above_50 = snapshot.get("above_50ma")
    if direction == "bullish" and above_200 is False and above_50 is False:
        categories.append("counter_trend")
    elif direction == "bearish" and above_200 is True and above_50 is True:
        categories.append("counter_trend")

    rsi_zone = snapshot.get("rsi_zone")
    if direction == "bullish" and rsi_zone == "overbought":
        categories.append("low_conviction_zone")
    elif direction == "bearish" and rsi_zone == "oversold":
        categories.append("low_conviction_zone")

    if snapshot.get("volume_surge") is True:
        categories.append("volume_anomaly")

    if snapshot.get("strong_trend") is False:
        categories.append("choppy_market")

    rsi_divergence = snapshot.get("rsi_divergence")
    if direction == "bullish" and rsi_divergence == "bearish_divergence":
        categories.append("rsi_divergence_present")
    elif direction == "bearish" and rsi_divergence == "bullish_divergence":
        categories.append("rsi_divergence_present")

    if snapshot.get("vix_regime") in ("Elevated Fear", "Crisis"):
        categories.append("elevated_vol_regime")

    return categories


def _to_naive_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _is_near_earnings(at: Any, earnings_dates: List[Any], window_days: int) -> bool:
    target = _to_naive_timestamp(at)
    if target is None or not earnings_dates:
        return False
    for raw_date in earnings_dates:
        d = _to_naive_timestamp(raw_date)
        if d is not None and abs((target - d).days) <= window_days:
            return True
    return False


def _recompute_snapshot_at(price_df: pd.DataFrame, at: Any) -> Dict[str, Any]:
    """
    price_df: raw OHLCV (not yet run through calculate_indicators). Slices to
    bars at-or-before `at` after running indicators, and snapshots the last
    such bar. Returns {} if `at` predates all available history, or on any
    failure (never raises).
    """
    target = _to_naive_timestamp(at)
    if price_df is None or price_df.empty or target is None:
        return {}
    try:
        indicators_df = calculate_indicators(price_df)
        idx = pd.DatetimeIndex(indicators_df.index)
        naive_idx = idx.tz_localize(None) if idx.tz is not None else idx
        sliced = indicators_df.loc[naive_idx <= target]
        if sliced.empty:
            return {}
        return build_indicator_snapshot(sliced)
    except Exception as exc:
        logger.debug(f"_recompute_snapshot_at: failed at {at}: {exc}")
        return {}


def categorize_incorrect_predictions(
    history: pd.DataFrame,
    *,
    price_history_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    ticker: Optional[str] = None,
    earnings_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    earnings_window_days: int = DEFAULT_EARNINGS_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Filters `history` to correct == False (same predicate
    prediction_performance.py uses — never re-derived), then categorizes
    each row's likely failure reason.

    Per row: uses row["indicator_snapshot"] if it's a non-empty dict
    (logged at prediction time); else, if price_history_fetcher+ticker are
    given, re-fetches and recomputes indicators for the bar nearest the
    row's date; else leaves the row uncategorized-by-indicators. The
    earnings_window category is checked separately (needs ticker + an
    earnings calendar, not indicator-dependent) if earnings_fetcher+ticker
    are given. Never raises — any fetch/compute failure degrades to
    "uncategorized" for that row plus a "unavailable" snapshot_source.

    Returns `history` filtered to the incorrect rows, with two added
    columns: `failure_categories` (list[str], non-empty) and
    `snapshot_source` ("logged"|"recomputed"|"unavailable").
    """
    empty_cols = list(history.columns) + ["failure_categories", "snapshot_source"] if history is not None else [
        "failure_categories", "snapshot_source",
    ]
    if history is None or history.empty or "correct" not in history.columns:
        return pd.DataFrame(columns=empty_cols)

    incorrect = history[history["correct"] == False].copy()  # noqa: E712 — explicit False, not falsy
    if incorrect.empty:
        incorrect["failure_categories"] = []
        incorrect["snapshot_source"] = []
        return incorrect

    earnings_dates: List[Any] = []
    if ticker and earnings_fetcher:
        try:
            earnings_hist = earnings_fetcher(ticker)
            if earnings_hist is not None and not earnings_hist.empty:
                earnings_dates = list(earnings_hist.index)
        except Exception as exc:
            logger.debug(f"categorize_incorrect_predictions: earnings fetch failed for {ticker}: {exc}")

    price_df_cache: Optional[pd.DataFrame] = None
    fetch_attempted = False

    categories_col: List[List[str]] = []
    sources_col: List[str] = []
    for _, row in incorrect.iterrows():
        direction = row.get("direction")
        logged_snapshot = row.get("indicator_snapshot") if "indicator_snapshot" in incorrect.columns else None

        snapshot: Dict[str, Any] = {}
        source = "unavailable"
        if isinstance(logged_snapshot, dict) and logged_snapshot:
            snapshot = logged_snapshot
            source = "logged"
        elif price_history_fetcher is not None and ticker is not None:
            if not fetch_attempted:
                fetch_attempted = True
                try:
                    price_df_cache = price_history_fetcher(ticker)
                except Exception as exc:
                    logger.debug(f"categorize_incorrect_predictions: price fetch failed for {ticker}: {exc}")
                    price_df_cache = None
            if price_df_cache is not None:
                snapshot = _recompute_snapshot_at(price_df_cache, row.get("date"))
                source = "recomputed" if snapshot else "unavailable"

        categories: List[str] = []
        if snapshot and direction in ("bullish", "bearish"):
            categories = categorize_failure(snapshot, direction)
        if earnings_dates and _is_near_earnings(row.get("date"), earnings_dates, earnings_window_days):
            categories.append("earnings_window")
        if not categories:
            categories = ["uncategorized"]

        categories_col.append(categories)
        sources_col.append(source)

    incorrect["failure_categories"] = categories_col
    incorrect["snapshot_source"] = sources_col
    return incorrect


def aggregate_failure_categories(categorized: pd.DataFrame) -> Dict[str, Any]:
    """{"n_incorrect", "category_counts", "category_rates", "top_category"} —
    a row can count toward more than one category, so counts can sum to more
    than n_incorrect. top_category ties break by CATEGORIES order."""
    if categorized is None or categorized.empty or "failure_categories" not in categorized.columns:
        return {
            "n_incorrect": 0,
            "category_counts": {c: 0 for c in CATEGORIES},
            "category_rates": {c: 0.0 for c in CATEGORIES},
            "top_category": None,
        }

    n = len(categorized)
    counts = {c: 0 for c in CATEGORIES}
    for cats in categorized["failure_categories"]:
        for c in cats:
            if c in counts:
                counts[c] += 1

    rates = {c: round(v / n, 4) for c, v in counts.items()}
    top_category = max(CATEGORIES, key=lambda c: counts[c])
    return {"n_incorrect": n, "category_counts": counts, "category_rates": rates, "top_category": top_category}
