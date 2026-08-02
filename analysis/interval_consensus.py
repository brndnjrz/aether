"""
Cross-horizon consensus (Roadmap Item 1) — what do all five direction models
say about one ticker right now, how much is each worth trusting, and when does
each signal expire?

The app has five models (5m / 15m / 30m / 1h / daily) and, before this module,
no way to see them together: comparing horizons meant changing a dropdown four
times and holding the results in your head. Worse, nothing said which of them
could actually pay for itself once options costs were priced in — so the horizon
most likely to be acted on was often the one least able to clear its own spread
and decay.

**Read-only, and deliberately so.** `predict()` and `predict_intraday()` both
call `save_*_prediction()` unconditionally, so a view that generated predictions
on render would append a row to the prediction log every time it was displayed —
inflating N and corrupting the live win rate that Model Lab, the retrain
triggers, and the horizon scoreboard all read off it. This module therefore
reads saved predictions and persisted metadata only. It never fetches, never
predicts, never writes. Refreshing is an explicit user action in the page.

Streamlit-free (see CLAUDE.md layer rules). Heavy imports (`ml_prediction` pulls
in xgboost) are deferred into the function bodies, matching the pattern
`analysis/retrain_triggers.py` established, so importing this module is cheap and
does not require the ML stack to be installed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.horizon_clock import daily_expiry, intraday_expiry
from config.tz import MARKET_TZ, now_et, now_et_iso

logger = logging.getLogger(__name__)

DAILY_KEY = "daily"

# A daily prediction is stale after a full day; an intraday one after a single
# bar of its own interval, since that is when a fresh signal becomes available.
DAILY_STALE_AFTER_MINUTES = 1440


def _latest_row(history: pd.DataFrame) -> Optional[pd.Series]:
    """
    Newest logged prediction, or None. Both history functions already sort
    descending by date, so row 0 is the latest — but an empty or all-NaT frame
    has to be handled rather than indexed into.
    """
    if history is None or history.empty:
        return None
    if "date" in history.columns:
        dated = history[history["date"].notna()]
        if dated.empty:
            return None
        return dated.iloc[0]
    return history.iloc[0]


def _age_minutes(when: Any, now_ts: pd.Timestamp) -> Optional[float]:
    """Minutes since `when`. Stored `date` values are UTC-parsed by the history
    functions; a naive value is therefore treated as UTC, not market-local."""
    if when is None:
        return None
    try:
        ts = pd.Timestamp(when)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    return (now_ts - ts.tz_convert(MARKET_TZ)).total_seconds() / 60.0


def _blank_horizon(label: str, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "horizon": label,
        "has_model": False,
        "has_prediction": False,
        "direction": None,
        "probability": None,
        "confidence": None,
        "trained_accuracy": None,
        "live_accuracy": None,
        "n_resolved": 0,
        "expires_at": None,
        "expires_at_str": None,
        "minutes_remaining": None,
        "is_expired": False,
        "crosses_session_close": False,
        "is_gradeable": None,
        "net_edge_pct": None,
        "is_tradeable": None,
        "cost_model": None,
        "dominant_cost": None,
        "best_dte": None,
        "prediction_age_minutes": None,
        "is_stale": False,
        "note": None,
    }
    row.update(extra)
    return row


def _cost_verdict(
    *,
    row: pd.Series,
    accuracy: Optional[float],
    horizon_minutes: Optional[float],
    underlying_price: Optional[float],
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Options cost verdict when live ladder quotes were supplied by the caller,
    otherwise the shares verdict already stored on the prediction record.

    Never silently presents one as the other — `cost_model` says which was used.
    Deriving `horizon_sigma` back out of the stored `avg_move_pct` keeps the two
    models anchored to the same volatility estimate rather than re-deriving it
    from price data this module is not allowed to fetch.
    """
    stored = row.get("tradeability") if row is not None else None
    stored = stored if isinstance(stored, dict) else {}

    if not ladder_quotes or accuracy is None or not horizon_minutes or not underlying_price:
        return {
            "net_edge_pct": stored.get("net_edge_pct"),
            "is_tradeable": stored.get("is_tradeable"),
            "cost_model": "shares" if stored else None,
            "dominant_cost": None,
            "best_dte": None,
        }

    avg_move_pct = stored.get("avg_move_pct")
    if not avg_move_pct or avg_move_pct <= 0:
        return {
            "net_edge_pct": stored.get("net_edge_pct"),
            "is_tradeable": stored.get("is_tradeable"),
            "cost_model": "shares" if stored else None,
            "dominant_cost": None,
            "best_dte": None,
        }
    horizon_sigma = float(avg_move_pct) / 100.0 / 0.8

    from analysis.options_pricing import sweep_expiries

    sweep = sweep_expiries(
        float(accuracy), horizon_sigma,
        underlying_price=float(underlying_price),
        horizon_minutes=float(horizon_minutes),
        quotes=ladder_quotes,
    )
    best = sweep.get("best") or {}
    return {
        "net_edge_pct": best.get("net_edge_pct"),
        "is_tradeable": best.get("is_tradeable"),
        "cost_model": "options",
        "dominant_cost": best.get("dominant_cost"),
        "best_dte": sweep.get("best_dte"),
        "sweep": sweep,
    }


def _intraday_horizon(
    ticker: str,
    interval: str,
    interval_minutes: int,
    now_ts: pd.Timestamp,
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]],
    underlying_price: Optional[float],
) -> Dict[str, Any]:
    from analysis.intraday_prediction import (
        get_intraday_prediction_history,
        load_metadata,
        model_exists,
    )
    from analysis.prediction_performance import compute_intraday_prediction_metrics

    if not model_exists(ticker, interval):
        return _blank_horizon(interval, note="No model trained.")

    meta = load_metadata(ticker, interval)
    # resolve=False: grading needs a network fetch, and this module must not do
    # one. Whatever the last page view already resolved is what we report.
    history = get_intraday_prediction_history(ticker, interval, resolve=False)
    metrics = compute_intraday_prediction_metrics(ticker, interval, history=history)

    out = _blank_horizon(
        interval,
        has_model=True,
        trained_accuracy=meta.get("directional_accuracy"),
        live_accuracy=metrics.get("win_rate"),
        n_resolved=metrics.get("n_resolved", 0),
    )

    row = _latest_row(history)
    if row is None:
        out["note"] = "Trained, but no prediction logged yet."
        return out

    horizon_minutes = row.get("horizon_minutes") or meta.get("horizon_minutes")
    clock = intraday_expiry(
        row.get("bar_timestamp"), horizon_minutes or 0, interval_minutes, now=now_ts,
    )
    age = _age_minutes(row.get("date"), now_ts)

    out.update({
        "has_prediction": True,
        "direction": (row.get("direction") or None),
        "probability": row.get("probability"),
        "confidence": row.get("confidence"),
        "expires_at": clock["expires_at"],
        "expires_at_str": clock["expires_at_str"],
        "minutes_remaining": clock["minutes_remaining"],
        "is_expired": clock["is_expired"],
        "crosses_session_close": clock["crosses_session_close"],
        "is_gradeable": clock["is_gradeable"],
        "prediction_age_minutes": round(age, 1) if age is not None else None,
        "is_stale": bool(age is not None and age > interval_minutes),
    })
    if clock["reason"]:
        out["note"] = clock["reason"]

    out.update(_cost_verdict(
        row=row,
        accuracy=meta.get("directional_accuracy"),
        horizon_minutes=horizon_minutes,
        underlying_price=underlying_price,
        ladder_quotes=ladder_quotes,
    ))
    return out


def _daily_horizon(
    ticker: str,
    now_ts: pd.Timestamp,
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]],
    underlying_price: Optional[float],
) -> Dict[str, Any]:
    from analysis.ml_prediction import (
        _load_model_metadata,
        _rf_path,
        _xgb_path,
        get_prediction_history,
    )
    from analysis.prediction_performance import compute_daily_prediction_metrics

    if not (_xgb_path(ticker).exists() and _rf_path(ticker).exists()):
        return _blank_horizon(DAILY_KEY, note="No model trained.")

    meta = _load_model_metadata(ticker)
    history = get_prediction_history(ticker)
    metrics = compute_daily_prediction_metrics(ticker, history=history)

    out = _blank_horizon(
        DAILY_KEY,
        has_model=True,
        trained_accuracy=meta.get("directional_accuracy"),
        live_accuracy=metrics.get("win_rate"),
        n_resolved=metrics.get("n_resolved", 0),
    )

    row = _latest_row(history)
    if row is None:
        out["note"] = "Trained, but no prediction logged yet."
        return out

    horizon_days = row.get("horizon_days") or meta.get("horizon_days") or 5
    # No trading calendar here (fetching one is not allowed), so this is the
    # business-day approximation and daily_expiry() labels it as such.
    clock = daily_expiry(row.get("date"), int(horizon_days), now=now_ts)
    age = _age_minutes(row.get("date"), now_ts)

    out.update({
        "has_prediction": True,
        "direction": (row.get("direction") or None),
        "probability": row.get("probability"),
        "confidence": row.get("confidence"),
        "expires_at": clock["expires_on"],
        "expires_at_str": clock["expires_on_str"],
        "minutes_remaining": (
            clock["days_remaining"] * 1440 if clock["days_remaining"] is not None else None
        ),
        "is_expired": clock["is_expired"],
        "is_gradeable": clock["is_gradeable"],
        "prediction_age_minutes": round(age, 1) if age is not None else None,
        "is_stale": bool(age is not None and age > DAILY_STALE_AFTER_MINUTES),
        "expiry_method": clock["method"],
    })

    # The daily model stores no tradeability record, so a cost verdict is only
    # possible when the caller supplied live quotes. A 30 DTE contract against a
    # 5-day signal is a coherent trade, so this is worth computing when we can.
    if ladder_quotes and meta.get("directional_accuracy") and underlying_price:
        sigma_daily = meta.get("neutral_threshold")
        if sigma_daily:
            from analysis.options_pricing import sweep_expiries

            sweep = sweep_expiries(
                float(meta["directional_accuracy"]),
                float(sigma_daily) * (float(horizon_days) ** 0.5),
                underlying_price=float(underlying_price),
                horizon_minutes=float(horizon_days) * DAILY_STALE_AFTER_MINUTES,
                quotes=ladder_quotes,
            )
            best = sweep.get("best") or {}
            out.update({
                "net_edge_pct": best.get("net_edge_pct"),
                "is_tradeable": best.get("is_tradeable"),
                "cost_model": "options",
                "dominant_cost": best.get("dominant_cost"),
                "best_dte": sweep.get("best_dte"),
                "sweep": sweep,
            })
    return out


def _alignment(horizons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reduce the per-horizon rows to one read. Only live (non-expired) directional
    calls vote — an expired signal is not evidence, which is the whole premise
    of analysis/horizon_clock.py.
    """
    live = [
        h for h in horizons
        if h["has_prediction"] and not h["is_expired"] and h["direction"] in ("bullish", "bearish")
    ]
    n_bull = sum(1 for h in live if h["direction"] == "bullish")
    n_bear = sum(1 for h in live if h["direction"] == "bearish")
    n_neutral = sum(
        1 for h in horizons
        if h["has_prediction"] and not h["is_expired"] and h["direction"] == "neutral"
    )

    if n_bull and not n_bear:
        net = "bullish"
    elif n_bear and not n_bull:
        net = "bearish"
    elif n_bull or n_bear:
        net = "mixed"
    else:
        net = "none"

    tradeable = [h for h in live if h["is_tradeable"]]
    agreeing = [h for h in live if net in ("bullish", "bearish") and h["direction"] == net]

    # Shortest-horizon agreeing signal that also clears costs. Horizons arrive in
    # ascending order, so the first match is the tightest.
    tightest = next(
        (h["horizon"] for h in agreeing if h["is_tradeable"] and not h["crosses_session_close"]),
        None,
    )
    # The line that matters most: horizons pointing the same way that cannot pay
    # for themselves. Without this the view would quietly invite acting on them.
    uneconomic = [
        h["horizon"] for h in agreeing
        if h["is_tradeable"] is False or h["crosses_session_close"]
    ]

    return {
        "n_bullish": n_bull,
        "n_bearish": n_bear,
        "n_neutral": n_neutral,
        "n_tradeable_bullish": sum(1 for h in tradeable if h["direction"] == "bullish"),
        "n_tradeable_bearish": sum(1 for h in tradeable if h["direction"] == "bearish"),
        "net_direction": net,
        "is_unanimous": bool(len(live) >= 2 and net in ("bullish", "bearish") and len(agreeing) == len(live)),
        "tightest_tradeable": tightest,
        "agree_but_uneconomic": uneconomic,
    }


def build_consensus(
    ticker: str,
    *,
    include_daily: bool = True,
    ladder_quotes: Optional[Dict[Any, Dict[str, Any]]] = None,
    underlying_price: Optional[float] = None,
    now: Any = None,
) -> Dict[str, Any]:
    """
    One read across every horizon for `ticker`.

    Parameters
    ----------
    include_daily : include the daily/swing model as a context row.
    ladder_quotes : optional expiry-ladder quotes from
        `data.options_data.get_expiry_ladder_quotes(...)["quotes"]`. Supplied ->
        each horizon gets a real **options** cost verdict via `sweep_expiries`.
        Omitted -> each horizon falls back to the stored **shares** verdict, and
        `cost_model` says so. Passing them in (rather than fetching here) is what
        keeps this function free of network access.
    underlying_price : required alongside `ladder_quotes` to compute elasticity.
    now : injectable clock; defaults to `now_et()`.

    Returns
    -------
    {"ticker", "as_of", "horizons": [...], "alignment": {...}, "warnings": [...]}
    See the module docstring; every rate is paired with an N.
    """
    ticker = ticker.upper().strip()
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp(now_et())
    now_ts = now_ts.tz_localize(MARKET_TZ) if now_ts.tz is None else now_ts.tz_convert(MARKET_TZ)

    from analysis.intraday_prediction import INTERVAL_SPECS

    horizons: List[Dict[str, Any]] = []
    for interval, spec in INTERVAL_SPECS.items():
        try:
            horizons.append(_intraday_horizon(
                ticker, interval, int(spec["minutes"]), now_ts, ladder_quotes, underlying_price,
            ))
        except Exception as exc:
            logger.warning("build_consensus: %s %s failed: %s", ticker, interval, exc)
            horizons.append(_blank_horizon(interval, note=f"Could not read: {exc}"))

    if include_daily:
        try:
            horizons.append(_daily_horizon(ticker, now_ts, ladder_quotes, underlying_price))
        except Exception as exc:
            logger.warning("build_consensus: %s daily failed: %s", ticker, exc)
            horizons.append(_blank_horizon(DAILY_KEY, note=f"Could not read: {exc}"))

    warnings: List[str] = []
    untrained = [h["horizon"] for h in horizons if not h["has_model"]]
    if untrained:
        warnings.append(f"No model trained for: {', '.join(untrained)}.")
    stale = [h["horizon"] for h in horizons if h["is_stale"]]
    if stale:
        warnings.append(
            f"Prediction older than one bar for: {', '.join(stale)} — refresh before acting."
        )
    expired = [h["horizon"] for h in horizons if h["has_prediction"] and h["is_expired"]]
    if expired:
        warnings.append(f"Signal already past its horizon for: {', '.join(expired)}.")
    dead = [h["horizon"] for h in horizons if h["crosses_session_close"]]
    if dead:
        warnings.append(
            f"Horizon runs past the 4:00 PM close for: {', '.join(dead)} — those can never be graded."
        )
    thin = [
        h["horizon"] for h in horizons
        if h["has_model"] and 0 < (h["n_resolved"] or 0) < 20
    ]
    if thin:
        warnings.append(
            f"Fewer than 20 resolved predictions for: {', '.join(thin)} — live accuracy is not yet meaningful."
        )
    if ladder_quotes is None:
        warnings.append(
            "No options quotes supplied — cost verdicts fall back to the shares model, "
            "which ignores delta leverage and theta."
        )

    return {
        "ticker": ticker,
        "as_of": now_et_iso() if now is None else str(now_ts),
        "horizons": horizons,
        "alignment": _alignment(horizons),
        "warnings": warnings,
    }


# ── Agreement backtest (Roadmap Item 7) ──────────────────────────────────────

# Two predictions count as concurrent if their bar timestamps fall within this
# many minutes. Predictions are only logged when the user presses a button, so
# timestamps are irregular and will rarely line up exactly.
DEFAULT_JOIN_TOLERANCE_MINUTES = 30

# Below this many resolved pairs a bucket's accuracy is noise, and is reported as
# None rather than as a number that invites a conclusion.
MIN_PAIRS_PER_BUCKET = 10


def backtest_agreement(
    ticker: str,
    *,
    base_interval: str = "15m",
    confirm_interval: str = "30m",
    tolerance_minutes: float = DEFAULT_JOIN_TOLERANCE_MINUTES,
) -> Dict[str, Any]:
    """
    When two horizons agreed, was the base horizon more accurate than it was on
    its own?

    If agreement measurably helps, "wait for two-horizon confirmation" becomes a
    measured rule rather than a hunch — the same discipline ORBC's second-close
    rule encodes. For an options trader it is also the cheapest possible edge
    improvement, since waiting costs only theta while a better hit rate multiplies
    against leverage.

    **This is not a clean backtest, and must never be presented as one.**
    Predictions exist in the log only because someone pressed a button, so the
    sample is irregular in time and self-selected — you were more likely to
    generate a prediction when the setup already looked interesting. It measures
    a correlation in your own logged history, nothing stronger. Every returned
    bucket carries its own `n` for exactly this reason.

    Returns
    -------
    {
      "ticker", "base_interval", "confirm_interval", "tolerance_minutes",
      "n_base_resolved": int,
      "buckets": {
        "agree":      {"n", "accuracy"},   # confirm agreed with base
        "conflict":   {"n", "accuracy"},   # confirm called the opposite
        "unconfirmed":{"n", "accuracy"},   # no concurrent confirm prediction
      },
      "baseline_accuracy": float | None,   # base alone, all resolved rows
      "agreement_lift": float | None,      # agree - baseline, None if either thin
      "warnings": [str],
    }
    """
    from analysis.intraday_prediction import get_intraday_prediction_history

    out: Dict[str, Any] = {
        "ticker": ticker.upper().strip(),
        "base_interval": base_interval,
        "confirm_interval": confirm_interval,
        "tolerance_minutes": tolerance_minutes,
        "n_base_resolved": 0,
        "buckets": {
            "agree": {"n": 0, "accuracy": None},
            "conflict": {"n": 0, "accuracy": None},
            "unconfirmed": {"n": 0, "accuracy": None},
        },
        "baseline_accuracy": None,
        "agreement_lift": None,
        "warnings": [
            "Predictions are logged only when generated by hand, so this sample is "
            "irregular in time and self-selected. It is a correlation in your own "
            "history, not a controlled backtest.",
        ],
    }

    base = get_intraday_prediction_history(out["ticker"], base_interval, resolve=False)
    confirm = get_intraday_prediction_history(out["ticker"], confirm_interval, resolve=False)

    if base is None or base.empty:
        out["warnings"].append(f"No {base_interval} predictions logged.")
        return out

    base = base[base["correct"].notna() & base["direction"].isin(["bullish", "bearish"])].copy()
    if base.empty:
        out["warnings"].append(f"No resolved directional {base_interval} predictions yet.")
        return out

    base["correct"] = base["correct"].astype(bool)
    out["n_base_resolved"] = len(base)
    out["baseline_accuracy"] = round(float(base["correct"].mean()), 4)

    have_confirm = (
        confirm is not None and not confirm.empty
        and confirm["direction"].isin(["bullish", "bearish"]).any()
    )
    if not have_confirm:
        out["buckets"]["unconfirmed"] = {
            "n": len(base), "accuracy": out["baseline_accuracy"],
        }
        out["warnings"].append(
            f"No directional {confirm_interval} predictions to confirm against — every "
            f"{base_interval} row is unconfirmed."
        )
        return out

    confirm = confirm[confirm["direction"].isin(["bullish", "bearish"])].copy()

    # Nearest-in-time join on bar timestamp. merge_asof needs both sides sorted
    # ascending; the history functions return newest-first.
    tol = pd.Timedelta(minutes=tolerance_minutes)
    base_sorted = base.sort_values("date")
    confirm_sorted = confirm.sort_values("date")[["date", "direction"]].rename(
        columns={"direction": "confirm_direction"}
    )
    joined = pd.merge_asof(
        base_sorted, confirm_sorted, on="date", tolerance=tol, direction="nearest",
    )

    agree_mask = joined["confirm_direction"] == joined["direction"]
    conflict_mask = (
        joined["confirm_direction"].notna() & (joined["confirm_direction"] != joined["direction"])
    )
    unconfirmed_mask = joined["confirm_direction"].isna()

    for name, mask in (
        ("agree", agree_mask), ("conflict", conflict_mask), ("unconfirmed", unconfirmed_mask),
    ):
        subset = joined[mask]
        n = int(len(subset))
        out["buckets"][name] = {
            "n": n,
            "accuracy": (
                round(float(subset["correct"].mean()), 4)
                if n >= MIN_PAIRS_PER_BUCKET else None
            ),
        }
        if 0 < n < MIN_PAIRS_PER_BUCKET:
            out["warnings"].append(
                f"'{name}' bucket has only {n} pair(s) — under {MIN_PAIRS_PER_BUCKET}, "
                "so no accuracy is reported for it."
            )

    agree_acc = out["buckets"]["agree"]["accuracy"]
    if agree_acc is not None and out["baseline_accuracy"] is not None:
        out["agreement_lift"] = round(agree_acc - out["baseline_accuracy"], 4)

    return out
