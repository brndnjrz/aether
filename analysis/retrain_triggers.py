"""
Retraining trigger checks (Prediction Improvement Engine, Phase 8) — pure
functions over already-persisted storage, Streamlit-free. Centralizes the
staleness check pages/trading.py's _retrain_overdue() used to own alone
(now reading config.settings.RETRAIN_STALENESS_DAYS instead of a bare
hardcoded 30), and adds two more signals: a live-vs-trained accuracy drop,
and a coarse market-volatility flag.
"""
import logging
import time
from typing import Any, Dict, Optional

from config.settings import (
    RETRAIN_STALENESS_DAYS,
    RETRAIN_ACCURACY_DROP_THRESHOLD,
    RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK,
)

logger = logging.getLogger(__name__)


def check_staleness_trigger(ticker: str, *, interval: Optional[str] = None) -> Dict[str, Any]:
    """
    Age of the deployed model, in days — prefers the accuracy.json's mtime,
    falling back to the .pkl's mtime if no accuracy file exists yet.
    interval=None checks the daily model; otherwise the interval-scoped
    intraday model. {"triggered", "age_days", "threshold_days", "reason"}.
    """
    ticker = ticker.upper()
    if interval is None:
        from analysis.ml_prediction import _xgb_path, _STORAGE_DIR
        acc_path = _STORAGE_DIR / f"{ticker}_accuracy.json"
        check_path = acc_path if acc_path.exists() else _xgb_path(ticker)
    else:
        from analysis.intraday_prediction import _accuracy_path, _xgb_path
        acc_path = _accuracy_path(ticker, interval)
        check_path = acc_path if acc_path.exists() else _xgb_path(ticker, interval)

    if not check_path.exists():
        return {
            "triggered": False, "age_days": None,
            "threshold_days": RETRAIN_STALENESS_DAYS, "reason": "No model trained yet.",
        }

    try:
        age_days = (time.time() - check_path.stat().st_mtime) / 86400
    except Exception as exc:
        logger.debug("check_staleness_trigger: could not stat %s: %s", check_path, exc)
        return {
            "triggered": False, "age_days": None,
            "threshold_days": RETRAIN_STALENESS_DAYS, "reason": "Could not determine model age.",
        }

    triggered = age_days > RETRAIN_STALENESS_DAYS
    reason = (
        f"Model is {age_days:.0f} day(s) old (> {RETRAIN_STALENESS_DAYS})."
        if triggered else
        f"Model is {age_days:.0f} day(s) old (<= {RETRAIN_STALENESS_DAYS})."
    )
    return {
        "triggered": triggered, "age_days": round(age_days, 1),
        "threshold_days": RETRAIN_STALENESS_DAYS, "reason": reason,
    }


def check_performance_drop_trigger(ticker: str, *, interval: Optional[str] = None) -> Dict[str, Any]:
    """
    Compares the model's persisted trained-in directional_accuracy against
    the LIVE win_rate from analysis.prediction_performance — never
    re-derives correctness, reads it via the same compute_*_prediction_metrics()
    chain Phases 1/2 already established.
    {"triggered", "trained_accuracy", "live_accuracy", "n_resolved", "reason"}.
    """
    ticker = ticker.upper()
    if interval is None:
        from analysis.ml_prediction import _load_model_metadata
        from analysis.prediction_performance import compute_daily_prediction_metrics
        trained_accuracy = _load_model_metadata(ticker).get("directional_accuracy")
        metrics = compute_daily_prediction_metrics(ticker)
    else:
        from analysis.intraday_prediction import load_metadata
        from analysis.prediction_performance import compute_intraday_prediction_metrics
        trained_accuracy = load_metadata(ticker, interval).get("directional_accuracy")
        metrics = compute_intraday_prediction_metrics(ticker, interval)

    live_accuracy = metrics.get("win_rate")
    n_resolved = metrics.get("n_resolved", 0)

    out: Dict[str, Any] = {
        "triggered": False, "trained_accuracy": trained_accuracy,
        "live_accuracy": live_accuracy, "n_resolved": n_resolved, "reason": None,
    }
    if trained_accuracy is None or live_accuracy is None:
        out["reason"] = "Not enough data to compare trained-in vs. live accuracy."
        return out
    if n_resolved < RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK:
        out["reason"] = (
            f"Only {n_resolved} resolved prediction(s) — need >= "
            f"{RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK} before judging a drop."
        )
        return out

    gap = trained_accuracy - live_accuracy
    out["triggered"] = gap >= RETRAIN_ACCURACY_DROP_THRESHOLD
    out["reason"] = (
        f"Live accuracy {live_accuracy * 100:.1f}% is {gap * 100:.1f} points below "
        f"the trained-in {trained_accuracy * 100:.1f}% "
        f"(threshold {RETRAIN_ACCURACY_DROP_THRESHOLD * 100:.0f} points)."
        if out["triggered"] else
        f"Live accuracy {live_accuracy * 100:.1f}% is within "
        f"{RETRAIN_ACCURACY_DROP_THRESHOLD * 100:.0f} points of the trained-in "
        f"{trained_accuracy * 100:.1f}%."
    )
    return out


def check_regime_change_trigger(ticker: str, df=None) -> Dict[str, Any]:
    """
    Honest-scope version: there is no persisted "regime at train time" to
    compare against — data.macro_data.get_vix_data() only returns the
    CURRENT regime, not a historical label — so this is NOT a true
    before/after detector. "triggered" is conservative: True only when the
    CURRENT VIX regime is "Elevated Fear"/"Crisis" (a real,
    currently-decidable condition). The ticker's own regime_markov
    Bull/Bear state (if `df` is given) is surfaced as informational
    context only, never as its own trigger — a real regime-*change*
    detector needs Phase 2's indicator_snapshot to have been collecting
    trained-at VIX regimes for a while first.
    """
    out: Dict[str, Any] = {
        "triggered": False, "vix_regime": None, "vix_elevated": False,
        "ticker_regime_state": None, "reason": None,
    }
    try:
        from data.macro_data import get_vix_data
        vix = get_vix_data()
    except Exception as exc:
        logger.debug("check_regime_change_trigger: VIX lookup failed: %s", exc)
        out["reason"] = "Could not fetch VIX data."
        return out

    regime = vix.get("regime")
    out["vix_regime"] = regime
    out["vix_elevated"] = regime in ("Elevated Fear", "Crisis")
    out["triggered"] = out["vix_elevated"]

    if df is not None:
        try:
            from analysis.regime_markov import analyze_regime_markov
            markov = analyze_regime_markov(df, ticker)
            if markov.get("available"):
                out["ticker_regime_state"] = markov.get("current_state")
        except Exception as exc:
            logger.debug("check_regime_change_trigger: regime_markov lookup failed: %s", exc)

    out["reason"] = (
        f"VIX regime is {regime} — market-wide volatility is elevated."
        if out["triggered"] else
        f"VIX regime is {regime} — no market-wide volatility flag."
    )
    return out


def check_all_retrain_triggers(
    ticker: str, *, interval: Optional[str] = None, df=None,
) -> Dict[str, Any]:
    """Runs all three checks. {"ticker", "interval", "should_retrain": True if
    ANY triggered, "triggers": {"staleness", "performance_drop", "regime_change"}}."""
    staleness = check_staleness_trigger(ticker, interval=interval)
    performance_drop = check_performance_drop_trigger(ticker, interval=interval)
    regime_change = check_regime_change_trigger(ticker, df=df)

    return {
        "ticker": ticker.upper(),
        "interval": interval,
        "should_retrain": staleness["triggered"] or performance_drop["triggered"] or regime_change["triggered"],
        "triggers": {
            "staleness": staleness,
            "performance_drop": performance_drop,
            "regime_change": regime_change,
        },
    }
