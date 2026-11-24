"""
Meta-labeling: a second model that predicts whether the *first* model is right.

The problem it addresses
------------------------
The primary model (ml_prediction / intraday_prediction) answers "which way?" and
answers it on every bar it is asked about. It has no way to say "I don't know
this one." The only abstention it has is a dead band on the ensemble probability
— 0.45 to 0.55 — which is a threshold on confidence, not a judgement about
whether *this particular setup* is one the model handles well.

Meta-labeling, from Lopez de Prado (AFML ch. 3), separates the two questions:

    primary model   ->  direction        (which way)
    meta model      ->  take / skip      (is this call trustworthy)

The meta model is trained on a binary label that could not be simpler: **was the
primary model's call correct?** Its features are the same market state the
primary model saw, plus what the primary model said about it. So it learns the
conditions under which the primary model fails — chop, low ADX, a call that
contradicts the higher-timeframe trend — without needing to predict price itself.

Why this raises accuracy
------------------------
It does not make the primary model better at any individual call. It raises the
accuracy of the calls you *act on*, by declining the ones the primary model is
historically bad at. Coverage falls, precision rises. That trade is the point,
and it is why coverage is reported next to accuracy everywhere below: 80%
accuracy on 10% of signals is not obviously better than 60% on all of them, and
which you want depends on whether you are short of opportunities or short of
capital.

Direction, not dollars
----------------------
The label here is **"was the direction right"**, deliberately, because this app's
purpose is prediction accuracy. Labelling by "did the trade clear costs" is the
other standard choice, and it makes the meta model a trade filter rather than an
accuracy filter — a different product, coupled to spread, contract choice, and
expiry. Costs stay where they belong: displayed beside the signal, for the trade
decision. See assess_options_tradeability.

What it cannot fix
------------------
A meta model cannot manufacture signal. If the primary model is at baseline, its
errors are unpredictable and the meta model will find nothing — correctly. This
module is built to report that outcome plainly rather than dress it up: every
fit is scored out-of-sample against the do-nothing alternative of taking every
signal, and refuses to claim an improvement it cannot demonstrate.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# A meta model needs enough *errors* to learn from, not just enough rows. At 60%
# primary accuracy, 200 rows carries about 80 failures — the floor at which a
# handful of features can find structure rather than memorize noise.
MIN_META_SAMPLES: int = 200

# Below this the meta model is not consulted at inference: too few examples for
# its probability to mean anything.
MIN_META_SAMPLES_FOR_USE: int = 300

# A meta model that keeps almost everything is not filtering, and one that keeps
# almost nothing has overfit to a handful of failures. Outside this range the
# result is reported but flagged unusable.
MIN_USABLE_COVERAGE: float = 0.20
MAX_USABLE_COVERAGE: float = 0.95

# Precision must beat take-everything by at least this much, out of sample,
# before the filter is worth applying.
MIN_PRECISION_GAIN: float = 0.03


def _meta_config() -> Dict[str, Any]:
    """
    Deliberately smaller than the primary models.

    The meta model has one binary target, far fewer effective observations (only
    as many as the primary made calls on), and exists to find coarse structure —
    "this model is unreliable in chop" — not a fine decision surface. A deep
    forest here would memorize which specific days went wrong.
    """
    return {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }


def build_meta_features(
    X_primary: pd.DataFrame,
    primary_prob: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Features for the meta model: what the primary model saw, plus what it said.

    The primary model's own output is the important addition. `primary_conviction`
    is distance from 0.5, so it reads the same for a bullish 0.9 and a bearish
    0.1 — both are strong calls. Feeding raw probability instead would make the
    meta model learn "bearish calls fail more" when what is really there is "weak
    calls fail more", and those are different claims.
    """
    names = feature_names or list(X_primary.columns)
    out = X_primary[names].copy()
    prob = np.asarray(primary_prob, dtype="float64")
    out["primary_prob"] = prob
    out["primary_conviction"] = np.abs(prob - 0.5) * 2.0
    out["primary_is_bullish"] = (prob >= 0.5).astype("int8")
    return out


def build_meta_labels(
    primary_prob: np.ndarray,
    y_true_binary: np.ndarray,
) -> np.ndarray:
    """
    1 where the primary model called the direction correctly, else 0.

    Note what this is *not*: it is not the market direction. The meta model is
    indifferent to whether price rose — it predicts whether the primary model got
    it right, which is a different and easier question, because model failure has
    structure that price does not.
    """
    predicted = (np.asarray(primary_prob) >= 0.5).astype(int)
    return (predicted == np.asarray(y_true_binary)).astype(int)


def _precision_at_threshold(
    meta_prob: np.ndarray, meta_label: np.ndarray, threshold: float,
) -> Tuple[Optional[float], float, int]:
    """(precision on taken signals, coverage, n_taken) at one threshold."""
    taken = meta_prob >= threshold
    n_taken = int(taken.sum())
    coverage = n_taken / len(meta_prob) if len(meta_prob) else 0.0
    if n_taken == 0:
        return None, 0.0, 0
    return float(meta_label[taken].mean()), coverage, n_taken


def evaluate_meta_model(
    X_meta: pd.DataFrame,
    meta_labels: np.ndarray,
    n_splits: int = 5,
    gap: int = 5,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Walk-forward evaluation of the meta filter against taking every signal.

    The comparison is the whole point, so it is built into the return rather than
    left to the caller: `base_precision` is what you get by acting on every
    primary signal, and `by_threshold` shows precision and coverage at each
    cutoff. A meta model that cannot beat base_precision has found nothing, and
    saying so is a valid result.

    Same anchored TimeSeriesSplit with a gap as the primary models — the meta
    label inherits the primary's forward window, so the same overlap applies.

    Returns {"n", "base_precision", "by_threshold", "best", "usable", "reason"}.
    """
    thresholds = thresholds or [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    out: Dict[str, Any] = {
        "n": int(len(X_meta)), "base_precision": None, "by_threshold": [],
        "best": None, "usable": False, "reason": None,
    }

    if len(X_meta) < MIN_META_SAMPLES:
        out["reason"] = (
            f"{len(X_meta)} primary calls available — need at least "
            f"{MIN_META_SAMPLES} before a meta model can be fitted. It learns from "
            "the primary model's mistakes, so it needs enough of them."
        )
        return out

    labels = np.asarray(meta_labels).astype(int)
    if len(np.unique(labels)) < 2:
        out["reason"] = (
            "The primary model was right on every call in this sample, or wrong on "
            "every call. With one class there is nothing for a meta model to separate."
        )
        return out

    X_arr = X_meta.values.astype("float32")
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    oos_prob: List[np.ndarray] = []
    oos_label: List[np.ndarray] = []
    for train_idx, val_idx in tscv.split(X_arr):
        y_tr, y_val = labels[train_idx], labels[val_idx]
        if len(np.unique(y_tr)) < 2 or len(val_idx) < 10:
            continue
        model = RandomForestClassifier(**_meta_config())
        model.fit(X_arr[train_idx], y_tr)
        oos_prob.append(model.predict_proba(X_arr[val_idx])[:, 1])
        oos_label.append(y_val)

    if not oos_prob:
        out["reason"] = "Walk-forward produced no usable folds — need more history."
        return out

    prob = np.concatenate(oos_prob)
    label = np.concatenate(oos_label)
    base = float(label.mean())
    out["base_precision"] = round(base, 4)

    rows = []
    for t in thresholds:
        precision, coverage, n_taken = _precision_at_threshold(prob, label, t)
        rows.append({
            "threshold": t,
            "precision": round(precision, 4) if precision is not None else None,
            "coverage": round(coverage, 4),
            "n_taken": n_taken,
            "gain": round(precision - base, 4) if precision is not None else None,
        })
    out["by_threshold"] = rows

    # Pick among thresholds that keep a sane share of signals. Taking the raw
    # argmax over precision would always drift to the highest cutoff, where two
    # surviving signals happen to have both been right.
    viable = [
        r for r in rows
        if r["precision"] is not None
        and MIN_USABLE_COVERAGE <= r["coverage"] <= MAX_USABLE_COVERAGE
    ]
    if not viable:
        out["reason"] = (
            "No threshold keeps a workable share of signals — the filter either "
            "passes nearly everything or almost nothing, so it is not separating "
            "good calls from bad ones."
        )
        return out

    best = max(viable, key=lambda r: r["gain"])
    out["best"] = best

    if best["gain"] < MIN_PRECISION_GAIN:
        out["reason"] = (
            f"Best out-of-sample gain is {best['gain']:+.1%} over the "
            f"{base:.1%} you get by taking every signal — under the "
            f"{MIN_PRECISION_GAIN:.0%} needed to be worth filtering. The primary "
            "model's errors do not look predictable from this feature set."
        )
        return out

    out["usable"] = True
    out["reason"] = (
        f"At threshold {best['threshold']:.2f}: {best['precision']:.1%} precision on "
        f"{best['coverage']:.0%} of signals, versus {base:.1%} taking all of them "
        f"({best['gain']:+.1%}). Fewer trades, better ones."
    )
    return out


def fit_meta_model(
    X_meta: pd.DataFrame,
    meta_labels: np.ndarray,
) -> Optional[RandomForestClassifier]:
    """Fit on everything, for deployment. Returns None when there is too little."""
    labels = np.asarray(meta_labels).astype(int)
    if len(X_meta) < MIN_META_SAMPLES or len(np.unique(labels)) < 2:
        return None
    model = RandomForestClassifier(**_meta_config())
    model.fit(X_meta.values.astype("float32"), labels)
    return model


def meta_verdict(
    model: Optional[RandomForestClassifier],
    X_meta_row: pd.DataFrame,
    threshold: float,
    n_training_samples: int,
) -> Dict[str, Any]:
    """
    Apply the filter to one live signal.

    Defaults to taking the signal whenever the meta model is missing, untrained,
    or fitted on too little. A filter that silently suppresses signals because it
    failed to load would be worse than no filter — the failure has to be visible,
    and the safe direction is to leave the primary model's call alone.
    """
    out = {
        "take": True, "meta_probability": None, "threshold": threshold,
        "reason": "No meta model — primary signal passed through unfiltered.",
    }
    if model is None:
        return out
    if n_training_samples < MIN_META_SAMPLES_FOR_USE:
        out["reason"] = (
            f"Meta model trained on {n_training_samples} calls (need "
            f"{MIN_META_SAMPLES_FOR_USE} to act on it) — signal passed through."
        )
        return out
    try:
        prob = float(model.predict_proba(X_meta_row.values.astype("float32"))[0, 1])
    except Exception as exc:
        logger.warning("meta_verdict: inference failed (%s) — passing signal through", exc)
        out["reason"] = f"Meta model inference failed ({exc}) — signal passed through."
        return out

    out["meta_probability"] = round(prob, 4)
    out["take"] = prob >= threshold
    out["reason"] = (
        f"Meta model rates this setup {prob:.0%} likely to be a correct call "
        f"(threshold {threshold:.0%}) — "
        + ("take it." if out["take"] else "skip it; the primary model is historically "
                                          "unreliable in conditions like these.")
    )
    return out
