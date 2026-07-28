"""
ML direction prediction service for Aether.

Architecture
------------
Two-model ensemble:
  - XGBoostClassifier  (primary, binary:logistic objective, outputs probability)
  - RandomForestClassifier (secondary calibration / ensemble member)

Ensemble bull probability:
  P_bull = 0.65 * xgb_prob + 0.35 * rf_prob

Both models predict binary direction on a ternary-labelled dataset:
  y label mapping for training:
    1  (bullish)  → binary class 1
   -1  (bearish)  → binary class 0
    0  (neutral)  → EXCLUDED from training (noisy flat-return rows)

Persistence
-----------
Models are saved to:
  storage/{TICKER}_xgb.pkl   — serialised XGBClassifier (or RF fallback)
  storage/{TICKER}_rf.pkl    — serialised RandomForestClassifier

Predictions are appended to:
  storage/{TICKER}_predictions.jsonl  — one JSON object per line

Walk-forward validation
-----------------------
TimeSeriesSplit(n_splits=10, gap=horizon_days)  — anchored expanding window.
The gap always equals the label horizon so the forward-return target never
bleeds into training features.

Per-ticker auto-tuning
-----------------------
train_model() no longer trains against one fixed label definition and one
fixed hyperparameter set. Instead it runs two small grid searches, each
scored with a cheaper reduced-fold walk-forward (see _SEARCH_N_SPLITS):

  1. select_label_scheme() tries a few (horizon_days, neutral_threshold)
     combinations (LABEL_SEARCH_GRID) and keeps whichever the ensemble is
     most consistently accurate at predicting for this ticker.
  2. select_hyperparams() tries a few XGBoost hyperparameter overrides
     (HYPERPARAM_SEARCH_GRID, first entry = current defaults as baseline)
     on top of the winning label scheme.

The winning combination is then re-validated with the full n_splits=10
walk-forward for official reporting, and the final models are trained on
it. horizon_days / neutral_threshold / hyperparam_overrides are persisted
in {ticker}_accuracy.json so predict() and evaluate_model() can reuse the
exact configuration a model was trained with. Models trained before this
existed have no such keys — every reader defaults to horizon_days=5,
neutral_threshold=0.005, hyperparam_overrides={} for backward compatibility.

A model is considered reliable when:
  - mean directional accuracy >= 0.52
  - std-dev of accuracy across folds <= 0.08

Dependencies
-----------
  scikit-learn>=1.3.0
  xgboost>=1.7.0        (add to requirements.txt if not present)
  joblib                (ships with scikit-learn)

If xgboost is unavailable, the module degrades gracefully — XGBoost is
replaced by a second RandomForestClassifier with a warning.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.tz import now_et_iso
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# XGBoost: optional but strongly preferred.
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed. Install with: pip install xgboost>=1.7.0. "
        "Falling back to a second RandomForestClassifier (degraded performance).",
        ImportWarning,
        stacklevel=2,
    )

from data.feature_engineering import (
    build_features,
    build_predict_row,
    class_balance_check,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# ── Storage paths ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STORAGE_DIR = _PROJECT_ROOT / "storage"
_STORAGE_DIR.mkdir(exist_ok=True)

# Allow the app-wide STORAGE_DIR setting to override the computed path.
try:
    from config.settings import STORAGE_DIR as _SETTINGS_STORAGE_DIR
    _STORAGE_DIR = Path(_SETTINGS_STORAGE_DIR)
    _STORAGE_DIR.mkdir(exist_ok=True)
except Exception:
    pass


def _xgb_path(ticker: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_xgb.pkl"


def _rf_path(ticker: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_rf.pkl"


def _predictions_path(ticker: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_predictions.jsonl"


# ── Model configuration ───────────────────────────────────────────────────────

def _xgb_config(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    XGBoost hyperparameters for daily financial direction classification.

    Key regularisation decisions:
    - max_depth=4: shallow trees prevent memorising individual bars
    - min_child_weight=10: prevents leaves fit to <10 samples (most important)
    - subsample=0.8, colsample_bytree=0.8: row/column bagging for diversity
    - learning_rate=0.05 with n_estimators=200: slow learning, adequate capacity
    """
    return {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _rf_config(n_features: int = 18) -> Dict[str, Any]:
    """
    RandomForest calibration-check ensemble member.
    Conservative regularisation (min_samples_leaf=20) to complement XGBoost.
    """
    return {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }


# ── Auto-tuning search grids ───────────────────────────────────────────────────
# Candidate (horizon_days, neutral_threshold) label definitions. Tried in order;
# the first entry (5, 0.005) matches the app's original fixed defaults, so a
# ticker only moves away from it when another combo scores measurably better.
LABEL_SEARCH_GRID: List[Tuple[int, float]] = [
    (5, 0.005),
    (3, 0.004),
    (10, 0.008),
]

# Candidate XGBoost hyperparameter overrides, merged on top of _xgb_config()'s
# defaults. The first entry ({}) is the current default config, kept as the
# baseline so a search can never do worse than not searching at all.
HYPERPARAM_SEARCH_GRID: List[Dict[str, Any]] = [
    {},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300},
    {"max_depth": 5, "learning_rate": 0.08, "n_estimators": 150},
]

# Reduced fold count used only during the search phase — cheaper than the
# full n_splits=10 walk-forward used for final reporting.
_SEARCH_N_SPLITS: int = 4

# A search candidate must clear this bar to be eligible to win at all;
# otherwise the grid falls back to the first (default) entry.
_MIN_SEARCH_ACCURACY: float = 0.50


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_binary_labels(y: pd.Series) -> np.ndarray:
    """Convert {-1, 1} direction labels to {0, 1} for sklearn/XGBoost.
    Neutral (0) rows must be removed before calling this function."""
    return ((y + 1) // 2).values.astype(int)   # -1 → 0, 1 → 1


def _directional_accuracy(y_true_binary: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Fraction of predictions where predicted direction matched actual direction."""
    predicted_class = (y_pred_prob >= 0.5).astype(int)
    return float(np.mean(predicted_class == y_true_binary))


def _filter_directional(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove neutral (0) rows from the training set for binary classification."""
    mask = y != 0
    return X.loc[mask], y.loc[mask]


def _run_walk_forward(
    X: pd.DataFrame,
    y_dir: pd.Series,
    xgb_cfg: Dict[str, Any],
    rf_cfg: Dict[str, Any],
    n_splits: int = 10,
    gap: int = 5,
) -> Dict[str, Any]:
    """
    Run anchored walk-forward validation. Returns a summary dict.
    y_dir contains {-1, 0, 1} labels; neutral rows are excluded per fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X_arr = X.values.astype("float32")
    y_arr = y_dir.values

    fold_accs: List[float] = []
    fold_aucs: List[float] = []
    total_val_samples = 0

    for train_idx, val_idx in tscv.split(X_arr):
        if len(val_idx) < 10:
            continue

        # Filter neutrals
        y_train_all = y_arr[train_idx]
        y_val_all = y_arr[val_idx]

        train_dir_mask = y_train_all != 0
        val_dir_mask = y_val_all != 0

        if train_dir_mask.sum() < 20 or val_dir_mask.sum() < 5:
            continue

        X_train = X_arr[train_idx][train_dir_mask].astype("float32")
        y_train = _to_binary_labels(pd.Series(y_train_all[train_dir_mask]))
        X_val = X_arr[val_idx][val_dir_mask].astype("float32")
        y_val = _to_binary_labels(pd.Series(y_val_all[val_dir_mask]))

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        # Train XGBoost (or fallback RF)
        if _XGBOOST_AVAILABLE:
            xgb_m = XGBClassifier(**xgb_cfg)
            xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            xgb_probs = xgb_m.predict_proba(X_val)[:, 1]
        else:
            rf_fb = RandomForestClassifier(**rf_cfg)
            rf_fb.fit(X_train, y_train)
            xgb_probs = rf_fb.predict_proba(X_val)[:, 1]

        rf_m = RandomForestClassifier(**rf_cfg)
        rf_m.fit(X_train, y_train)
        rf_probs = rf_m.predict_proba(X_val)[:, 1]

        ensemble = 0.65 * xgb_probs + 0.35 * rf_probs

        fold_accs.append(_directional_accuracy(y_val, ensemble))
        try:
            fold_aucs.append(float(roc_auc_score(y_val, ensemble)))
        except ValueError:
            fold_aucs.append(0.5)
        total_val_samples += len(y_val)

    if not fold_accs:
        return {
            "n_folds": 0,
            "mean_directional_accuracy": 0.0,
            "std_directional_accuracy": 0.0,
            "mean_auc": 0.5,
            "n_validation_samples": 0,
            "is_reliable": False,
            "reliability_reason": "Walk-forward produced no valid folds — need more data",
            "fold_accuracies": [],
        }

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_auc = float(np.mean(fold_aucs))
    is_reliable = mean_acc >= 0.52 and std_acc <= 0.08

    if is_reliable:
        reason = (
            f"Consistent across {len(fold_accs)} folds — "
            f"mean accuracy {mean_acc:.1%} ± {std_acc:.1%}"
        )
    elif mean_acc < 0.52:
        reason = (
            f"Below minimum threshold — mean accuracy {mean_acc:.1%} "
            f"(need >=52%). Treat signal as weak."
        )
    else:
        reason = (
            f"High variance across folds — std {std_acc:.1%} (need <=8%). "
            f"Model is unstable across market regimes."
        )

    return {
        "n_folds": len(fold_accs),
        "mean_directional_accuracy": round(mean_acc, 4),
        "std_directional_accuracy": round(std_acc, 4),
        "mean_auc": round(mean_auc, 4),
        "n_validation_samples": total_val_samples,
        "is_reliable": is_reliable,
        "reliability_reason": reason,
        "fold_accuracies": [round(a, 4) for a in fold_accs],
    }


def _load_model_metadata(ticker: str) -> Dict[str, Any]:
    """
    Read {ticker}_accuracy.json and backfill defaults for keys that older
    models (trained before label-search/hyperparameter-search existed)
    never wrote.

    Returns a dict always containing at least: horizon_days, neutral_threshold,
    hyperparam_overrides — safe to use even if the accuracy file is missing.
    """
    from data.feature_engineering import FORWARD_BARS, NEUTRAL_THRESHOLD

    meta: Dict[str, Any] = {}
    acc_log_path = _STORAGE_DIR / f"{ticker.upper()}_accuracy.json"
    if acc_log_path.exists():
        try:
            with open(acc_log_path) as f_log:
                meta = json.load(f_log)
        except Exception:
            meta = {}

    meta.setdefault("horizon_days", FORWARD_BARS)
    meta.setdefault("neutral_threshold", NEUTRAL_THRESHOLD)
    meta.setdefault("hyperparam_overrides", {})
    return meta


def select_label_scheme(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Search LABEL_SEARCH_GRID for the (horizon_days, neutral_threshold) combo
    this ticker's ensemble predicts most consistently, using a reduced-fold
    walk-forward (_SEARCH_N_SPLITS) to keep the search fast.

    Falls back to the grid's first entry (the original fixed defaults) if no
    candidate clears _MIN_SEARCH_ACCURACY.

    Returns
    -------
    dict with keys: horizon_days, neutral_threshold, X_dir_18, y_dir,
    candidates (list of {horizon_days, neutral_threshold, mean_accuracy,
    std_accuracy, n_folds} for every combo tried, best first).
    """
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for horizon_days, neutral_threshold in LABEL_SEARCH_GRID:
        try:
            X, y = build_features(
                df, ticker=ticker,
                forward_bars=horizon_days, neutral_threshold=neutral_threshold,
            )
        except ValueError as exc:
            logger.debug(f"select_label_scheme: {ticker} skipped horizon_days={horizon_days} neutral_threshold={neutral_threshold}: {exc}")
            continue

        X_dir, y_dir = _filter_directional(X, y)
        if len(X_dir) < 50:
            continue

        X_dir_18 = X_dir[FEATURE_NAMES]
        balance = class_balance_check(y_dir)
        xgb_cfg = _xgb_config(scale_pos_weight=balance["recommended_scale_pos_weight"])
        rf_cfg = _rf_config(n_features=len(FEATURE_NAMES))
        wf = _run_walk_forward(
            X_dir_18, y_dir, xgb_cfg, rf_cfg,
            n_splits=_SEARCH_N_SPLITS, gap=horizon_days,
        )

        entry = {
            "horizon_days": horizon_days,
            "neutral_threshold": neutral_threshold,
            "mean_accuracy": wf["mean_directional_accuracy"],
            "std_accuracy": wf["std_directional_accuracy"],
            "n_folds": wf["n_folds"],
            "X_dir_18": X_dir_18,
            "y_dir": y_dir,
        }
        candidates.append(entry)

        if wf["n_folds"] == 0:
            continue
        if best is None or entry["mean_accuracy"] > best["mean_accuracy"]:
            best = entry

    if best is None or best["mean_accuracy"] < _MIN_SEARCH_ACCURACY:
        logger.warning(f"select_label_scheme: {ticker} no candidate cleared {_MIN_SEARCH_ACCURACY} accuracy — falling back to default label scheme")
        default_horizon, default_threshold = LABEL_SEARCH_GRID[0]
        X, y = build_features(
            df, ticker=ticker,
            forward_bars=default_horizon, neutral_threshold=default_threshold,
        )
        X_dir, y_dir = _filter_directional(X, y)
        best = {
            "horizon_days": default_horizon,
            "neutral_threshold": default_threshold,
            "X_dir_18": X_dir[FEATURE_NAMES],
            "y_dir": y_dir,
        }

    return {
        "horizon_days": best["horizon_days"],
        "neutral_threshold": best["neutral_threshold"],
        "X_dir_18": best["X_dir_18"],
        "y_dir": best["y_dir"],
        "candidates": sorted(
            [{k: v for k, v in c.items() if k not in ("X_dir_18", "y_dir")} for c in candidates],
            key=lambda c: c.get("mean_accuracy", 0.0), reverse=True,
        ),
    }


def select_hyperparams(
    X_dir_18: pd.DataFrame,
    y_dir: pd.Series,
    scale_pos_weight: float,
    gap: int,
) -> Dict[str, Any]:
    """
    Search HYPERPARAM_SEARCH_GRID for the XGBoost override set that scores
    best on the winning label scheme's data, using the same reduced-fold
    walk-forward as select_label_scheme().

    The first grid entry ({}) is the current default config, so this search
    can never choose worse-than-baseline hyperparameters.

    Returns
    -------
    dict with keys: overrides (the winning dict, possibly {}), candidates
    (list of {overrides, mean_accuracy, std_accuracy, n_folds}, best first).
    """
    rf_cfg = _rf_config(n_features=len(FEATURE_NAMES))
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for overrides in HYPERPARAM_SEARCH_GRID:
        xgb_cfg = {**_xgb_config(scale_pos_weight=scale_pos_weight), **overrides}
        wf = _run_walk_forward(
            X_dir_18, y_dir, xgb_cfg, rf_cfg,
            n_splits=_SEARCH_N_SPLITS, gap=gap,
        )
        entry = {
            "overrides": overrides,
            "mean_accuracy": wf["mean_directional_accuracy"],
            "std_accuracy": wf["std_directional_accuracy"],
            "n_folds": wf["n_folds"],
        }
        candidates.append(entry)

        if wf["n_folds"] == 0:
            continue
        if best is None or entry["mean_accuracy"] > best["mean_accuracy"]:
            best = entry

    if best is None or best["mean_accuracy"] < _MIN_SEARCH_ACCURACY:
        logger.warning(f"select_hyperparams: no candidate cleared {_MIN_SEARCH_ACCURACY} accuracy — falling back to default hyperparameters")
        best = {"overrides": HYPERPARAM_SEARCH_GRID[0]}

    return {
        "overrides": best["overrides"],
        "candidates": sorted(candidates, key=lambda c: c.get("mean_accuracy", 0.0), reverse=True),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(ticker: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Train the XGBoost + RandomForest ensemble on price data for a given ticker.

    If df is None, the function fetches 2 years of daily price history using
    get_price_history() and runs calculate_indicators() before feature engineering.

    Before fitting, this runs select_label_scheme() and select_hyperparams()
    to auto-tune the label horizon/threshold and XGBoost hyperparameters for
    this specific ticker (see module docstring). This makes training slower
    (multiple reduced-fold walk-forwards instead of one) but each ticker gets
    whichever definition/config its own price history rewards most.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    df : pd.DataFrame, optional
        Pre-fetched DataFrame that has already been passed through
        calculate_indicators(). If None, data is fetched automatically.

    Returns
    -------
    dict with keys:
        directional_accuracy : float  — mean WF validation accuracy
        accuracy_std : float          — std-dev across folds
        mean_auc : float
        n_train : int                 — directional samples used for final model
        n_test : int                  — total WF validation samples
        trained_at : str              — ISO timestamp
        is_reliable : bool
        reliability_reason : str
        horizon_days : int            — winning label lookahead (3, 5, or 10)
        neutral_threshold_pct : float — winning neutral zone, as a percent
        hyperparam_overrides : dict   — winning XGBoost overrides ({} = defaults)
        label_search : list           — every label combo tried, best first
        hyperparam_search : list      — every hyperparameter combo tried, best first
        error : str or None
    """
    ticker = ticker.upper()
    result: Dict[str, Any] = {
        "ticker": ticker,
        "directional_accuracy": None,
        "accuracy_std": None,
        "mean_auc": None,
        "n_train": 0,
        "n_test": 0,
        "trained_at": None,
        "is_reliable": False,
        "reliability_reason": "",
        "horizon_days": None,
        "neutral_threshold_pct": None,
        "hyperparam_overrides": {},
        "label_search": [],
        "hyperparam_search": [],
        "error": None,
    }

    # ── Fetch data if not provided ────────────────────────────────────────────
    if df is None:
        try:
            from data.price_data import get_price_history
            from analysis.indicators import calculate_indicators
            df_raw = get_price_history(ticker, period="2y")
            if df_raw is None or df_raw.empty:
                logger.warning(f"train_model: no price data available for {ticker}")
                result["error"] = f"No price data available for {ticker}"
                return result
            df = calculate_indicators(df_raw)
        except Exception as exc:
            logger.error(f"train_model: data fetch failed for {ticker}: {exc}")
            result["error"] = f"Data fetch failed: {exc}"
            return result

    logger.info("train_model: starting for %s (%d bars)", ticker, len(df) if df is not None else 0)

    # ── Search for the best label horizon/threshold for this ticker ───────────
    # Catches both ValueError (raised deliberately by build_features() for
    # too-few-rows) and KeyError (raised by pandas when df is missing expected
    # indicator columns, e.g. a caller passed raw OHLCV without ever running
    # it through calculate_indicators()). Both are caller-input problems, not
    # bugs in the search itself, so both should degrade to a structured error
    # dict rather than propagate as an uncaught exception.
    try:
        label_choice = select_label_scheme(df, ticker)
    except (ValueError, KeyError) as exc:
        logger.warning(f"train_model: label scheme search failed for {ticker}: {exc}")
        if isinstance(exc, KeyError):
            result["error"] = (
                f"Missing expected column {exc}. df must be the output of "
                "calculate_indicators() — raw OHLCV is not sufficient."
            )
        else:
            result["error"] = str(exc)
        return result

    horizon_days = label_choice["horizon_days"]
    neutral_threshold = label_choice["neutral_threshold"]
    X_dir_18 = label_choice["X_dir_18"]
    y_dir = label_choice["y_dir"]

    if len(X_dir_18) < 50:
        logger.warning(f"train_model: only {len(X_dir_18)} directional samples for {ticker} after neutral-zone removal — need >=50")
        result["error"] = (
            f"Only {len(X_dir_18)} directional samples for {ticker} after neutral-zone removal. "
            "Provide at least 250 bars of price data."
        )
        return result

    logger.info(
        "train_model: %s label scheme selected — horizon_days=%d neutral_threshold=%.3f",
        ticker, horizon_days, neutral_threshold,
    )

    balance = class_balance_check(y_dir)
    spw = balance["recommended_scale_pos_weight"]

    # ── Search for the best XGBoost hyperparameters on the winning label scheme ─
    hp_choice = select_hyperparams(X_dir_18, y_dir, scale_pos_weight=spw, gap=horizon_days)
    hp_overrides = hp_choice["overrides"]
    logger.info("train_model: %s hyperparam overrides selected — %s", ticker, hp_overrides or "defaults")

    xgb_cfg = {**_xgb_config(scale_pos_weight=spw), **hp_overrides}
    rf_cfg = _rf_config(n_features=len(FEATURE_NAMES))

    # ── Full walk-forward validation BEFORE fitting the final model ───────────
    logger.info("train_model: running full walk-forward validation for %s", ticker)
    wf = _run_walk_forward(X_dir_18, y_dir, xgb_cfg, rf_cfg, n_splits=10, gap=horizon_days)
    logger.info(
        "train_model: validation complete — mean_acc=%.3f std=%.3f reliable=%s",
        wf["mean_directional_accuracy"],
        wf["std_directional_accuracy"],
        wf["is_reliable"],
    )

    # ── Final model trained on ALL directional data (18 core features only) ───
    X_arr = X_dir_18.values.astype("float32")
    y_binary = _to_binary_labels(y_dir)

    if _XGBOOST_AVAILABLE:
        xgb_final = XGBClassifier(**xgb_cfg)
        xgb_final.fit(X_arr, y_binary, verbose=False)
    else:
        xgb_final = RandomForestClassifier(**rf_cfg)
        xgb_final.fit(X_arr, y_binary)

    rf_final = RandomForestClassifier(**rf_cfg)
    rf_final.fit(X_arr, y_binary)

    # ── Persist models ────────────────────────────────────────────────────────
    now_iso = now_et_iso()
    try:
        joblib.dump(xgb_final, _xgb_path(ticker))
        joblib.dump(rf_final, _rf_path(ticker))
        # Also persist the accuracy metrics so predict() can load them without retraining
        acc_record = {
            "ticker": ticker,
            "directional_accuracy": wf["mean_directional_accuracy"],
            "accuracy_std": wf["std_directional_accuracy"],
            "mean_auc": wf["mean_auc"],
            "is_reliable": wf["is_reliable"],
            "trained_at": now_iso,
            "horizon_days": horizon_days,
            "neutral_threshold": neutral_threshold,
            "hyperparam_overrides": hp_overrides,
        }
        with open(_STORAGE_DIR / f"{ticker}_accuracy.json", "w") as f_acc:
            json.dump(acc_record, f_acc)
        logger.info(
            "train_model: saved models to %s / %s",
            _xgb_path(ticker), _rf_path(ticker),
        )
    except Exception as exc:
        logger.error("train_model: failed to save models: %s", exc)
        result["error"] = f"Model save failed: {exc}"
        return result

    # ── Return training summary ───────────────────────────────────────────────
    result.update({
        "directional_accuracy": wf["mean_directional_accuracy"],
        "accuracy_std": wf["std_directional_accuracy"],
        "mean_auc": wf["mean_auc"],
        "n_train": len(X_dir_18),
        "n_test": wf["n_validation_samples"],
        "trained_at": now_iso,
        "is_reliable": wf["is_reliable"],
        "reliability_reason": wf["reliability_reason"],
        "class_balance": balance,
        "horizon_days": horizon_days,
        "neutral_threshold_pct": round(neutral_threshold * 100, 2),
        "hyperparam_overrides": hp_overrides,
        "label_search": label_choice["candidates"],
        "hyperparam_search": hp_choice["candidates"],
        "error": None,
    })
    return result


def predict(ticker: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Load saved models and produce a directional prediction for the latest bar.

    If models do not exist on disk, train_model() is called automatically.
    If df is None, price data is fetched using get_price_history().

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    df : pd.DataFrame, optional
        Output of calculate_indicators(). At least 60 bars required for
        feature warm-up. Fetched automatically if None.

    Returns
    -------
    dict with keys:

        direction : str
            "bullish" | "bearish" | "neutral"

        probability : float
            Bull probability, range 0.0–1.0. Values in [0.45, 0.55] → neutral.

        expected_move_pct : float
            Median 5-day return historically in setups where the model predicted
            the same direction. Look-back statistic — not a price forecast.

        confidence : str
            "high" if probability > 0.65 or < 0.35
            "medium" if probability > 0.55 or < 0.45
            "low" otherwise

        top_features : dict
            {feature_name: importance_value} for the top 8 XGBoost features
            by gain-based importance score.

        model_accuracy : float
            Mean walk-forward directional accuracy (from training summary).
            Loaded from the accuracy log if available, else 0.0.

        last_trained : str
            ISO timestamp from when the model files were last written.

        horizon_days : int
            Label lookahead this model was trained with (3, 5, or 10). Drives
            expected_move_pct and is what "N-day forecast" refers to in the UI.

        neutral_threshold_pct : float
            Neutral zone this model was trained with, as a percent.

        hyperparam_overrides : dict
            XGBoost hyperparameter overrides this model was trained with
            ({} means the defaults were used).

        error : str or None
            Non-None means prediction failed; all other values are defaults.
    """
    ticker = ticker.upper()
    result: Dict[str, Any] = {
        "ticker": ticker,
        "direction": "neutral",
        "probability": 0.50,
        "expected_move_pct": None,
        "confidence": "low",
        "top_features": {},
        "model_accuracy": 0.0,
        "last_trained": None,
        "horizon_days": None,
        "neutral_threshold_pct": None,
        "hyperparam_overrides": {},
        "error": None,
    }

    # ── Fetch data if not provided ────────────────────────────────────────────
    if df is None:
        try:
            from data.price_data import get_price_history
            from analysis.indicators import calculate_indicators
            df_raw = get_price_history(ticker, period="2y")
            if df_raw is None or df_raw.empty:
                logger.warning(f"predict: no price data available for {ticker}")
                result["error"] = f"No price data available for {ticker}"
                return result
            df = calculate_indicators(df_raw)
        except Exception as exc:
            logger.error(f"predict: data fetch failed for {ticker}: {exc}")
            result["error"] = f"Data fetch failed: {exc}"
            return result

    # ── Train if models do not exist ──────────────────────────────────────────
    if not _xgb_path(ticker).exists() or not _rf_path(ticker).exists():
        logger.info("predict: no saved models for %s — training now", ticker)
        train_result = train_model(ticker, df)
        if train_result.get("error"):
            logger.error(f"predict: auto-training failed for {ticker}: {train_result['error']}")
            result["error"] = f"Auto-training failed: {train_result['error']}"
            return result
        # Carry accuracy into predict result
        result["model_accuracy"] = train_result.get("directional_accuracy", 0.0) or 0.0
        result["last_trained"] = train_result.get("trained_at")
    else:
        # Read last-modified timestamp as proxy for training date
        try:
            mtime = _xgb_path(ticker).stat().st_mtime
            result["last_trained"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception as exc:
            logger.debug(f"predict: could not read model mtime for {ticker}: {exc}")

    # ── Load the trained model's metadata (horizon/threshold/hyperparams) ─────
    metadata = _load_model_metadata(ticker)
    horizon_days = metadata["horizon_days"]
    neutral_threshold = metadata["neutral_threshold"]
    result["horizon_days"] = horizon_days
    result["neutral_threshold_pct"] = round(neutral_threshold * 100, 2)
    result["hyperparam_overrides"] = metadata["hyperparam_overrides"]

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        xgb_model = joblib.load(_xgb_path(ticker))
        rf_model = joblib.load(_rf_path(ticker))
    except Exception as exc:
        logger.error(f"predict: failed to load models for {ticker}: {exc}")
        result["error"] = f"Failed to load models: {exc}"
        return result

    # ── Build feature row ─────────────────────────────────────────────────────
    X_row = build_predict_row(df)
    if X_row is None:
        logger.warning(f"predict: could not build feature row for {ticker} — insufficient bars or missing indicator columns")
        result["error"] = (
            "Could not build feature row — ensure df has >=60 bars and all "
            "required indicator columns are present (RSI, MACD_hist, ADX, etc.)."
        )
        return result

    # Use only the 18 core features the model was trained on
    try:
        X_arr = X_row[FEATURE_NAMES].values.astype("float32")
    except KeyError as exc:
        logger.error(f"predict: feature column mismatch for {ticker}: {exc}")
        result["error"] = f"Feature column mismatch: {exc}. Retrain the model."
        return result

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        xgb_prob = float(xgb_model.predict_proba(X_arr)[0, 1])
        rf_prob = float(rf_model.predict_proba(X_arr)[0, 1])
        ensemble_prob = 0.65 * xgb_prob + 0.35 * rf_prob
    except Exception as exc:
        logger.error(f"predict: model inference error for {ticker}: {exc}")
        result["error"] = f"Model inference error: {exc}"
        return result

    # ── Signal derivation ─────────────────────────────────────────────────────
    # Neutral dead-band: [0.45, 0.55]
    if ensemble_prob > 0.65 or ensemble_prob < 0.35:
        confidence = "high"
    elif ensemble_prob > 0.55 or ensemble_prob < 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    if 0.45 <= ensemble_prob <= 0.55:
        direction = "neutral"
    elif ensemble_prob > 0.55:
        direction = "bullish"
    else:
        direction = "bearish"

    # ── Top features ──────────────────────────────────────────────────────────
    top_features: Dict[str, float] = {}
    try:
        if hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
        else:
            importances = rf_model.feature_importances_

        total = importances.sum()
        norm = importances / total if total > 0 else importances
        # Trim to 18-feature FEATURE_NAMES if model was trained on more columns
        feature_labels = FEATURE_NAMES[: len(norm)]
        sorted_idx = np.argsort(norm)[::-1][:8]
        top_features = {
            feature_labels[i]: round(float(norm[i]), 4)
            for i in sorted_idx
            if i < len(feature_labels)
        }
    except Exception as exc:
        logger.debug(f"predict: top_features computation skipped for {ticker}: {exc}")

    # ── Expected move estimate ────────────────────────────────────────────────
    # Uses the same horizon_days/neutral_threshold this model was trained with,
    # so the N-day forward return matches the model's own label definition.
    expected_move_pct = None
    try:
        X_full, y_full = build_features(
            df, ticker=ticker,
            forward_bars=horizon_days, neutral_threshold=neutral_threshold,
        )
        X_dir, y_dir = _filter_directional(X_full, y_full)
        X_dir_18 = X_dir[FEATURE_NAMES].values.astype("float32")

        xgb_probs_full = xgb_model.predict_proba(X_dir_18)[:, 1]
        rf_probs_full = rf_model.predict_proba(X_dir_18)[:, 1]
        ensemble_full = 0.65 * xgb_probs_full + 0.35 * rf_probs_full

        predicted_mask = ensemble_full >= 0.5 if direction == "bullish" else ensemble_full < 0.5

        fwd_ret = df["Close"].pct_change(horizon_days).shift(-horizon_days).reindex(X_dir.index)
        if predicted_mask.sum() >= 5:
            subset_ret = fwd_ret.values[predicted_mask]
            subset_ret = subset_ret[~np.isnan(subset_ret)]
            if len(subset_ret) >= 5:
                expected_move_pct = round(float(np.median(subset_ret)) * 100, 2)
    except Exception as exc:
        logger.debug(f"predict: expected_move_pct computation skipped for {ticker}: {exc}")   # non-critical

    # ── Load accuracy from last training run ──────────────────────────────────
    if result["model_accuracy"] == 0.0:
        result["model_accuracy"] = metadata.get("directional_accuracy", 0.0) or 0.0

    # ── Assemble result ───────────────────────────────────────────────────────
    result.update({
        "direction": direction,
        "probability": round(ensemble_prob, 4),
        "expected_move_pct": expected_move_pct,
        "confidence": confidence,
        "top_features": top_features,
        "price_at_prediction": float(df["Close"].iloc[-1]),
    })

    # ── Persist prediction ────────────────────────────────────────────────────
    save_prediction(ticker, result)

    logger.info(
        f"predict: {ticker} complete — direction={direction} probability={result['probability']} "
        f"confidence={confidence}"
    )

    return result


def get_prediction_history(ticker: str) -> pd.DataFrame:
    """
    Return all stored predictions for a ticker as a sorted DataFrame.

    Returns an empty DataFrame (with correct columns) if no predictions file exists.

    Columns
    -------
    date : datetime (UTC)
    direction : str
    probability : float
    confidence : str
    actual_outcome : str or None  (filled in retrospectively)
    correct : bool or None
    """
    ticker = ticker.upper()
    resolve_predictions(ticker)

    path = _predictions_path(ticker)
    empty_cols = ["date", "direction", "probability", "confidence", "actual_outcome", "correct"]

    if not path.exists():
        return pd.DataFrame(columns=empty_cols)

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(records)

    # Normalise date column — may be stored as "predicted_at" or "date"
    if "predicted_at" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"predicted_at": "date"})
    if "date" in df.columns:
        # format="mixed" is required: older rows were logged with naive
        # datetime.utcnow().isoformat() timestamps, newer rows with
        # tz-aware now_et_iso() timestamps. Without it, pandas infers a
        # single format from the first row and silently coerces every
        # later row that doesn't match to NaT (dropped by the caller).
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
        df = df.sort_values("date", ascending=False)

    # Ensure all expected columns exist
    for col in empty_cols:
        if col not in df.columns:
            df[col] = None

    return df[empty_cols].reset_index(drop=True)


def save_prediction(ticker: str, prediction: Dict[str, Any]) -> None:
    """
    Append a prediction record to the JSONL log for a ticker.

    Each line in the JSONL file is a self-contained prediction event with a
    timestamp. The `actual_outcome` and `correct` fields are initially None
    and are back-filled by resolve_predictions() once horizon_days has
    elapsed (that function is called automatically from get_prediction_history()).

    Parameters
    ----------
    ticker : str
    prediction : dict
        Output of predict(). Must contain at minimum 'direction' and 'probability'.
    """
    path = _predictions_path(ticker.upper())
    try:
        record = {
            "predicted_at": now_et_iso(),
            "date": now_et_iso(),
            "ticker": ticker.upper(),
            "direction": prediction.get("direction"),
            "probability": prediction.get("probability"),
            "confidence": prediction.get("confidence"),
            "model_accuracy": prediction.get("model_accuracy"),
            "expected_move_pct": prediction.get("expected_move_pct"),
            "horizon_days": prediction.get("horizon_days"),
            "price_at_prediction": prediction.get("price_at_prediction"),
            "actual_outcome": None,
            "correct": None,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("save_prediction: failed to write log for %s: %s", ticker, exc)


def resolve_predictions(ticker: str) -> int:
    """
    Back-fill actual_outcome/correct for logged predictions whose horizon
    has elapsed, by comparing price_at_prediction to the close price
    horizon_days trading bars later. Rewrites the JSONL log in place.

    Predictions logged before this function existed have no
    price_at_prediction and are skipped (nothing to compare against).
    Neutral-direction predictions are left unresolved, matching how
    training excludes the neutral class from directional accuracy.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    int
        Number of records newly resolved.
    """
    ticker = ticker.upper()
    path = _predictions_path(ticker)
    if not path.exists():
        return 0

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    pending = [
        r for r in records
        if r.get("correct") is None
        and r.get("direction") in ("bullish", "bearish")
        and r.get("price_at_prediction") is not None
    ]
    if not pending:
        return 0

    try:
        from data.price_data import get_price_history
        price_df = get_price_history(ticker, period="2y")
    except Exception as exc:
        logger.warning("resolve_predictions: price fetch failed for %s: %s", ticker, exc)
        return 0
    if price_df is None or price_df.empty:
        logger.warning(f"resolve_predictions: no price data returned for {ticker} — cannot resolve pending predictions")
        return 0

    closes = price_df["Close"]
    dates = price_df.index
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    resolved_count = 0
    for record in pending:
        try:
            predicted_at = pd.Timestamp(record["predicted_at"])
            if predicted_at.tz is not None:
                predicted_at = predicted_at.tz_localize(None)
        except Exception:
            continue
        horizon = record.get("horizon_days") or 5

        future_closes = closes[dates > predicted_at]
        if len(future_closes) < horizon:
            continue  # horizon hasn't elapsed yet — leave unresolved for now

        entry_price = float(record["price_at_prediction"])
        exit_price = float(future_closes.iloc[horizon - 1])
        actual_return_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        record["actual_outcome"] = actual_return_pct
        record["correct"] = bool(
            actual_return_pct > 0 if record["direction"] == "bullish" else actual_return_pct < 0
        )
        resolved_count += 1

    if resolved_count:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.debug(f"resolve_predictions: {ticker} resolved {resolved_count} pending prediction(s)")

    return resolved_count


def evaluate_model(ticker: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics for the direction model.

    Combines walk-forward accuracy metrics with historical prediction log analysis.
    This is the detailed evaluation function used by the Model Details expander
    in the UI — separate from the quick validation that runs inside train_model().

    Parameters
    ----------
    ticker : str
    df : pd.DataFrame, optional
        Output of calculate_indicators(). At least 250 bars recommended.
        Fetched automatically if None.

    Returns
    -------
    dict with keys:

        directional_accuracy : float
            Mean directional accuracy across walk-forward folds (0.0–1.0).

        win_rate_by_confidence : dict
            {confidence_level: win_rate} computed from prediction history.
            Confidence levels: "high", "medium", "low".

        last_n_accuracy : float
            Directional accuracy over the most recent 20 predictions in the log.
            None if fewer than 20 predictions exist.

        total_predictions : int
            Total predictions logged for this ticker.

        is_reliable : bool
            True if mean WF accuracy >= 0.52 and std <= 0.08.

        class_balance : dict
            Output of class_balance_check().

        horizon_days : int
            Label lookahead used for this recomputation, loaded from the
            trained model's persisted metadata (defaults to 5 if the model
            predates label-search).

        error : str or None
    """
    ticker = ticker.upper()
    out: Dict[str, Any] = {
        "ticker": ticker,
        "directional_accuracy": None,
        "accuracy_std": None,
        "win_rate_by_confidence": {},
        "last_n_accuracy": None,
        "total_predictions": 0,
        "is_reliable": False,
        "class_balance": {},
        "horizon_days": None,
        "error": None,
    }

    # ── Walk-forward accuracy from current data ───────────────────────────────
    if df is None:
        try:
            from data.price_data import get_price_history
            from analysis.indicators import calculate_indicators
            df_raw = get_price_history(ticker, period="2y")
            if df_raw is None or df_raw.empty:
                logger.warning(f"evaluate_model: no price data available for {ticker}")
                out["error"] = f"No price data for {ticker}"
                return out
            df = calculate_indicators(df_raw)
        except Exception as exc:
            logger.error(f"evaluate_model: data fetch failed for {ticker}: {exc}")
            out["error"] = f"Data fetch failed: {exc}"
            return out

    # Reuse the horizon/threshold/hyperparameters the model was actually
    # trained with, so this recomputation matches what's deployed rather
    # than silently re-scoring against a different label definition.
    metadata = _load_model_metadata(ticker)
    horizon_days = metadata["horizon_days"]
    neutral_threshold = metadata["neutral_threshold"]
    hp_overrides = metadata["hyperparam_overrides"]
    out["horizon_days"] = horizon_days

    try:
        X, y = build_features(
            df, ticker=ticker,
            forward_bars=horizon_days, neutral_threshold=neutral_threshold,
        )
    except (ValueError, KeyError) as exc:
        # KeyError happens when df is missing expected indicator columns
        # (e.g. a caller passed raw OHLCV without running calculate_indicators()
        # first) — same caller-input problem as ValueError, so it should
        # degrade to a structured error instead of propagating uncaught.
        logger.warning(f"evaluate_model: build_features failed for {ticker}: {exc}")
        if isinstance(exc, KeyError):
            out["error"] = (
                f"Missing expected column {exc}. df must be the output of "
                "calculate_indicators() — raw OHLCV is not sufficient."
            )
        else:
            out["error"] = str(exc)
        return out

    X_dir, y_dir = _filter_directional(X, y)
    balance = class_balance_check(y_dir)
    out["class_balance"] = balance

    spw = balance["recommended_scale_pos_weight"]
    xgb_cfg = {**_xgb_config(scale_pos_weight=spw), **hp_overrides}
    rf_cfg = _rf_config(n_features=len(FEATURE_NAMES))

    # Use only the 18 core features, consistent with train_model() and inference
    X_dir_18 = X_dir[FEATURE_NAMES]
    wf = _run_walk_forward(X_dir_18, y_dir, xgb_cfg, rf_cfg, n_splits=10, gap=horizon_days)
    out["directional_accuracy"] = wf["mean_directional_accuracy"]
    out["accuracy_std"] = wf["std_directional_accuracy"]
    out["is_reliable"] = wf["is_reliable"]
    out["reliability_reason"] = wf["reliability_reason"]
    out["mean_auc"] = wf["mean_auc"]
    out["n_validation_samples"] = wf["n_validation_samples"]
    out["n_training_samples"] = len(X_dir)


    # ── Historical prediction log analysis ────────────────────────────────────
    history = get_prediction_history(ticker)
    out["total_predictions"] = len(history)

    if not history.empty and "correct" in history.columns and "confidence" in history.columns:
        # Win rate by confidence level (from logged predictions that have been resolved)
        resolved = history.dropna(subset=["correct"])
        if not resolved.empty:
            by_conf: Dict[str, float] = {}
            for conf_level in ["high", "medium", "low"]:
                subset = resolved[resolved["confidence"] == conf_level]
                if len(subset) > 0:
                    by_conf[conf_level] = round(float(subset["correct"].mean()), 3)
            out["win_rate_by_confidence"] = by_conf

        # Last 20 predictions accuracy
        if len(resolved) >= 20:
            recent = resolved.head(20)
            out["last_n_accuracy"] = round(float(recent["correct"].mean()), 3)
        elif len(resolved) >= 5:
            out["last_n_accuracy"] = round(float(resolved.head(len(resolved))["correct"].mean()), 3)

    logger.info(
        f"evaluate_model: {ticker} complete — directional_accuracy={out['directional_accuracy']} "
        f"is_reliable={out['is_reliable']} total_predictions={out['total_predictions']}"
    )

    return out
