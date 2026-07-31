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
import shutil
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.tz import now_et_iso
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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


# ── Model versioning (Prediction Improvement Engine, Phase 7) ────────────────
# A parallel archive tree alongside the flat "latest" files above —
# _xgb_path()/_rf_path() are never touched by any of this, so every
# existing reader keeps working on the exact same paths regardless of
# whether a ticker has version history.

_VERSION_HISTORY_COLUMNS = [
    "version", "ticker", "archived_at", "directional_accuracy", "accuracy_std",
    "mean_auc", "is_reliable", "horizon_days", "neutral_threshold",
    "hyperparam_overrides", "rf_hyperparam_overrides", "ensemble_weights",
    "trained_at", "rolled_back_from_latest",
]


def _versions_dir(ticker: str) -> Path:
    d = _STORAGE_DIR / "versions" / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version_history_path(ticker: str) -> Path:
    return _versions_dir(ticker) / "history.jsonl"


def _next_version_number(ticker: str) -> int:
    path = _version_history_path(ticker)
    if not path.exists():
        return 1
    max_version = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                max_version = max(max_version, int(record.get("version", 0)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return max_version + 1


def _archive_current_version(ticker: str, acc_record: Dict[str, Any]) -> int:
    """
    Archive whatever is CURRENTLY at _xgb_path/_rf_path/the accuracy JSON —
    the model about to be replaced by this training run — into
    storage/versions/{TICKER}/v{N}_*, before the caller overwrites the
    "latest" files. No-op (returns 0) if nothing exists yet to archive —
    a ticker's first-ever train has no prior model to preserve, so v1 is
    the first model that got REPLACED, not the first trained.
    """
    ticker = ticker.upper()
    xgb_path = _xgb_path(ticker)
    rf_path = _rf_path(ticker)
    acc_path = _STORAGE_DIR / f"{ticker}_accuracy.json"
    if not (xgb_path.exists() and rf_path.exists()):
        return 0

    version = _next_version_number(ticker)
    versions_dir = _versions_dir(ticker)
    try:
        shutil.copy2(xgb_path, versions_dir / f"v{version}_xgb.pkl")
        shutil.copy2(rf_path, versions_dir / f"v{version}_rf.pkl")
        if acc_path.exists():
            shutil.copy2(acc_path, versions_dir / f"v{version}_accuracy.json")
        history_record = {
            "version": version,
            "ticker": ticker,
            "archived_at": now_et_iso(),
            "directional_accuracy": acc_record.get("directional_accuracy"),
            "accuracy_std": acc_record.get("accuracy_std"),
            "mean_auc": acc_record.get("mean_auc"),
            "is_reliable": acc_record.get("is_reliable"),
            "horizon_days": acc_record.get("horizon_days"),
            "neutral_threshold": acc_record.get("neutral_threshold"),
            "hyperparam_overrides": acc_record.get("hyperparam_overrides"),
            "rf_hyperparam_overrides": acc_record.get("rf_hyperparam_overrides"),
            "ensemble_weights": acc_record.get("ensemble_weights"),
            "trained_at": acc_record.get("trained_at"),
            "rolled_back_from_latest": False,
        }
        with open(_version_history_path(ticker), "a") as f:
            f.write(json.dumps(history_record) + "\n")
        logger.info("_archive_current_version: %s archived as v%d", ticker, version)
        return version
    except Exception as exc:
        logger.warning("_archive_current_version: failed to archive %s: %s", ticker, exc)
        return 0


def get_version_history(ticker: str) -> pd.DataFrame:
    """
    Read storage/versions/{TICKER}/history.jsonl, sorted version-descending.
    An append-only EVENT log, not a snapshot-per-version table — a rollback
    appends a second row for a version number that was already archived
    once, distinguished by rolled_back_from_latest=True, rather than
    overwriting or deduplicating the earlier entry.

    Empty DataFrame (documented columns) if the ticker has never been
    retrained — not an error.
    """
    path = _version_history_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=_VERSION_HISTORY_COLUMNS)

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
        return pd.DataFrame(columns=_VERSION_HISTORY_COLUMNS)

    df = pd.DataFrame(records)
    for col in _VERSION_HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_VERSION_HISTORY_COLUMNS].sort_values("version", ascending=False).reset_index(drop=True)


def rollback_to_version(ticker: str, version: int) -> Dict[str, Any]:
    """
    Restore an archived version's model files as the current "latest" —
    a straight file copy (shutil.copy2), never a retrain. Archives the
    CURRENT latest as a new version FIRST (via _archive_current_version),
    so rolling back never discards the model being rolled back FROM — it
    becomes retrievable too. Returns a structured {"error": ...} on a
    missing/corrupt version rather than raising, matching every other
    public function's contract in this module.
    """
    ticker = ticker.upper()
    out: Dict[str, Any] = {
        "ticker": ticker, "rolled_back_to_version": None,
        "new_current_version_archived": None, "error": None,
    }

    versions_dir = _versions_dir(ticker)
    src_xgb = versions_dir / f"v{version}_xgb.pkl"
    src_rf = versions_dir / f"v{version}_rf.pkl"
    src_acc = versions_dir / f"v{version}_accuracy.json"
    if not (src_xgb.exists() and src_rf.exists()):
        out["error"] = f"Version {version} not found for {ticker}."
        return out

    current_meta = _load_model_metadata(ticker)
    archived_version = _archive_current_version(ticker, current_meta)
    out["new_current_version_archived"] = archived_version or None

    try:
        shutil.copy2(src_xgb, _xgb_path(ticker))
        shutil.copy2(src_rf, _rf_path(ticker))
        if src_acc.exists():
            shutil.copy2(src_acc, _STORAGE_DIR / f"{ticker}_accuracy.json")
    except Exception as exc:
        out["error"] = f"Rollback failed: {exc}"
        return out

    try:
        with open(_version_history_path(ticker), "a") as f:
            f.write(json.dumps({
                "version": version, "ticker": ticker,
                "archived_at": now_et_iso(), "rolled_back_from_latest": True,
            }) + "\n")
    except Exception as exc:
        logger.warning("rollback_to_version: could not append history record for %s: %s", ticker, exc)

    out["rolled_back_to_version"] = version
    logger.info(
        "rollback_to_version: %s rolled back to v%d (archived previous latest as v%s)",
        ticker, version, archived_version,
    )
    return out


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


def _logreg_config() -> Dict[str, Any]:
    """
    Linear baseline for model comparison (Prediction Improvement Engine,
    Phase 4) — a cheap sanity floor other models should beat. Needs
    StandardScaler-transformed input, unlike the tree models; handled inside
    _fit_predict_proba(), never exposed to callers. `penalty` is left at
    sklearn's default (l2) rather than passed explicitly — sklearn 1.8+
    deprecates the standalone `penalty` param in favor of `l1_ratio`.
    """
    return {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }


def _gbc_config() -> Dict[str, Any]:
    """
    sklearn's own gradient boosting — a second, differently-regularized
    boosted-tree comparison point against XGBoost, at zero new dependency
    cost. Deliberately similar regularization philosophy to _xgb_config()
    for a fair walk-forward comparison.
    """
    return {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
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
    {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 400},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200},
]

# Candidate RandomForest hyperparameter overrides, merged on top of
# _rf_config()'s defaults. RF was never tuned before this grid existed —
# same "first entry is the baseline" guarantee as HYPERPARAM_SEARCH_GRID.
RF_HYPERPARAM_SEARCH_GRID: List[Dict[str, Any]] = [
    {},
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 4, "min_samples_leaf": 30},
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20},
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


def _price_sanity_error(df: pd.DataFrame, ticker: str, threshold: float = 0.15) -> Optional[str]:
    """
    Guards against a single corrupted/glitched final bar from the data
    provider silently poisoning price_at_prediction. Observed once in
    production: a fetched last close of ~$108 for SPY, saved and displayed
    alongside every adjacent prediction showing ~$750 — every other
    prediction across five separate logs (90+ daily, 30+ intraday) had a
    normal price, so this reads as a one-off bad fetch, not a systemic
    issue, but it's cheap to guard against recurring.

    Compares the latest close to the median of the preceding ~20 bars
    (excluding itself) rather than a caller-supplied "last known good"
    price, so a real multi-day trend never trips it — only a single bar
    that jumps far outside its own recent neighborhood. A move this size
    in one bar is not a real price for any equity/ETF; it's bad data.

    Returns an error message if the check fails, else None. Skips the
    check (returns None) when there isn't enough history to judge from.
    """
    recent = df["Close"].tail(21)
    if len(recent) < 11:
        return None
    recent_median = float(recent.iloc[:-1].median())
    latest_close = float(recent.iloc[-1])
    if recent_median <= 0:
        return None
    deviation = abs(latest_close - recent_median) / recent_median
    if deviation > threshold:
        return (
            f"Latest price for {ticker} ({latest_close:.2f}) deviates {deviation:.0%} from "
            f"its own recent median ({recent_median:.2f}) — likely a bad data fetch. Try again."
        )
    return None


_ENSEMBLE_SOFTMAX_TEMPERATURE = 0.05


def _softmax_ensemble_weights(
    mean_accuracies: Dict[str, float],
    temperature: float = _ENSEMBLE_SOFTMAX_TEMPERATURE,
) -> Dict[str, float]:
    """
    {model_name: mean_directional_accuracy} -> {model_name: weight}, summing
    to 1.0. Softmax over (accuracy - 0.5), so a model exactly at the
    coin-flip baseline gets a near-zero relative score rather than the ~40-45%
    share a raw-accuracy-proportional split would give it, and the gap
    between e.g. a 54% and a 58% model is sharpened rather than compressed.
    A model at or below 0.50 still gets a small positive weight — never
    fully zeroed, matching the ensemble's existing philosophy of blending
    rather than hard model-selection. Generalizes the historical hardcoded
    0.65/0.35 XGB/RF split to N models; temperature=0.05 approximates that
    split for a typical XGB-ahead-of-RF accuracy gap.
    """
    names = list(mean_accuracies.keys())
    if not names:
        return {}
    scores = np.array([mean_accuracies[n] - 0.5 for n in names])
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / exp_scores.sum()
    return {n: round(float(w), 4) for n, w in zip(names, weights)}


def _fit_predict_proba(
    kind: str,
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    """
    Fit one model kind for one walk-forward fold, return predict_proba's
    positive-class column. "xgb" falls back to a default-config
    RandomForest if xgboost isn't importable — same degrade-not-crash
    convention as the rest of this module for that optional dependency.
    "logreg" needs StandardScaler-transformed input (fit on this fold's
    training split only, never the validation split) — the only model kind
    that does; tree-based kinds take the raw features.
    """
    if kind == "xgb":
        if _XGBOOST_AVAILABLE:
            m = XGBClassifier(**config)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m = RandomForestClassifier(**_rf_config())
            m.fit(X_train, y_train)
        return m.predict_proba(X_val)[:, 1]
    if kind == "rf":
        m = RandomForestClassifier(**config)
        m.fit(X_train, y_train)
        return m.predict_proba(X_val)[:, 1]
    if kind == "logreg":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        m = LogisticRegression(**config)
        m.fit(X_train_scaled, y_train)
        return m.predict_proba(X_val_scaled)[:, 1]
    if kind == "gbc":
        m = GradientBoostingClassifier(**config)
        m.fit(X_train, y_train)
        return m.predict_proba(X_val)[:, 1]
    raise ValueError(f"Unknown model kind: {kind!r}")


def _summarize_fold_scores(fold_accs: List[float], fold_aucs: List[float], n_validation_samples: int) -> Dict[str, Any]:
    """Shared by every per-model and ensemble summary in _run_walk_forward_multi
    — identical math/thresholds to the original _run_walk_forward's summary."""
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
        "n_validation_samples": n_validation_samples,
        "is_reliable": is_reliable,
        "reliability_reason": reason,
        "fold_accuracies": [round(a, 4) for a in fold_accs],
    }


def _run_walk_forward_multi(
    X: pd.DataFrame,
    y_dir: pd.Series,
    model_specs: Dict[str, Tuple[str, Dict[str, Any]]],
    n_splits: int = 10,
    gap: int = 5,
    ensemble_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generalizes the anchored walk-forward validation to N models.
    model_specs: {model_name: (model_kind, config_dict)}, model_kind in
    {"xgb", "rf", "logreg", "gbc"}. y_dir contains {-1, 0, 1} labels;
    neutral rows are excluded per fold — identical CV-splitting/neutral-
    filtering machinery as the original 2-model _run_walk_forward.

    ensemble_weights is the FIXED weight dict used to blend every fold's
    per-model probabilities (defaults to equal 1/N weight per model if not
    given) — deliberately not adaptive per fold, so a caller that wants the
    historical hardcoded 0.65/0.35 XGB/RF blend gets bit-identical results
    to the original function on every fold, not a look-ahead-free but
    numerically different adaptive scheme.

    Returns {"n_folds", "per_model": {name: {...same shape as the original
    return dict...}}, "ensemble_weights": <the weights actually used for
    blending>, "ensemble": {...same shape...}}.
    """
    names = list(model_specs.keys())
    if ensemble_weights is None:
        ensemble_weights = {name: 1.0 / len(names) for name in names}

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X_arr = X.values.astype("float32")
    y_arr = y_dir.values

    per_model_fold_accs: Dict[str, List[float]] = {name: [] for name in names}
    per_model_fold_aucs: Dict[str, List[float]] = {name: [] for name in names}
    ensemble_fold_accs: List[float] = []
    ensemble_fold_aucs: List[float] = []
    total_val_samples = 0

    for train_idx, val_idx in tscv.split(X_arr):
        if len(val_idx) < 10:
            continue

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

        fold_probs: Dict[str, np.ndarray] = {}
        for name, (kind, config) in model_specs.items():
            probs = _fit_predict_proba(kind, config, X_train, y_train, X_val, y_val)
            fold_probs[name] = probs
            per_model_fold_accs[name].append(_directional_accuracy(y_val, probs))
            try:
                per_model_fold_aucs[name].append(float(roc_auc_score(y_val, probs)))
            except ValueError:
                per_model_fold_aucs[name].append(0.5)

        ensemble_probs = sum(ensemble_weights[name] * fold_probs[name] for name in names)
        ensemble_fold_accs.append(_directional_accuracy(y_val, ensemble_probs))
        try:
            ensemble_fold_aucs.append(float(roc_auc_score(y_val, ensemble_probs)))
        except ValueError:
            ensemble_fold_aucs.append(0.5)
        total_val_samples += len(y_val)

    per_model_summary = {
        name: _summarize_fold_scores(per_model_fold_accs[name], per_model_fold_aucs[name], total_val_samples)
        for name in names
    }
    ensemble_summary = _summarize_fold_scores(ensemble_fold_accs, ensemble_fold_aucs, total_val_samples)

    return {
        "n_folds": ensemble_summary["n_folds"],
        "per_model": per_model_summary,
        "ensemble_weights": ensemble_weights,
        "ensemble": ensemble_summary,
    }


def _run_walk_forward(
    X: pd.DataFrame,
    y_dir: pd.Series,
    xgb_cfg: Dict[str, Any],
    rf_cfg: Dict[str, Any],
    n_splits: int = 10,
    gap: int = 5,
) -> Dict[str, Any]:
    """
    Run anchored walk-forward validation for the original 2-model XGB/RF
    ensemble. UNCHANGED signature/return shape — now a thin wrapper over
    _run_walk_forward_multi() with the historical fixed 0.65/0.35 blend, so
    every existing caller (select_label_scheme, select_hyperparams,
    select_rf_hyperparams, evaluate_model, intraday_prediction.py's
    read-only import) keeps working unmodified.
    """
    return _run_walk_forward_multi(
        X, y_dir, {"xgb": ("xgb", xgb_cfg), "rf": ("rf", rf_cfg)},
        n_splits=n_splits, gap=gap, ensemble_weights={"xgb": 0.65, "rf": 0.35},
    )["ensemble"]


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
    meta.setdefault("rf_hyperparam_overrides", {})
    meta.setdefault("ensemble_weights", {"xgb": 0.65, "rf": 0.35})
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


def select_rf_hyperparams(
    X_dir_18: pd.DataFrame,
    y_dir: pd.Series,
    xgb_cfg: Dict[str, Any],
    gap: int,
) -> Dict[str, Any]:
    """
    Search RF_HYPERPARAM_SEARCH_GRID for the RandomForest override set that
    scores best, with the already-chosen xgb_cfg held fixed — the reverse
    pairing of select_hyperparams() (which holds RF fixed while searching
    XGB). Same reduced-fold walk-forward, same "first entry is the
    can't-do-worse-than-baseline default" guarantee, same return shape.

    Returns
    -------
    dict with keys: overrides (the winning dict, possibly {}), candidates
    (list of {overrides, mean_accuracy, std_accuracy, n_folds}, best first).
    """
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for overrides in RF_HYPERPARAM_SEARCH_GRID:
        rf_cfg = {**_rf_config(n_features=len(FEATURE_NAMES)), **overrides}
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
        logger.warning(f"select_rf_hyperparams: no candidate cleared {_MIN_SEARCH_ACCURACY} accuracy — falling back to default RF hyperparameters")
        best = {"overrides": RF_HYPERPARAM_SEARCH_GRID[0]}

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
        "rf_hyperparam_overrides": {},
        "ensemble_weights": {},
        "label_search": [],
        "hyperparam_search": [],
        "rf_hyperparam_search": [],
        "archived_version": None,
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

    # ── Search for the best RandomForest hyperparameters with xgb_cfg fixed ────
    rf_hp_choice = select_rf_hyperparams(X_dir_18, y_dir, xgb_cfg, gap=horizon_days)
    rf_hp_overrides = rf_hp_choice["overrides"]
    logger.info("train_model: %s RF hyperparam overrides selected — %s", ticker, rf_hp_overrides or "defaults")

    rf_cfg = {**_rf_config(n_features=len(FEATURE_NAMES)), **rf_hp_overrides}

    # ── Full walk-forward validation BEFORE fitting the final model ───────────
    # Runs through _run_walk_forward_multi directly (rather than the
    # _run_walk_forward wrapper) so per-model accuracies are available to
    # compute a learned ensemble weight below — "ensemble" is bit-identical
    # to what _run_walk_forward(...) would have returned, since the fixed
    # 0.65/0.35 weight is passed explicitly.
    logger.info("train_model: running full walk-forward validation for %s", ticker)
    wf_multi = _run_walk_forward_multi(
        X_dir_18, y_dir, {"xgb": ("xgb", xgb_cfg), "rf": ("rf", rf_cfg)},
        n_splits=10, gap=horizon_days, ensemble_weights={"xgb": 0.65, "rf": 0.35},
    )
    wf = wf_multi["ensemble"]
    ensemble_weights = _softmax_ensemble_weights({
        name: m["mean_directional_accuracy"] for name, m in wf_multi["per_model"].items()
    })
    logger.info(
        "train_model: validation complete — mean_acc=%.3f std=%.3f reliable=%s ensemble_weights=%s",
        wf["mean_directional_accuracy"],
        wf["std_directional_accuracy"],
        wf["is_reliable"],
        ensemble_weights,
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
        # Archive whatever model is currently deployed BEFORE overwriting it —
        # a no-op on a ticker's first-ever train (nothing to archive yet).
        archived_version = _archive_current_version(ticker, _load_model_metadata(ticker))
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
            "rf_hyperparam_overrides": rf_hp_overrides,
            "ensemble_weights": ensemble_weights,
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
        "rf_hyperparam_overrides": rf_hp_overrides,
        "ensemble_weights": ensemble_weights,
        "label_search": label_choice["candidates"],
        "hyperparam_search": hp_choice["candidates"],
        "rf_hyperparam_search": rf_hp_choice["candidates"],
        "archived_version": archived_version or None,
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
        "rf_hyperparam_overrides": {},
        "ensemble_weights": {},
        "indicator_snapshot": {},
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

    sanity_error = _price_sanity_error(df, ticker)
    if sanity_error:
        logger.error(f"predict: {ticker} failed price sanity check: {sanity_error}")
        result["error"] = sanity_error
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
    result["rf_hyperparam_overrides"] = metadata["rf_hyperparam_overrides"]
    ensemble_weights = metadata["ensemble_weights"]
    result["ensemble_weights"] = ensemble_weights

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
        ensemble_prob = ensemble_weights.get("xgb", 0.65) * xgb_prob + ensemble_weights.get("rf", 0.35) * rf_prob
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
        ensemble_full = ensemble_weights.get("xgb", 0.65) * xgb_probs_full + ensemble_weights.get("rf", 0.35) * rf_probs_full

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

    # ── Indicator snapshot (Prediction Improvement Engine, Phase 2) ──────────
    # Point-in-time context for failure categorization later — a snapshot
    # failure must never block the prediction itself, hence the try/except
    # rather than letting build_indicator_snapshot's own internals leak here.
    try:
        from analysis.prediction_errors import build_indicator_snapshot
        indicator_snapshot = build_indicator_snapshot(df)
    except Exception as exc:
        logger.debug(f"predict: indicator snapshot skipped for {ticker}: {exc}")
        indicator_snapshot = {}

    # ── Assemble result ───────────────────────────────────────────────────────
    result.update({
        "direction": direction,
        "probability": round(ensemble_prob, 4),
        "expected_move_pct": expected_move_pct,
        "confidence": confidence,
        "top_features": top_features,
        "price_at_prediction": float(df["Close"].iloc[-1]),
        "indicator_snapshot": indicator_snapshot,
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
    model_accuracy : float or None
    expected_move_pct : float or None
    horizon_days : int or None
    price_at_prediction : float or None
    indicator_snapshot : dict or None  (point-in-time context for failure
        categorization — see analysis/prediction_errors.py; None for
        predictions logged before this field existed)
    """
    ticker = ticker.upper()
    resolve_predictions(ticker)

    path = _predictions_path(ticker)
    # model_accuracy/expected_move_pct/price_at_prediction were being read
    # into the intermediate DataFrame below and then silently dropped by the
    # final df[empty_cols] slice, since they were never listed here — the
    # bug behind "Exp Move"/"Model Acc" always showing empty in the UI even
    # though the underlying JSONL records had real values the whole time.
    # horizon_days has the same problem — save_prediction() always writes it,
    # but it was never in this list either.
    empty_cols = [
        "date", "direction", "probability", "confidence", "actual_outcome", "correct",
        "model_accuracy", "expected_move_pct", "horizon_days", "price_at_prediction",
        "indicator_snapshot",
    ]

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
            "indicator_snapshot": prediction.get("indicator_snapshot"),
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
    rf_hp_overrides = metadata["rf_hyperparam_overrides"]
    ensemble_weights = metadata["ensemble_weights"]
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
    rf_cfg = {**_rf_config(n_features=len(FEATURE_NAMES)), **rf_hp_overrides}

    # Use only the 18 core features, consistent with train_model() and inference
    X_dir_18 = X_dir[FEATURE_NAMES]
    wf = _run_walk_forward_multi(
        X_dir_18, y_dir, {"xgb": ("xgb", xgb_cfg), "rf": ("rf", rf_cfg)},
        n_splits=10, gap=horizon_days, ensemble_weights=ensemble_weights,
    )["ensemble"]
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


def compare_models(ticker: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Score XGBoost, RandomForest, LogisticRegression, and GradientBoosting
    against each other through the same walk-forward harness training uses
    (_run_walk_forward_multi, n_splits=10) — an informational comparison
    panel (Prediction Improvement Engine, Phase 4). Pure read/compare: never
    retrains or overwrites any persisted model, never touches _xgb_path()/
    _rf_path()/the accuracy JSON.

    Uses the ticker's already-selected label scheme and hyperparameters
    (from _load_model_metadata) so the comparison is apples-to-apples with
    what's actually deployed, not a differently-tuned re-run.

    Returns
    -------
    dict with keys: ticker, horizon_days, neutral_threshold,
    models ({name: per-model walk-forward summary}), best_single_model
    (highest mean_directional_accuracy), ensemble_weights (softmax-over-
    accuracy weights an ensemble of all 4 models would use — informational;
    not wired into predict()'s actual 2-model inference), ensemble (an
    equal-weighted blend of all 4 — a baseline reference point, not the
    recommended weighting), error.
    """
    ticker = ticker.upper()
    out: Dict[str, Any] = {
        "ticker": ticker,
        "horizon_days": None,
        "neutral_threshold": None,
        "models": {},
        "best_single_model": None,
        "ensemble_weights": {},
        "ensemble": {},
        "error": None,
    }

    if df is None:
        try:
            from data.price_data import get_price_history
            from analysis.indicators import calculate_indicators
            df_raw = get_price_history(ticker, period="2y")
            if df_raw is None or df_raw.empty:
                out["error"] = f"No price data for {ticker}"
                return out
            df = calculate_indicators(df_raw)
        except Exception as exc:
            out["error"] = f"Data fetch failed: {exc}"
            return out

    metadata = _load_model_metadata(ticker)
    horizon_days = metadata["horizon_days"]
    neutral_threshold = metadata["neutral_threshold"]
    out["horizon_days"] = horizon_days
    out["neutral_threshold"] = neutral_threshold

    try:
        X, y = build_features(df, ticker=ticker, forward_bars=horizon_days, neutral_threshold=neutral_threshold)
    except (ValueError, KeyError) as exc:
        out["error"] = str(exc)
        return out

    X_dir, y_dir = _filter_directional(X, y)
    if len(X_dir) < 250:
        out["error"] = (
            f"Only {len(X_dir)} directional samples for {ticker} after neutral-zone removal. "
            "Provide at least 250 bars of price data."
        )
        return out
    X_dir_18 = X_dir[FEATURE_NAMES]

    balance = class_balance_check(y_dir)
    spw = balance["recommended_scale_pos_weight"]
    xgb_cfg = {**_xgb_config(scale_pos_weight=spw), **metadata["hyperparam_overrides"]}
    rf_cfg = {**_rf_config(n_features=len(FEATURE_NAMES)), **metadata["rf_hyperparam_overrides"]}

    model_specs = {
        "xgb": ("xgb", xgb_cfg),
        "rf": ("rf", rf_cfg),
        "logreg": ("logreg", _logreg_config()),
        "gbc": ("gbc", _gbc_config()),
    }
    wf_multi = _run_walk_forward_multi(X_dir_18, y_dir, model_specs, n_splits=10, gap=horizon_days)
    out["models"] = wf_multi["per_model"]

    if wf_multi["n_folds"] == 0:
        out["error"] = "Walk-forward produced no valid folds — need more data"
        return out

    accuracies = {name: m["mean_directional_accuracy"] for name, m in wf_multi["per_model"].items()}
    out["best_single_model"] = max(accuracies, key=accuracies.get)
    out["ensemble_weights"] = _softmax_ensemble_weights(accuracies)
    out["ensemble"] = wf_multi["ensemble"]

    logger.info(
        "compare_models: %s best_single_model=%s accuracies=%s",
        ticker, out["best_single_model"], accuracies,
    )
    return out

