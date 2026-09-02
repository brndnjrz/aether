"""
ML direction prediction service for Aether.

Architecture
------------
Two-model ensemble:
  - XGBoostClassifier  (primary, binary:logistic objective, outputs probability)
  - RandomForestClassifier (secondary calibration / ensemble member)

Ensemble bull probability:
  P_bull = w_xgb * xgb_prob + w_rf * rf_prob

The weights are learned per model by softmax over per-model fold accuracy and
persisted in {ticker}_accuracy.json; 0.65/0.35 is only the fallback for a model
trained before they were recorded. At _ENSEMBLE_SOFTMAX_TEMPERATURE=0.05 the
learned weights can be far more lopsided than 0.65/0.35 (a 12-point accuracy gap
gives ~0.92/0.08), which is why the reported accuracy is re-scored under them
rather than under the fixed blend.

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

A model is considered reliable when all three hold:
  - mean directional accuracy >= 0.52
  - std-dev of accuracy across folds <= 0.08
  - accuracy exceeds the majority-class baseline by >= MIN_EDGE_OVER_BASELINE

That third condition is the load-bearing one. The 0.52 floor is not a real bar:
once the neutral band drops small moves, bull-market drift leaves the majority
class at 54-60% on a trending ticker, so "always predict up" clears 0.52 on its
own. `_majority_class_baseline()` computes what a constant guess scores on the
same labels, and a model that cannot beat it is reported unreliable no matter how
its raw accuracy reads.

Reported accuracy is measured under the ensemble weights that are actually
deployed (see _score_ensemble_weights) — not the fixed 0.65/0.35 blend used to
run the folds, which can differ sharply from the softmax weights inference uses.

One caveat this module does not correct for: train_model runs three sequential
argmax searches (label scheme, XGB hyperparameters, RF hyperparameters) all
scored on the same history with no holdout, so the winner's accuracy is the
maximum of many tries and is optimistic — worth roughly 7 points on data with no
signal at all. `selection_lift` is persisted alongside the accuracy so the
component attributable to picking the best label scheme is at least visible.

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
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.tz import now_et_iso
from config.settings import (
    HOLDOUT_FRACTION,
    MIN_EDGE_OVER_BASELINE,
    MIN_HOLDOUT_SAMPLES,
)
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


def _versions_dir(ticker: str, create: bool = False) -> Path:
    """Path accessor. Pass create=True only from a writer.

    This used to mkdir unconditionally, so merely *reading* version history for a
    mistyped ticker permanently created storage/versions/<TYPO>/.
    """
    d = _STORAGE_DIR / "versions" / ticker.upper()
    if create:
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
    versions_dir = _versions_dir(ticker, create=True)
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

    versions_dir = _versions_dir(ticker, create=True)
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


def _average_uniqueness(
    retained_index: pd.Index,
    full_index: pd.Index,
    horizon: int,
) -> np.ndarray:
    """
    Per-row weight correcting for overlapping label windows.

    Row t's label is the return over bars (t, t+horizon], so consecutive rows
    share horizon-1 bars of outcome. Fitting them as independent observations
    lets redundant information dominate.

    With a fixed horizon and *every* bar labelled this would be near-constant and
    pointless. It is not constant here: _filter_directional drops neutral rows
    first, so the retained rows are unevenly spaced. A row inside a dense cluster
    — a volatile stretch where many moves cleared the band — overlaps many other
    retained rows and currently carries as much weight as an isolated row whose
    outcome is entirely its own. That systematically over-weights high-volatility
    regimes in the fit.

    Follows the average-uniqueness construction in Lopez de Prado, AFML ch. 4:
    concurrency at a bar is how many retained labels span it; a row's weight is
    the mean of 1/concurrency across its own span. Normalized to mean 1 so the
    effective learning rate does not change with horizon.
    """
    positions = full_index.get_indexer(retained_index)
    valid = positions >= 0
    if not valid.all():
        # Rows whose bars aren't in the reference frame can't have their overlap
        # measured; give them neutral weight rather than dropping them.
        positions = np.where(valid, positions, -1)

    span_end = len(full_index) + horizon + 2
    concurrency = np.zeros(span_end, dtype=np.float64)
    for p in positions[valid]:
        concurrency[p + 1 : p + 1 + horizon] += 1.0

    weights = np.ones(len(positions), dtype=np.float64)
    for i, p in enumerate(positions):
        if p < 0:
            continue
        span = concurrency[p + 1 : p + 1 + horizon]
        span = span[span > 0]
        if len(span):
            weights[i] = float(np.mean(1.0 / span))

    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def _fit_predict_proba(
    kind: str,
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
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
            m.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)
        else:
            m = RandomForestClassifier(**_rf_config())
            m.fit(X_train, y_train, sample_weight=sample_weight)
        return m.predict_proba(X_val)[:, 1]
    if kind == "rf":
        m = RandomForestClassifier(**config)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        return m.predict_proba(X_val)[:, 1]
    if kind == "logreg":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        m = LogisticRegression(**config)
        m.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        return m.predict_proba(X_val_scaled)[:, 1]
    if kind == "gbc":
        m = GradientBoostingClassifier(**config)
        m.fit(X_train, y_train, sample_weight=sample_weight)
        return m.predict_proba(X_val)[:, 1]
    raise ValueError(f"Unknown model kind: {kind!r}")


def _summarize_fold_scores(
    fold_accs: List[float],
    fold_aucs: List[float],
    n_validation_samples: int,
    baseline_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Shared by every per-model and ensemble summary in _run_walk_forward_multi
    — identical math/thresholds to the original _run_walk_forward's summary.

    `baseline_accuracy` is the majority-class rate on the directional rows: what
    "always predict the more common direction" scores without a model. After the
    neutral band drops the small moves, bull-market drift leaves that at 54-60%
    on a trending ticker — well above the flat 52% floor, so a model could clear
    52% while being strictly worse than a constant guess. When supplied, the gate
    additionally requires beating it. Left None (the per-model summaries, the
    hyperparameter searches) the behaviour is unchanged.
    """
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
            "baseline_accuracy": baseline_accuracy,
            "edge_over_baseline": None,
        }

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_auc = float(np.mean(fold_aucs))

    edge_over_baseline = None
    beats_baseline = True
    if baseline_accuracy is not None:
        edge_over_baseline = mean_acc - baseline_accuracy
        beats_baseline = edge_over_baseline >= MIN_EDGE_OVER_BASELINE

    is_reliable = mean_acc >= 0.52 and std_acc <= 0.08 and beats_baseline

    if is_reliable:
        reason = (
            f"Consistent across {len(fold_accs)} folds — "
            f"mean accuracy {mean_acc:.1%} ± {std_acc:.1%}"
        )
        if baseline_accuracy is not None:
            reason += f", {edge_over_baseline:+.1%} vs the {baseline_accuracy:.1%} always-one-way baseline"
    elif mean_acc < 0.52:
        reason = (
            f"Below minimum threshold — mean accuracy {mean_acc:.1%} "
            f"(need >=52%). Treat signal as weak."
        )
    elif not beats_baseline:
        reason = (
            f"No edge over the naive baseline — mean accuracy {mean_acc:.1%} vs "
            f"{baseline_accuracy:.1%} for always predicting the more common "
            f"direction ({edge_over_baseline:+.1%}, need >="
            f"{MIN_EDGE_OVER_BASELINE:.0%}). The model is not beating a constant guess."
        )
    else:
        reason = (
            f"High variance across folds — std {std_acc:.1%} (need <=8%) over "
            f"{n_validation_samples} validation samples. Note that with small "
            f"folds this is largely sampling noise, not regime instability."
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
        "baseline_accuracy": (
            round(baseline_accuracy, 4) if baseline_accuracy is not None else None
        ),
        "edge_over_baseline": (
            round(edge_over_baseline, 4) if edge_over_baseline is not None else None
        ),
    }


def _majority_class_baseline(y_dir: pd.Series) -> float:
    """
    What "always predict the more common direction" scores on these labels — the
    reference any model has to beat to be worth running. Neutral rows are
    excluded, matching the population the model is validated on.
    """
    directional = y_dir[y_dir != 0]
    if len(directional) == 0:
        return 0.5
    bull_share = float((directional == 1).mean())
    return max(bull_share, 1.0 - bull_share)


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
    # (y_val, {model: probs}) per fold, kept so a caller can re-score the blend
    # under different weights without refitting — see _score_ensemble_weights.
    fold_records: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []

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
        fold_records.append((y_val, fold_probs))

    per_model_summary = {
        name: _summarize_fold_scores(per_model_fold_accs[name], per_model_fold_aucs[name], total_val_samples)
        for name in names
    }
    ensemble_summary = _summarize_fold_scores(
        ensemble_fold_accs, ensemble_fold_aucs, total_val_samples,
        baseline_accuracy=_majority_class_baseline(y_dir),
    )

    return {
        "n_folds": ensemble_summary["n_folds"],
        "per_model": per_model_summary,
        "ensemble_weights": ensemble_weights,
        "ensemble": ensemble_summary,
        "_fold_records": fold_records,
        "_total_val_samples": total_val_samples,
        "_baseline_accuracy": _majority_class_baseline(y_dir),
    }


def _score_ensemble_weights(
    wf_multi: Dict[str, Any],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Re-score an already-run walk-forward under a different blend, reusing each
    fold's stored per-model probabilities. No refitting.

    Needed because train_model derives its deployed ensemble weights by softmax
    over per-model fold accuracy *after* validating, and used to keep reporting
    the fixed-0.65/0.35 score — describing a blend inference never uses. At
    _ENSEMBLE_SOFTMAX_TEMPERATURE=0.05 those diverge hard (a 12-point model gap
    gives ~0.92/0.08), so the two were not interchangeable.

    Caveat kept deliberately visible: `weights` fitted on these same folds makes
    the result optimistic. It is still the right number to report, because it is
    the one describing the model that actually runs.
    """
    fold_records = wf_multi.get("_fold_records") or []
    if not fold_records:
        return wf_multi["ensemble"]

    accs: List[float] = []
    aucs: List[float] = []
    for y_val, fold_probs in fold_records:
        blended = sum(weights[name] * fold_probs[name] for name in weights)
        accs.append(_directional_accuracy(y_val, blended))
        try:
            aucs.append(float(roc_auc_score(y_val, blended)))
        except ValueError:
            aucs.append(0.5)
    return _summarize_fold_scores(
        accs, aucs, wf_multi.get("_total_val_samples", 0),
        baseline_accuracy=wf_multi.get("_baseline_accuracy"),
    )


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

def _evaluate_on_holdout(
    df: pd.DataFrame,
    ticker: str,
    horizon_days: int,
    neutral_threshold: float,
    xgb_cfg: Dict[str, Any],
    rf_cfg: Dict[str, Any],
    ensemble_weights: Dict[str, float],
    search_end: pd.Timestamp,
    uniqueness_weights: bool = False,
) -> Dict[str, Any]:
    """
    Fit on everything up to `search_end` and score the untouched tail.

    This is the only number in train_model that no search stage influenced.
    Everything else — the label scheme, both hyperparameter grids, the ensemble
    weights — was chosen by maximizing over data that the walk-forward then
    reported on, so those figures are upper bounds. Here the configuration is
    fixed first and the data is seen once.

    A `horizon_days` gap is dropped after `search_end`: the last training rows'
    label windows extend forward, so without it the first holdout rows overlap
    outcomes the model was fitted on.

    Returns {"accuracy", "n", "baseline_accuracy", "edge_over_baseline",
    "ci95_halfwidth", "reason"} — accuracy None when the tail is too small to say
    anything, with `reason` explaining why.
    """
    out: Dict[str, Any] = {
        "accuracy": None, "n": 0, "baseline_accuracy": None,
        "edge_over_baseline": None, "ci95_halfwidth": None, "reason": None,
        "holdout_start": None,
    }
    try:
        X_all, y_all = build_features(
            df, ticker=ticker,
            forward_bars=horizon_days, neutral_threshold=neutral_threshold,
        )
    except (ValueError, KeyError) as exc:
        out["reason"] = f"Could not build holdout features: {exc}"
        return out

    X_dir, y_dir = _filter_directional(X_all, y_all)
    if X_dir.empty:
        out["reason"] = "No directional rows available."
        return out

    # Features are built over the whole frame so rolling windows keep their
    # warm-up; the split is applied afterwards, by timestamp.
    gap_end = search_end + pd.Timedelta(days=horizon_days)
    train_mask = X_dir.index <= search_end
    test_mask = X_dir.index > gap_end

    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    out["n"] = n_test
    if n_test:
        out["holdout_start"] = X_dir.index[test_mask][0].isoformat()
    if n_train < 50:
        out["reason"] = f"Only {n_train} rows before the holdout split — need >=50."
        return out
    if n_test < MIN_HOLDOUT_SAMPLES:
        out["reason"] = (
            f"Only {n_test} holdout rows (need >={MIN_HOLDOUT_SAMPLES}) — too few to "
            "estimate accuracy from. Reliability falls back to walk-forward."
        )
        return out

    X_tr = X_dir.loc[train_mask, FEATURE_NAMES].values.astype("float32")
    y_tr = _to_binary_labels(y_dir.loc[train_mask])
    X_te = X_dir.loc[test_mask, FEATURE_NAMES].values.astype("float32")
    y_te = _to_binary_labels(y_dir.loc[test_mask])

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        out["reason"] = "Holdout or training split is single-class."
        return out

    try:
        fit_w = (
            _average_uniqueness(X_dir.index[train_mask], df.index, horizon_days)
            if uniqueness_weights else None
        )
        xgb_probs = _fit_predict_proba("xgb", xgb_cfg, X_tr, y_tr, X_te, y_te, fit_w)
        rf_probs = _fit_predict_proba("rf", rf_cfg, X_tr, y_tr, X_te, y_te, fit_w)
    except Exception as exc:
        out["reason"] = f"Holdout fit failed: {exc}"
        return out

    blended = (
        ensemble_weights.get("xgb", 0.65) * xgb_probs
        + ensemble_weights.get("rf", 0.35) * rf_probs
    )
    accuracy = _directional_accuracy(y_te, blended)
    baseline = _majority_class_baseline(y_dir.loc[test_mask])

    out.update({
        "accuracy": round(float(accuracy), 4),
        "baseline_accuracy": round(float(baseline), 4),
        "edge_over_baseline": round(float(accuracy - baseline), 4),
        # Binomial standard error at the observed rate. Reported so a 70% over 45
        # rows is not read as though it were 70% over 4500.
        "ci95_halfwidth": round(
            float(1.96 * np.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n_test)), 4
        ),
        "reason": f"Fitted on {n_train} rows through {search_end.date()}, scored on {n_test} later rows.",
    })
    return out


def train_model(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    uniqueness_weights: bool = False,
) -> Dict[str, Any]:
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
    uniqueness_weights : bool, default False
        Weight each training row by the average uniqueness of its label window
        (see _average_uniqueness), correcting for the fact that retained rows in
        volatile clusters overlap each other far more than isolated rows do.

        **Off by default because it was measured and did not help.** An A/B over
        eight synthetic series — same split, same configuration, only the fit
        weights differing — gave a mean holdout delta of -0.5 points, a median of
        0.0, and 3 wins in 8, against a run-to-run spread of 2.5 points. The
        weights themselves were active (0.82 to 3.7), so the mechanism worked; it
        simply produced no edge.

        That test is weak evidence rather than a refutation: synthetic random
        walks contain no signal, and no improvement to *how* a model fits can
        recover signal that is not there. Re-run scripts/ab_uniqueness_weights.py
        against real tickers before deciding. Kept wired rather than deleted so
        that test costs one flag instead of a reimplementation.

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

    # ── Carve the holdout out BEFORE any search sees the data ─────────────────
    # Every search stage below maximizes over what it is scored on, so a number
    # produced from the same rows is an upper bound. df_search is what the
    # searches get; the tail after it is scored once, at the end, by
    # _evaluate_on_holdout. If there isn't enough history to spare, the holdout
    # is skipped and reporting falls back to walk-forward with that stated.
    df_search = df
    search_end = None
    if len(df) >= 300:
        split_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
        df_search = df.iloc[:split_idx]
        search_end = df_search.index[-1]
        logger.info(
            "train_model: %s holding out %d of %d bars (searches see through %s)",
            ticker, len(df) - split_idx, len(df), search_end.date(),
        )
    else:
        logger.info(
            "train_model: %s only %d bars — no holdout, walk-forward only",
            ticker, len(df),
        )

    # ── Search for the best label horizon/threshold for this ticker ───────────
    # Catches both ValueError (raised deliberately by build_features() for
    # too-few-rows) and KeyError (raised by pandas when df is missing expected
    # indicator columns, e.g. a caller passed raw OHLCV without ever running
    # it through calculate_indicators()). Both are caller-input problems, not
    # bugs in the search itself, so both should degrade to a structured error
    # dict rather than propagate as an uncaught exception.
    try:
        label_choice = select_label_scheme(df_search, ticker)
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
    # Re-score under the weights that will actually be deployed. Without this,
    # directional_accuracy/is_reliable describe the fixed 0.65/0.35 blend above
    # while predict() runs the softmax blend — two different models.
    wf = _score_ensemble_weights(wf_multi, ensemble_weights)

    # How much of the reported accuracy came from picking the best of the label
    # grid rather than from signal. select_label_scheme scores every candidate on
    # the same history and keeps the argmax with no holdout, which on random
    # walks alone is worth ~7 points — so the winner's score is not a clean
    # estimate. Recording the gap against the grid's default entry makes the
    # selection component visible instead of silently baked into the headline.
    selection_lift = None
    try:
        default_scheme = LABEL_SEARCH_GRID[0]
        default_score = next(
            (
                c["mean_accuracy"] for c in label_choice.get("candidates", [])
                if c.get("horizon_days") == default_scheme[0]
                and c.get("neutral_threshold") == default_scheme[1]
            ),
            None,
        )
        if default_score is not None:
            selection_lift = round(wf["mean_directional_accuracy"] - default_score, 4)
    except Exception as exc:
        logger.debug("train_model: selection_lift unavailable for %s: %s", ticker, exc)

    # ── Out-of-sample check on the withheld tail ──────────────────────────────
    # The configuration is fixed at this point, so this is the one number no
    # search stage influenced. When it has enough rows it, not the walk-forward,
    # decides is_reliable — the walk-forward figure carries the selection bias
    # that selection_lift only makes visible.
    holdout = {"accuracy": None, "n": 0, "reason": "No holdout — insufficient history."}
    if search_end is not None:
        holdout = _evaluate_on_holdout(
            df, ticker, horizon_days, neutral_threshold,
            xgb_cfg, rf_cfg, ensemble_weights, search_end,
            uniqueness_weights=uniqueness_weights,
        )
        if holdout["accuracy"] is None:
            logger.warning("train_model: %s holdout unusable — %s", ticker, holdout["reason"])
        else:
            logger.info(
                "train_model: %s holdout accuracy=%.3f (n=%d, +/-%.1f%% at 95%%) "
                "vs walk-forward %.3f — gap %+.3f",
                ticker, holdout["accuracy"], holdout["n"],
                holdout["ci95_halfwidth"] * 100,
                wf["mean_directional_accuracy"],
                holdout["accuracy"] - wf["mean_directional_accuracy"],
            )

    # Prefer the honest estimate for the gate when it is usable.
    if holdout["accuracy"] is not None:
        edge = holdout["edge_over_baseline"]
        # The edge must clear both a floor and the sampling noise of the holdout
        # itself. A fixed 2-point bar is meaningless at n=75, where the 95%
        # interval on the accuracy is around +/-11 points — a model with no edge
        # at all lands several points above baseline by luck routinely. Measured
        # on drift-free random walks, the fixed bar alone passed 2 of 5; adding
        # this term rejects the ones whose apparent edge is inside their own
        # error bar.
        #
        # This uses the interval on the accuracy as a stand-in for the interval on
        # the difference. Baseline is estimated from the same rows, so the two are
        # correlated and this is a heuristic rather than an exact test — but it is
        # scaled to the sample, which the fixed threshold was not.
        required_edge = max(MIN_EDGE_OVER_BASELINE, holdout["ci95_halfwidth"] or 0.0)
        is_reliable = bool(
            holdout["accuracy"] >= 0.52
            and edge is not None
            and edge >= required_edge
        )
        if is_reliable:
            reliability_reason = (
                f"Holdout accuracy {holdout['accuracy']:.1%} on {holdout['n']} unseen rows "
                f"(+/-{holdout['ci95_halfwidth']:.1%}), {edge:+.1%} vs the "
                f"{holdout['baseline_accuracy']:.1%} always-one-way baseline — "
                "an edge larger than the holdout's own error bar."
            )
        elif holdout["accuracy"] < 0.52:
            reliability_reason = (
                f"Holdout accuracy {holdout['accuracy']:.1%} on {holdout['n']} unseen rows "
                f"is below the 52% floor, despite {wf['mean_directional_accuracy']:.1%} "
                "in-search. Treat the in-search figure as selection, not signal."
            )
        elif edge is not None and edge < MIN_EDGE_OVER_BASELINE:
            reliability_reason = (
                f"Holdout accuracy {holdout['accuracy']:.1%} does not beat the "
                f"{holdout['baseline_accuracy']:.1%} always-one-way baseline by "
                f"{MIN_EDGE_OVER_BASELINE:.0%} ({edge:+.1%}). Not better than a constant guess."
            )
        else:
            reliability_reason = (
                f"Holdout edge {edge:+.1%} over the {holdout['baseline_accuracy']:.1%} "
                f"baseline is inside the sampling noise of {holdout['n']} rows "
                f"(+/-{holdout['ci95_halfwidth']:.1%}). Indistinguishable from luck — "
                "needs more history, not a retrain."
            )
        wf = {**wf, "is_reliable": is_reliable, "reliability_reason": reliability_reason}

    logger.info(
        "train_model: validation complete — mean_acc=%.3f std=%.3f reliable=%s ensemble_weights=%s",
        wf["mean_directional_accuracy"],
        wf["std_directional_accuracy"],
        wf["is_reliable"],
        ensemble_weights,
    )

    # ── Final model trained on ALL directional data (18 core features only) ───
    # Including the holdout. The holdout exists to produce an unbiased *estimate*;
    # once that estimate is recorded there is no reason to ship a model fitted on
    # 80% of the history. Standard practice: measure on unseen data, then refit on
    # everything for deployment. X_dir_18 above came from df_search only, so
    # rebuild over the full frame using the already-chosen label scheme.
    X_fit, y_fit = X_dir_18, y_dir
    if search_end is not None:
        try:
            X_full, y_full = build_features(
                df, ticker=ticker,
                forward_bars=horizon_days, neutral_threshold=neutral_threshold,
            )
            X_full_dir, y_full_dir = _filter_directional(X_full, y_full)
            if len(X_full_dir) > len(X_dir_18):
                X_fit, y_fit = X_full_dir[FEATURE_NAMES], y_full_dir
                logger.info(
                    "train_model: %s final fit on %d rows (search set was %d)",
                    ticker, len(X_fit), len(X_dir_18),
                )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "train_model: %s could not rebuild full training set (%s) — "
                "final model fits the search set only", ticker, exc,
            )

    X_arr = X_fit.values.astype("float32")
    y_binary = _to_binary_labels(y_fit)
    fit_weights = (
        _average_uniqueness(X_fit.index, df.index, horizon_days)
        if uniqueness_weights else None
    )

    if _XGBOOST_AVAILABLE:
        xgb_final = XGBClassifier(**xgb_cfg)
        xgb_final.fit(X_arr, y_binary, sample_weight=fit_weights, verbose=False)
    else:
        xgb_final = RandomForestClassifier(**rf_cfg)
        xgb_final.fit(X_arr, y_binary, sample_weight=fit_weights)

    rf_final = RandomForestClassifier(**rf_cfg)
    rf_final.fit(X_arr, y_binary, sample_weight=fit_weights)

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
            "baseline_accuracy": wf.get("baseline_accuracy"),
            "edge_over_baseline": wf.get("edge_over_baseline"),
            "selection_lift": selection_lift,
            # The unbiased estimate. directional_accuracy above is an upper
            # bound; this is what the model scored on rows no search stage saw.
            "holdout_accuracy": holdout["accuracy"],
            "holdout_n": holdout["n"],
            "holdout_baseline_accuracy": holdout.get("baseline_accuracy"),
            "holdout_edge_over_baseline": holdout.get("edge_over_baseline"),
            "holdout_ci95_halfwidth": holdout.get("ci95_halfwidth"),
            "holdout_note": holdout.get("reason"),
            "uniqueness_weights": uniqueness_weights,
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
        "n_train": len(X_fit),
        "n_test": wf["n_validation_samples"],
        "trained_at": now_iso,
        "is_reliable": wf["is_reliable"],
        "reliability_reason": wf["reliability_reason"],
        "baseline_accuracy": wf.get("baseline_accuracy"),
        "edge_over_baseline": wf.get("edge_over_baseline"),
        "selection_lift": selection_lift,
        "holdout_accuracy": holdout["accuracy"],
        "holdout_n": holdout["n"],
        "holdout_baseline_accuracy": holdout.get("baseline_accuracy"),
        "holdout_edge_over_baseline": holdout.get("edge_over_baseline"),
        "holdout_ci95_halfwidth": holdout.get("ci95_halfwidth"),
        "holdout_note": holdout.get("reason"),
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


MIN_SIMILAR_SETUPS_FOR_WIN_RATE = 20


def _summarize_similar_setups(subset_ret: np.ndarray, direction: str) -> Dict[str, Any]:
    """
    Distribution of forward returns across every historical bar where this model
    made the same directional call (Roadmap Item 6).

    `subset_ret` is a fraction-return array; everything returned is in percent.

    Two honesty constraints:

    - **In-sample.** These bars come from the model's own training history, so the
      win rate is optimistic — the same caveat the "Signal Sharpe (IS)" metric
      carries. `is_in_sample` is always True; it exists so the UI cannot forget.
    - **`win_rate` is None below MIN_SIMILAR_SETUPS_FOR_WIN_RATE matches.** A rate
      over 6 samples reads as precision it does not have. `n` is always reported,
      so a thin sample is visible rather than hidden.
    """
    returns_pct = subset_ret * 100.0
    wins = returns_pct > 0 if direction == "bullish" else returns_pct < 0
    n = int(len(returns_pct))

    return {
        "n": n,
        "is_thin": bool(n < MIN_SIMILAR_SETUPS_FOR_WIN_RATE),
        "win_rate": (
            round(float(wins.mean()), 4) if n >= MIN_SIMILAR_SETUPS_FOR_WIN_RATE else None
        ),
        "mean_pct": round(float(np.mean(returns_pct)), 2),
        "median_pct": round(float(np.median(returns_pct)), 2),
        "p25_pct": round(float(np.percentile(returns_pct, 25)), 2),
        "p75_pct": round(float(np.percentile(returns_pct, 75)), 2),
        "worst_pct": round(float(np.min(returns_pct)), 2),
        "best_pct": round(float(np.max(returns_pct)), 2),
        "returns_pct": [round(float(r), 3) for r in returns_pct],
        "is_in_sample": True,
    }


def predict(
    ticker: str, df: Optional[pd.DataFrame] = None, *, persist: bool = True,
) -> Dict[str, Any]:
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
        "similar_setups": None,
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
    result["neutral_threshold"] = neutral_threshold
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

    # ── Expected move estimate + similar historical setups ───────────────────
    # Uses the same horizon_days/neutral_threshold this model was trained with,
    # so the N-day forward return matches the model's own label definition.
    #
    # The `predicted_mask` below IS a similar-setups search: every historical bar
    # where this model would have made the same directional call. It was already
    # being computed and then thrown away except for its median, so
    # similar_setups reports the rest of the distribution (Roadmap Item 6).
    #
    # In-sample by construction — the model is scoring bars from its own training
    # history, so the win rate here is optimistic in the same way the "Signal
    # Sharpe (IS)" metric is. Flagged as such rather than presented as a forecast.
    expected_move_pct = None
    similar_setups = None
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
                similar_setups = _summarize_similar_setups(subset_ret, direction)
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
        "similar_setups": similar_setups,
        "confidence": confidence,
        "top_features": top_features,
        "price_at_prediction": float(df["Close"].iloc[-1]),
        "indicator_snapshot": indicator_snapshot,
    })

    # ── Persist prediction ────────────────────────────────────────────────────
    # persist=False exists for automated callers (alert sweeps, scanners). Every
    # row appended here counts toward the live win rate that Model Lab, the
    # retrain triggers, and the horizon scoreboard all read, so a background loop
    # writing on every poll would silently corrupt all three.
    if persist:
        save_prediction(ticker, result)

    logger.info(
        f"predict: {ticker} complete — direction={direction} probability={result['probability']} "
        f"confidence={confidence}"
    )

    return result


def _rewrite_jsonl_atomic(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Replace a JSONL file's contents without a window in which it is truncated.

    `open(path, "w")` zeroes the file before the first record lands, so a crash
    or a full disk mid-write leaves an empty prediction log — and `storage/` is
    gitignored with no backup. Writing to a sibling temp file and renaming makes
    the swap atomic at the filesystem level: readers see either the old contents
    or the new ones, never nothing.

    The lock is held across the read-modify-write in resolve_predictions, so a
    concurrent Streamlit render and cron sweep cannot interleave and lose the
    appends that landed between one's read and its write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def _jsonl_lock(path: Path):
    """
    Advisory exclusive lock for a read-modify-write cycle on a JSONL log.

    resolve_predictions reads every record, makes a network call, then rewrites
    the whole file. Without a lock, a save_prediction() landing inside that
    window — seconds wide, because of the fetch — is erased on rewrite.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def get_prediction_history(ticker: str, resolve: bool = True) -> pd.DataFrame:
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

    Parameters
    ----------
    resolve : bool
        When True (the default, preserving existing behaviour) back-fill any
        elapsed predictions first. Read-only callers should pass False: resolving
        fetches price history and rewrites this file, which is not what a caller
        asking to *read* the log expects.
    """
    ticker = ticker.upper()
    if resolve:
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
            # Band the model was trained under, so resolve_predictions() grades
            # the same population directional_accuracy was measured on.
            "neutral_threshold": prediction.get("neutral_threshold"),
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
        and not r.get("neutral_outcome")
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

        # A prediction older than the fetched window has no bar at or before it
        # in `dates`, so `dates > predicted_at` would start at the very first
        # bar and grade it against a close ~2 years after it was made.
        if predicted_at < dates[0]:
            logger.debug(
                "resolve_predictions: %s prediction at %s predates the fetched "
                "window — left unresolved", ticker, predicted_at,
            )
            continue

        future_closes = closes[dates > predicted_at]
        if len(future_closes) < horizon:
            continue  # horizon hasn't elapsed yet — leave unresolved for now

        entry_price = float(record["price_at_prediction"])
        exit_price = float(future_closes.iloc[horizon - 1])
        actual_raw = (exit_price - entry_price) / entry_price * 100
        actual_return_pct = round(actual_raw, 2)
        record["actual_outcome"] = actual_return_pct

        # Grade inside the neutral band the model was trained under.
        # _filter_directional drops every neutral row before validation, so
        # directional_accuracy is conditional on the move clearing the band;
        # grading sub-band outcomes here made live accuracy measure a strictly
        # harder question and opened a permanent gap that check_performance_
        # drop_trigger reads as degradation.
        neutral_threshold = record.get("neutral_threshold")
        if neutral_threshold is None:
            neutral_threshold = _load_model_metadata(ticker).get("neutral_threshold")
        if neutral_threshold is not None and abs(actual_raw) <= float(neutral_threshold) * 100:
            record["correct"] = None
            record["neutral_outcome"] = True
            resolved_count += 1
            continue

        # Sign the unrounded return. Rounding to 2dp first mapped any move under
        # 0.005% to exactly 0.0, which satisfied neither branch and was stored as
        # incorrect regardless of direction.
        record["correct"] = bool(
            actual_raw > 0 if record["direction"] == "bullish" else actual_raw < 0
        )
        resolved_count += 1

    if not resolved_count:
        return 0

    # Apply the resolutions to a fresh read taken under the lock, rather than
    # writing back the snapshot captured before the price fetch. That snapshot is
    # seconds stale — the fetch sits inside the window — so writing it back
    # erased any prediction save_prediction() appended in the meantime.
    updates = {
        r["predicted_at"]: {
            k: r[k] for k in ("actual_outcome", "correct", "neutral_outcome") if k in r
        }
        for r in pending
        if r.get("predicted_at")
        and (r.get("correct") is not None or r.get("neutral_outcome"))
    }

    applied = 0
    with _jsonl_lock(path):
        current: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    current.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        for rec in current:
            update = updates.get(rec.get("predicted_at"))
            if update and rec.get("correct") is None and not rec.get("neutral_outcome"):
                rec.update(update)
                applied += 1
        if applied:
            _rewrite_jsonl_atomic(path, current)

    logger.debug(
        "resolve_predictions: %s resolved %d pending prediction(s)", ticker, applied
    )
    return applied


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
    out["baseline_accuracy"] = wf.get("baseline_accuracy")
    out["edge_over_baseline"] = wf.get("edge_over_baseline")


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

