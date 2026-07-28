"""
Intraday ML direction prediction — 15-minute (and other intraday) bars.

Deliberately a **separate module** from analysis/ml_prediction.py rather than an
interval parameter threaded through it. `ml_prediction.predict()` is called from
three pages (Trading Desk, Research, Screener), so modifying it to
serve intraday bars would put all three in the blast radius of every change here.
This module instead **reuses ml_prediction's scaffolding read-only** — the
walk-forward runner, the model configs, and the label helpers are all pure and
bar-size agnostic, so they can be imported and called without being edited.

Nothing in this module writes to a daily model's files. Storage paths carry the
interval (`SPY_15m_xgb.pkl`), so a daily `SPY_xgb.pkl` and its prediction
history are never opened for writing from here.

Two things genuinely differ from the daily model, and both are correctness
issues rather than plumbing:

1. **Volatility-scaled neutral thresholds.** The daily grid uses a fixed +/-0.5%
   band. On 15-minute bars a 5-bar move spans 75 minutes, where 0.5% is a very
   large move — it would label 66-93% of bars neutral, and `_filter_directional`
   drops every neutral row before training. The survivors would all come from
   high-volatility windows: a biased sample that won't generalize. Here the
   threshold is a multiple of realized per-bar sigma scaled by sqrt(horizon), so
   the directional/neutral split stays sane at any bar size.

2. **Session-boundary label masking.** A forward return computed with
   `pct_change(n).shift(-n)` across an intraday index silently spans the
   overnight gap for the last n bars of every session. That gap is not something
   intraday features can predict, so those rows are labelled NaN and dropped
   rather than training the model on noise.

Intraday features add what actually matters within a session — minutes since the
open, position relative to session VWAP and the opening range — and drop
`day_of_week`, which carries almost no information inside a 60-day window.

Data ceiling: the provider caps 15m history at roughly 60 calendar days (~1560
bars). That is more rows than the daily model's 2-year set, but only 60 days of
regime diversity, so an intraday model goes stale far faster than a daily one.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config.tz import MARKET_TZ, now_et_iso

# ── Read-only reuse from the daily module ────────────────────────────────────
# These are pure, stateless, and make no assumption about bar size. Importing
# them keeps the ensemble/validation logic identical between daily and intraday
# without editing analysis/ml_prediction.py.
from analysis.ml_prediction import (
    _directional_accuracy,
    _filter_directional,
    _rf_config,
    _run_walk_forward,
    _to_binary_labels,
    _xgb_config,
    _STORAGE_DIR,
)
from data.feature_engineering import class_balance_check

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning("intraday_prediction: xgboost unavailable — ensemble falls back to RandomForest only")

from sklearn.ensemble import RandomForestClassifier

# ── Supported intervals ──────────────────────────────────────────────────────
# bars_per_session is used to scale volatility and normalization windows. Values
# assume a 6.5-hour regular session (09:30-16:00 ET).
INTERVAL_SPECS: Dict[str, Dict[str, Any]] = {
    "5m":  {"minutes": 5,  "bars_per_session": 78, "max_period": "60d"},
    "15m": {"minutes": 15, "bars_per_session": 26, "max_period": "60d"},
    "30m": {"minutes": 30, "bars_per_session": 13, "max_period": "60d"},
    "1h":  {"minutes": 60, "bars_per_session": 7,  "max_period": "180d"},
}
DEFAULT_INTERVAL = "15m"

# Horizons in *bars*, not days. At 15m these are 45 / 75 / 150 minutes.
HORIZON_SEARCH_GRID: List[int] = [3, 5, 10]

# Neutral band as a multiple of the realized sigma of an h-bar move. 0.75 keeps
# roughly 45-55% of rows directional at typical index volatility — enough signal
# to train on without discarding most of the sample.
SIGMA_MULTIPLE_GRID: List[float] = [0.5, 0.75, 1.0]

MIN_ROWS_REQUIRED: int = 200
MIN_DIRECTIONAL_SAMPLES: int = 150
_SEARCH_N_SPLITS: int = 4
_MIN_SEARCH_ACCURACY: float = 0.50

# Feature list for intraday models. Independent of feature_engineering's
# FEATURE_NAMES, which must stay frozen for daily model compatibility.
INTRADAY_FEATURE_NAMES: List[str] = [
    "rsi_norm",
    "rsi_5_norm",
    "macd_hist_sign",
    "adx_norm",
    "atr_pct",
    "bb_pct",
    "vol_ratio",
    "price_vs_sma20",
    "price_vs_sma50",
    "ret_1b",              # 1-bar return
    "ret_3b",
    "ret_5b",
    "hv_ratio",
    "obv_slope",
    "stoch_k_norm",
    "hl_range_pct",
    # ── Intraday-specific ────────────────────────────────────────────────────
    "minutes_since_open_norm",   # 0 at the open, 1 at the close
    "price_vs_vwap",             # (Close - session VWAP) / session VWAP
    "vwap_dist_atr",             # same distance expressed in ATRs
    "pos_in_session_range",      # 0 at session low, 1 at session high
    "pos_in_opening_range",      # position relative to the first 30 min
    "ret_since_open",            # return from the session's first close
]


# ── Storage (interval-scoped — never collides with daily model files) ────────

def _interval_tag(interval: str) -> str:
    if interval not in INTERVAL_SPECS:
        raise ValueError(f"interval must be one of {sorted(INTERVAL_SPECS)}; got {interval!r}")
    return interval


def _xgb_path(ticker: str, interval: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_{_interval_tag(interval)}_xgb.pkl"


def _rf_path(ticker: str, interval: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_{_interval_tag(interval)}_rf.pkl"


def _accuracy_path(ticker: str, interval: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_{_interval_tag(interval)}_accuracy.json"


def _predictions_path(ticker: str, interval: str) -> Path:
    return _STORAGE_DIR / f"{ticker.upper()}_{_interval_tag(interval)}_predictions.jsonl"


def model_exists(ticker: str, interval: str = DEFAULT_INTERVAL) -> bool:
    return _xgb_path(ticker, interval).exists() and _rf_path(ticker, interval).exists()


def load_metadata(ticker: str, interval: str = DEFAULT_INTERVAL) -> Dict[str, Any]:
    """Read the saved accuracy/config record, or {} if the model is untrained."""
    path = _accuracy_path(ticker, interval)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("load_metadata: could not read %s: %s", path, exc)
        return {}


# ── Session helpers ──────────────────────────────────────────────────────────

def _to_market_tz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the index to America/New_York. A naive index is assumed to already
    be market-local (the provider returns exchange-local timestamps for
    intraday intervals) and is localized rather than shifted — converting from
    UTC instead would move 09:30 ET to 05:30 and break every session boundary.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(MARKET_TZ)
    else:
        idx = idx.tz_convert(MARKET_TZ)
    out = df.copy()
    out.index = idx
    return out


def _session_ids(index: pd.DatetimeIndex) -> pd.Series:
    """Integer session id per bar, so boundaries can be compared cheaply."""
    dates = index.normalize()
    return pd.Series(pd.factorize(dates)[0], index=index)


def _minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    mins = (index.hour - 9) * 60 + (index.minute - 30)
    return pd.Series(mins, index=index).astype("float32")


# ── Labels ───────────────────────────────────────────────────────────────────

def build_intraday_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    sigma_multiple: float,
    vol_window: int = 130,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Forward-return direction labels for intraday bars.

    Three differences from the daily labeller, all correctness rather than
    plumbing:

    - The forward window must stay inside one session. A bar whose h-bar-ahead
      close belongs to a later session gets a NaN label and is dropped, so the
      model is never trained to "predict" an overnight gap it cannot see.
    - The neutral band is `sigma_multiple * sigma_t * sqrt(horizon_bars)`. A
      fixed percentage band calibrated for daily bars labels the overwhelming
      majority of intraday bars neutral, and the survivors come only from
      high-volatility windows — a biased training sample.
    - `sigma_t` is a **trailing** estimate over `vol_window` bars, using only
      returns up to and including bar t. A full-series sigma would make bar t's
      label depend on future bars, which is label-definition leakage, and would
      also fix one threshold across changing volatility regimes.

    Sigma is estimated from within-session returns only. The first return of
    each session spans the overnight gap, which is far larger than an intraday
    bar move and would inflate the estimate.

    Returns (labels, info) where labels holds {1, -1, 0, NaN}.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    close = df["Close"]
    fwd_ret = close.pct_change(horizon_bars).shift(-horizon_bars)
    sessions = _session_ids(df.index)

    intra_session_ret = close.pct_change().where(sessions.shift(1) == sessions)
    sigma_t = intra_session_ret.rolling(vol_window, min_periods=30).std()
    sigma_t = sigma_t.fillna(intra_session_ret.expanding(min_periods=30).std())
    if not np.isfinite(sigma_t.dropna()).any() or (sigma_t.dropna() <= 0).all():
        raise ValueError("Cannot estimate bar volatility — price series is flat or too short")

    horizon_sigma_t = sigma_t * np.sqrt(horizon_bars)
    threshold_t = sigma_multiple * horizon_sigma_t

    # Session mask: keep only bars whose forward window stays in-session.
    same_session = sessions.shift(-horizon_bars) == sessions
    n_before = int(fwd_ret.notna().sum())
    fwd_ret = fwd_ret.where(same_session)
    n_after = int(fwd_ret.notna().sum())

    labels = pd.Series(np.nan, index=df.index, dtype="float32")
    decidable = fwd_ret.notna() & threshold_t.notna() & (threshold_t > 0)
    labels[decidable & (fwd_ret > threshold_t)] = 1
    labels[decidable & (fwd_ret < -threshold_t)] = -1
    labels[decidable & (fwd_ret.abs() <= threshold_t)] = 0

    median_sigma = float(horizon_sigma_t.median())
    info = {
        "horizon_bars": horizon_bars,
        "sigma_multiple": sigma_multiple,
        "vol_window": vol_window,
        "bar_sigma": float(sigma_t.median()),
        "horizon_sigma": median_sigma,
        "threshold": float(threshold_t.median()),
        "threshold_pct": round(float(threshold_t.median()) * 100, 4),
        "dropped_to_session_mask": n_before - n_after,
    }
    return labels, info


# ── Features ─────────────────────────────────────────────────────────────────

def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute INTRADAY_FEATURE_NAMES onto a copy of `df`, which must already carry
    the indicator columns from calculate_indicators(). Every value is derived
    from bar t or earlier — no forward-looking references.
    """
    out = df.copy()
    sessions = _session_ids(out.index)
    grp = out.groupby(sessions.values, sort=False)

    # Standard technicals, bar-count based rather than day based.
    out["rsi_norm"] = out["RSI"] / 100.0
    delta = out["Close"].diff()
    gain5 = delta.clip(lower=0).rolling(5).mean()
    loss5 = (-delta.clip(upper=0)).rolling(5).mean()
    rs5 = gain5 / loss5.replace(0, np.nan)
    out["rsi_5_norm"] = (100 - (100 / (1 + rs5))) / 100.0
    out["macd_hist_sign"] = np.sign(out["MACD_hist"].fillna(0)).astype("float32")
    out["adx_norm"] = (out["ADX"] / 100.0).clip(0, 1)
    out["atr_pct"] = out["ATR_pct"]
    out["bb_pct"] = out["BB_pct"].clip(-0.2, 1.2)
    out["vol_ratio"] = out["vol_ratio"].clip(0, 10)
    out["price_vs_sma20"] = (out["Close"] - out["SMA_20"]) / out["SMA_20"]
    out["price_vs_sma50"] = (out["Close"] - out["SMA_50"]) / out["SMA_50"]
    out["ret_1b"] = out["Close"].pct_change(1)
    out["ret_3b"] = out["Close"].pct_change(3)
    out["ret_5b"] = out["Close"].pct_change(5)

    hv_short = out["Close"].pct_change().rolling(10).std()
    hv_long = out["Close"].pct_change().rolling(30).std()
    out["hv_ratio"] = (hv_short / hv_long.replace(0, np.nan)).clip(0.2, 3.0)

    out["obv_slope"] = np.sign(out["OBV"].diff(5).fillna(0)).astype("float32")
    out["stoch_k_norm"] = (out["STOCH_K"] / 100.0).clip(0, 1)
    out["hl_range_pct"] = (out["High"] - out["Low"]) / out["Close"]

    # ── Intraday-specific ────────────────────────────────────────────────────
    mins = _minutes_since_open(out.index)
    out["minutes_since_open_norm"] = (mins / 390.0).clip(0, 1)

    vwap = out["VWAP"] if "VWAP" in out.columns else out["Close"]
    out["price_vs_vwap"] = (out["Close"] - vwap) / vwap.replace(0, np.nan)
    atr = out["ATR"].replace(0, np.nan)
    out["vwap_dist_atr"] = ((out["Close"] - vwap) / atr).clip(-5, 5)

    # Position within the session's range so far (expanding, so no lookahead).
    sess_high = grp["High"].cummax()
    sess_low = grp["Low"].cummin()
    span = (sess_high - sess_low).replace(0, np.nan)
    out["pos_in_session_range"] = ((out["Close"] - sess_low) / span).clip(0, 1)

    # Position relative to the first 30 minutes of the session.
    in_or = mins < 30
    or_high = out["High"].where(in_or).groupby(sessions.values).cummax().ffill()
    or_low = out["Low"].where(in_or).groupby(sessions.values).cummin().ffill()
    or_span = (or_high - or_low).replace(0, np.nan)
    out["pos_in_opening_range"] = ((out["Close"] - or_low) / or_span).clip(-3, 4)

    first_close = grp["Close"].transform("first")
    out["ret_since_open"] = (out["Close"] - first_close) / first_close

    return out


def build_intraday_features(
    df: pd.DataFrame,
    ticker: str = "",
    horizon_bars: int = 5,
    sigma_multiple: float = 0.75,
    vol_window: Optional[int] = None,
    interval: str = DEFAULT_INTERVAL,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Build (X, y, label_info) for intraday training.

    `df` must be the output of calculate_indicators() on intraday bars. The
    index is normalized to market time here, so a naive or UTC index is fine.

    `vol_window` defaults to five sessions' worth of bars for `interval`, which
    keeps the trailing volatility estimate comparable across bar sizes.
    """
    if df is None or len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"build_intraday_features requires at least {MIN_ROWS_REQUIRED} bars; "
            f"got {0 if df is None else len(df)}"
        )

    if vol_window is None:
        vol_window = INTERVAL_SPECS[_interval_tag(interval)]["bars_per_session"] * 5

    market_df = _to_market_tz(df)
    labels, info = build_intraday_labels(market_df, horizon_bars, sigma_multiple, vol_window)
    featured = _add_intraday_features(market_df)

    X_raw = featured[INTRADAY_FEATURE_NAMES]
    y_raw = labels.reindex(X_raw.index)

    valid = y_raw.notna() & X_raw.notna().all(axis=1) & np.isfinite(X_raw).all(axis=1)
    X = X_raw.loc[valid].astype("float32")
    y = y_raw.loc[valid].astype("int8")

    if len(X) == 0:
        raise ValueError(
            f"{ticker + ': ' if ticker else ''}build_intraday_features produced zero "
            "valid samples. Supply more intraday history."
        )

    info["n_samples"] = len(X)
    info["n_input_bars"] = len(df)
    logger.debug(
        "build_intraday_features: %s %d bars -> %d samples (threshold %.4f%%, "
        "session mask dropped %d)",
        ticker, len(df), len(X), info["threshold_pct"], info["dropped_to_session_mask"],
    )
    return X, y, info


def normalize_intraday_features(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling z-score using only past observations. `window` is in *bars* and
    should be set from the interval (see train_intraday_model), not inherited
    from the daily module's 252 — which would mean ~10 days at 15m, not a year.
    """
    skip = {"macd_hist_sign", "obv_slope", "minutes_since_open_norm"}
    result = X.copy()
    for col in X.columns:
        if col in skip:
            continue
        roll_mean = X[col].rolling(window, min_periods=30).mean()
        roll_std = X[col].rolling(window, min_periods=30).std()
        exp_mean = X[col].expanding(min_periods=10).mean()
        exp_std = X[col].expanding(min_periods=10).std()
        mean = roll_mean.fillna(exp_mean)
        std = roll_std.fillna(exp_std).replace(0, 1e-8)
        result[col] = (X[col] - mean) / std
    return result.astype("float32").fillna(0.0)


# ── Cost-aware reliability ───────────────────────────────────────────────────

def assess_tradeability(
    mean_accuracy: float,
    horizon_sigma: float,
    round_trip_cost_pct: float = 0.02,
) -> Dict[str, Any]:
    """
    Accuracy alone is misleading at intraday horizons. A 53%-accurate model
    whose average move is 0.2% can be statistically real and still lose money
    once spread and commission are paid.

    Models the edge as: (2 * accuracy - 1) * average favourable move, i.e. the
    expected signed return per trade from taking the predicted direction, then
    subtracts an assumed round-trip cost.

    `round_trip_cost_pct` defaults to 0.02% (2 bps), roughly spread plus
    commission on a liquid index ETF. Raise it for anything less liquid.
    """
    avg_move_pct = horizon_sigma * 100 * 0.8   # mean |move| ~ 0.8 sigma
    gross_edge_pct = (2 * mean_accuracy - 1) * avg_move_pct
    net_edge_pct = gross_edge_pct - round_trip_cost_pct

    breakeven_accuracy = (
        (round_trip_cost_pct / avg_move_pct + 1) / 2 if avg_move_pct > 0 else 1.0
    )

    return {
        "avg_move_pct": round(avg_move_pct, 4),
        "gross_edge_pct": round(gross_edge_pct, 4),
        "round_trip_cost_pct": round_trip_cost_pct,
        "net_edge_pct": round(net_edge_pct, 4),
        "breakeven_accuracy": round(breakeven_accuracy, 4),
        "is_tradeable": bool(net_edge_pct > 0),
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_intraday_history(ticker: str, interval: str = DEFAULT_INTERVAL) -> Optional[pd.DataFrame]:
    """Fetch the maximum intraday history the provider allows for `interval`."""
    from data.price_data import get_price_history
    from analysis.indicators import calculate_indicators

    spec = INTERVAL_SPECS[_interval_tag(interval)]
    raw = get_price_history(ticker, period=spec["max_period"], interval=interval)
    if raw is None or raw.empty:
        logger.warning("load_intraday_history: no %s data for %s", interval, ticker)
        return None
    return calculate_indicators(raw)


# ── Label / hyperparameter search ────────────────────────────────────────────

def select_intraday_label_scheme(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
) -> Dict[str, Any]:
    """
    Search (horizon_bars, sigma_multiple) for the combination this ticker's
    ensemble validates best on, using a reduced-fold walk-forward.

    Falls back to (5 bars, 0.75 sigma) if nothing clears _MIN_SEARCH_ACCURACY.
    """
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    norm_window = INTERVAL_SPECS[interval]["bars_per_session"] * 10

    for horizon_bars in HORIZON_SEARCH_GRID:
        for sigma_multiple in SIGMA_MULTIPLE_GRID:
            try:
                X, y, info = build_intraday_features(
                    df, ticker=ticker, horizon_bars=horizon_bars,
                    sigma_multiple=sigma_multiple, interval=interval,
                )
            except ValueError as exc:
                logger.debug(
                    "select_intraday_label_scheme: %s skipped h=%d m=%.2f: %s",
                    ticker, horizon_bars, sigma_multiple, exc,
                )
                continue

            X_dir, y_dir = _filter_directional(X, y)
            if len(X_dir) < MIN_DIRECTIONAL_SAMPLES:
                continue

            X_norm = normalize_intraday_features(X_dir, window=norm_window)
            balance = class_balance_check(y_dir)
            xgb_cfg = _xgb_config(scale_pos_weight=balance["recommended_scale_pos_weight"])
            rf_cfg = _rf_config(n_features=len(INTRADAY_FEATURE_NAMES))
            wf = _run_walk_forward(
                X_norm, y_dir, xgb_cfg, rf_cfg,
                n_splits=_SEARCH_N_SPLITS, gap=horizon_bars,
            )
            if wf["n_folds"] == 0:
                continue

            entry = {
                "horizon_bars": horizon_bars,
                "sigma_multiple": sigma_multiple,
                "threshold_pct": info["threshold_pct"],
                "mean_accuracy": wf["mean_directional_accuracy"],
                "std_accuracy": wf["std_directional_accuracy"],
                "n_folds": wf["n_folds"],
                "n_directional": len(X_dir),
                "neutral_pct": balance["neutral_pct"],
                "_X": X_norm,
                "_y": y_dir,
                "_info": info,
            }
            candidates.append(entry)
            if best is None or entry["mean_accuracy"] > best["mean_accuracy"]:
                best = entry

    if best is None or best["mean_accuracy"] < _MIN_SEARCH_ACCURACY:
        logger.warning(
            "select_intraday_label_scheme: %s no candidate cleared %.2f — using defaults",
            ticker, _MIN_SEARCH_ACCURACY,
        )
        X, y, info = build_intraday_features(
            df, ticker=ticker, horizon_bars=5, sigma_multiple=0.75, interval=interval,
        )
        X_dir, y_dir = _filter_directional(X, y)
        if len(X_dir) < MIN_DIRECTIONAL_SAMPLES:
            raise ValueError(
                f"Only {len(X_dir)} directional samples at the default label scheme "
                f"(need {MIN_DIRECTIONAL_SAMPLES}). Not enough intraday history."
            )
        best = {
            "horizon_bars": 5,
            "sigma_multiple": 0.75,
            "threshold_pct": info["threshold_pct"],
            "_X": normalize_intraday_features(X_dir, window=norm_window),
            "_y": y_dir,
            "_info": info,
        }

    return {
        "horizon_bars": best["horizon_bars"],
        "sigma_multiple": best["sigma_multiple"],
        "threshold_pct": best["threshold_pct"],
        "X": best["_X"],
        "y": best["_y"],
        "info": best["_info"],
        "candidates": sorted(
            [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates],
            key=lambda c: c.get("mean_accuracy", 0.0), reverse=True,
        ),
    }


# ── Train ────────────────────────────────────────────────────────────────────

def train_intraday_model(
    ticker: str,
    interval: str = DEFAULT_INTERVAL,
    df: Optional[pd.DataFrame] = None,
    round_trip_cost_pct: float = 0.02,
) -> Dict[str, Any]:
    """
    Train the intraday XGBoost + RandomForest ensemble for one ticker/interval.

    Writes only `{TICKER}_{interval}_*` files — a daily model's `{TICKER}_xgb.pkl`
    and prediction history are never touched.

    Returns a summary dict; on failure returns {"error": "..."} rather than
    raising, matching the daily module's contract so the UI can render either.
    """
    ticker = ticker.upper().strip()
    interval = _interval_tag(interval)

    result: Dict[str, Any] = {
        "ticker": ticker, "interval": interval, "error": None,
        "horizon_bars": None, "directional_accuracy": 0.0,
    }

    if df is None:
        df = load_intraday_history(ticker, interval)
    if df is None or df.empty:
        result["error"] = f"No {interval} price data available for {ticker}."
        return result
    if len(df) < MIN_ROWS_REQUIRED:
        result["error"] = (
            f"Only {len(df)} bars of {interval} history for {ticker}; "
            f"need at least {MIN_ROWS_REQUIRED}."
        )
        return result

    try:
        choice = select_intraday_label_scheme(df, ticker, interval)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    X, y_dir = choice["X"], choice["y"]
    horizon_bars = choice["horizon_bars"]
    balance = class_balance_check(y_dir)
    xgb_cfg = _xgb_config(scale_pos_weight=balance["recommended_scale_pos_weight"])
    rf_cfg = _rf_config(n_features=len(INTRADAY_FEATURE_NAMES))

    wf = _run_walk_forward(X, y_dir, xgb_cfg, rf_cfg, n_splits=10, gap=horizon_bars)
    if wf["n_folds"] == 0:
        result["error"] = "Walk-forward validation produced no valid folds — need more history."
        return result

    tradeability = assess_tradeability(
        wf["mean_directional_accuracy"], choice["info"]["horizon_sigma"], round_trip_cost_pct,
    )

    # Final fit on all directional data.
    X_arr = X.values.astype("float32")
    y_binary = _to_binary_labels(y_dir)
    if _XGBOOST_AVAILABLE:
        xgb_final = XGBClassifier(**xgb_cfg)
        xgb_final.fit(X_arr, y_binary, verbose=False)
    else:
        xgb_final = RandomForestClassifier(**rf_cfg)
        xgb_final.fit(X_arr, y_binary)
    rf_final = RandomForestClassifier(**rf_cfg)
    rf_final.fit(X_arr, y_binary)

    now_iso = now_et_iso()
    record = {
        "ticker": ticker,
        "interval": interval,
        "horizon_bars": horizon_bars,
        "horizon_minutes": horizon_bars * INTERVAL_SPECS[interval]["minutes"],
        "sigma_multiple": choice["sigma_multiple"],
        "threshold_pct": choice["threshold_pct"],
        "directional_accuracy": wf["mean_directional_accuracy"],
        "accuracy_std": wf["std_directional_accuracy"],
        "mean_auc": wf["mean_auc"],
        "is_reliable": wf["is_reliable"],
        "trained_at": now_iso,
        "n_train": len(X),
        "tradeability": tradeability,
    }
    try:
        joblib.dump(xgb_final, _xgb_path(ticker, interval))
        joblib.dump(rf_final, _rf_path(ticker, interval))
        with open(_accuracy_path(ticker, interval), "w") as f:
            json.dump(record, f)
    except Exception as exc:
        logger.error("train_intraday_model: save failed: %s", exc)
        result["error"] = f"Model save failed: {exc}"
        return result

    logger.info(
        "train_intraday_model: %s %s h=%d bars acc=%.3f+/-%.3f reliable=%s tradeable=%s",
        ticker, interval, horizon_bars, wf["mean_directional_accuracy"],
        wf["std_directional_accuracy"], wf["is_reliable"], tradeability["is_tradeable"],
    )

    result.update(record)
    result.update({
        "reliability_reason": wf["reliability_reason"],
        "fold_accuracies": wf["fold_accuracies"],
        "n_validation_samples": wf["n_validation_samples"],
        "class_balance": balance,
        "label_search": choice["candidates"],
        "session_mask_dropped": choice["info"]["dropped_to_session_mask"],
    })
    return result


# ── Predict ──────────────────────────────────────────────────────────────────

def predict_intraday(
    ticker: str,
    interval: str = DEFAULT_INTERVAL,
    df: Optional[pd.DataFrame] = None,
    auto_train: bool = True,
    round_trip_cost_pct: float = 0.02,
) -> Dict[str, Any]:
    """
    Direction prediction for the most recent intraday bar.

    Trains first if no model exists for this ticker/interval and `auto_train`.
    Returns {"error": ...} rather than raising, so the UI can always render.
    """
    ticker = ticker.upper().strip()
    interval = _interval_tag(interval)

    if df is None:
        df = load_intraday_history(ticker, interval)
    if df is None or df.empty:
        return {"error": f"No {interval} price data available for {ticker}.",
                "ticker": ticker, "interval": interval}

    if not model_exists(ticker, interval):
        if not auto_train:
            return {"error": f"No {interval} model trained for {ticker}.",
                    "ticker": ticker, "interval": interval}
        train_result = train_intraday_model(
            ticker, interval, df=df, round_trip_cost_pct=round_trip_cost_pct,
        )
        if train_result.get("error"):
            return {"error": train_result["error"], "ticker": ticker, "interval": interval}

    meta = load_metadata(ticker, interval)
    horizon_bars = int(meta.get("horizon_bars") or 5)
    sigma_multiple = float(meta.get("sigma_multiple") or 0.75)
    norm_window = INTERVAL_SPECS[interval]["bars_per_session"] * 10

    try:
        X, _y, info = build_intraday_features(
            df, ticker=ticker, horizon_bars=horizon_bars,
            sigma_multiple=sigma_multiple, interval=interval,
        )
    except ValueError as exc:
        return {"error": str(exc), "ticker": ticker, "interval": interval}

    # The most recent bar has no label (its forward window hasn't happened), so
    # build the inference row from the full feature frame rather than from X.
    market_df = _to_market_tz(df)
    featured = _add_intraday_features(market_df)[INTRADAY_FEATURE_NAMES]
    featured = featured.replace([np.inf, -np.inf], np.nan)
    normed = normalize_intraday_features(featured.ffill().dropna(), window=norm_window)
    if normed.empty:
        return {"error": "Not enough warm-up data to build an inference row.",
                "ticker": ticker, "interval": interval}

    latest = normed.iloc[[-1]].values.astype("float32")

    try:
        xgb_model = joblib.load(_xgb_path(ticker, interval))
        rf_model = joblib.load(_rf_path(ticker, interval))
    except Exception as exc:
        return {"error": f"Could not load model: {exc}", "ticker": ticker, "interval": interval}

    xgb_prob = float(xgb_model.predict_proba(latest)[0, 1])
    rf_prob = float(rf_model.predict_proba(latest)[0, 1])
    ensemble = 0.65 * xgb_prob + 0.35 * rf_prob

    if ensemble >= 0.55:
        direction = "bullish"
    elif ensemble <= 0.45:
        direction = "bearish"
    else:
        direction = "neutral"

    distance = abs(ensemble - 0.5)
    confidence = "high" if distance >= 0.20 else ("medium" if distance >= 0.10 else "low")

    accuracy = float(meta.get("directional_accuracy") or 0.0)
    tradeability = assess_tradeability(accuracy, info["horizon_sigma"], round_trip_cost_pct)
    bar_ts = market_df.index[-1]

    result = {
        "ticker": ticker,
        "interval": interval,
        "error": None,
        "direction": direction,
        "probability": round(ensemble, 4),
        "confidence": confidence,
        "horizon_bars": horizon_bars,
        "horizon_minutes": horizon_bars * INTERVAL_SPECS[interval]["minutes"],
        "threshold_pct": info["threshold_pct"],
        "model_accuracy": accuracy,
        "is_reliable": bool(meta.get("is_reliable", False)),
        "tradeability": tradeability,
        "price_at_prediction": float(market_df["Close"].iloc[-1]),
        "bar_timestamp": bar_ts.isoformat(),
        "last_trained": meta.get("trained_at"),
    }
    save_intraday_prediction(ticker, interval, result)
    logger.info(
        "predict_intraday: %s %s direction=%s prob=%.4f confidence=%s",
        ticker, interval, direction, ensemble, confidence,
    )
    return result


# ── Prediction history ───────────────────────────────────────────────────────

_HISTORY_COLS = ["date", "direction", "probability", "confidence",
                 "horizon_minutes", "actual_outcome", "correct"]


def save_intraday_prediction(ticker: str, interval: str, prediction: Dict[str, Any]) -> None:
    """Append one prediction to the interval-scoped JSONL log."""
    path = _predictions_path(ticker, interval)
    try:
        record = {
            "predicted_at": now_et_iso(),
            "date": now_et_iso(),
            "ticker": ticker.upper(),
            "interval": interval,
            "bar_timestamp": prediction.get("bar_timestamp"),
            "direction": prediction.get("direction"),
            "probability": prediction.get("probability"),
            "confidence": prediction.get("confidence"),
            "horizon_bars": prediction.get("horizon_bars"),
            "horizon_minutes": prediction.get("horizon_minutes"),
            "model_accuracy": prediction.get("model_accuracy"),
            "price_at_prediction": prediction.get("price_at_prediction"),
            "actual_outcome": None,
            "correct": None,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("save_intraday_prediction: write failed for %s: %s", ticker, exc)


def get_intraday_prediction_history(
    ticker: str,
    interval: str = DEFAULT_INTERVAL,
    resolve: bool = True,
) -> pd.DataFrame:
    """
    All logged predictions for one ticker/interval, newest first.

    Timestamps are parsed with format="mixed" so a log that accumulates both
    naive and tz-aware rows can never silently coerce the newer ones to NaT —
    the failure that hid weeks of daily predictions before it was fixed.
    """
    ticker = ticker.upper()
    interval = _interval_tag(interval)
    if resolve:
        resolve_intraday_predictions(ticker, interval)

    path = _predictions_path(ticker, interval)
    if not path.exists():
        return pd.DataFrame(columns=_HISTORY_COLS)

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
        return pd.DataFrame(columns=_HISTORY_COLS)

    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed", errors="coerce")
        df = df.sort_values("date", ascending=False)
    for col in _HISTORY_COLS:
        if col not in df.columns:
            df[col] = None
    return df[_HISTORY_COLS].reset_index(drop=True)


def resolve_intraday_predictions(ticker: str, interval: str = DEFAULT_INTERVAL) -> int:
    """
    Back-fill actual_outcome/correct for predictions whose horizon has elapsed,
    comparing against bars **at the prediction's own interval**. Resolving an
    intraday prediction against daily closes would score a 75-minute call over
    five days.

    A prediction whose horizon would cross the session close is left unresolved
    rather than scored across the overnight gap, matching how the labels were
    built. Neutral predictions are left unresolved.
    """
    ticker = ticker.upper()
    interval = _interval_tag(interval)
    path = _predictions_path(ticker, interval)
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
        and r.get("bar_timestamp")
    ]
    if not pending:
        return 0

    from data.price_data import get_price_history
    spec = INTERVAL_SPECS[interval]
    raw = get_price_history(ticker, period=spec["max_period"], interval=interval)
    if raw is None or raw.empty:
        logger.warning("resolve_intraday_predictions: no %s data for %s", interval, ticker)
        return 0

    price_df = _to_market_tz(raw)
    sessions = _session_ids(price_df.index)
    resolved = 0

    for record in pending:
        try:
            bar_ts = pd.Timestamp(record["bar_timestamp"])
            if bar_ts.tz is None:
                bar_ts = bar_ts.tz_localize(MARKET_TZ)
            else:
                bar_ts = bar_ts.tz_convert(MARKET_TZ)
        except Exception:
            continue

        horizon = int(record.get("horizon_bars") or 5)
        positions = price_df.index.get_indexer([bar_ts], method="nearest")
        if len(positions) == 0 or positions[0] < 0:
            continue
        start = int(positions[0])
        end = start + horizon
        if end >= len(price_df):
            continue  # horizon hasn't elapsed yet
        if sessions.iloc[end] != sessions.iloc[start]:
            continue  # would span the overnight gap — leave unresolved

        entry = float(record["price_at_prediction"])
        exit_price = float(price_df["Close"].iloc[end])
        actual_pct = round((exit_price - entry) / entry * 100, 4)
        record["actual_outcome"] = actual_pct
        record["correct"] = bool(
            actual_pct > 0 if record["direction"] == "bullish" else actual_pct < 0
        )
        resolved += 1

    if resolved:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.debug("resolve_intraday_predictions: %s %s resolved %d", ticker, interval, resolved)
    return resolved
