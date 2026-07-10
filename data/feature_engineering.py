"""
Feature engineering for ML direction prediction.

Input:  DataFrame already processed by calculate_indicators() — all OHLCV columns
        plus RSI, MACD_hist, ADX, ATR_pct, BB_pct, OBV, hv_10, hv_21, vol_ratio,
        SMA_20, SMA_50, SMA_200, above_200ma, STOCH_K are expected to be present.

Output: (X, y) where
        X  — pd.DataFrame, shape (n_samples, 18+), no NaN rows, index-aligned with df
        y  — pd.Series, int8, values {1, -1, 0} where (forward_bars/neutral_threshold
             are configurable — see build_features() — defaulting to 5 days / 0.5%):
              1  = N-day forward return > +threshold  (bullish)
             -1  = N-day forward return < -threshold  (bearish)
              0  = neutral zone (|return| < threshold), kept in dataset for compatibility

Design rules:
- Zero future leakage: every feature is computed from data available at bar t.
  The forward return target uses .shift(-N) — the last N rows will have NaN targets.
- Features are either normalised per-row or are unit-free ratios (returns, percentages,
  binary flags) that are already comparable across price regimes.
- The functions are stateless — same inputs always produce the same outputs.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FORWARD_BARS: int = 5           # predict direction 5 trading days out
NEUTRAL_THRESHOLD: float = 0.005   # rows where |5d return| < 0.5% labelled neutral
MIN_ROWS_REQUIRED: int = 60     # minimum bars to produce any samples

# Ordered list of the 18 features (must stay stable for saved model compatibility).
# Additional derived features may follow; only these 18 are used in the model.
FEATURE_NAMES: List[str] = [
    "rsi_norm",          # RSI / 100  (0–1)
    "rsi_5_norm",        # 5-period RSI / 100 (short momentum)
    "macd_hist_sign",    # sign(-1/0/+1) of MACD histogram
    "adx_norm",          # ADX / 100
    "atr_pct",           # ATR / Close * 100 — already in indicators output
    "bb_pct",            # BB percent position (0=lower band, 1=upper band)
    "vol_ratio",         # Volume / 20d avg volume
    "price_vs_sma20",    # (Close - SMA_20) / SMA_20
    "price_vs_sma50",    # (Close - SMA_50) / SMA_50
    "ret_1d",            # 1-day return
    "ret_5d",            # 5-day return (lagged — not the target)
    "ret_10d",           # 10-day return
    "hv_ratio",          # hv_10 / hv_21 — short vol / medium vol
    "obv_slope",         # sign of 5-bar OBV change
    "stoch_k_norm",      # STOCH_K / 100
    "day_of_week",       # 0=Mon … 4=Fri (categorical treated as ordinal)
    "above_200ma",       # binary 0/1
    "hl_range_pct",      # (High - Low) / Close — daily bar range normalised
]


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    ticker: str = "",
    forward_bars: int = FORWARD_BARS,
    neutral_threshold: float = NEUTRAL_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix (X) and target vector (y) from a fully-enriched
    DataFrame returned by calculate_indicators().

    Parameters
    ----------
    df : pd.DataFrame
        Output of calculate_indicators(). Must contain at minimum:
        Close, High, Low, Volume, RSI, MACD_hist, ADX, ATR_pct, BB_pct,
        vol_ratio, SMA_20, SMA_50, SMA_200, OBV, hv_10, hv_21, STOCH_K,
        above_200ma.
    ticker : str
        Optional ticker symbol for logging context.
    forward_bars : int
        Label lookahead in trading days. Defaults to FORWARD_BARS (5).
        ml_prediction.select_label_scheme() searches a small grid of
        alternatives (3/5/10) and passes the winner here.
    neutral_threshold : float
        Return magnitude below which a row is labelled neutral. Defaults to
        NEUTRAL_THRESHOLD (0.5%).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix. Columns are exactly FEATURE_NAMES in order.
        No NaN values. Index is a subset of df.index.
    y : pd.Series
        Direction labels:
          1  = bullish (forward_bars-day forward return >  +neutral_threshold)
         -1  = bearish (forward_bars-day forward return <  -neutral_threshold)
          0  = neutral (|return| <= neutral_threshold, kept for completeness)
        Same index as X.

    Raises
    ------
    ValueError
        If df has fewer than MIN_ROWS_REQUIRED rows or produces zero valid samples.
    """
    if df is None or len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"build_features requires at least {MIN_ROWS_REQUIRED} rows; "
            f"got {0 if df is None else len(df)}"
        )

    out = df.copy()
    label = f"[{ticker}] " if ticker else ""

    # ── Target (computed first — shifted so features see no future data) ──────
    fwd_ret = out["Close"].pct_change(forward_bars).shift(-forward_bars)
    # Use float dtype so NaN can be stored; int8 cannot hold NaN in pandas
    target = pd.Series(np.nan, index=out.index, dtype="float32")
    target[fwd_ret > neutral_threshold] = 1
    target[fwd_ret < -neutral_threshold] = -1
    target[(fwd_ret <= neutral_threshold) & (fwd_ret >= -neutral_threshold)] = 0
    # Rows where fwd_ret is NaN (last forward_bars rows) remain NaN and are dropped later

    # ── Feature 1: rsi_norm ───────────────────────────────────────────────────
    out["rsi_norm"] = out["RSI"] / 100.0

    # ── Feature 2: rsi_5_norm — 5-period RSI (build inline) ──────────────────
    delta5 = out["Close"].diff()
    gain5 = delta5.clip(lower=0).rolling(5).mean()
    loss5 = (-delta5.clip(upper=0)).rolling(5).mean()
    rs5 = gain5 / loss5.replace(0, np.nan)
    out["rsi_5_norm"] = (100 - (100 / (1 + rs5))) / 100.0

    # ── Feature 3: macd_hist_sign ─────────────────────────────────────────────
    out["macd_hist_sign"] = np.sign(out["MACD_hist"].fillna(0)).astype("float32")

    # ── Feature 4: adx_norm ───────────────────────────────────────────────────
    out["adx_norm"] = (out["ADX"] / 100.0).clip(0, 1)

    # ── Feature 5: atr_pct — passthrough from calculate_indicators ───────────
    out["atr_pct"] = out["ATR_pct"]

    # ── Feature 6: bb_pct — clipped to handle extreme band breaches ──────────
    out["bb_pct"] = out["BB_pct"].clip(-0.2, 1.2)

    # ── Feature 7: vol_ratio — cap vol spikes ────────────────────────────────
    out["vol_ratio"] = out["vol_ratio"].clip(0, 10)

    # ── Features 8–9: price vs moving averages ────────────────────────────────
    out["price_vs_sma20"] = (out["Close"] - out["SMA_20"]) / out["SMA_20"]
    out["price_vs_sma50"] = (out["Close"] - out["SMA_50"]) / out["SMA_50"]

    # ── Features 10–12: lagged returns — pure look-back, no leakage ──────────
    out["ret_1d"] = out["Close"].pct_change(1)
    out["ret_5d"] = out["Close"].pct_change(5)
    out["ret_10d"] = out["Close"].pct_change(10)

    # ── Feature 13: hv_ratio (short vol / medium vol) ─────────────────────────
    hv_10_col = (
        out["hv_10"]
        if "hv_10" in out.columns
        else out["Close"].pct_change().rolling(10).std() * np.sqrt(252) * 100
    )
    hv_21_col = (
        out["hv_21"]
        if "hv_21" in out.columns
        else out["Close"].pct_change().rolling(21).std() * np.sqrt(252) * 100
    )
    out["hv_ratio"] = (hv_10_col / hv_21_col.replace(0, np.nan)).clip(0.2, 3.0)

    # ── Feature 14: obv_slope — sign of 5-bar OBV change ─────────────────────
    out["obv_slope"] = np.sign(out["OBV"].diff(5).fillna(0)).astype("float32")

    # ── Feature 15: stoch_k_norm ──────────────────────────────────────────────
    out["stoch_k_norm"] = (out["STOCH_K"] / 100.0).clip(0, 1)

    # ── Feature 16: day_of_week ───────────────────────────────────────────────
    out["day_of_week"] = out.index.dayofweek.astype("float32")

    # ── Feature 17: above_200ma ───────────────────────────────────────────────
    out["above_200ma"] = out["above_200ma"].astype("float32")

    # ── Feature 18: hl_range_pct ──────────────────────────────────────────────
    out["hl_range_pct"] = (out["High"] - out["Low"]) / out["Close"]

    # ── Additional lag return features (supplement the 18 core) ──────────────
    # These go AFTER FEATURE_NAMES so the 18-column model is not disturbed.
    for lag in [2, 3]:
        out[f"ret_{lag}d"] = out["Close"].pct_change(lag)

    # Rolling volatility features
    out["vol_5d_std"] = out["returns"].rolling(5).std() if "returns" in out.columns else out["Close"].pct_change().rolling(5).std()
    out["vol_10d_std"] = out["returns"].rolling(10).std() if "returns" in out.columns else out["Close"].pct_change().rolling(10).std()
    out["vol_20d_std"] = out["returns"].rolling(20).std() if "returns" in out.columns else out["Close"].pct_change().rolling(20).std()

    # Regime features
    out["above_50ma"] = (out["above_50ma"].astype("float32") if "above_50ma" in out.columns
                         else (out["Close"] > out["SMA_50"]).astype("float32"))
    if "SMA_20" in out.columns:
        out["sma20_slope"] = out["SMA_20"].pct_change(5)

    # OBV rate of change (more granular than sign)
    out["obv_roc_5"] = out["OBV"].pct_change(5).fillna(0).clip(-1, 1)

    # ── Assemble feature matrix ───────────────────────────────────────────────
    all_feature_cols = FEATURE_NAMES.copy()
    extended_cols = ["ret_2d", "ret_3d", "vol_5d_std", "vol_10d_std", "vol_20d_std",
                     "above_50ma", "sma20_slope", "obv_roc_5"]
    for col in extended_cols:
        if col in out.columns:
            all_feature_cols.append(col)

    X_raw = out[all_feature_cols].copy()

    # Align target
    y_raw = target.reindex(X_raw.index)

    # Drop rows where target is NaN (last FORWARD_BARS rows) or any feature is NaN
    valid_mask = y_raw.notna() & X_raw.notna().all(axis=1)
    X = X_raw.loc[valid_mask].astype("float32")
    # y_raw is float32 (to allow NaN); cast to int8 only after NaN rows have been removed
    y = y_raw.loc[valid_mask].astype("int8")

    if len(X) == 0:
        raise ValueError(
            f"{label}build_features produced zero valid samples after NaN removal. "
            "Supply at least 250 bars of data for reliable feature computation."
        )

    logger.debug(
        "%sbuild_features: %d input rows → %d valid samples "
        "(dropped %d NaN rows). "
        "Class balance: bullish=%d neutral=%d bearish=%d",
        label,
        len(df),
        len(X),
        len(df) - len(X),
        int((y == 1).sum()),
        int((y == 0).sum()),
        int((y == -1).sum()),
    )

    return X, y


def normalize_features(X: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Apply rolling z-score normalisation to each feature column using only
    past observations (no look-ahead bias).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from build_features(). Float dtype expected.
    window : int
        Rolling window for mean/std estimation. Default 252 (1 trading year).
        Rows in the first `window` bars will have NaN — handled by forward-filling
        with the global mean/std from that initial window.

    Returns
    -------
    pd.DataFrame
        Same shape as X. Values are z-scored per column using a rolling window.
        Binary columns (0/1 values only) and already-bounded features
        (day_of_week, above_200ma, macd_hist_sign, obv_slope) are excluded
        from normalisation — their ranges are already interpretable.
    """
    # Columns that should not be z-scored (binary or ordinal with fixed meaning)
    skip_cols = {"day_of_week", "above_200ma", "above_50ma", "macd_hist_sign",
                 "obv_slope", "obv_roc_5"}

    result = X.copy()
    for col in X.columns:
        if col in skip_cols:
            continue
        rolling_mean = X[col].rolling(window, min_periods=30).mean()
        rolling_std = X[col].rolling(window, min_periods=30).std()
        # Fallback for the warm-up period: use expanding mean/std
        exp_mean = X[col].expanding(min_periods=10).mean()
        exp_std = X[col].expanding(min_periods=10).std()
        mean = rolling_mean.fillna(exp_mean)
        std = rolling_std.fillna(exp_std).replace(0, 1e-8)
        result[col] = (X[col] - mean) / std

    return result.astype("float32")


# ── Internal helpers shared with ml_prediction.py ────────────────────────────

def build_predict_row(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build a single-row feature matrix for inference from the latest bar of df.

    Inference-time counterpart to build_features(). Computes all features for
    the most recent bar only — no target variable is computed.

    Parameters
    ----------
    df : pd.DataFrame
        Output of calculate_indicators() with at least MIN_ROWS_REQUIRED bars.

    Returns
    -------
    pd.DataFrame of shape (1, 18) with columns matching FEATURE_NAMES exactly.
    Returns None if any required feature produces NaN (warm-up insufficient).
    """
    if df is None or len(df) < MIN_ROWS_REQUIRED:
        logger.warning(
            "build_predict_row: insufficient data (%s rows)", len(df) if df is not None else 0
        )
        return None

    try:
        out = df.copy()

        # 5-period RSI inline
        delta5 = out["Close"].diff()
        gain5 = delta5.clip(lower=0).rolling(5).mean()
        loss5 = (-delta5.clip(upper=0)).rolling(5).mean()
        rs5 = gain5 / loss5.replace(0, np.nan)
        rsi_5 = (100 - (100 / (1 + rs5))) / 100.0

        hv_10_col = (
            out["hv_10"]
            if "hv_10" in out.columns
            else out["Close"].pct_change().rolling(10).std() * np.sqrt(252) * 100
        )
        hv_21_col = (
            out["hv_21"]
            if "hv_21" in out.columns
            else out["Close"].pct_change().rolling(21).std() * np.sqrt(252) * 100
        )

        row = {
            "rsi_norm":       out["RSI"].iloc[-1] / 100.0,
            "rsi_5_norm":     rsi_5.iloc[-1],
            "macd_hist_sign": float(np.sign(out["MACD_hist"].fillna(0).iloc[-1])),
            "adx_norm":       float(np.clip(out["ADX"].iloc[-1] / 100.0, 0, 1)),
            "atr_pct":        out["ATR_pct"].iloc[-1],
            "bb_pct":         float(np.clip(out["BB_pct"].iloc[-1], -0.2, 1.2)),
            "vol_ratio":      float(np.clip(out["vol_ratio"].iloc[-1], 0, 10)),
            "price_vs_sma20": (out["Close"].iloc[-1] - out["SMA_20"].iloc[-1]) / out["SMA_20"].iloc[-1],
            "price_vs_sma50": (out["Close"].iloc[-1] - out["SMA_50"].iloc[-1]) / out["SMA_50"].iloc[-1],
            "ret_1d":         out["Close"].pct_change(1).iloc[-1],
            "ret_5d":         out["Close"].pct_change(5).iloc[-1],
            "ret_10d":        out["Close"].pct_change(10).iloc[-1],
            "hv_ratio":       float(np.clip(hv_10_col.iloc[-1] / (hv_21_col.iloc[-1] + 1e-9), 0.2, 3.0)),
            "obv_slope":      float(np.sign(out["OBV"].diff(5).fillna(0).iloc[-1])),
            "stoch_k_norm":   float(np.clip(out["STOCH_K"].iloc[-1] / 100.0, 0, 1)),
            "day_of_week":    float(out.index[-1].dayofweek),
            "above_200ma":    float(out["above_200ma"].iloc[-1]),
            "hl_range_pct":   (out["High"].iloc[-1] - out["Low"].iloc[-1]) / out["Close"].iloc[-1],
        }

        result = pd.DataFrame([row], columns=FEATURE_NAMES, dtype="float32")

        if result.isnull().any().any():
            logger.warning("build_predict_row: NaN in feature row — insufficient warm-up data")
            return None

        return result

    except Exception as exc:
        logger.error("build_predict_row error: %s", exc, exc_info=True)
        return None


def class_balance_check(y: pd.Series) -> dict:
    """
    Return class distribution stats for a target vector.

    Handles both binary ({-1, 1}) and ternary ({-1, 0, 1}) label sets.
    The 0-class (neutral) rows are tracked but excluded from the
    recommended_scale_pos_weight calculation (which concerns the model's
    binary bull/bear edge only).

    Returns
    -------
    dict with keys:
        n_bullish, n_bearish, n_neutral, n_total,
        bull_pct, bear_pct, neutral_pct,
        imbalance_flag,
        recommended_scale_pos_weight
    """
    # For binary training (neutral rows dropped), y contains only {-1, 1}
    # For ternary (neutral rows kept), y contains {-1, 0, 1}
    n_bull = int((y == 1).sum())
    n_bear = int((y == -1).sum())
    n_neutral = int((y == 0).sum())
    n_directional = n_bull + n_bear
    n_total = n_directional + n_neutral

    bull_pct = n_bull / n_directional * 100 if n_directional > 0 else 50.0
    bear_pct = n_bear / n_directional * 100 if n_directional > 0 else 50.0
    neutral_pct = n_neutral / n_total * 100 if n_total > 0 else 0.0

    return {
        "n_bullish": n_bull,
        "n_bearish": n_bear,
        "n_neutral": n_neutral,
        "n_total": n_total,
        "bull_pct": round(bull_pct, 1),
        "bear_pct": round(bear_pct, 1),
        "neutral_pct": round(neutral_pct, 1),
        "imbalance_flag": bull_pct < 35 or bull_pct > 65,
        "recommended_scale_pos_weight": round(n_bear / n_bull, 3) if n_bull > 0 else 1.0,
    }
