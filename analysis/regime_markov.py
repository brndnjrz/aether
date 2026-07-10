"""
Markov-chain regime model — Phase 1 of the probabilistic regime framework.

Discretizes price history into a small state chain (Bear/Neutral/Bull) using
the same trend thresholds as analysis/regime.py, then fits a first-order
Markov transition matrix over that history. From the matrix we derive
regime persistence, a multi-step forecast, the chain's stationary
distribution, and a Bull-minus-Bear signal with a sample-size-based
confidence score. No new dependencies — pure numpy/pandas.

This operates on a single ticker's own price history (not a shared market
regime), so it answers "how has THIS ticker's trend historically evolved"
rather than a broad market-regime question.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_ORDER = ["Bear", "Neutral", "Bull"]

MIN_BARS = 60  # need enough history for a transition matrix to mean anything


def label_trend_states(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized per-bar trend state across full history, using the same
    thresholds as analysis.regime.detect_regime's trend classification
    (above 200-day SMA, 21-day price change, ADX). Sideways and Choppy are
    collapsed into one "Neutral" state — a 4-state chain fragments
    transition counts too thin to be reliable at typical single-ticker
    sample sizes.
    """
    close = df["Close"]
    above_200 = close > df["SMA_200"] if "SMA_200" in df.columns else pd.Series(False, index=df.index)
    price_21d = close.pct_change(21) * 100
    adx = df["ADX"] if "ADX" in df.columns else pd.Series(20.0, index=df.index)

    uptrend = above_200 & (price_21d > 3) & (adx > 20)
    downtrend = (~above_200) & (price_21d < -3) & (adx > 20)

    states = pd.Series("Neutral", index=df.index)
    states[uptrend] = "Bull"
    states[downtrend] = "Bear"
    return states


def transition_counts(states: pd.Series) -> pd.DataFrame:
    """Raw state(t) -> state(t+1) counts, reindexed to a fixed state order."""
    pairs = pd.DataFrame({"from": states, "to": states.shift(-1)}).dropna()
    counts = pd.crosstab(pairs["from"], pairs["to"])
    return counts.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)


def build_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """
    Row-normalized transition matrix P(state(t+1) | state(t)). A state with
    no observed outgoing transitions (e.g. it only ever occurred on the
    final bar) gets a self-loop of 1.0 so every row stays a valid
    probability distribution — required for matrix powers and the
    stationary distribution to be well-defined.
    """
    counts = transition_counts(states)
    row_sums = counts.sum(axis=1)
    matrix = counts.div(row_sums.replace(0, np.nan), axis=0)
    for state in STATE_ORDER:
        if row_sums[state] == 0:
            matrix.loc[state] = 0.0
            matrix.loc[state, state] = 1.0
    return matrix.fillna(0.0)


def regime_persistence(transition_matrix: pd.DataFrame) -> Dict[str, float]:
    """P(state -> same state) per state — how "sticky" each regime is."""
    return {state: round(float(transition_matrix.loc[state, state]), 4) for state in STATE_ORDER}


def stationary_distribution(transition_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Long-run unconditional probability of each state — the left eigenvector
    of the transition matrix for eigenvalue 1, normalized to sum to 1.
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix.to_numpy().T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    if vec.sum() < 0:
        vec = -vec
    vec = np.clip(vec, 0, None)
    total = vec.sum()
    vec = vec / total if total > 0 else np.full(len(STATE_ORDER), 1 / len(STATE_ORDER))
    return {state: round(float(v), 4) for state, v in zip(STATE_ORDER, vec)}


def forecast_regime_path(transition_matrix: pd.DataFrame, current_state: str, steps: int = 10) -> pd.DataFrame:
    """Multi-step forecast: repeatedly apply the transition matrix to the current state's one-hot vector."""
    vec = np.array([1.0 if s == current_state else 0.0 for s in STATE_ORDER])
    matrix = transition_matrix.to_numpy()
    rows = []
    for step in range(1, steps + 1):
        vec = vec @ matrix
        rows.append(vec.copy())
    return pd.DataFrame(rows, index=range(1, steps + 1), columns=STATE_ORDER)


def bull_bear_signal(transition_matrix: pd.DataFrame, current_state: str, state_counts: pd.Series) -> Dict[str, Any]:
    """
    Signal = P(next=Bull | current) - P(next=Bear | current), in [-1, 1].
    Confidence scales with how many times this state has been observed
    historically (thin samples produce a noisy transition-matrix row) —
    30+ occurrences is treated as a fully reliable estimate.
    """
    next_probs = transition_matrix.loc[current_state]
    signal = float(next_probs.get("Bull", 0.0) - next_probs.get("Bear", 0.0))
    n_obs = int(state_counts.get(current_state, 0))
    confidence = round(min(1.0, n_obs / 30), 4)
    return {
        "signal": round(signal, 4),
        "confidence": confidence,
        "n_obs": n_obs,
        "next_step_probs": {s: round(float(next_probs[s]), 4) for s in STATE_ORDER},
    }


def analyze_regime_markov(df: pd.DataFrame, ticker: str = "", forecast_steps: int = 10) -> Dict[str, Any]:
    """
    Full Phase 1 Markov regime analysis for a single ticker's price history.
    `df` must already have indicators computed (analysis.indicators.calculate_indicators)
    — needs SMA_200, ADX, Close.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return {"available": False, "reason": f"Need at least {MIN_BARS} bars of history, got {len(df) if df is not None else 0}."}

    states = label_trend_states(df)
    current_state = states.iloc[-1]
    matrix = build_transition_matrix(states)
    counts = transition_counts(states)
    state_counts = counts.sum(axis=1)

    signal_info = bull_bear_signal(matrix, current_state, state_counts)
    persistence = regime_persistence(matrix)
    stationary = stationary_distribution(matrix)
    forecast = forecast_regime_path(matrix, current_state, steps=forecast_steps)

    return {
        "available": True,
        "ticker": ticker,
        "current_state": current_state,
        "n_bars": len(df),
        "transition_matrix": matrix,
        "state_counts": state_counts.to_dict(),
        "persistence": persistence,
        "stationary_distribution": stationary,
        "forecast_path": forecast,
        "signal": signal_info["signal"],
        "confidence": signal_info["confidence"],
        "next_step_probs": signal_info["next_step_probs"],
    }
