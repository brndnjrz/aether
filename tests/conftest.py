"""
Shared pytest fixtures for the aether test suite.

Ensures the project root is importable (mirrors the sys.path.insert() pattern
used by pages/*.py) and provides a synthetic OHLCV+indicators DataFrame that
lets analysis/ml_prediction.py be exercised end-to-end without any network
access (yfinance is unreachable in CI/sandboxed environments).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_ohlcv(n: int = 520, seed: int = 42, drift: float = 0.0004) -> pd.DataFrame:
    """
    Geometric random-walk OHLCV series with the same column shape yfinance
    returns (Open, High, Low, Close, Volume — DatetimeIndex, business days).

    drift=0.0 produces a pure random walk with no real directional signal —
    used to sanity-check that the reliability gate in ml_prediction doesn't
    rubber-stamp a model as reliable when there's nothing to learn.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-01", periods=n)
    returns = rng.normal(drift, 0.014, n)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def synthetic_price_df():
    """Raw OHLCV synthetic data, pre-indicators."""
    return _make_synthetic_ohlcv()


@pytest.fixture
def synthetic_indicators_df():
    """
    Synthetic OHLCV run through the real calculate_indicators() — this is
    the exact shape train_model()/predict()/evaluate_model() expect as `df`.
    """
    from analysis.indicators import calculate_indicators

    raw = _make_synthetic_ohlcv()
    return calculate_indicators(raw)


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """
    Redirect analysis.ml_prediction's module-level _STORAGE_DIR to a temp
    directory for the duration of a test, so tests never read/write the
    real storage/ directory (which holds live AAPL/SPY model + prediction
    data that must not be touched or polluted by test tickers).
    """
    import analysis.ml_prediction as ml_prediction

    monkeypatch.setattr(ml_prediction, "_STORAGE_DIR", tmp_path)
    return tmp_path


# ── Intraday fixtures (ORBC strategy) ───────────────────────────────────────

RTH_BARS_5M = 78  # 09:30-16:00 inclusive of the 09:30 bar, 5-minute interval


def make_intraday_session(
    closes: list,
    date: str = "2026-07-06",
    interval_minutes: int = 5,
    volume: int = 1_000_000,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Build one ET trading session of intraday OHLCV from an explicit list of
    closes, starting at 09:30. High/Low are padded a hair beyond the
    open/close so each bar has a real range without perturbing the closes the
    ORBC state machine keys off of.

    Explicit closes (rather than a random walk) are what make the breakout
    state machine testable: a test can lay out exactly which bars close
    inside/outside the opening range.
    """
    n = len(closes)
    start = pd.Timestamp(f"{date} 09:30", tz=tz)
    idx = pd.date_range(start, periods=n, freq=f"{interval_minutes}min")

    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": np.full(n, volume, dtype=np.int64)},
        index=idx,
    )


@pytest.fixture
def intraday_session_factory():
    """Exposes make_intraday_session() to tests as a fixture."""
    return make_intraday_session
