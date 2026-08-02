"""
Tests for prediction_performance.rolling_accuracy() — Roadmap Item 8.

No network. Pure function over a history frame.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.prediction_performance import DEFAULT_ROLLING_WINDOW, rolling_accuracy


def _history(flags, *, direction="bullish"):
    """flags: list of True/False/None for `correct`, oldest first."""
    return pd.DataFrame({
        "date": pd.to_datetime(
            [f"2026-08-{3 + i:02d}T14:30:00+00:00" for i in range(len(flags))], utc=True,
        ),
        "direction": [direction] * len(flags),
        "correct": flags,
        "probability": [0.6] * len(flags),
        "confidence": ["high"] * len(flags),
        "actual_outcome": [0.2 if f else -0.2 for f in flags],
    })


def test_empty_history_returns_empty_frame_with_columns():
    out = rolling_accuracy(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["date", "accuracy", "n_window"]


def test_history_with_no_resolved_rows_returns_empty():
    out = rolling_accuracy(_history([None, None, None]))
    assert out.empty


def test_all_correct_gives_a_flat_hundred_percent_line():
    out = rolling_accuracy(_history([True] * 5))
    assert len(out) == 5
    assert (out["accuracy"] == 1.0).all()


def test_all_wrong_gives_a_flat_zero_line():
    out = rolling_accuracy(_history([False] * 5))
    assert (out["accuracy"] == 0.0).all()


def test_line_starts_immediately_rather_than_after_a_full_window():
    """
    min_periods=1: a new ticker with 4 resolved predictions should still plot,
    with n_window showing how thin each point is.
    """
    out = rolling_accuracy(_history([True, True, False, True]))
    assert len(out) == 4
    assert out["n_window"].tolist() == [1, 2, 3, 4]


def test_accuracy_tracks_a_genuine_decline():
    """Was good, now bad — the pattern a single current number cannot show."""
    out = rolling_accuracy(_history([True] * 10 + [False] * 10))
    assert out["accuracy"].iloc[0] == pytest.approx(1.0)
    assert out["accuracy"].iloc[-1] < out["accuracy"].iloc[9]


def test_window_is_capped_at_the_available_sample():
    out = rolling_accuracy(_history([True] * 5))
    assert out["n_window"].max() <= 5


def test_window_size_is_configurable():
    flags = [True] * 10 + [False] * 10
    wide = rolling_accuracy(_history(flags), window=20)
    narrow = rolling_accuracy(_history(flags), window=3)
    # A narrow window reacts faster, so its final value is nearer to pure 0.
    assert narrow["accuracy"].iloc[-1] <= wide["accuracy"].iloc[-1]


def test_unresolved_rows_are_excluded_not_counted_as_wrong():
    with_pending = rolling_accuracy(_history([True, True, None, True]))
    without = rolling_accuracy(_history([True, True, True]))
    assert len(with_pending) == 3
    assert with_pending["accuracy"].tolist() == without["accuracy"].tolist()


def test_neutral_predictions_are_excluded():
    """Neutral calls are never graded, matching every other metric here."""
    hist = _history([True, True, True])
    hist.loc[1, "direction"] = "neutral"
    out = rolling_accuracy(hist)
    assert len(out) == 2


def test_rows_are_sorted_oldest_first_regardless_of_input_order():
    hist = _history([True, False, True]).iloc[::-1].reset_index(drop=True)
    out = rolling_accuracy(hist)
    assert out["date"].is_monotonic_increasing


def test_rows_with_missing_dates_are_dropped():
    hist = _history([True, True, True])
    hist.loc[1, "date"] = pd.NaT
    out = rolling_accuracy(hist)
    assert len(out) == 2


def test_missing_correct_column_returns_empty_rather_than_raising():
    out = rolling_accuracy(pd.DataFrame({"date": pd.to_datetime(["2026-08-03"], utc=True)}))
    assert out.empty


def test_default_window_is_the_documented_twenty():
    assert DEFAULT_ROLLING_WINDOW == 20
