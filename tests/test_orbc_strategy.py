"""
Tests for analysis/orbc_strategy.py — the Opening Range Breakout Confirmation
strategy.

The core value here is pinning down the confirmation state machine, which is
where the strategy spec was ambiguous and where an off-by-one would silently
turn "waits for confirmation" back into "enters on the first breakout" — the
exact false-breakout behavior the strategy exists to avoid.

All data is synthetic and hand-laid-out (see conftest.make_intraday_session)
so each test controls exactly which bars close inside/outside the opening
range. No network access.
"""
from __future__ import annotations

from datetime import time as dtime

import pandas as pd
import pytest

from analysis.orbc_strategy import (
    ORBCConfig,
    backtest_orbc,
    compute_opening_range,
    detect_orbc_signals,
    detect_session_signals,
    evaluate_orbc_trade,
    latest_session_state,
    to_market_tz,
)
from tests.conftest import make_intraday_session


# A config with the filters off, so state-machine tests isolate the
# confirmation logic from volume/VWAP/ATR gating.
NO_FILTERS = dict(require_volume=False, require_vwap=False, require_atr=False)


def test_page_import_contract_is_satisfied():
    """
    pages/strategy_lab.py imports these four names from analysis.orbc_strategy
    at module level. The page calls render() on import, so it can't be imported
    directly in a test without firing network fetches — this asserts the same
    import contract instead, so a rename here fails loudly rather than taking
    the whole Strategy Lab page down at runtime.
    """
    import analysis.orbc_strategy as orbc

    for name in ("ORBCConfig", "backtest_orbc", "latest_session_state", "to_market_tz"):
        assert hasattr(orbc, name), f"analysis.orbc_strategy is missing {name}"


def _session(closes, **kwargs):
    return make_intraday_session(closes, **kwargs)


# ── Opening range ───────────────────────────────────────────────────────────

def test_opening_range_uses_first_15_minutes_of_5m_bars():
    # 09:30, 09:35, 09:40 are the 15-minute range; 09:45 onward is not.
    df = _session([100, 101, 102, 150, 60])
    orange = compute_opening_range(df, opening_range_minutes=15)

    assert orange["bar_count"] == 3, "15-minute range over 5m bars must be exactly 3 bars"
    # High/low include the 0.05 padding from the fixture.
    assert orange["opening_high"] == pytest.approx(102.05)
    assert orange["opening_low"] == pytest.approx(99.95)
    assert orange["range_size"] == pytest.approx(2.10)
    # The 09:45 spike to 150 must NOT be in the range.
    assert orange["opening_high"] < 150


def test_opening_range_duration_is_configurable():
    df = _session([100, 101, 102, 103, 104, 105, 106])
    assert compute_opening_range(df, 15)["bar_count"] == 3
    assert compute_opening_range(df, 30)["bar_count"] == 6
    assert compute_opening_range(df, 5)["bar_count"] == 1


def test_opening_range_none_when_session_starts_after_window():
    df = _session([100, 101, 102])
    df.index = df.index + pd.Timedelta(hours=2)  # session data starts at 11:30
    assert compute_opening_range(df, 15) is None


# ── Confirmation state machine ──────────────────────────────────────────────

def test_first_breakout_close_alone_does_not_signal():
    """The whole point of the strategy: one close outside is not enough."""
    # Range ~99.95-102.05. One close above, then back inside.
    df = _session([100, 101, 102, 103, 100, 100, 100])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    assert state["signals"] == [], "a single breakout close must not fire a signal"


def test_second_consecutive_close_outside_fires_long():
    df = _session([100, 101, 102, 103, 104, 100, 100])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))

    assert len(state["signals"]) == 1
    sig = state["signals"][0]
    assert sig["direction"] == "long"
    assert sig["action"] == "BUY"
    assert sig["confirmation_count"] == 2
    # Fires on the 2nd close outside = the 104 bar (09:50), not the 103 bar.
    assert sig["entry_price"] == pytest.approx(104.0)
    assert sig["timestamp"].strftime("%H:%M") == "09:50"


def test_second_consecutive_close_below_fires_short():
    df = _session([100, 101, 102, 99, 98, 100, 100])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))

    assert len(state["signals"]) == 1
    sig = state["signals"][0]
    assert sig["direction"] == "short"
    assert sig["action"] == "SELL"
    assert sig["entry_price"] == pytest.approx(98.0)


def test_close_back_inside_range_resets_the_count():
    """
    Two non-consecutive closes above the range must not accumulate into a
    confirmation — a breakout that round-trips has to start over.
    """
    # above, inside, above -> counts should be 1, reset, 1. Never reaches 2.
    df = _session([100, 101, 102, 103, 100, 103, 100, 100])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    assert state["signals"] == []


def test_direction_flip_resets_the_count():
    """A close above then a close below is not 2 confirmations."""
    df = _session([100, 101, 102, 103, 98, 100, 100])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    assert state["signals"] == []


def test_confirmation_closes_is_configurable_to_three():
    df = _session([100, 101, 102, 103, 104, 105, 100])
    two = detect_session_signals(df, ORBCConfig(confirmation_closes=2, **NO_FILTERS))
    three = detect_session_signals(df, ORBCConfig(confirmation_closes=3, max_confirmation_closes=4, **NO_FILTERS))

    assert two["signals"][0]["entry_price"] == pytest.approx(104.0)   # 2nd close out
    assert three["signals"][0]["entry_price"] == pytest.approx(105.0)  # 3rd close out


def test_only_one_signal_per_direction_per_session():
    """
    The resolved 'fire at 2, fall through to 3' rule must not double-signal:
    a long run of closes outside the range produces exactly one entry.
    """
    df = _session([100, 101, 102] + [103, 104, 105, 106, 107, 108])
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    assert len(state["signals"]) == 1, "count 2 and count 3 must not both fire"


def test_reversal_day_yields_one_trade_when_one_signal_per_session():
    """
    A day that confirms long, then reverses and confirms short, must produce a
    single trade while one_signal_per_session is on — the flag means one signal
    per session, not one per direction.
    """
    df = _session([100, 101, 102, 103, 104, 90, 90], date="2026-07-06")
    state = detect_session_signals(df, ORBCConfig(one_signal_per_session=True, **NO_FILTERS))

    assert len(state["signals"]) == 1
    assert state["signals"][0]["direction"] == "long"


def test_reversal_day_yields_both_trades_when_re_entry_allowed():
    df = _session([100, 101, 102, 103, 104, 90, 90], date="2026-07-06")
    state = detect_session_signals(df, ORBCConfig(one_signal_per_session=False, **NO_FILTERS))

    assert [s["direction"] for s in state["signals"]] == ["long", "short"]


def test_filter_block_at_count_two_falls_through_to_count_three():
    """
    The 'second OR third candle' rule: if a filter blocks the 2nd close, the
    3rd close may still fire. Volume is used as the blocking filter — the 2nd
    breakout bar has below-average volume, the 3rd has a surge.
    """
    df = _session([100, 101, 102, 103, 104, 105, 100])
    # vol_sma_20 needs history; supply it directly so the filter is decidable.
    df["vol_sma_20"] = 1_000_000.0
    df["Volume"] = 1_000_000
    df.iloc[4, df.columns.get_loc("Volume")] = 500_000    # 2nd close out: weak
    df.iloc[5, df.columns.get_loc("Volume")] = 3_000_000  # 3rd close out: surge

    cfg = ORBCConfig(require_volume=True, require_vwap=False, require_atr=False)
    state = detect_session_signals(df, cfg)

    assert len(state["signals"]) == 1
    assert state["signals"][0]["confirmation_count"] == 3
    assert state["signals"][0]["entry_price"] == pytest.approx(105.0)
    # The blocked attempt is recorded so the UI can explain it.
    assert len(state["rejections"]) == 1
    assert state["rejections"][0]["confirmation_count"] == 2


def test_signal_expires_after_max_confirmation_closes():
    """Beyond max_confirmation_closes the episode is dead even if closes continue."""
    df = _session([100, 101, 102, 103, 104, 105, 106, 107])
    df["vol_sma_20"] = 1_000_000.0
    df["Volume"] = 100  # every breakout bar fails the volume filter

    cfg = ORBCConfig(require_volume=True, require_vwap=False, require_atr=False,
                     confirmation_closes=2, max_confirmation_closes=3)
    state = detect_session_signals(df, cfg)

    assert state["signals"] == []
    # Only counts 2 and 3 are candidates; counts 4+ never get evaluated.
    assert [r["confirmation_count"] for r in state["rejections"]] == [2, 3]


# ── Filters ─────────────────────────────────────────────────────────────────

def test_time_filter_blocks_entries_after_cutoff():
    # Breakout confirms at 09:50; a 09:45 cutoff must exclude it.
    df = _session([100, 101, 102, 103, 104, 105])
    late = detect_session_signals(df, ORBCConfig(entry_cutoff=dtime(9, 45), **NO_FILTERS))
    assert late["signals"] == []

    early = detect_session_signals(df, ORBCConfig(entry_cutoff=dtime(11, 0), **NO_FILTERS))
    assert len(early["signals"]) == 1


def test_vwap_filter_blocks_long_below_vwap():
    df = _session([100, 101, 102, 103, 104, 100])
    df["VWAP"] = 200.0  # price is far below VWAP — no long allowed

    cfg = ORBCConfig(require_volume=False, require_vwap=True, require_atr=False)
    state = detect_session_signals(df, cfg)
    assert state["signals"] == []
    assert "VWAP" in state["rejections"][0]["reasons"][0]


def test_atr_filter_blocks_tight_opening_range():
    df = _session([100, 101, 102, 103, 104, 100])
    df["ATR"] = 100.0  # range must exceed 0.5 * 100 = 50; it's ~2.1

    cfg = ORBCConfig(require_volume=False, require_vwap=False, require_atr=True)
    state = detect_session_signals(df, cfg)
    assert state["signals"] == []
    assert "too tight" in state["rejections"][0]["reasons"][0]


def test_missing_indicator_columns_skip_filters_rather_than_blocking():
    """
    A short-history frame with no ATR/VWAP/vol_sma_20 must not silently
    suppress every signal — filters with unavailable inputs are skipped.
    """
    df = _session([100, 101, 102, 103, 104, 100])
    cfg = ORBCConfig(require_volume=True, require_vwap=True, require_atr=True)
    state = detect_session_signals(df, cfg)

    assert len(state["signals"]) == 1
    filters = state["signals"][0]["filters"]
    assert "skipped" in str(filters["volume"])
    assert "skipped" in str(filters["vwap"])


def test_allow_short_false_suppresses_short_signals():
    df = _session([100, 101, 102, 99, 98, 100])
    cfg = ORBCConfig(allow_long=True, allow_short=False, **NO_FILTERS)
    assert detect_session_signals(df, cfg)["signals"] == []


# ── Stop / target ───────────────────────────────────────────────────────────

def test_long_stop_at_opening_low_target_at_2r():
    df = _session([100, 101, 102, 103, 104, 100])
    sig = detect_session_signals(df, ORBCConfig(**NO_FILTERS))["signals"][0]

    assert sig["stop_price"] == pytest.approx(99.95)   # opening low
    risk = 104.0 - 99.95
    assert sig["risk"] == pytest.approx(risk, abs=1e-3)
    assert sig["target_price"] == pytest.approx(104.0 + 2 * risk, abs=1e-3)
    assert sig["rr_ratio"] == pytest.approx(2.0, abs=0.01)


def test_short_stop_and_target_are_mirrored():
    df = _session([100, 101, 102, 99, 98, 100])
    sig = detect_session_signals(df, ORBCConfig(**NO_FILTERS))["signals"][0]

    assert sig["stop_price"] == pytest.approx(102.05)  # opening high, above entry
    assert sig["stop_price"] > sig["entry_price"]
    assert sig["target_price"] < sig["entry_price"], "short target must be below entry"
    assert sig["risk"] > 0 and sig["reward"] > 0
    assert sig["rr_ratio"] == pytest.approx(2.0, abs=0.01)


def test_atr_stop_method_uses_atr_distance():
    df = _session([100, 101, 102, 103, 104, 100])
    df["ATR"] = 2.0
    cfg = ORBCConfig(stop_method="atr", stop_atr_multiple=1.0, **NO_FILTERS)
    sig = detect_session_signals(df, cfg)["signals"][0]
    assert sig["stop_price"] == pytest.approx(102.0)  # 104 - 1*2


# ── Trade simulation ────────────────────────────────────────────────────────

def test_long_trade_exits_at_target():
    closes = [100, 101, 102, 103, 104] + [120, 120]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    trade = evaluate_orbc_trade(df, state["signals"][0], ORBCConfig(**NO_FILTERS))

    assert trade["exit_reason"] == "target"
    assert trade["return_pct"] > 0
    assert trade["win"] is True


def test_long_trade_exits_at_stop_with_negative_return():
    closes = [100, 101, 102, 103, 104] + [90, 90]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    trade = evaluate_orbc_trade(df, state["signals"][0], ORBCConfig(**NO_FILTERS))

    assert trade["exit_reason"] == "stop"
    assert trade["return_pct"] < 0
    assert trade["win"] is False


def test_short_trade_profits_when_price_falls():
    """Direction-aware sign convention: a short that drops must show a gain."""
    closes = [100, 101, 102, 99, 98] + [80, 80]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    trade = evaluate_orbc_trade(df, state["signals"][0], ORBCConfig(**NO_FILTERS))

    assert trade["direction"] == "short"
    assert trade["exit_reason"] == "target"
    assert trade["return_pct"] > 0, "short into a falling market must be a gain, not a loss"


def test_short_trade_loses_when_price_rises():
    closes = [100, 101, 102, 99, 98] + [120, 120]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    trade = evaluate_orbc_trade(df, state["signals"][0], ORBCConfig(**NO_FILTERS))

    assert trade["exit_reason"] == "stop"
    assert trade["return_pct"] < 0


def test_trade_force_closes_at_session_end_never_holds_overnight():
    # Price drifts but never reaches stop or target.
    closes = [100, 101, 102, 103, 104, 104.1, 104.2, 104.3]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    trade = evaluate_orbc_trade(df, state["signals"][0], ORBCConfig(**NO_FILTERS))

    assert trade["exit_reason"] == "session_end"
    assert trade["exit_price"] == pytest.approx(104.3)
    assert trade["return_pct"] is not None


def test_simulation_starts_after_the_signal_bar_no_lookahead():
    """
    Entry is the signal bar's close and the walk starts at the next bar, so a
    target touched *on* the signal bar itself must not count as an exit.
    """
    closes = [100, 101, 102, 103, 104, 104.05]
    df = _session(closes)
    state = detect_session_signals(df, ORBCConfig(**NO_FILTERS))
    sig = state["signals"][0]
    trade = evaluate_orbc_trade(df, sig, ORBCConfig(**NO_FILTERS))

    assert trade["holding_period_bars"] >= 1
    assert trade["exit_reason"] != "target"


# ── Confidence score ────────────────────────────────────────────────────────

def test_confidence_score_is_bounded_and_has_components():
    df = _session([100, 101, 102, 103, 104, 100])
    sig = detect_session_signals(df, ORBCConfig(**NO_FILTERS))["signals"][0]

    assert 0.0 <= sig["confidence_score"] <= 100.0
    assert set(sig["score_components"]) == {
        "confirmation_strength", "volume_thrust", "vwap_alignment",
        "range_quality", "trend_alignment",
    }
    for value in sig["score_components"].values():
        assert 0.0 <= value <= 1.0


def test_volume_surge_scores_higher_than_weak_volume():
    def score_with_volume(breakout_volume):
        df = _session([100, 101, 102, 103, 104, 100])
        df["vol_sma_20"] = 1_000_000.0
        df["Volume"] = 1_000_000
        df.iloc[4, df.columns.get_loc("Volume")] = breakout_volume
        cfg = ORBCConfig(require_volume=False, require_vwap=False, require_atr=False)
        return detect_session_signals(df, cfg)["signals"][0]["confidence_score"]

    assert score_with_volume(3_000_000) > score_with_volume(400_000)


# ── Multi-session / timezone ────────────────────────────────────────────────

def test_state_does_not_carry_across_sessions():
    """
    Each session is independent: a breakout at the end of day 1 must not
    combine with a breakout at the start of day 2 into one confirmation.
    """
    day1 = _session([100, 101, 102, 103], date="2026-07-06")  # 1 close out only
    day2 = _session([100, 101, 102, 103], date="2026-07-07")  # 1 close out only
    df = pd.concat([day1, day2])

    assert detect_orbc_signals(df, ORBCConfig(**NO_FILTERS)) == []


def test_signals_found_across_multiple_sessions():
    day1 = _session([100, 101, 102, 103, 104], date="2026-07-06")
    day2 = _session([100, 101, 102, 103, 104], date="2026-07-07")
    df = pd.concat([day1, day2])

    signals = detect_orbc_signals(df, ORBCConfig(**NO_FILTERS))
    assert len(signals) == 2
    assert signals[0]["timestamp"] < signals[1]["timestamp"]


def test_naive_index_is_treated_as_market_time_not_shifted():
    """
    A tz-naive index (some providers/CSV round-trips) must be localized to ET,
    not converted from UTC — otherwise 09:30 ET data would be read as 05:30
    and the opening-range window would miss entirely.
    """
    df = _session([100, 101, 102, 103, 104])
    naive = df.copy()
    naive.index = naive.index.tz_localize(None)

    aware_signals = detect_orbc_signals(df, ORBCConfig(**NO_FILTERS))
    naive_signals = detect_orbc_signals(naive, ORBCConfig(**NO_FILTERS))

    assert len(naive_signals) == len(aware_signals) == 1
    assert naive_signals[0]["entry_price"] == aware_signals[0]["entry_price"]


def test_utc_index_is_converted_to_market_time():
    """A UTC-labeled index must convert to ET so 13:30 UTC reads as 09:30 ET."""
    df = _session([100, 101, 102, 103, 104])
    utc = df.copy()
    utc.index = utc.index.tz_convert("UTC")

    assert len(detect_orbc_signals(utc, ORBCConfig(**NO_FILTERS))) == 1


def test_latest_session_state_returns_only_the_last_day():
    day1 = _session([100, 101, 102, 103, 104], date="2026-07-06")
    day2 = _session([100, 101, 102, 99, 98], date="2026-07-07")
    df = pd.concat([day1, day2])

    state = latest_session_state(df, ORBCConfig(**NO_FILTERS))
    assert state["session_date"].strftime("%Y-%m-%d") == "2026-07-07"
    assert state["signals"][0]["direction"] == "short"


# ── Backtest aggregation ────────────────────────────────────────────────────

def test_backtest_aggregates_across_sessions():
    winner = _session([100, 101, 102, 103, 104, 120, 120], date="2026-07-06")
    loser = _session([100, 101, 102, 103, 104, 90, 90], date="2026-07-07")
    df = pd.concat([winner, loser])

    result = backtest_orbc("TEST", ORBCConfig(**NO_FILTERS), df=df)

    assert result["num_trades"] == 2
    assert result["num_sessions"] == 2
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["win_rate"] == pytest.approx(50.0)
    assert set(result["exit_breakdown"]) == {"target", "stop"}
    assert len(result["equity_curve"]) == 3  # seed + 2 closed trades


def test_backtest_reports_zero_trades_without_crashing():
    quiet = _session([100, 100, 100, 100, 100, 100], date="2026-07-06")
    result = backtest_orbc("TEST", ORBCConfig(**NO_FILTERS), df=quiet)

    assert result["num_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_return_pct"] == 0.0


def test_backtest_splits_stats_by_direction():
    long_day = _session([100, 101, 102, 103, 104, 120, 120], date="2026-07-06")
    short_day = _session([100, 101, 102, 99, 98, 80, 80], date="2026-07-07")
    df = pd.concat([long_day, short_day])

    result = backtest_orbc("TEST", ORBCConfig(**NO_FILTERS), df=df)
    assert result["by_direction"]["long"]["trades"] == 1
    assert result["by_direction"]["short"]["trades"] == 1
    # Both were winners in their own direction.
    assert result["by_direction"]["short"]["win_rate"] == pytest.approx(100.0)


# ── Config validation ───────────────────────────────────────────────────────

def test_invalid_config_raises():
    with pytest.raises(ValueError):
        ORBCConfig(opening_range_minutes=0)
    with pytest.raises(ValueError):
        ORBCConfig(confirmation_closes=3, max_confirmation_closes=2)
    with pytest.raises(ValueError):
        ORBCConfig(stop_method="nonsense")
    with pytest.raises(ValueError):
        ORBCConfig(allow_long=False, allow_short=False)
