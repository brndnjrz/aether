"""
Tests for the horizon-scaled stop helpers in analysis/risk.py (Roadmap Item 2B).

No network, no fixtures needed — these are pure functions over numbers.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.risk import contract_loss_at_stop, horizon_stop
from config.settings import HORIZON_STOP_ATR_MULTIPLE

PRICE = 600.0
AVG_MOVE = 0.35          # percent, the mean |move| over a 75-minute horizon


def test_horizon_stop_scales_with_the_horizons_own_average_move():
    """distance = k * (0.35 / 100) * 600 = k * 2.10"""
    s = horizon_stop(PRICE, AVG_MOVE)
    assert s["distance"] == pytest.approx(HORIZON_STOP_ATR_MULTIPLE * 2.10, abs=1e-4)
    assert s["k"] == HORIZON_STOP_ATR_MULTIPLE


def test_horizon_stop_is_far_tighter_than_a_daily_atr_stop():
    """
    The reason this exists. A daily ATR on SPY near $600 runs ~$5-6, so 1.5x ATR
    is ~$8 — roughly 4x the entire average move over a 75-minute window, which
    means the stop is unreachable and the R:R computed from it is fiction.
    """
    horizon = horizon_stop(PRICE, AVG_MOVE)["distance"]
    daily_atr_stop = 1.5 * 5.50
    assert horizon < daily_atr_stop / 3


def test_long_and_short_stops_sit_on_opposite_sides_of_entry():
    long_stop = horizon_stop(PRICE, AVG_MOVE, direction="long")
    short_stop = horizon_stop(PRICE, AVG_MOVE, direction="short")
    assert long_stop["stop_price"] < PRICE < short_stop["stop_price"]
    assert long_stop["distance"] == short_stop["distance"]


@pytest.mark.parametrize("direction", ["long", "bull", "bullish", "LONG"])
def test_long_synonyms_all_place_the_stop_below_entry(direction):
    assert horizon_stop(PRICE, AVG_MOVE, direction=direction)["stop_price"] < PRICE


@pytest.mark.parametrize("direction", ["short", "bear", "bearish"])
def test_short_synonyms_all_place_the_stop_above_entry(direction):
    assert horizon_stop(PRICE, AVG_MOVE, direction=direction)["stop_price"] > PRICE


def test_k_is_overridable_and_scales_linearly():
    a = horizon_stop(PRICE, AVG_MOVE, k=1.0)["distance"]
    b = horizon_stop(PRICE, AVG_MOVE, k=2.0)["distance"]
    assert b == pytest.approx(2 * a)


def test_stop_pct_is_consistent_with_the_distance():
    s = horizon_stop(PRICE, AVG_MOVE)
    assert s["stop_pct"] == pytest.approx(s["distance"] / PRICE * 100, abs=1e-3)


def test_basis_string_names_both_inputs_so_the_stop_is_auditable():
    s = horizon_stop(PRICE, AVG_MOVE)
    assert "0.350%" in s["basis"]
    assert f"{HORIZON_STOP_ATR_MULTIPLE:g}" in s["basis"]


@pytest.mark.parametrize("price,avg_move", [
    (0.0, AVG_MOVE), (-1.0, AVG_MOVE), (PRICE, 0.0), (PRICE, -0.1),
])
def test_unusable_input_returns_empty_rather_than_a_misleading_stop(price, avg_move):
    assert horizon_stop(price, avg_move) == {}


def test_non_numeric_input_returns_empty_without_raising():
    assert horizon_stop("abc", AVG_MOVE) == {}
    assert horizon_stop(PRICE, None) == {}


def test_zero_k_is_rejected():
    assert horizon_stop(PRICE, AVG_MOVE, k=0) == {}


# ── contract-level loss ───────────────────────────────────────────────────────

def test_contract_loss_is_delta_weighted():
    """A $2.00 adverse move at 0.50 delta costs $1.00 of premium."""
    loss = contract_loss_at_stop(2.00, delta=0.50, option_mid=3.00)
    assert loss["premium_loss"] == pytest.approx(1.00)
    assert loss["premium_loss_pct"] == pytest.approx(33.33, abs=1e-2)


def test_contract_loss_reveals_leverage_the_underlying_stop_hides():
    """
    A 0.35% stop on the underlying is a third of the premium on a near-dated
    contract. Showing only the underlying distance understates risk by roughly
    the elasticity factor.
    """
    distance = horizon_stop(PRICE, AVG_MOVE)["distance"]
    underlying_pct = distance / PRICE * 100
    loss = contract_loss_at_stop(distance, delta=0.50, option_mid=3.00)
    assert loss["premium_loss_pct"] > underlying_pct * 20


def test_put_delta_sign_does_not_flip_the_loss():
    call = contract_loss_at_stop(2.0, delta=0.50, option_mid=3.00)
    put = contract_loss_at_stop(2.0, delta=-0.50, option_mid=3.00)
    assert call["premium_loss"] == put["premium_loss"]


def test_total_loss_is_flagged_when_the_stop_exceeds_the_premium():
    loss = contract_loss_at_stop(10.00, delta=0.50, option_mid=3.00)
    assert loss["is_total_loss"] is True


def test_partial_loss_is_not_flagged_as_total():
    loss = contract_loss_at_stop(2.00, delta=0.50, option_mid=3.00)
    assert loss["is_total_loss"] is False


@pytest.mark.parametrize("kwargs", [
    {"delta": 0.0, "option_mid": 3.0},
    {"delta": 0.5, "option_mid": 0.0},
])
def test_contract_loss_guards_unusable_input(kwargs):
    assert contract_loss_at_stop(2.0, **kwargs) == {}


def test_contract_loss_rejects_a_nonpositive_distance():
    assert contract_loss_at_stop(0.0, delta=0.5, option_mid=3.0) == {}
