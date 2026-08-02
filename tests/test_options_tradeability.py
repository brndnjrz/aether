"""
Tests for the options cost model in analysis/options_pricing.py.

No network — Greeks and quotes are passed in directly, which is the whole point
of keeping `assess_options_tradeability` dependency-pure.

The numbers below are hand-computed in the docstrings so a future change that
shifts a result has to justify itself against arithmetic, not against a
previously-recorded output.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.options_pricing import (
    assess_options_tradeability,
    sweep_expiries,
)

# horizon_sigma such that avg_move_pct lands on exactly 0.35%:
#   avg_move_pct = sigma * 100 * 0.8  =>  sigma = 0.35 / 80 = 0.004375
SIGMA_035 = 0.004375
ACC = 0.558          # (2 * 0.558 - 1) = 0.116
SPY = 600.0
H15M = 75.0          # 15m interval, 5-bar horizon


def _zero_dte(**overrides):
    kwargs = dict(
        underlying_price=SPY, option_mid=3.00, option_bid=2.99, option_ask=3.01,
        delta=0.50, horizon_minutes=H15M, days_to_expiry=240 / 1440,   # 4h to close
    )
    kwargs.update(overrides)
    return assess_options_tradeability(ACC, SIGMA_035, **kwargs)


def _thirty_dte(**overrides):
    kwargs = dict(
        underlying_price=SPY, option_mid=16.00, option_bid=15.97, option_ask=16.03,
        delta=0.50, horizon_minutes=H15M, days_to_expiry=30.0, theta_per_day=-0.20,
    )
    kwargs.update(overrides)
    return assess_options_tradeability(ACC, SIGMA_035, **kwargs)


# ── elasticity and gross edge ─────────────────────────────────────────────────

def test_elasticity_is_delta_weighted_underlying_over_premium():
    """L = 0.50 * 600 / 3.00 = 100 at 0DTE; = 0.50 * 600 / 16 = 18.75 at 30 DTE."""
    assert _zero_dte()["elasticity"] == pytest.approx(100.0)
    assert _thirty_dte()["elasticity"] == pytest.approx(18.75)


def test_gross_edge_is_leveraged_far_above_the_underlying_edge():
    """
    Underlying edge = 0.116 * 0.35% = 0.0406%. The shares model stops there.
    Leveraged: 0.0406 * 100 = 4.06% at 0DTE, 0.0406 * 18.75 = 0.761% at 30 DTE.
    Understating this by ~100x is bug (1) the options model exists to fix.
    """
    assert _zero_dte()["gross_edge_pct"] == pytest.approx(4.06, abs=1e-2)
    assert _thirty_dte()["gross_edge_pct"] == pytest.approx(0.7613, abs=1e-3)


def test_avg_move_pct_matches_the_shares_model_convention():
    """Same 0.8-sigma convention as intraday_prediction.assess_tradeability."""
    assert _zero_dte()["avg_move_pct"] == pytest.approx(0.35)


# ── spread ────────────────────────────────────────────────────────────────────

def test_spread_cost_is_a_percent_of_premium_not_of_the_underlying():
    """2 cents on a $3.00 contract = 0.667%, vs the 0.02% the shares model assumed."""
    v = _zero_dte()
    assert v["spread_cost_pct"] == pytest.approx(0.6667, abs=1e-3)
    assert v["spread_source"] == "live_quotes"


def test_wide_spread_alone_can_kill_the_edge():
    """
    Gross edge at 30 DTE is 0.761%. A $1.00-wide book on a $16 contract costs
    6.25% -- the edge dies on spread alone, with theta a rounding error.
    """
    v = _thirty_dte(option_bid=15.50, option_ask=16.50)
    assert v["spread_cost_pct"] == pytest.approx(6.25, abs=1e-2)
    assert v["is_tradeable"] is False
    assert v["dominant_cost"] == "spread"


@pytest.mark.parametrize("bid,ask", [
    (None, None),        # no quotes at all
    (0.0, 3.01),         # zero bid -- real, and common on illiquid strikes
    (3.01, 2.99),        # crossed book
    (3.00, 3.00),        # zero-width
])
def test_degenerate_quotes_fall_back_to_the_settings_assumption(bid, ask):
    v = _zero_dte(option_bid=bid, option_ask=ask, fallback_spread_pct=0.02)
    assert v["spread_source"] == "fallback_assumption"
    assert v["spread_cost_pct"] == pytest.approx(2.0)


def test_fallback_spread_is_labelled_so_it_cannot_pass_as_a_live_quote():
    live = _zero_dte()
    fallback = _zero_dte(option_bid=None, option_ask=None)
    assert live["spread_source"] == "live_quotes"
    assert fallback["spread_source"] == "fallback_assumption"


# ── theta ─────────────────────────────────────────────────────────────────────

def test_near_dated_theta_uses_the_sqrt_extrinsic_model_not_black_scholes():
    """
    Black-Scholes theta diverges as T -> 0, so below OPTIONS_MIN_DTE_FOR_BS_THETA
    the model switches. With 240 min to the close and a 75-min horizon:
        retained = sqrt(165 / 240) = 0.82916  ->  drag = 17.08% of premium
    """
    v = _zero_dte()
    assert v["theta_method"] == "sqrt_extrinsic"
    assert v["theta_drag_pct"] == pytest.approx(17.084, abs=1e-2)


def test_zero_dte_fails_on_theta_alone_even_with_a_penny_spread():
    """
    The headline finding. Gross 4.06%, spread 0.67%, theta 17.08% -> deeply
    negative, and the reason is time, not fills. A trader who tightens their
    fills here is solving the wrong problem, which is why dominant_cost exists.
    """
    v = _zero_dte()
    assert v["is_tradeable"] is False
    assert v["dominant_cost"] == "theta"
    assert v["net_edge_pct"] == pytest.approx(4.06 - 0.6667 - 17.084, abs=2e-2)


def test_sqrt_model_ignores_a_supplied_black_scholes_theta_when_near_expiry():
    """A caller passing a (meaningless) 0DTE BS theta must not override the guard."""
    v = _zero_dte(theta_per_day=-0.01)
    assert v["theta_method"] == "sqrt_extrinsic"


def test_horizon_longer_than_time_to_expiry_loses_all_extrinsic():
    v = _zero_dte(days_to_expiry=30 / 1440, horizon_minutes=H15M)   # 30 min left, 75 min horizon
    assert v["theta_drag_pct"] == pytest.approx(100.0)


def test_expired_contract_reports_total_decay_rather_than_dividing_by_zero():
    v = _zero_dte(days_to_expiry=0.0)
    assert v["theta_drag_pct"] == pytest.approx(100.0)
    assert v["theta_method"] == "sqrt_extrinsic"


def test_thirty_dte_uses_black_scholes_theta():
    """0.20/day * (75/390) / 16.00 = 0.2404% on the trading basis."""
    v = _thirty_dte()
    assert v["theta_method"] == "black_scholes"
    assert v["theta_drag_pct"] == pytest.approx(0.2404, abs=1e-3)


def test_theta_basis_changes_the_verdict_magnitude():
    """
    Calendar basis spreads a day's decay over 1440 min instead of 390, so it is
    ~3.7x gentler. Trading is the conservative default.
    """
    trading = _thirty_dte(theta_basis="trading")
    calendar = _thirty_dte(theta_basis="calendar")
    assert trading["theta_drag_pct"] > calendar["theta_drag_pct"]
    assert calendar["theta_drag_pct"] == pytest.approx(0.2404 * 390 / 1440, abs=1e-3)
    assert trading["theta_basis"] == "trading"


def test_missing_theta_falls_back_to_the_empirical_model_rather_than_zero():
    """Silently treating an absent theta as zero would flatter every verdict."""
    v = _thirty_dte(theta_per_day=None)
    assert v["theta_method"] == "sqrt_extrinsic"
    assert v["theta_drag_pct"] > 0


# ── the flip ──────────────────────────────────────────────────────────────────

def test_the_same_edge_flips_sign_across_the_expiry_ladder():
    """
    The central finding of Roadmap Item 3: leverage and theta both scale
    inversely with time to expiry, so they partly cancel and the answer is
    expiry-dependent. 0DTE deeply negative, 30 DTE positive.
    """
    near, far = _zero_dte(), _thirty_dte()
    assert near["is_tradeable"] is False
    assert far["is_tradeable"] is True
    assert far["net_edge_pct"] == pytest.approx(0.7613 - 0.375 - 0.2404, abs=1e-2)
    assert near["dominant_cost"] == "theta"
    assert far["dominant_cost"] == "spread"


# ── breakeven ─────────────────────────────────────────────────────────────────

def test_breakeven_accuracy_inverts_the_net_edge_formula_exactly():
    """
    Feeding breakeven_accuracy back in as the accuracy must produce net == 0.
    Guards against the two formulas drifting apart.
    """
    v = _thirty_dte()
    at_breakeven = _thirty_dte()
    recomputed = assess_options_tradeability(
        v["breakeven_accuracy"], SIGMA_035,
        underlying_price=SPY, option_mid=16.00, option_bid=15.97, option_ask=16.03,
        delta=0.50, horizon_minutes=H15M, days_to_expiry=30.0, theta_per_day=-0.20,
    )
    assert recomputed["net_edge_pct"] == pytest.approx(0.0, abs=1e-3)
    assert at_breakeven["breakeven_accuracy"] == v["breakeven_accuracy"]


def test_zero_dte_breakeven_is_far_beyond_what_the_model_achieves():
    """
    total_cost = 0.6667 + 17.084 = 17.75; L * avg_move = 100 * 0.35 = 35
    breakeven = (17.75 / 35 + 1) / 2 = 0.7536

    Note this is *not* above 1.0 -- 100x leverage means even a 17.75% cost only
    demands 75% accuracy. It is still hopeless (the model scores 55.8%), and
    stating it as a required accuracy rather than a raw negative edge is what
    makes it legible: "you would need 75% to break even here."
    """
    v = _zero_dte()
    assert v["breakeven_accuracy"] == pytest.approx(0.7536, abs=1e-3)
    assert v["breakeven_accuracy"] > ACC + 0.15


def test_impossible_breakeven_is_reported_above_one_not_clamped_to_one():
    """
    Low leverage against a very wide book does exceed 100%. Clamping to 1.0
    would read as "just barely possible" when the honest answer is "impossible."
        L * avg_move = 18.75 * 0.35 = 6.5625
        spread ($1.20 on $16) = 7.5%, theta = 0.24%  ->  total 7.74%
        breakeven = (7.74 / 6.5625 + 1) / 2 = 1.09
    """
    v = _thirty_dte(option_bid=15.40, option_ask=16.60)
    assert v["breakeven_accuracy"] > 1.0
    assert v["is_tradeable"] is False


# ── guards ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,fragment", [
    ({"option_mid": 0.0}, "mid price"),
    ({"option_mid": -1.0}, "mid price"),
    ({"delta": 0.0}, "Delta is zero"),
    ({"underlying_price": 0.0}, "Underlying price"),
    ({"horizon_minutes": 0.0}, "Horizon must be positive"),
])
def test_unusable_input_returns_none_not_false(kwargs, fragment):
    """
    is_tradeable must be None, never False, when nothing was computed --
    otherwise "couldn't compute" renders identically to "computed, and it's bad."
    """
    v = _zero_dte(**kwargs)
    assert v["is_tradeable"] is None
    assert v["net_edge_pct"] is None
    assert fragment in v["reason"]


def test_zero_volatility_is_rejected_rather_than_dividing_by_zero():
    v = assess_options_tradeability(
        ACC, 0.0, underlying_price=SPY, option_mid=3.00, delta=0.50,
        horizon_minutes=H15M, days_to_expiry=1.0,
    )
    assert v["is_tradeable"] is None
    assert "volatility is zero" in v["reason"]


def test_non_numeric_input_degrades_without_raising():
    v = assess_options_tradeability(
        "abc", SIGMA_035, underlying_price=SPY, option_mid=3.0, delta=0.5,
        horizon_minutes=H15M, days_to_expiry=1.0,
    )
    assert v["reason"] == "Non-numeric input."


def test_put_delta_sign_does_not_flip_elasticity():
    """A -0.50 put delta has the same leverage as a +0.50 call delta."""
    call = _thirty_dte(delta=0.50)
    put = _thirty_dte(delta=-0.50)
    assert call["elasticity"] == put["elasticity"]


# ── vega breakeven (Item 11's honest half) ────────────────────────────────────

def test_iv_breakeven_is_none_without_vega():
    """No vega supplied means no breakeven — do not invent one."""
    assert _thirty_dte()["iv_points_to_erase_edge"] is None


def test_iv_breakeven_inverts_vega_against_the_edge():
    """
    net edge 0.1459% of a $16 contract = $0.02334 of edge. With vega 0.0234
    (price change per 1 vol point), a ~1.0-point IV decline erases it.
    """
    v = _thirty_dte(vega=0.0234)
    expected = v["net_edge_pct"] / 100 * 16.00 / 0.0234
    assert v["iv_points_to_erase_edge"] == pytest.approx(expected, abs=1e-3)
    assert v["iv_points_to_erase_edge"] == pytest.approx(1.0, abs=0.15)


def test_iv_breakeven_is_none_when_there_is_no_edge_left_to_erase():
    """A negative edge has nothing to wipe out; a number there would imply one."""
    v = _zero_dte(vega=0.05)
    assert v["net_edge_pct"] < 0
    assert v["iv_points_to_erase_edge"] is None


def test_larger_vega_means_a_smaller_iv_move_erases_the_edge():
    low = _thirty_dte(vega=0.01)["iv_points_to_erase_edge"]
    high = _thirty_dte(vega=0.05)["iv_points_to_erase_edge"]
    assert high < low


def test_iv_breakeven_ignores_vega_sign():
    assert _thirty_dte(vega=0.0234)["iv_points_to_erase_edge"] == pytest.approx(
        _thirty_dte(vega=-0.0234)["iv_points_to_erase_edge"]
    )


@pytest.mark.parametrize("bad_vega", [0.0, "abc", None])
def test_unusable_vega_yields_no_breakeven(bad_vega):
    assert _thirty_dte(vega=bad_vega)["iv_points_to_erase_edge"] is None


def test_sweep_passes_each_rungs_vega_through():
    quotes = {
        30: {"mid": 16.00, "bid": 15.97, "ask": 16.03, "delta": 0.50,
             "theta_per_day": -0.20, "vega": 0.0234, "days_to_expiry": 30.0},
    }
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=quotes,
    )
    assert out["by_dte"][30]["iv_points_to_erase_edge"] is not None


# ── sweep ─────────────────────────────────────────────────────────────────────

def _ladder():
    return {
        0: {"mid": 3.00, "bid": 2.99, "ask": 3.01, "delta": 0.50,
            "days_to_expiry": 240 / 1440, "expiry_date": "2026-08-03"},
        2: {"mid": 6.00, "bid": 5.97, "ask": 6.03, "delta": 0.50,
            "theta_per_day": -1.20, "expiry_date": "2026-08-05"},
        7: {"mid": 9.00, "bid": 8.96, "ask": 9.04, "delta": 0.50,
            "theta_per_day": -0.55, "expiry_date": "2026-08-10"},
        30: {"mid": 16.00, "bid": 15.97, "ask": 16.03, "delta": 0.50,
             "theta_per_day": -0.20, "expiry_date": "2026-09-02"},
    }


def test_sweep_returns_a_verdict_for_every_rung():
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=_ladder(),
    )
    assert set(out["by_dte"]) == {0, 2, 7, 30}
    assert out["ladder"] == [0, 2, 7, 30]
    assert all("net_edge_pct" in v for v in out["by_dte"].values())


def test_sweep_picks_the_highest_net_edge_rung():
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=_ladder(),
    )
    best_by_hand = max(
        out["by_dte"].items(), key=lambda kv: kv[1]["net_edge_pct"],
    )[0]
    assert out["best_dte"] == best_by_hand
    assert out["best"]["net_edge_pct"] == out["by_dte"][best_by_hand]["net_edge_pct"]


def test_sweep_uses_each_rungs_own_days_to_expiry_when_supplied():
    """
    Rung 0 carries days_to_expiry=240/1440 (4h to the close), not the literal
    key 0 -- otherwise a 0DTE rung would report an already-expired contract.
    """
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=_ladder(),
    )
    assert out["by_dte"][0]["days_to_expiry"] == pytest.approx(240 / 1440, abs=1e-3)
    assert out["by_dte"][0]["theta_drag_pct"] < 100.0


def test_sweep_breaks_ties_toward_the_shorter_expiry():
    identical = {
        7: {"mid": 9.00, "bid": 8.96, "ask": 9.04, "delta": 0.50, "theta_per_day": -0.55},
        30: {"mid": 9.00, "bid": 8.96, "ask": 9.04, "delta": 0.50, "theta_per_day": -0.55},
    }
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=identical,
    )
    assert out["by_dte"][7]["net_edge_pct"] == out["by_dte"][30]["net_edge_pct"]
    assert out["best_dte"] == 7


def test_sweep_survives_a_rung_with_unusable_quotes():
    quotes = _ladder()
    quotes[2] = {"mid": 0.0, "delta": 0.50}      # unusable
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=quotes,
    )
    assert out["by_dte"][2]["net_edge_pct"] is None
    assert out["best_dte"] is not None           # other rungs still ranked


def test_sweep_with_no_computable_rung_reports_no_best():
    quotes = {0: {"mid": 0.0, "delta": 0.5}, 2: {"mid": 0.0, "delta": 0.5}}
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=quotes,
    )
    assert out["best_dte"] is None
    assert out["best"] is None
    assert out["any_tradeable"] is False


def test_sweep_orders_rungs_ascending_even_from_unordered_input():
    unordered = {30: _ladder()[30], 0: _ladder()[0], 7: _ladder()[7]}
    out = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=H15M, quotes=unordered,
    )
    assert out["ladder"] == [0, 7, 30]


def test_longer_horizons_need_longer_expiries():
    """
    The diagonal the scoreboard is meant to expose: as the signal horizon grows,
    theta at short expiries grows with it, pushing the best rung further out.
    """
    short = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=15.0, quotes=_ladder(),
    )
    long = sweep_expiries(
        ACC, SIGMA_035, underlying_price=SPY, horizon_minutes=300.0, quotes=_ladder(),
    )
    assert short["by_dte"][0]["theta_drag_pct"] < long["by_dte"][0]["theta_drag_pct"]
