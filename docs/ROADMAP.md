# Aether Roadmap — SPY Horizon Cockpit

> **Status: all 14 items implemented** on branch `feat/horizon-cockpit`.
> Keep this document as the design record — the *why* behind each item, the
> decisions settled before building, the ideas deliberately rejected, and the four
> traps the implementation had to avoid. Read it that way, not as pending work.
>
> Two things did not land as originally written, both deliberately:
>
> - **Item 11 (vega)** ships only its honest half. `iv_points_to_erase_edge`
>   reports how many volatility points of IV decline would wipe out the modelled
>   edge — arithmetic from a Greek the app already had. *Forecasting* the IV move
>   still needs the calibration study Item 11 names (how far does SPY ATM IV
>   actually move after a 0.35% 75-minute move?); inventing that number instead
>   would have been [trap 1](#trap-1--hardcoded-signal-weights-are-false-precision).
> - **Item 12's analysis is gated, not live.** Logging and comparison are both
>   built, but `compare_followed_vs_discretionary()` refuses to report until both
>   arms clear their minimums — see
>   [trap 3](#trap-3--personalized-learning-needs-hundreds-of-trades). The gate is
>   the feature.
>
> **Verification caveat.** No environment on the development machine has the full
> `requirements.txt` (`~/anaconda3/envs/stock-app/` is empty), so tests that
> actually train a model could not be run locally: 392 of 436 pass, and all 44
> failures share one cause — a missing `xgboost` — with zero assertion failures.
> Run `pytest tests/ -q` in a complete env before trusting the suite, and treat
> anything touching live options quotes as unexercised until it has run against a
> real chain during market hours.

**Bottom line:** the app had five direction models (5m / 15m / 30m / 1h / daily) and no
way to see them at once; it told you when to get in but never when to get out; and its
"is this tradeable after costs" check was modelled on trading SPY shares when the actual
instrument is SPY options. Everything in v2.0 addressed one of those three gaps.

## Table of Contents

- [Context: the use case this roadmap serves](#context-the-use-case-this-roadmap-serves)
- [Decisions made](#decisions-made)
- [Already built — do not rebuild](#already-built--do-not-rebuild)
- [Tier 1 — v2.0](#tier-1--v20)
- [Tier 2 — v2.5](#tier-2--v25)
- [Tier 3 — v3.0](#tier-3--v30)
- [Explicitly rejected, and why](#explicitly-rejected-and-why)
- [Four traps to avoid while building this](#four-traps-to-avoid-while-building-this)
- [Open questions](#open-questions)
- [Sequencing](#sequencing)

---

## Context: the use case this roadmap serves

One trader, one ticker (SPY), trading **SPY options contracts**, using the per-interval
direction models to decide **which horizon to trade and when to be flat**. Not a
multi-user product, not a multi-ticker screener.

That reframe drives the priority order below. The app's information is currently sliced
**by ticker** — every page takes a ticker input and shows one symbol deeply. The actual
decision is sliced **by horizon**. Features that rank symbols (scanners, watchlists,
letter grades, sector exposure) solve a problem this user does not have; features that
rank or reconcile *horizons* are the ones that matter.

Second observation: the app is ~90% entry logic. Of everything on Trading Desk — VWAP
read, momentum, volume, trend alignment, candlestick pattern, suggested entry, opening
range, pivots, position size, prediction gauge — only the static `1.5 x ATR` stop and
target level concern the exit. Yet the exit is already implicit in every prediction and
simply is not surfaced (see [Item 2](#item-2--exit-clock-and-horizon-scaled-stops)).

Third, and most consequential: the tradeability check assumes a 2 bps round trip, which
is right for the ETF and badly wrong for contracts. See
[Item 3](#item-3--options-aware-tradeability).

### Design principles to preserve

The app's credibility rests on not overstating what it knows. Existing examples: the
gauge windowed to 35-65%, the "small, real edge" callout, the explicit
`not tradeable after costs` verdict, ORBC's small-sample warning below 30 trades,
`RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK = 20`, `_safe_div` returning `None` rather than
`0.0` on a 0/0. Every item below must hold that line. Specifically:

- **No invented composite scores.** A number on screen looks measured. If it is not
  fitted or counted, do not show it.
- **Always show N** next to any rate. Per-horizon sample sizes will differ by an order
  of magnitude (5m generates far more predictions than daily).
- **Distinguish "not computable" from "zero."**
- `analysis/` and `data/` stay Streamlit-free; `pages/` stay display-only.

---

## Decisions made

Settled 2026-08-02, before implementation started.

| Question | Decision | Consequence |
|---|---|---|
| **Instrument** | SPY **options contracts** | The 2 bps cost model is structurally wrong, not just mis-parameterized. Promotes options-aware tradeability to Tier 1 as [Item 3](#item-3--options-aware-tradeability). |
| **Cockpit placement** | Top of **Trading Desk -> Predictions**, above the Daily/Intraday radio | No new page, no change to `home.py`. Sits directly above the train/predict controls that feed it. See the ticker-input note in [Item 1](#item-1--spy-horizon-cockpit). |
| **Refresh behavior** | **Read-only + staleness flag + one manual "Refresh all horizons" button** | No `persist` parameter needed on `predict()`/`predict_intraday()`. The prediction log keeps meaning "calls I deliberately made." Avoids [trap 4](#trap-4--auto-generating-predictions-corrupts-the-track-record) entirely. |
| **Expiry** | **Varies by trade** | Expiry becomes an *input* to Item 3, not a fixed assumption. The trust scoreboard becomes a **horizon x expiry grid** rather than a single net-edge column. More work, but it answers the better question: "given I want to act on the 15m signal, which expiry should I buy?" |
| **Spread** | **Unknown — read it live, fall back to a conservative constant** | Better than a user-supplied number anyway: `get_atm_greeks` already returns live `bid`/`ask`, so the cost model self-calibrates. A settings constant covers market-closed and fetch-failure paths only. |

---

## Already built — do not rebuild

Four items commonly proposed for an app like this already exist here in whole or in part.

| Proposed feature | Current state |
|---|---|
| **Market overview dashboard** | Done. `pages/home.py` — regime banner vs 200MA, index cards (SPY/QQQ/IWM/VIX), Markov regime second opinion on the S&P, sector performance chart + table, recent activity. |
| **Historical similar setups** | Engine exists, output discarded. `ml_prediction.predict()` (lines ~1371-1400) runs the trained model across *all* history, masks to bars where it predicted the same direction (`predicted_mask`), and takes the **median** forward return as `expected_move_pct`. That mask *is* the similar-setups search. Count, win rate, and distribution are already in hand and thrown away. See [Item 6](#item-6--similar-setups-panel). |
| **Trade journal schema** | Tables exist, UI was cut. `portfolio/db.py` `positions` already has `thesis`, `conviction`, `target_price`, `stop_price`, `exit_date`, `exit_price`, `exit_reason`. There is also an entirely unused `watchlist` table. |
| **Indicator explanations** | Present where it matters — "How to read this" expanders on Options (IV Rank / IV-RV / IV-GARCH / term structure), Predictions (gauge / confidence / expected move / feature importance), and Model Lab (all six panels). A full glossary is a feature for other users. |

---

## Tier 1 — v2.0

Five items. Items 1, 2, 4, 5 are wiring already-computed values into one place. Item 3 is
the only one that needs real new math, and it is the one that most changes what the app
tells you.

### Item 1 — SPY Horizon Cockpit

**Problem.** To compare horizons today you change a dropdown four times and hold the
results in your head. `pages/strategy_lab.py::_render_intraday_predictions_reference`
gets closest — it loops all four intervals and reads each history — but it only
tabulates. It never concludes anything: no agreement read, no cost verdict, no live
accuracy, no expiry.

**Target view.**

```
SPY -- 10:42 AM ET | VIX 17.2 (Low) | Regime: Uptrend            [Refresh all horizons]

Horizon  Signal    Prob  Conf   Live acc (n)  Expires    Net edge (opt)  Age
5m       BULLISH   58%   med    46.2% (31)    10:57 AM   -2.4%   no      3m
15m      BULLISH   61%   high   56.8% (44)    11:45 AM   -0.9%   no      3m
30m      BULLISH   57%   med    54.5% (22)     1:12 PM   +0.3%   yes     3m
1h       NEUTRAL   52%   low    -- (8)         --            --          3m
daily    BULLISH   59%   med    55.7% (61)    Aug 7          n/a         6h

Alignment: 3 of 4 intraday horizons bullish, 0 opposing, 1 neutral.
Tightest tradeable horizon: 30m -- expires 1:12 PM ET (150 min).
Note: 5m and 15m agree directionally but do not clear options costs.
```

That last line is the point of the whole feature. Today nothing tells you that the
horizon you are most likely to act on is the one least able to pay for itself.

**Where.** New `analysis/interval_consensus.py`, one public function:

```python
def build_consensus(ticker: str, *, include_daily: bool = True) -> dict:
    """
    Read-only synthesis across every interval in INTERVAL_SPECS plus (optionally)
    the daily model. Reads saved predictions and persisted metadata only --
    never calls predict()/predict_intraday(), never fetches price data, never
    writes. Streamlit-free.

    Returns
    -------
    {
      "ticker": str,
      "as_of": str,                    # now_et_iso()
      "horizons": [
        {
          "horizon": "15m",            # or "daily"
          "has_model": bool,
          "has_prediction": bool,
          "direction": "bullish"|"bearish"|"neutral"|None,
          "probability": float|None,
          "confidence": "high"|"medium"|"low"|None,
          "trained_accuracy": float|None,     # from metadata
          "live_accuracy": float|None,        # compute_*_prediction_metrics win_rate
          "n_resolved": int,
          "expires_at": str|None,             # ET ISO; see Item 2
          "minutes_remaining": float|None,    # negative once expired
          "is_expired": bool,
          "net_edge_pct": float|None,         # OPTION-return terms; see Item 3
          "is_tradeable": bool|None,
          "cost_model": "options"|"shares"|None,
          "prediction_age_minutes": float|None,
          "is_stale": bool,                   # age > 1 bar interval
        },
        ...
      ],
      "alignment": {
        "n_bullish": int, "n_bearish": int, "n_neutral": int,
        "n_tradeable_bullish": int, "n_tradeable_bearish": int,
        "net_direction": "bullish"|"bearish"|"mixed"|"none",
        "is_unanimous": bool,
        "tightest_tradeable": str|None,       # shortest non-expired, tradeable,
                                              # non-neutral horizon agreeing with
                                              # net_direction
        "agree_but_uneconomic": [str],        # horizons agreeing with net_direction
                                              # that fail the cost check
      },
      "warnings": [str],                      # stale/expired/untrained/low-n notes
    }
    """
```

**Inputs, all existing:**

| Field | Source |
|---|---|
| direction / probability / confidence | `get_intraday_prediction_history(t, iv).iloc[0]`; `get_prediction_history(t).iloc[0]` |
| trained_accuracy, is_reliable | `intraday_prediction.load_metadata(t, iv)`; `ml_prediction._load_model_metadata(t)` |
| live_accuracy, n_resolved | `compute_intraday_prediction_metrics` / `compute_daily_prediction_metrics` -> `win_rate`, `n_resolved` |
| net_edge_pct, is_tradeable | Item 3's `assess_options_tradeability(...)`; falls back to the stored `tradeability` dict when no live chain is available |
| expires_at | `bar_timestamp` + `horizon_minutes` (intraday); prediction date + `horizon_days` trading days (daily) |
| has_model | `intraday_prediction.model_exists(t, iv)`; both `.pkl` paths for daily |

**Placement detail.** The Predictions tab currently has two *separate* ticker inputs
(`predictions_ticker_input` for daily, `intraday_pred_ticker_input` for intraday). The
cockpit sits above the Daily/Intraday radio, so it needs a ticker before either exists.
Recommended: the cockpit reads `st.session_state.get("predictions_ticker", "SPY")` and
does **not** add a third input — the existing inputs keep writing that key, so switching
ticker in either sub-tab updates the cockpit on the next rerun. Avoids a third source of
truth for "which ticker."

**Read-only, per the decision above.** Renders from saved predictions; marks anything
older than one bar interval as stale; one explicit **Refresh all horizons** button loops
`predict_intraday` across `INTERVAL_SPECS` (reusing the existing "Generate All
Predictions" code path) and those *do* get logged, because you asked for them.

**Acceptance criteria.**
- Renders with zero trained models (all rows "no model," no exception).
- Renders with a mix of trained/untrained, predicted/not-predicted.
- `n_resolved` shown next to every accuracy; `--` where not computable, never `0.0%`.
- A prediction past its horizon renders as expired, not as a live signal.
- `agree_but_uneconomic` is surfaced in prose, not just as a column value.
- No network calls, no writes, on plain render.

**Tests** (`tests/test_interval_consensus.py`, no network, `isolated_storage`): hand-built
JSONL fixtures for the empty case, the all-agree case, the split 15m-bull/30m-bear case,
the expired-prediction case, and the stale-prediction case. Assert `tightest_tradeable`
skips expired, non-tradeable, and neutral horizons, and that a directionally-agreeing but
uneconomic horizon lands in `agree_but_uneconomic` rather than `tightest_tradeable`.

---

### Item 2 — Exit clock and horizon-scaled stops

**Problem.** Every prediction already has a hard expiry and the UI never states it.
`resolve_predictions()` grades a 15m/5-bar call by comparing `price_at_prediction` to
the close exactly 75 minutes later. That is the model's entire contract: it has
validated evidence about a 75-minute move and **none at all** about minute 76. Holding
past the horizon is trading an expired signal.

For options this matters far more than for shares, because time is not neutral — every
minute past the horizon is paid for in theta. The exit clock and Item 3 are two views of
the same fact.

**Part A — expiry surfacing.** Two fields already returned by `predict_intraday`
(`bar_timestamp`, `horizon_minutes`) give this for free:

- On the prediction result: `Signal expires 11:45 AM ET (63 min remaining)`.
- Once elapsed: an explicit expired state, visually distinct from a live signal.
- Daily model: expiry as a date, advanced by `horizon_days` **trading** days (not
  calendar days — reuse the same forward-bar walk `resolve_predictions` uses so the two
  cannot disagree).
- Same `minutes_remaining` / `is_expired` fields feed the cockpit table.

Put the arithmetic in `analysis/horizon_clock.py` so the Predictions tab, the cockpit,
and Item 3's theta drag all call one implementation.

**Part B — horizon-scaled stops.** The trade card's stop is `1.5 x daily ATR`
(`pages/trading.py:517`). For a 15m signal with a 75-minute horizon that stop is
enormous — it will essentially never be touched inside the window, so the displayed R:R
pairs a 75-minute target against a multi-day stop and is not meaningful.

`assess_tradeability` already computes `avg_move_pct` for the exact horizon
(`horizon_sigma * 100 * 0.8`, the mean absolute move over that many bars). Use it:

```
horizon_stop_distance = k * avg_move_pct * price      # k ~ 1.0-1.5, configurable
```

Keep the daily-ATR stop for the daily/swing path, show both when they disagree
materially rather than silently swapping one for the other, and put `k` in
`config/settings.py` — not inline in the page.

**Options note:** the stop is expressed on the underlying (that is what you watch), but
the *loss* at that stop is leveraged by delta. Show the implied contract-level loss
alongside it so the risk number is not quietly understated by a factor of 10.

**Acceptance criteria.**
- Expiry shown in ET on every non-neutral prediction, intraday and daily.
- Expired predictions cannot be mistaken for live ones.
- Daily expiry advances by trading days; verified against `resolve_predictions`'s own
  bar walk on a fixture spanning a weekend and a holiday.
- Intraday R:R is computed against the horizon-scaled stop, with the basis labeled.

**Tests:** `tests/test_horizon_clock.py` — expiry math across a session close, across a
weekend, and for each of the four intervals; `k`-scaling of the stop.

---

### Item 3 — Options-aware tradeability

**Problem.** `assess_tradeability` (`analysis/intraday_prediction.py:582`) models the
edge as:

```
avg_move_pct  = horizon_sigma * 100 * 0.8
gross_edge    = (2 * accuracy - 1) * avg_move_pct
net_edge      = gross_edge - round_trip_cost_pct        # default 0.02 (2 bps)
```

Every term there is in **underlying** percentage points. That is correct for trading SPY
shares and wrong in three separate ways for trading SPY contracts:

1. **The payoff is leveraged.** A 0.35% SPY move is not a 0.35% move in an ATM contract —
   it is roughly `delta * S / P` times larger. With SPY near $600, an ATM contract at
   $3.00 gives an elasticity around 100x; the same strike 30 days out at $16 gives
   around 19x. The current model understates the gross edge by one to two orders of
   magnitude.
2. **The cost is far larger.** A 1-2 cent spread on a $2.00 contract is a 0.5-1.0% round
   trip, not 0.02%.
3. **Time is a cost, and it is missing entirely.** Theta is often the dominant term at
   short expiries, and the current model has no time-decay concept at all.

Raising the constant from 0.02 to 1.0 would fix (2) while leaving (1) and (3) wrong — and
because (1) works in your favor and (3) against you, the sign of the answer is genuinely
not obvious. It has to be modeled properly.

**Target model.** New `assess_options_tradeability()`, in `analysis/options_pricing.py`
(it belongs with the Black-Scholes code, and keeps `intraday_prediction.py` from growing
an options dependency):

```python
def assess_options_tradeability(
    mean_accuracy: float,
    horizon_sigma: float,
    *,
    underlying_price: float,
    option_mid: float,
    option_bid: float,
    option_ask: float,
    delta: float,
    theta_per_day: float,
    horizon_minutes: float,
    days_to_expiry: float,            # labels the result; guards the theta model
    theta_basis: str = "trading",     # "trading" | "calendar"
) -> dict:
    """
    Tradeability in OPTION-return terms rather than underlying-return terms.

        elasticity      L = |delta| * underlying_price / option_mid
        gross_edge_pct  = (2 * accuracy - 1) * L * avg_underlying_move_pct
        spread_cost_pct = (ask - bid) / option_mid
        theta_drag_pct  = |theta_per_day| * day_fraction / option_mid
        net_edge_pct    = gross_edge_pct - spread_cost_pct - theta_drag_pct

        breakeven_accuracy =
            ((spread_cost_pct + theta_pct) / (L * avg_move_pct) + 1) / 2

    Returns every term separately, not just the net -- the whole value of this
    function is seeing WHICH cost kills the edge.

    One (horizon, expiry) pair per call. The caller sweeps expiries; see
    "Expiry sweep" below.
    """
```

**Expiry sweep.** Because expiry varies by trade, one verdict per horizon is not enough —
the function above is evaluated across a small expiry ladder and the results form a grid:

```python
OPTIONS_EXPIRY_LADDER_DTE = [0, 2, 7, 30]      # config/settings.py
```

A companion `sweep_expiries(...)` (same module) returns `{dte: verdict}` per horizon, plus
the best-scoring DTE. Practical notes:

- **Live quotes per expiry.** `get_options_chain(ticker, expiry)` is already per-expiry, so
  the ladder needs one fetch per rung. Respect `OPTIONS_CACHE_TTL` (600s) and fetch the
  ladder once per render, not once per horizon — the same four chains serve all five
  horizons. Five horizons x four expiries is 20 cells from **four** fetches.
- **Map DTE to a real expiry date.** `chain_data["expirations"]` gives actual listed dates;
  snap each ladder rung to the nearest listed expiry rather than assuming one exists at
  exactly 2 or 7 days, and report the date actually used.
- **Spread source.** Live `bid`/`ask` from the fetched chain. Fall back to
  `OPTIONS_FALLBACK_SPREAD_PCT` (settings) when the market is closed or a fetch fails, and
  label which source produced the number — never present a fallback figure as a live quote.
- **Degenerate quotes.** A `bid` of 0, a zero-width spread, or `mid == 0` all appear in real
  chain data. Guard rather than divide.

**Data sources, all existing:** `data/options_data.py::get_atm_greeks(ticker)` already
returns `bid`, `ask`, `strike`, `delta`, `theta`, `iv` for both the ATM call and put.
`analysis/options_pricing.py::black_scholes_greeks` returns `theta` already normalized to
**per calendar day** (`theta_annual / 365`, line 68) — verified, no unit guessing needed.

**`theta_basis` is a real modeling choice, not a detail.** Pro-rating calendar theta
across a 75-minute window (`75 / 1440`) assumes decay runs overnight at the same rate as
midday, which is not how the market prices it — most decay is realized during trading
hours. `"trading"` pro-rates over a 390-minute session (`75 / 390`) and is the more
conservative, more realistic intraday choice; `"calendar"` is available for comparison.
Show which basis is in use, and consider showing both as a range rather than implying
precision the model does not have.

**Known omissions to document in the docstring, not paper over:**
- **Gamma ignored.** Delta is held constant over the horizon; fine for small moves,
  optimistic for large ones (in your favor on a winner, against you on a loser).
- **Vega ignored, and this is the big one.** An IV crush after a directional move can
  erase a correct call entirely. The app already computes IV Rank, IV/RV, and an
  IV/GARCH forecast on the Options tab — a v2.5 refinement could pull a vega term in.
  Until then, state plainly that a "tradeable" verdict assumes IV holds.
- **Single ATM contract.** No spreads, no multi-leg structures. Selling premium has an
  entirely different cost profile and is out of scope here.
- **Black-Scholes theta breaks down as expiry approaches.** `theta_annual / 365` is
  unstable as `T -> 0`, so a 0DTE contract cannot be costed from the closed-form Greek
  alone. Guard it: below some minimum `T`, fall back to an empirical decay estimate
  (extrinsic value remaining, divided by minutes left in the session) and label which
  method produced the number. Do not let a divide-by-near-zero print a confident figure.

**Integration.**
- `assess_tradeability` (the shares model) **stays** and keeps its current behavior and
  tests. Nothing that currently calls it changes.
- Intraday train/predict output gains the options verdict **alongside** the shares one,
  labeled by `cost_model`, so the Predictions tab can show both and the difference is
  visible rather than swapped in silently.
- Defaults live in `config/settings.py`: `OPTIONS_COST_MODEL_DEFAULT`,
  `OPTIONS_THETA_BASIS`, and a fallback spread assumption for when no live chain is
  available. Not hardcoded in a page — same rule the retrain thresholds follow.
- Graceful degradation: no live chain (after hours, fetch failure) -> fall back to the
  stored shares verdict and say so. Never silently show a shares verdict labeled as
  options.

**Expected finding — and why the expiry sweep is the whole point.** Elasticity and theta
both scale inversely with time to expiry, so they partly cancel and the verdict genuinely
flips depending on which expiry is bought. Rough 15m arithmetic, SPY near $600, ATM,
`delta = 0.50`, illustrative only:

```
                                    0DTE (P ~ $3)      30 DTE (P ~ $16)
avg SPY move over 75 min             0.35%              0.35%
accuracy 55.8% -> underlying edge    0.041%             0.041%
elasticity  delta * S / P           ~100x              ~19x
  -> gross option edge              ~4.1%              ~0.78%
spread cost                         ~0.7% ($0.02)      ~0.2% ($0.03)
theta over 75/390 of a session      ~20-30%            ~0.24%
                                    -----------------  -----------------
net                                 deeply negative    ~ +0.3%
```

So the honest statement is **not** "options make everything untradeable." It is that at
very short expiries theta swamps a real edge by an order of magnitude, while at 30ish days
the same edge plausibly survives. Since expiry varies by trade, that is exactly the thing
the grid has to show rather than assume — and when a cell is deeply negative the useful
output is not a red badge but a *reason*: "no intraday horizon can pay for 0DTE decay; buy
more time or trade shares." Far more valuable than `+0.008%`.

Today's green "15m: tradeable, net edge +0.008%" is wrong in both directions at once
(leverage understated, theta absent), and correcting it is the highest-value change in
this roadmap.

The grid also reframes the strategy question productively: **longer-dated contracts trade
leverage away for theta relief**, so each signal horizon likely has a minimum viable
expiry. Finding that boundary per horizon is the concrete, actionable output of Items 3
and 4 together.

**Acceptance criteria.**
- Every term returned separately; the UI names which cost dominates.
- `breakeven_accuracy` reported in the same terms as the model's actual accuracy.
- Shares and options verdicts both available and clearly labeled.
- Expiry sweep returns a full `{horizon: {dte: verdict}}` grid plus the best rung per
  horizon; the DTE actually used is the nearest *listed* expiry and is reported as a date.
- Ladder fetched once per render (four chains), not once per horizon-expiry cell.
- No live chain -> explicit fallback, never a mislabeled verdict.
- Zero/degenerate quotes (`bid == 0`, `mid == 0`, zero-width spread) guarded, not divided.

**Tests** (`tests/test_options_tradeability.py`, no network — Greeks passed in directly):
elasticity math against a hand-computed case; a wide-spread contract failing on spread
alone; a 0DTE case failing on theta alone; `theta_basis` switching changing the verdict;
`option_mid == 0` and `delta == 0` guarded rather than dividing by zero; breakeven
accuracy inverting the net-edge formula exactly. For `sweep_expiries`: the grid's shape
across a 5-horizon x 4-rung ladder, best-rung selection when two rungs tie, a ladder rung
with no nearby listed expiry snapping to the nearest available one, and the fallback-spread
path being labeled as such.

---

### Item 4 — Horizon trust scoreboard (Model Lab)

**Problem.** Model Lab answers "how is the 15m model doing" only after you select 15m.
It never answers "**which** horizon should I trade," which is the actual question.

**Target view** — five rows, each showing the *best* expiry for that horizon, with the
full grid one expander down. Every column is already computed once Item 3 lands:

```
Horizon  Walk-fwd  Live acc   N   Best expiry  Net edge  Dominant cost  Calibrated   Verdict
5m         54.1%     46.2%   31   30 DTE        -0.4%    theta          no           Do not trade
15m        55.8%     56.8%   44   30 DTE        +0.31%   spread         yes          Primary
30m        54.9%     54.5%   22    7 DTE        +0.44%   spread         unknown (n)  Primary
1h         52.3%        --    8   --                --   --             unknown (n)  Insufficient data
daily      57.2%     55.7%   61   30 DTE        +1.9%    spread         yes          Swing context
```

```
[expander] Net edge % -- every horizon x expiry combination

           0DTE      2 DTE     7 DTE    30 DTE
5m        -28.4%     -6.1%     -1.9%    -0.4%
15m       -19.7%     -3.2%     -0.6%    +0.31%
30m        -9.8%     -1.4%     +0.44%   +0.28%
1h         -4.1%     +0.2%     +0.51%   +0.22%
daily        n/a     -0.8%     +1.1%    +1.9%

Quotes: live (SPY chain, 10:42 AM ET). Theta basis: trading-hours.
```

That grid is the single most actionable artifact in the roadmap. It answers, in one
glance, both "which horizon should I trade" *and* "which expiry should I buy to trade it"
— and it will very likely show a diagonal: the shorter the signal horizon, the longer the
expiry needed to outrun decay. Nothing in the app says anything like this today.

**Columns.**
- **Walk-fwd** — `load_metadata` / `_load_model_metadata`.
- **Live acc / N** — `compute_*_prediction_metrics` -> `win_rate`, `n_resolved`.
- **Best expiry / Net edge** — Item 3's `sweep_expiries`, best-scoring rung.
- **Dominant cost** — what makes the table actionable: a spread-dominated failure is
  fixable with better fills or a higher-priced contract; a theta-dominated failure means
  the expiry is wrong for the horizon, not that the model is bad.
- **Calibrated** — does HIGH beat LOW in that interval's calibration table? Requires a
  minimum N per bucket, else `unknown (n)`. This column says whether you may size on the
  confidence badge at all.
- **Verdict** — an explicit, readable rule chain, never a score. Draft:
  `n_resolved < 20` -> Insufficient data;
  `best net_edge <= 0 and accuracy <= 0.52` -> Do not trade;
  `best net_edge <= 0 and accuracy > 0.52` -> **Uneconomic** (real edge, costs eat it at
  *every* expiry — the distinction that matters most here);
  `live_acc < trained_acc - RETRAIN_ACCURACY_DROP_THRESHOLD` -> Degraded, retrain;
  else rank survivors by best net edge -> Primary / Secondary.

**Where.** `analysis/prediction_performance.py` gains
`compare_horizons(ticker) -> list[dict]` (pure, Streamlit-free); Model Lab renders it as
a new top-level panel above the Daily/Intraday tabs, since it spans both.

**Tests:** extend `tests/test_prediction_performance.py` — the verdict rule chain at each
branch (especially "Do not trade" vs "Uneconomic"), `n/a` handling for daily net edge,
`unknown (n)` when a calibration bucket is underpopulated.

---

### Item 5 — Fix two stale caption numbers

Small, but Items 1-4 will quote these numbers, so fix first.

| Location | Says | Actual | Fix |
|---|---|---|---|
| `pages/trading.py:1088` and the gauge's gray step at `:1070` | neutral zone 47-53% | `predict()` uses `[0.45, 0.55]` (`ml_prediction.py:1344`) | Widen the shaded band and the caption to 45-55%. A 46% reading currently renders inside a red band while the model calls it neutral. |
| `pages/trading.py:1540` (disclaimer) | probabilities "are clipped" to 35-65% | Nothing clips them. `predict()` returns the raw ensemble probability; `confidence == "high"` is *defined* as `> 0.65 or < 0.35`, i.e. outside those bounds. The 35-65 range is only the gauge's axis. | Reword to "displayed on a 35-65% axis" / "display range." |

`docs/ML_PREDICTION.md` repeats the 47-53% figure and needs the same correction.

---

## Tier 2 — v2.5

### Item 6 — Similar setups panel

Surface what `predict()` already computes and discards. From the existing
`predicted_mask` and `fwd_ret`, return alongside `expected_move_pct`:

```
n_similar, win_rate, mean_return, median_return, p25, p75, worst, best
```

Rendered as a stat row plus a histogram of `subset_ret`. Roughly 30 lines of
already-computed data, and the most persuasive panel available.

Two cautions: the mask is generated by the **current** model over its own training
history, so it is in-sample and optimistic — label it in the same register as the
existing "Signal Sharpe (IS)" help text. Keep the existing `>= 5` sample floor; below
~20 report N without the win rate.

### Item 7 — Interval-agreement backtest

The research question the logs can already answer: **when 15m and 30m agreed, was
accuracy higher than 15m alone?**

Join the per-interval prediction JSONLs on overlapping timestamps, bucket by agreement
state (unanimous / partial / conflicting), report accuracy and N per bucket. If
agreement measurably improves accuracy, "wait for two-horizon confirmation" becomes a
measured rule rather than a hunch — the same discipline ORBC's second-close rule encodes.
For an options trader it is also the cheapest possible edge improvement, since waiting
costs only theta while a better hit rate multiplies against leverage.

Where: `analysis/interval_consensus.py` gains `backtest_agreement(ticker) -> dict`.

Build the caveats in from the start: predictions are only logged when you press the
button, so timestamps are irregular and self-selected — this is **not** a clean backtest
and must not be presented as one. Needs a generous join tolerance and an explicit N
floor.

### Item 8 — Accuracy over time (Model Lab)

Rolling 20-prediction win rate as a line per horizon, with retrain dates from
`storage/versions/{TICKER}/history.jsonl` drawn as vlines. Answers "improving or
drifting" and makes each retrain's effect visible instead of inferred. Reuses the version
history the page already reads.

### Item 9 — Section the Day Trading tab

`_render_daytrading` is ~590 lines of continuous scroll. Collapse into Overview /
Technicals / Patterns / Risk / Backtest with the summary always expanded. Pure display
refactor, no logic change — the import-contract test
(`test_page_import_contract_is_satisfied`) is the guard.

### Item 10 — Signal contribution, fitted

Only ship this if the weights are **fitted**. Fit a logistic regression on the logged
history — features being the recorded signal states (`vwap_direction`,
`momentum_direction`, `trend_direction`, pattern present, flag confidence), label being
the realized outcome — and display the fitted coefficients. `log_activity` already
records most of these per Analyze click.

If N is too small to fit, ship the vote count the trade card already shows
(`3 of 5 signals bullish`) and nothing more. See
[trap 1](#trap-1--hardcoded-signal-weights-are-false-precision).

### Item 11 — Vega term for the options cost model

Item 3 ships ignoring vega. The refinement: pull an IV-change term into
`assess_options_tradeability` so an IV crush after a correct directional call is priced
in. The inputs exist — the Options tab already computes IV Rank, IV/RV, and an
IV/GARCH forward forecast. Deferred because it needs its own calibration work: how much
does SPY ATM IV actually move after a 0.35% 75-minute move? That is a measurable
question and worth answering before modeling it.

---

## Tier 3 — v3.0

### Item 12 — Prediction-to-trade linkage

Tag a logged trade with the prediction that motivated it, so the app can eventually
report "you followed the 15m model 18 times for +2.1% average; you overrode it 7 times
for -1.4%." Schema is largely present (`positions.thesis`, `conviction`, `exit_reason`);
needs a prediction-ID foreign key and a small UI.

For options specifically this closes a gap Item 3 can only estimate: the linkage records
your **actual** fills, which is the real spread you paid, so the modeled cost can be
checked against reality rather than assumed.

**Build the logging now, hold the analysis.** See
[trap 3](#trap-3--personalized-learning-needs-hundreds-of-trades).

### Item 13 — Grounded AI chat

"Why are you bearish?" answered from the **already-computed** consensus dict and nothing
else — the prompt must forbid introducing facts not present in the payload. Ungrounded,
it will confidently narrate a VWAP reclaim that never happened. Routes through the
existing `ai/client.py` multi-provider path; gate on `ai_available()`.

### Item 14 — Alerts via cron

Streamlit has no background loop — a page computes only while you are looking at it, so
in-app alerting is not possible. The honest path is a standalone script on the
`scripts/scheduled_retrain.py` pattern (not imported by the app, run from cron/launchd)
that evaluates trigger conditions and writes to a notification file or pushes to a
device.

Worth alerting on: consensus flip, a horizon crossing from uneconomic to tradeable,
regime shift, retrain trigger firing, IV Rank crossing its band.

Note this is the one feature that would need the `persist=False` parameter the read-only
cockpit decision let us skip — an alert loop generating predictions would corrupt the
log. Handle it when Item 14 starts, not before.

---

## Explicitly rejected, and why

Recorded so these do not get re-litigated from scratch.

| Idea | Why not |
|---|---|
| **Multi-ticker scanner** | Ranks symbols; this user trades one. The useful transform is to scan **horizons**, which is Item 1. |
| **Watchlist intelligence** | Same multi-ticker premise. "Confidence changed since yesterday" is worth having *for SPY* — that is a column in the cockpit, not a page. |
| **Portfolio sector exposure** | A sector breakdown of a 100%-SPY book is 100% "broad market." |
| **Letter grades (A+/A/B/C/D)** | The value of an ordinal grade is fast cross-ticker sorting. With one ticker it only destroys information, compressing a calibrated probability into a coarse bucket — the exact false precision the app avoids everywhere else. |
| **Full indicator glossary** | Covered where it matters by the existing "How to read this" expanders. |
| **AI mentor / lesson mode** | For a solo advanced user, one more thing to scroll past. |
| **Market overview dashboard** | Built: `pages/home.py`. |

---

## Four traps to avoid while building this

### Trap 1 — hardcoded signal weights are false precision

`Trend 35% / Momentum 20% / Volume 15% / Pattern 20% / News 10%` looks measured. Where
does 35 come from? Two honest options: fit the weights from the logged history (Item 10),
or show the vote as the count it already is. Do not ship invented weights into an app
whose credibility rests on not doing that.

### Trap 2 — "Confidence 82%" is not currently computable

The confidence tiers are **distance from the neutral band** — a spread measure, not
P(correct). `_render_prediction_card`'s help text is careful about this, and Model Lab
exists specifically to test whether HIGH means anything. A composite "82%" would silently
promise calibration that has not been verified.

The honest version, which the app can already compute: **the realized accuracy of that
confidence bucket at that horizon** — "HIGH confidence on 15m SPY has landed 56.8% of 44
times." Less exciting, actually true, directly sizeable.

### Trap 3 — personalized learning needs hundreds of trades

With 20-40 round trips, "you lose on countertrend setups" is noise that reads as
self-knowledge, and acting on it is worse than acting on nothing. The app already
enforces this bar elsewhere (`RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK = 20`, ORBC's
sub-30-trade warning). Apply it here: log from Item 12, gate the analysis behind a real N.

### Trap 4 — auto-generating predictions corrupts the track record

`predict()` and `predict_intraday()` both call `save_*_prediction()` unconditionally.
Any feature that generates predictions on page render appends to
`storage/*_predictions.jsonl` every time it runs, inflating N and distorting the live win
rate that Model Lab, the retrain triggers, and the trust scoreboard all depend on.

**Resolved for v2.0** by the read-only cockpit decision — generation stays behind an
explicit button. Item 14 (alerts) reopens it and will need `persist=False` on both
functions.

---

## Open questions

Resolved: instrument, cockpit placement, refresh behavior, expiry handling, spread
sourcing — see [Decisions made](#decisions-made). Nothing blocks Tier 1. Still open, all
with a working default:

1. **Daily model in the cockpit?** It has a multi-day horizon, so it is swing context
   rather than an intraday tradeable row. Default plan: include it, with a real net edge
   from the expiry sweep (a 30 DTE contract against a 5-day signal is a coherent trade)
   but marked as context rather than a session signal. Confirm or drop.
2. **Horizon-scaled stops — replace or coexist?** Default plan is coexist: show the
   horizon-scaled stop for intraday signals, with the daily-ATR stop alongside when they
   differ materially.
3. **`theta_basis` default** — plan is `"trading"` (390-minute session), the more
   conservative and more realistic intraday choice. Confirm, or show both as a range.
4. **Expiry ladder rungs** — plan is `[0, 2, 7, 30]` DTE. Add 45 or 60 if the grid shows
   net edge still climbing at the 30-day end.

### To confirm once the app is running during market hours

Not blocking — the code reads these live — but worth eyeballing against reality the first
time the grid renders:

- The ATM bid/ask actually returned for SPY (Trading Desk -> Options -> ATM Greeks).
- Whether the 0DTE theta guard fires and which method it reports.
- Whether the grid's diagonal matches intuition, i.e. shorter signal horizons needing
  longer expiries.

---

## Sequencing

```
Item 5  (caption fixes)          -- do first; Items 1-4 quote these numbers
   |
Item 2A (expiry math)            -- shared dependency of Items 1 and 4
   |
Item 3  (options tradeability)   -- assess_options_tradeability + sweep_expiries
   |
Item 1  (cockpit)  ---+
Item 4  (scoreboard)  |          -- both consume Items 2A and 3
   |                  |
Item 2B (horizon stops)
   |
-- v2.0 --
   |
Items 6, 7, 8, 9, 10, 11         -- independent; 6 is cheapest, 7 most novel,
   |                                11 refines Item 3
-- v2.5 --
   |
Items 12, 13, 14                 -- 14 needs 12's data to be worth alerting on,
                                    and reopens trap 4
```

Nothing in Tier 1 is blocked. Items 1, 2, 4, 5 are wiring existing computed values into
one place — no new modeling, no new data sources, no new dependencies. Item 3 is the only
genuinely new math, and it gates the honesty of everything downstream of it.
