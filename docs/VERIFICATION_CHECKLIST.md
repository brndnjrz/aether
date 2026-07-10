# Verification Checklist — Timezone / Activity Log / Options Log

Covers manual verification of three changes: the ET timezone fix (`config/tz.py`), the
activity log (`portfolio/activity_log.py`), and the options fill ledger + FIFO round-trip
matcher (`portfolio/option_fills.py`, `portfolio/round_trips.py`).

## 1. Timezone check

Launch the app (`streamlit run app.py`), go to Trading Desk → Day Trading. Confirm the
market-status banner and the "⏰" timestamp both show the correct current Eastern time
(compare against your phone's clock).

`now_et()` in `config/tz.py` uses `zoneinfo.ZoneInfo("America/New_York")`, which
auto-handles EDT/EST — there's no legacy `datetime.now()`/`utcnow()` left in that render
path (`pages/trading.py:155`). If the banner/clock is off by exactly 1 hour, that's stale
tzdata on the machine, not a bug in the code.

## 2. Day Trading activity logging

Click **Analyze** for a ticker, then click it again without changing ticker/interval, then
again after switching ticker. Check `storage/journal.db`:

```
sqlite3 storage/journal.db "select * from activity_log"
```

Confirm exactly **one row per distinct click** (same ticker/interval clicked twice = one
row, not two).

The Analyze button writes through `log_activity("day_trading_analyze", ...)` at
`pages/trading.py:454`, guarded by a dedup check right above it (`trading.py:452-453`):

```python
log_key = (ticker, interval)
if st.session_state.get("_last_logged_dt") != log_key:
    log_activity(...)
    st.session_state["_last_logged_dt"] = log_key
```

A row is written only when `(ticker, interval)` changes from the last logged value — a
Streamlit rerun on the *same* ticker/interval won't re-log. Test sequence:

- Click Analyze on AAPL/5m → 1 row.
- Click Analyze again on AAPL/5m (no change) → still 1 row (dedup suppresses it).
- Switch to MSFT or a different interval, click Analyze → a 2nd row.

If a second click of the *same* ticker/interval produces a 2nd row, the dedup key isn't
surviving the rerun.

## 3. Options & Predictions activity logging

On the Options tab, type a new ticker and pick a new expiry; on the Predictions tab, click
**Generate Prediction**. Re-run the same `activity_log` query and confirm new rows appear
(`options_view`, `options_expiry_view`, `prediction_generated`) with sensible
`detail_json`.

- Options tab view fires `log_activity("options_view", ...)` at `trading.py:671`; picking
  an expiry fires `log_activity("options_expiry_view", ticker, {"expiry": selected_expiry})`
  at `trading.py:728`. Neither has a dedup guard like step 2 — switching ticker/expiry logs
  every time, which is expected.
- Predictions tab fires `log_activity("prediction_generated", ...)` at `trading.py:1383`.

Confirm `event_type` and `detail_json` match what actually happened — e.g.
`options_expiry_view` should have `{"expiry": "<the date picked>"}`.

## 4. Options Log accuracy — the money-math check

Go to Portfolio → **📝 Options Log** and manually re-enter the 7 real fills from the
screenshots (SPY $754 put buy/sell, SPY $743 call buy/sell x2, SPY $753 put buy/sell) with
their actual EDT fill times. Confirm the computed round trips match the hand-calculated
P&L: **+$940, +$1,390, +$1,940**, and the still-open $743 call shows **no round trip**
(it stays an unmatched open lot).

This validates `portfolio/round_trips.py`'s FIFO matcher:

- The "Log a fill" form (`portfolio.py:265`) stores ticker, strike, call/put, expiry,
  buy/sell, contract qty, price per contract, and fill date+time (as ET via `MARKET_TZ`).
- `compute_round_trips()` groups fills by `(ticker, strike, option_type, expiry_date)`,
  sorts by `filled_at`, and FIFO-matches buys against sells (a deque per side). Each
  matched chunk becomes one round trip: `pnl_dollars = (exit - entry) * qty * 100`.
- Matching is FIFO per contract-key, so entry order matters — enter fills in actual
  chronological fill order, not grouped by buy/sell, especially for the $743 call which has
  2 buys and 2 sells.
- Expected: three closed round trips totaling +$940, +$1,390, +$1,940, and the still-open
  $743 call leg stays in the `open_buys`/`open_sells` deque and never gets emitted as a
  round trip — confirm it appears only in the fill ledger, not in the round trips table.

## 5. Analytics sanity check

Confirm the hold-time-bucket chart correctly separates the 26-min and 1h43m trades from the
overnight trade, and the win-rate-by-hour chart renders without error on this small sample.

`_render_round_trip_analytics()` (`portfolio.py:359`) buckets by `hold_bucket` (`<30min`,
`30min-2h`, `2h-24h (overnight)`, `>24h`, computed in `round_trips.py:_hold_bucket`) and
plots win rate per bucket, plus a second chart grouping by `entry_hour` (ET). With the
26-min trade (`<30min`), 1h43m trade (`30min-2h`), and overnight trade (`2h-24h`), confirm
each lands in the expected bar and the "win rate by entry hour" chart doesn't error with
only 3 data points.

## 6. Regression check

Confirm the existing ML Predictions history table (Trading Desk → Predictions) still
renders correctly and now shows times in **ET instead of UTC**, without breaking on old
UTC-stored rows in the prediction history file.

Prediction timestamps now render via `utc_iso_to_et_str(pred_ts, ...)`
(`trading.py:1319`, `824`, `833`). That function assumes naive input is UTC and converts to
ET, so old rows stored as plain `datetime.utcnow().isoformat()` (no tzinfo) should still
convert correctly via `tz_localize("UTC")`. Confirm old rows show a *plausible* ET time (not
shifted by an extra offset) and the table doesn't error on the historical row format.
