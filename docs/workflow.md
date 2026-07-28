# Aether — Investing Workflow

This app is a **research tool, not a trading platform**. There's no order execution, no
brokerage connection, and no equity position-entry workflow — the Dashboard's Open Positions
summary will show "no open positions" until the app is wired to a real data source. The value is
in the research, screening, and signal generation, not in logging equity trades. Options fills
are the exception: the Options Log page is a real, working trade journal.

A natural weekly/daily loop looks like:

**Screener → Research → Trading Desk (Day Trading / Options / Predictions)**, with **Options Log**
as where you actually record options fills and review your own trading patterns.

---

## 1. Screener — find candidates

You give it a ticker list (or use a built-in 50-stock S&P sample) and pick one of four filters:
Quality+Momentum, Oversold Quality (mean reversion), High IV Rank (options premium candidates),
or a custom watchlist scored by recent returns. Every screen enforces a baseline quality gate
(ROIC > 6%, or gross margin > 30%, or positive FCF yield) so you're not screening pure junk.
Output is a ranked table with a composite score.

**Can't do:** no backtested historical returns for the screen, no multi-factor optimization, and
it loops tickers one at a time — slow past ~100 names. ML signals only show up here if you
already trained that ticker's model elsewhere; it never trains inline.

## 2. Research — deep dive on one ticker

This is the core analysis page: fundamentals (P/E, PEG, EV/EBITDA, FCF yield, ROIC,
debt/equity), a 0–100 quality/value/growth scorecard with red-flag detection, full technical
picture (trend regime, RSI/MACD/ADX/Bollinger Bands, support/resistance), and — if you've
trained a model — the ML direction signal (auto-selected 3/5/10-day horizon). There's also an Options tab (IV rank quick
view) and an AI Brief tab that, if AI is configured (Claude or Ollama — see AI Setup in the
README), synthesizes everything into a one-page thesis with a BUY/HOLD/AVOID/WATCH call and
due-diligence questions.

**Can't do:** it's single-ticker only, fundamentals lag by a quarter (yfinance, not real-time
filings), and the AI brief is a synthesis of the data already shown — it doesn't add outside
information or predict a price.

## 3. Options Log — the one place trades actually get logged

A real fill-logging workflow, not a read-only view: log each option fill as your broker reports
it (ticker, strike, call/put, expiry, buy/sell, qty, price, fill date+time in ET), and
`portfolio/round_trips.py` FIFO-matches buys against sells per `(ticker, strike, type, expiry)`
group into a round-trips table with actual P&L, hold time, and win/loss.

The analytics below that are built specifically to find patterns worth acting on: a cumulative
P&L equity curve (is this working, overall?), win rate by hold-time bucket and by entry hour,
win rate by option type (calls vs. puts) and by day of week, and a per-ticker performance
breakdown (total/avg P&L, win rate) — plus an activity-correlation picker that shows what Day
Trading/Options/Predictions/Strategy Lab signals were on screen during a given trade's hold
window (fed by the activity log described under Trading Desk below).

**Can't do:** there is no UI to log an *equity* position at all — that's tracked only by the
Dashboard's Open Positions summary, which reflects whatever (if anything) is in the local
`positions` table, with no way to add to it from the app. This limitation does not apply to
options, which are fully loggable here. It's also FIFO-only — if your actual fill order in a
multi-lot contract isn't chronological in how you enter it, the matched round trips will be
wrong; enter fills in true chronological order, not grouped by buy/sell.

## 4. Trading Desk — Day Trading, Options, Predictions

One page, four tabs:

- **Day Trading**: a faster loop for intraday work on one ticker, with a market-status banner
  (PRE-MARKET / MARKET OPEN / AFTER-HOURS, ET clock) up top — VWAP deviation, momentum
  score, volume ratio, MA-alignment, and a candlestick pattern read (Doji, Engulfing, Inside Bar,
  NR4 — pure OHLC geometry, no ML) as a fifth signal, plus opening-range breakout tracking and
  pivot levels. VWAP deviation, momentum, and volume ratio are sourced from whichever bar you
  select (5m/15m/30m/1h intraday, or daily if no intraday data exists) and VWAP resets each session for
  intraday bars, so the interval you pick actually changes those three signals; MA-alignment
  stays anchored to daily SMA20/50/200 + EMA50 on purpose, as the swing-trend backdrop rather
  than a reactive signal. A **Chart Patterns** section runs Flag/Pennant continuation-pattern
  detection (`analysis/flag_pennant_detection.py`): swing points pair into a pole, the
  consolidation after the pole gets trendlines fit to it (parallel = Flag, converging = Pennant),
  and a pattern is only ever shown once its breakout has actually confirmed — never a forming,
  unconfirmed setup. Each confirmed pattern gets a 0–100 confidence score
  (`analysis/flag_pennant_scoring.py`, weighted on volume behavior, breakout momentum, pole
  strength, ATR compression/expansion, trend alignment, trendline fit, and flag symmetry) and is
  filterable by a confidence-threshold slider; the chart itself draws the shaded consolidation
  zone, trendlines, pole, and breakout marker for each one. A **Suggested Entry / Stop / Target**
  card takes a majority vote across those signals (volume excluded, since it reads conviction
  rather than direction; the most recent confirmed Flag/Pennant contributes a vote alongside
  VWAP/momentum/trend/candlestick) and proposes a bias, an ATR-based stop, and a
  reward-to-risk-aware target (a pivot level if it clears 1.5:1, a Flag/Pennant measured-move
  target if that's better, otherwise a fixed 2:1) — a rule-based suggestion, not a guaranteed
  setup; a confirmed Flag/Pennant in the card's direction can only tighten the stop or improve the
  target, never loosen either. An "AI Day Trading
  Brief" button synthesizes all the signals into a plain-English intraday read with a session
  bias (works with Ollama or Claude — see AI Setup in the README). Every Analyze click here (and
  every Options tab view/expiry pick and Predictions generate elsewhere on this page) writes to a
  local activity log (`portfolio/activity_log.py`) that the Options Log page's
  "what were you looking at during this trade?" picker reads back later.
  A "Backtest: MACD Bullish Cross" expander lets you test one concrete rule (MACD histogram
  turns up while MACD/Signal are still negative and price is above the 200-day average, 2%
  stop / 3% target, one trade at a time) against 2 years of history, showing win rate, trade
  count, and an equity curve — the first real answer in the app to "did this signal actually
  work historically?"
- **Options**: real chain data (calls/puts, IV, OI, volume) plus IV Rank (current HV vs 52-week
  range) and an IV-vs-realized-vol ratio to tell you whether premium looks rich or cheap. A
  GARCH(1,1) forward volatility forecast (fit on daily returns, horizon-matched to the nearest
  expiry) gives a second, forward-looking read via an IV/GARCH ratio — same rich/cheap framing as
  IV/RV, but comparing IV to where volatility is modeled to go rather than where it's already
  been. Rule-based
  strategy suggestions (Iron Condor when IV is high and price is sideways, Bull Put Spread when IV
  is high and trend is up, etc.) with a payoff diagram.
- **News**: recent headlines for the entered ticker, VADER-scored positive/neutral/negative, with
  an overall tone and compound-score summary. Display-only — headline-level sentiment, not a
  trading signal on its own, and it doesn't feed the Suggested Entry/Stop/Target card or any
  other signal on the page.
- **Predictions**: XGBoost + Random Forest ensemble trained per-ticker on 2 years of daily data.
  Training auto-selects, per ticker, whichever label horizon (3, 5, or 10 trading days) and
  XGBoost hyperparameters the ensemble is most consistently accurate with — a small grid search
  scored via reduced-fold walk-forward, rather than one fixed 5-day config for every ticker. Built
  with real discipline against overclaiming: the winning configuration still gets the full
  walk-forward validation (not a single train/test split), a neutral zone that excludes
  flat-return days from training, and probability output deliberately clipped to 35–65% so it
  never looks more confident than a ~52–58% historical edge actually is. Models expire after 30
  days and need retraining. Alongside the direction call, a **Price Path** simulation
  (Monte Carlo, seeded from the stock's own historical volatility and the model's own
  probability/expected-move as drift, run over the model's auto-selected horizon) reports a
  day-by-day 25th–75th percentile open/close band — a probability range, not a second
  point-forecast model, and it carries no walk-forward accuracy figure of its own. A **Prediction
  horizon** toggle switches to a separate, 15-minute-bar model with its own features/labels/
  storage — expect a much weaker edge there; see `docs/ML_PREDICTION.md`.

**Can't do:** no real-time tick data (intraday is cached 60s, still delayed by yfinance under the
hood); ATM Greeks (delta/gamma/theta/vega/rho) are computed via Black-Scholes off each option's
own implied volatility (or a solved fallback when yfinance's IV field is missing), not observed
market sensitivities, and they're only computed for the ATM call/put, not the full chain; the
P&L diagrams use an approximate premium estimate rather than a full options pricing model; the
ML signal does not predict news, earnings, or macro shocks, has no
long-horizon signal (auto-selected between 3 and 10 days only), and a 52–58% directional edge is
real but modest — it's a tilt, not a forecast to size a position around by itself. The Price Path
simulation is not independently backtested — it's a volatility-driven simulation, not a validated
forecasting model, so treat its band as a plausibility range, not an accuracy-scored prediction.
Flag/Pennant detection is unvalidated pattern geometry, same caveat as the candlestick reads —
its confidence score is an internally-consistent weighting of volume/momentum/trend/ATR/shape
factors, not a backtested win rate; nothing in the app currently reports the historical hit rate
of confirmed Flag/Pennant patterns the way the MACD Bullish Cross backtest does for its one rule.

---

## What the whole app fundamentally can't do

- No brokerage connection or order execution, and no UI to log *equity* positions — the equity
  Positions tab reflects whatever (if anything) is in the local `positions` table. Options fills
  are the exception — those have a real logging UI (the Options Log page) with automatic
  FIFO round-trip P&L.
- No real-time data — everything runs on yfinance, which is delayed and occasionally
  rate-limited.
- Equities only — no crypto, futures, forex, bonds.
- No backtesting of strategies, screens, or the fundamental scorecard — the one exception is the
  Day Trading tab's MACD Bullish Cross backtest, which validates a single rule, not the other
  signals shown alongside it (including Flag/Pennant detection), and not the screener or
  scorecard.
- AI features are opt-in and require your own `ANTHROPIC_API_KEY` (or a local Ollama instance)
  — without it, those buttons just don't appear, the rest of the app works fine.

## Practical use pattern

Run Screener weekly for ideas → Research the shortlist for fundamentals/technicals/AI brief →
use Trading Desk's Day Trading tab for intraday timing (VWAP/momentum/trend/candlestick/
Flag-Pennant signals), Options tab for defined-risk structures, and Predictions tab as a
secondary confirmation signal, never the primary reason to enter → log any options fills in
the Options Log page as you take them, so the round-trip P&L and win-rate-by-hold-time/
entry-hour, by-ticker, and day-of-week analytics build up over time instead of being reconstructed from memory later.

---

## Recommended Trading Workflow

## 1. Day Trading Workflow

### Step 1 — Regime and macro context (Dashboard)

1.1 Open the Dashboard first, every session, before looking at any single ticker. Read the regime banner: Bull, Uptrend, Sideways, Downtrend, or Bear Market, derived from S&P vs. its 200-day MA.

1.2 Check the live index cards for SPY, QQQ, IWM, and VIX. A rising VIX alongside a Downtrend/Bear regime means wider, faster intraday swings — expect your stops to get tested more often. A low, flat VIX in a Bull/Uptrend regime means tighter, slower ranges.

1.3 Scan sector performance (1mo/3mo). If your ticker's sector is lagging badly over 1mo while the index is fine, that's a headwind worth noting before you commit to a long.

1.3b Glance at the Market Regime (Markov) panel below the index cards — a probabilistic second opinion on the same trend (persistence odds and a Bull-minus-Bear signal, fit on the S&P's own history). Treat disagreement between it and the rule-based banner in 1.1 as a reason to weight Step 3's intraday signals more heavily, not as a tie-breaker to resolve on its own.

1.4 **Decision point:**

- If regime = Bull or Uptrend → long setups have the tape behind them; short setups are countertrend, treat them as lower-probability and require extra confirmation in Step 4.
- If regime = Downtrend or Bear → short setups are aligned; long setups are countertrend — same extra-confirmation rule applies.
- If regime = Sideways → neither direction has structural tailwind; weight the intraday signals in Step 3 more heavily than the regime itself.

### Step 2 — Open Trading Desk > Day Trading tab

2.1 Select your ticker and choose the bar interval (5m, 15m, 30m, or 1h; falls back to daily if no intraday data is available for that symbol).

2.2 Confirm the interval matches your intended holding period — a 5m chart for a trade you plan to hold two hours will generate noisy, contradictory reads. Roughly: 5m/15m for scalps of a few minutes to under an hour, 30m/1h for trades you plan to hold multiple hours to the full session.

2.3 Note that VWAP is session-anchored and resets each day, while Trend Alignment always uses daily SMA20/50/200 + EMA50 regardless of what interval you picked in 2.1 — do not expect Trend Alignment to react to your chosen interval; it's a fixed swing-trend backdrop.

### Step 3 — Read each of the six signals individually

3.1 **VWAP deviation** — if price is meaningfully above session VWAP (roughly >0.3%) and volume is not fading, that's a bullish read; meaningfully below is bearish. If price is oscillating within a tight band around VWAP (roughly ±0.1%), there is no signal — treat this ticker as rangebound intraday and consider skipping it.

3.2 **Momentum score (RSI/MACD/EMA20)** — check whether RSI/MACD/EMA20 agree with the VWAP read from 3.1. Momentum confirming VWAP direction strengthens the case; momentum flat or diverging while VWAP is stretched is a warning that the move may be exhausting.

3.3 **Volume ratio** — is current volume meaningfully above its recent average, or below? A directional VWAP/momentum read on below-average volume is a weak, low-conviction signal; the same read on elevated volume is a stronger one. Volume is descriptive context here — it is explicitly excluded from the majority vote in Step 4, so use it to size your confidence, not to override the vote.

3.4 **Trend Alignment (daily SMA20/50/200 + EMA50)** — is the intraday direction from 3.1/3.2 running with the daily trend or against it? Aligned = higher-confidence continuation setup. Against it = you're trading a countertrend bounce/fade; require the pattern read in 3.5 and the MACD backtest in Step 5 to both agree before proceeding.

3.5 **Candlestick pattern (Doji, Bullish/Bearish Engulfing, Inside Bar, NR4)** — this is pure OHLC geometry with no statistical validation behind it. Use it only as a tie-breaker or timing cue (e.g., a Bullish Engulfing right at VWAP support), never as a standalone reason to enter.

3.6 **Trendline & swing structure (chart overlay)** — the chart shows two dashed lines (green = floor/support, red = ceiling/resistance, fit off the last 30 bars) plus green/red triangle markers for confirmed swing lows/highs (a swing point only gets marked once price reverses by more than one ATR — small wiggles don't count). Read it in plain terms: is price near the floor (potential long entry), near the ceiling (potential short entry or take-profit for an existing long), or in open space between the two (no structural edge, lean on 3.1-3.5 instead)? A break through the ceiling on above-average volume (3.3) is a stronger continuation case than the same break on weak volume — treat a low-volume break as more likely to fail and revert.

3.7 **Flag & Pennant pattern (Chart Patterns section)** — check whether a confirmed Bull or Bear Flag/Pennant is showing, and at what confidence score. It's a continuation pattern, not a reversal — a Bull Flag/Pennant means the existing uptrend is expected to resume after a pause, so read it alongside the daily trend in 3.4: a Bull pattern with the daily trend already up is a clean continuation entry, the same pattern against a daily downtrend is a countertrend bounce needing the extra confirmation called for in 3.4. Only breakouts that have actually confirmed are ever shown, so you will never see a "forming" pattern to trade around — by the time it appears, the trendline has already broken on a closing basis. Treat anything below roughly 65 confidence as a minor input and anything above roughly 80 (marked with a star breakout icon instead of a diamond) as a stronger read worth weighting alongside 3.1-3.4.

### Step 4 — Suggested Entry/Stop/Target card

4.1 Read the majority-vote bias (LONG/SHORT/NEUTRAL) — this is the vote across VWAP, momentum, trend, candlestick pattern, and (if one confirmed within the last 10 bars) Flag/Pennant direction — volume excluded, per 3.3.

4.2 If the card returns NEUTRAL, or if your own read in Step 3 disagrees with the card's bias, stand down on this ticker for now rather than forcing a trade.

4.3 Note the ATR-based stop (1.5x ATR from entry) and the target (nearby pivot if it clears 1.5:1 reward:risk, a Flag/Pennant measured-move target if that's better, otherwise a fixed 2:1). Confirm the R:R is at or above 1.5:1 before moving forward — if the card had to fall back to the fixed 2:1 target because no pivot or pattern target qualified, treat that target as more theoretical and watch price action near it rather than assuming it will hold.

4.4 If a confirmed Flag/Pennant from 3.7 agrees with the card's direction, the card will already have tightened the stop to the pattern's own boundary and/or swapped in its measured-move target when it's better — check the caption line under the card to see whether that happened; it never loosens the stop or picks a worse target. Separately, cross-check the stop/target against the swing structure from 3.6: if the nearest swing low sits closer to entry than the (possibly already-tightened) stop, tighten it further to just below that swing low instead — it's a level the market has already defended, not an arbitrary multiple. Likewise, if a swing high or the resistance trendline sits before the suggested target, treat that as a realistic first target even if it's short of the card's current target — take partial profit there rather than assuming price sails through it.

### Step 5 — Cross-check against the one backtested rule

5.1 Open "Backtest: MACD Bullish Cross" for this ticker. This is the only validated signal in the entire app (rule: MACD histogram turns up while MACD/Signal are still negative, price above 200-day MA; 2% stop / 3% target; one trade at a time; 2 years of history). The Flag/Pennant confidence score from 3.7 is not backtested the same way — it's an internally-consistent weighting, not a measured historical win rate — so don't treat a high pattern confidence score as equivalent to this backtest's validation.

5.2 Check win rate and trade count together — a high win rate on only 4-5 trades over 2 years is not meaningful; look for a reasonable trade count (double digits) alongside a win rate clearly above 50%.

5.3 **Decision point:**

- If today's live setup matches an active MACD bullish cross AND Step 4's card also says LONG → this is the highest-confidence alignment the app can produce. Proceed to Step 6.
- If the backtest shows a poor win rate or thin sample for this ticker, or if there's no active cross today → treat Step 4's card as unvalidated and downsize accordingly, or skip.

### Step 6 — ML Predictions tab (confirmation only, never the trigger)

6.1 Check the direction call (bullish/neutral/bearish) and bull probability (capped 35-65% — it will never show false high confidence).

6.2 Check walk-forward accuracy for this specific ticker's trained model. A 52-58% edge is real but modest — a tilt, not a forecast.

6.3 **Decision point:** use this only to break a tie between two otherwise equally-supported setups. Never let a bullish ML read override a NEUTRAL or conflicting card from Step 4, and never enter a trade solely because the ML signal is bullish.

### Step 7 — AI Day Trading Brief (optional)

7.1 If available, run it and compare its stated bias and invalidation risk against your own conclusions from Steps 3-6. Use it to catch anything you missed, not as a new independent signal.

### Step 8 — Size the trade (Quick Risk Calculator, same Day Trading tab)

8.1 Open the "Quick Risk Calculator" expander at the bottom of the Day Trading tab — it's on the same page, no navigation needed. Enter the stop price from Step 4.3/4.4 and your risk % per trade.

8.2 Take the resulting share count as your position size. If this is an equity trade, positions still cannot be logged from the UI — record it manually outside the app (spreadsheet, journal), the app will not track it for you. If this is an options trade, log the fill on the Options Log page right after you're filled instead — that one persists, feeds the FIFO round-trip P&L, and lets the win-rate-by-hold-time/entry-hour analytics build up over time.

### Step 9 — Pre-trade checklist: Day Trading (final gate, run immediately before entry)

1. Regime (Step 1.1) and your setup's direction are aligned, or you've explicitly accepted the countertrend risk.
2. VWAP, momentum, and trend (3.1, 3.2, 3.4) agree on direction — no more than one of the three is neutral/conflicting.
3. Volume ratio (3.3) is at or above average — not a low-conviction, low-volume drift.
4. Entry is near the floor/support (long) or ceiling/resistance (short) from 3.6, not in open space between them — unless you're taking a confirmed breakout on strong volume.
5. If a Flag/Pennant is showing (3.7), its direction matches your trade direction and its confidence score isn't near the bottom of your threshold — a marginal pattern shouldn't be the deciding factor.
6. Card bias (4.1) is LONG or SHORT, not NEUTRAL.
7. Stop (4.4) sits just beyond the nearest swing low/high (and the pattern boundary, if tighter), and R:R to the target (4.3/4.4) is at least 1.5:1.
8. MACD backtest (5.1-5.3) either supports this setup directly, or you've consciously downsized because it doesn't apply today.
9. ML signal (Step 6) does not directly contradict your direction.
10. Share count from the risk sizer (8.1) is calculated and ready — you are not sizing on the fly.

## 2. Options Trading Workflow

### Step 1 — Regime and fundamentals context

1.1 Open the Dashboard and read the regime banner and VIX level, same as the day trading workflow. Elevated VIX generally means richer premium across the board — note this before you even check ticker-specific IV.

1.2 Open Research for your ticker and check the Quality/Value/Growth scorecard and any red flags.

1.3 **Decision point:** if there is an active red flag on the scorecard, treat any premium-selling strategy (Iron Condor, credit spreads) as higher-risk regardless of what the IV framework says in Step 2 — the app has no earnings/macro/news awareness, so a red flag plus an unknown near-term catalyst is a reason to reduce size or skip.

### Step 2 — Trading Desk > Options tab: read the IV framework

2.1 **IV Rank** — where current implied vol sits versus its own 52-week range. High IVR (roughly above 50, higher confidence above 70) means options are rich relative to this stock's own history. Low IVR means options are cheap relative to history.

2.2 **IV Percentile** — a complementary read to IVR; if the two disagree noticeably, treat the vol picture as less clean and lean on Step 3's dual rich/cheap check more heavily.

2.3 **ATM Implied Vol** — the real, chain-derived number. Sanity-check it against the ticker's typical historical vol level (you'll cross-check this properly in Step 3).

### Step 3 — Two independent rich/cheap reads

3.1 **IV/RV ratio** — above 1.15 means options are rich versus realized vol, favoring premium-selling strategies. Below 0.85 means options are cheap versus realized vol, favoring premium-buying strategies. Between 0.85 and 1.15 is a neutral read — no strong edge either way.

3.2 **IV/GARCH ratio** — the same rich/cheap framing, but forward-looking: GARCH(1,1) forecasts vol out to the nearest expiry rather than looking backward at realized vol. Read it with the same >1.15 rich / <0.85 cheap thresholds.

3.3 **Decision point:**

- If IV/RV and IV/GARCH agree (both rich or both cheap) → higher-conviction vol read; proceed with the strategy direction both point to.
- If they disagree (one rich, one cheap, or one/both neutral) → the vol signal is ambiguous. Either skip the trade, or move to a smaller, more defined-risk structure (e.g., a spread instead of a naked short) and size down.

### Step 4 — Strategy suggestion, cross-checked against trend

4.1 Read the rule-based strategy suggestion, driven by IVR + trend:

- High IVR + Sideways → Iron Condor
- High IVR + Uptrend → Bull Put Spread
- High IVR + Downtrend → Bear Call Spread
- Low IVR + Uptrend → Long Call / Bull Call Spread
- Low IVR + Downtrend → Long Put / Bear Put Spread

4.2 Open Research's Chart & Technicals tab for the same ticker. Check the auto-detected support/resistance levels, the trendline/swing overlay (floor/ceiling lines and confirmed swing highs/lows, same read as 3.6 in the Day Trading Workflow), RSI, and the trend/momentum panels.

4.3 **Decision point:** does the trend read in the strategy suggestion (4.1) match what you see on the actual chart (4.2)? If the app calls "Uptrend" but price is chopping below a declining SMA50 with RSI under 50, or swing highs/lows are flattening instead of climbing, treat the suggested strategy's directional assumption as suspect — either re-verify manually or default to the non-directional option (Iron Condor logic) if IVR supports it. If you're picking directional strikes (Bull Put Spread, Long Call, etc.), place the short/long strike beyond the nearest swing high/low or trendline rather than at an arbitrary distance — same logic as the day trading stop/target in 4.4 above.

### Step 5 — ATM Greeks and real chain data

5.1 Pull ATM delta/gamma/theta/vega/rho (Black-Scholes off the option's own implied vol, with a solved fallback if IV is missing). Use delta to gauge directional exposure and theta to gauge daily time decay for your structure.

5.2 Remember these Greeks are ATM only, not calculated across the whole chain — before selecting actual strikes, open the real options chain (calls/puts, IV, OI, volume, bid/ask) for the nearest expiration and confirm liquidity (tight bid/ask, reasonable open interest) at the strikes you actually intend to trade.

5.3 **Decision point:** if OI/volume at your intended strike is thin or the bid/ask is wide, either move to a more liquid strike/expiration or reduce size — illiquid strikes will erode any theoretical edge from Steps 3-4.

### Step 6 — ML Predictions (confirmation only, plausibility check on strikes)

6.1 Check the direction call and probability (capped 35-65%) and walk-forward accuracy, same caveats as the day trading workflow — a 52-58% edge is a tilt, not a forecast.

6.2 Check the Monte Carlo price-path band (25th-75th percentile, seeded from historical vol plus the model's probability/expected-move as drift, run over the model's own horizon).

6.3 **Decision point:** if you're selling premium (e.g., Iron Condor, credit spread), check whether your short strikes sit outside the 25th-75th percentile band — if a short strike sits inside that band, treat it as a signal to widen the strike or reduce size, since a "plausible" move could reach it. This band is explicitly not backtested — use it as a plausibility check only, never as validation of the trade itself.

### Step 7 — Payoff diagram and AI brief

7.1 Review the P&L payoff diagram for the exact structure and strikes you're considering. Remember it uses an approximate premium estimate, not a full pricing model — treat max-gain/max-loss figures as directional estimates, not exact numbers to rely on for precise breakeven math.

7.2 If available, run the AI Options Brief and compare its read against your own conclusions from Steps 2-6.

### Step 8 — Size the trade (Quick Risk Calculator, Day Trading tab)

8.1 There's no options-specific sizer — use the same "Quick Risk Calculator" on the Day Trading tab, treating the option's entry/stop premium as the price inputs, to get a contract count relative to account risk %.

8.2 There is no built-in stress-test tool anymore. Manually sanity-check the position against a SPY ±10% / VIX-shock scenario yourself before sizing up, especially for undefined-risk structures.

8.3 Once filled, log the trade on the Options Log page (ticker, strike, type, expiry, side, qty, price, fill time) rather than tracking it outside the app — this is what feeds the FIFO round-trip P&L matching and the win-rate-by-hold-time/entry-hour analytics there.

### Step 9 — Pre-trade checklist: Options Trading (final gate, run immediately before entry)

1. No unresolved red flag on the Research scorecard (1.3), or you've explicitly accepted that risk with reduced size.
2. IV/RV and IV/GARCH (3.1-3.3) either agree, or you've sized down to reflect the ambiguity.
3. Strategy suggestion's trend assumption (4.1) is confirmed against the actual chart (4.2-4.3).
4. Strikes selected have adequate liquidity on the real chain (5.2-5.3) — no wide bid/ask, reasonable OI.
5. If selling premium, short strikes sit outside the Monte Carlo 25th-75th percentile band (6.3).
6. Payoff diagram max-loss (7.1) is acceptable at your intended size, understanding it's an estimate.
7. Contract count from the risk sizer (8.1) is calculated, and the stress test (8.2) shows tolerable drawdown under a SPY ±10% / VIX shock scenario.
8. Trade is queued to be logged manually post-entry (8.3), since the app won't track it.

## What This App Cannot Replace

- **A live broker/data feed** — no real-time quotes, no real-time execution; nothing here should drive split-second decisions.
- **A backtested strategy** — outside of the single MACD Bullish Cross rule, every other signal (scorecard, screener, IVR strategy map, candlestick patterns, ML predictions) is unvalidated logic, not a proven edge.
- **Order execution and position tracking** — there is no way to place a trade or log an equity position from the UI; the Day Trading tab's Quick Risk Calculator is a standalone calculator, not a live book, and there is no portfolio-level stress-test tool.
