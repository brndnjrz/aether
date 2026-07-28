# ORBC — Opening Range Breakout Confirmation

Strategy definition, implementation notes, and the practical daily routine for
Strategy Lab's ORBC tab.

Companion to `workflow.md`, which covers the Screener / Research / Options Log /
Trading Desk loop. This doc covers only the intraday ORBC strategy.

- [The idea in plain terms](#the-idea-in-plain-terms)
- [Strategy rules](#strategy-rules)
- [Implementation decisions](#implementation-decisions)
- [Configurable parameters](#configurable-parameters)
- [Daily routine](#daily-routine)
- [Weekly review](#weekly-review)
- [Habits that matter](#habits-that-matter)
- [What it cannot do](#what-it-cannot-do)

---

## The idea in plain terms

After the 9:30 ET open, the first 15 minutes set a high and a low — the
**opening range**. If price later closes above that high, it might be a real
breakout, or it might be a head-fake, which is extremely common right after the
open.

So instead of buying the first candle that pokes out, ORBC waits: it only
signals when a **second consecutive candle** also closes outside the range. Same
logic mirrored for breakdowns below the opening low.

That one-bar wait is the entire edge. It costs you a slightly worse entry price
in exchange for skipping a large share of failed breakouts.

---

## Strategy rules

**1. Opening range** — from the first bars in `[09:30, 09:30 + N minutes)`,
store:

```
Opening High  = max High
Opening Low   = min Low
Range Size    = High - Low
```

With the 15-minute default on 5-minute bars, that is exactly the 09:30, 09:35,
and 09:40 bars. The 09:45 bar is the first one eligible to break out.

**2. Breakout** — a bar closing `> Opening High` (bullish) or `< Opening Low`
(bearish).

**3. Confirmation** — count consecutive closes outside the range. Signal on the
2nd. A close back **inside** the range resets the count to zero: a breakout that
round-trips has to start over.

**4. Entry** — the confirming bar's close. `BUY` for long, `SELL` for short.

**5. Filters** (all optional, all on by default):

| Filter | Rule | Why |
|---|---|---|
| Volume | breakout bar volume > 20-bar average | a breakout nobody participates in is not a breakout |
| VWAP | longs only above VWAP, shorts only below | don't fight the session's average price |
| ATR | Range Size > 0.5 × ATR(14) | skip dead, coiled opens |
| Time | entries only between range end and 11:00 ET | the edge decays into midday chop |

**6. Exits**

- Stop: opposite side of the opening range (default), or 1 ATR, or a fixed percent
- Target: 2:1 risk/reward (default), or an ATR multiple, or the range width
  projected from the break
- All positions flatten at the session close — this never holds overnight

---

## Implementation decisions

Logic lives in `analysis/orbc_strategy.py`; the page is display-only.

**The "second OR third candle" rule.** The original spec fired a signal at
`count == 2` *and* again at `count == 3`, which double-signals the same
breakout. Resolved as: fire once on the 2nd close; if a filter blocks it there,
later closes may still fire up to `max_confirmation_closes` (default 3). Exactly
one signal per breakout episode, but a filter miss on the second bar doesn't
kill a setup that confirms on the third. Blocked attempts are recorded and shown
in the UI with the reason.

**`one_signal_per_session` means per session, not per direction.** A day that
confirms long, stops out, then confirms short produces **one** trade. Set it
`False` to allow re-entry after a fresh breakout episode.

**Shorts are fully supported.** `evaluate_orbc_trade()` is direction-aware,
unlike the long-only simulators elsewhere in the app.

**No lookahead.** Entry is the signal bar's close and the P&L walk starts at the
*next* bar, so a target touched on the signal bar itself cannot count as an exit.

**Confidence score (0–100)** — weighted from volume thrust (25%), how decisively
the confirming closes cleared the range (25%), VWAP alignment (20%), range
quality vs. ATR (15%), and short-term EMA agreement (15%). Missing indicator
columns score a neutral 0.5 rather than penalizing.

**Filters skip rather than fail** when their input is unavailable (short history
means NaN ATR/VWAP/volume average). A skipped filter is reported as skipped —
it never silently suppresses every signal.

---

## Configurable parameters

In the ORBC tab's **Strategy parameters** panel:

| Parameter | Default | Notes |
|---|---|---|
| Bar interval | 5m | also 1m / 2m / 15m |
| Opening range | 15 min | 30 min is the common alternative |
| Confirm on close # | 2 | the core rule; 1 disables confirmation entirely |
| Give up after close # | 3 | filter-fallthrough window |
| Entry cutoff | 11:00 ET | |
| Volume confirmation | on, 1.0× | multiplier of the 20-bar average |
| VWAP alignment | on | |
| ATR range filter | on, 0.5× | see caveat below |
| Allow longs / shorts | both on | |
| Stop method | opening range | or ATR multiple, or percent |
| Target method | 2:1 R:R | or ATR multiple, or range projection |
| One signal per session | on | off allows re-entry |

**Caveat on the ATR filter.** At `0.5 × ATR` on 5-minute bars this barely binds,
because a 3-bar opening range is naturally wider than a 1-bar average true
range. It only screens out genuinely dead opens. Raise the multiplier (1.5–2.0)
if you want it to actually filter.

---

## Daily routine

### Before the open (9:00–9:25)

**Your own watchlist.** Aether doesn't maintain one — pull up whatever you use
externally (Webull or similar) for the 5–10 tickers you're tracking, each with
a target, stop, and one-line thesis.

You are answering one question: *which of my names is actually in play today?*
A name gapping 2% on 3× volume deserves attention. A name flat on 0.6% volume
does not. Aether has no automated gap/volume scan across a saved list — check
each candidate individually in Research, or use your external watchlist tool's
own screener for this pass.

**Dashboard.** Ten-second glance. Is the S&P above its 200-day? Where is VIX?
This sets **position size**, not direction. High VIX means smaller size and
wider stops.

### The open (9:30–9:45)

Do nothing. This is the point of the strategy — the first 15 minutes *define*
your reference range rather than being something to trade. Open **Strategy Lab →
ORBC** on your best candidate and let the range form.

### The window (9:45–11:00)

1. **Check Range Size.** Very small = coiled and indecisive; breakouts from it
   tend to fail. Very large = the move may be spent and your stop will be wide.
2. **Watch for closes outside the band** (hollow circles on the chart).
3. **Wait for the signal box.** One close outside is nothing. When the second
   confirms you get a green BUY or red SELL box with entry, stop, target, R:R,
   and a confidence score.
4. **Read the ✕ marks.** These are breakouts that reached the confirmation count
   but a filter rejected — hover for the reason. "Volume 0.6× the average" is
   the most useful thing on the chart: a breakout happened and nobody showed up.
5. **Open the Confidence breakdown** before acting. A 75+ with volume thrust and
   VWAP alignment both high is a different trade than a 45 that squeaked past.
6. **Click "Log this ORBC signal"** whether or not you take it.

After 11:00 the scanner stops signalling by design.

### Sizing — do not skip

Take the entry and stop from the ORBC card to **Trading Desk → Day Trading →
Quick Risk Calculator** (bottom of the tab). It returns the share/contract
count that risks 1% of the account *on that specific stop distance*. This step
decides whether you survive a losing streak — it matters more than signal
quality.

### Logging fills

Log every options fill on the **Options Log** page in chronological order as
the broker reports it. The FIFO matcher builds round trips with P&L, hold time,
and win rate by hold-time bucket and entry hour.

**Entry-hour win rate is the most actionable number in the app.** Most traders
have one hour of the day where they consistently lose money and don't know it.

---

## Weekly review

Roughly 30 minutes, Sunday.

- **Options Log analytics.** Which hold-time bucket and which entry
  hour actually make money? Cut the worst one next week.
- **Strategy Lab → ORBC → Backtest** on your top 3 tickers. Compare filters on
  vs. off, 15-min vs. 30-min range. Intraday history caps at ~60 days and ORBC
  fires at most once per session, so this yields a few dozen trades — the app
  warns when the sample is under 30. Treat results as *directional, not proof*.
- **Research page** on any name you're considering adding — fundamental
  scorecard, red flags, ML signal, AI brief.
- Rebuild next week's watchlist (external tool) with fresh targets and theses.

---

## Habits that matter

**Log signals you skip, not just ones you take.** Over a month the activity log
tells you whether your discretionary filtering adds value or destroys it. Most
people find they skip their best setups.

**Tune one parameter at a time.** The panel has a dozen knobs. Changing five at
once and getting a better backtest teaches you nothing except that 40 trades can
be overfit. Change one, test on 3 tickers, keep or revert.

---

## What it cannot do

Bars only. No Level 2, no order flow, no news awareness.

It will not see an earnings surprise or a Fed headline coming. A clean ORBC
signal into a scheduled 10:00 AM economic release is a coin flip regardless of
what the confidence score says — check the calendar yourself.

Backtest results are close-to-close with no intrabar fills, no slippage, and no
commissions. Real execution will be worse than the equity curve.

Not financial advice.
