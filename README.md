# Aether

**Bottom line:** a Streamlit dashboard for independent traders — live technical analysis, fundamental scoring, an ML direction model (daily + intraday), options analytics, and AI-generated briefs, in one multi-page app. All data is live from yfinance. No mocks.

Not a brokerage. Doesn't execute trades. Not financial advice.

## Table of Contents

- [What It Is](#what-it-is)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Libraries Used](#libraries-used)
- [AI & ML Model Overview](#ai--ml-model-overview)
- [Setup Environment Using Anaconda](#setup-environment-using-anaconda)
- [How to Run the Dashboard](#how-to-run-the-dashboard)
- [How to Use](#how-to-use)
- [Architecture & Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Extensibility & Customization](#extensibility--customization)
- [Future Ideas (Not Yet Built)](#future-ideas-not-yet-built)
- [Data Sources & Caching](#data-sources--caching)
- [Reliability & Verification](#reliability--verification)
- [Disclaimer & License](#disclaimer--license)
- [Appendix: Helpful Commands](#appendix-helpful-commands)

## What It Is

**Bottom line:** self-directed trading research, without paying for a Bloomberg terminal.

It pulls live market data and:

- Runs a machine-learning direction model
- Scores company fundamentals
- Detects intraday chart patterns
- Surfaces options strategies from current implied volatility
- Optionally has an LLM (Claude or local Ollama) turn the numbers into a plain-English brief

Not a brokerage. Doesn't execute trades. Not financial advice.

## Key Features

| Page | What It Does |
|------|--------------|
| **Dashboard** (`pages/home.py`) | Live market overview — index prices, VIX, S&P regime banner, sector performance, open positions summary |
| **Research** (`pages/research.py`) | Full single-stock deep dive: fundamental scorecard, technical chart, ML direction signal, options IV, news sentiment, and an AI investment brief |
| **Options Log** (`pages/portfolio.py`) | The trade journal. Manual fill entry → FIFO round-trip P&L, win-rate analytics by hold time/entry hour/ticker/option type/day of week, cumulative P&L curve, and **Model vs. Discretionary** — did following a signal beat overriding it |
| **Trading Desk** (`pages/trading.py`) | Four tabs. **Day Trading** (status banner, intraday signals, candlestick + Flag/Pennant detection with confidence scoring, suggested entry/stop/target; Market Regime / Risk Calculator / Oscillators / Backtest / Signal Weights / AI Brief as sub-tabs), **Options** (chain, IV Rank, GARCH forward vol, Greeks, P&L diagrams, AI brief), **News** (headline sentiment), **Predictions** (**Horizon Cockpit** across all five models, then daily or intraday signal + price path + similar setups + exit plan) |
| **Strategy Lab** (`pages/strategy_lab.py`) | **ORBC** (Opening Range Breakout Confirmation: requires a 2nd consecutive close outside the opening range before signalling, in `analysis/orbc_strategy.py`) with a Live Scanner and Backtest sub-tab, plus a read-only **Intraday Predictions** reference panel (latest saved intraday prediction per interval, so it can be checked without leaving this page) |
| **Model Lab** (`pages/model_lab.py`) | Read-only track record for both models. **Horizon Scoreboard** (which horizon to trade, with a horizon×expiry net-edge grid), agreement backtest, precision/recall/F1/calibration, accuracy-over-time, why-it-was-wrong failure analysis, 4-model comparison, version history with rollback, retrain triggers. Reads what Trading Desk logged; the **Prediction Improvement Engine** (see [AI & ML Model Overview](#ai--ml-model-overview)) |

Every Analyze click, options view, and prediction on the Trading Desk logs to a local activity log — later surfaced by Options Log's "what were you looking at" picker and the Dashboard's Recent Activity feed.

## Quick Start

**Prerequisites:** Python 3.12, conda (or any virtualenv manager).

```bash
conda create -n aether python=3.12 -y
conda activate aether
cd aether
pip install -r requirements.txt
cp .env.example .env
# Edit .env — see AI & ML Model Overview below for AI setup
streamlit run app.py
```

Runs at **http://localhost:8501**.

## Libraries Used

| Library | Purpose |
|---------|---------|
| `streamlit` | Dashboard UI and multi-page navigation |
| `yfinance` | Price history, fundamentals, options chains — the sole market data source |
| `pandas` / `numpy` | Data manipulation and numerical computation throughout |
| `plotly` | Candlestick charts, overlays, and analytics visualizations |
| `scikit-learn` | Random Forest model and preprocessing pipeline |
| `xgboost` | Gradient-boosted price-direction model (paired with Random Forest in the ensemble) |
| `scipy` | Black-Scholes pricing and implied-volatility solver (`analysis/options_pricing.py`) |
| `arch` | GARCH(1,1) forward volatility forecast (`analysis/volatility_forecast.py`) |
| `vaderSentiment` | Lexicon-based headline sentiment scoring |
| `feedparser` | Google News RSS parsing for headline sentiment |
| `anthropic` | Claude API client for AI briefs |
| `requests` | HTTP calls to a local Ollama server (no `ollama` pip package required) |
| `python-dotenv` | Loads `.env` into `config/settings.py` |
| `tzdata` | Ensures correct US/Eastern conversions via `zoneinfo` on all platforms |

- **`fpdf2`, `Pillow`** — unused, safe to drop.
- **`narwhals`** — pinned on purpose. Real transitive dependency of `scikit-learn`/`plotly`; a version mismatch there once took down the whole Predictions tab for 11 days before anyone noticed (see `tests/test_ml_prediction.py`).

## AI & ML Model Overview

### AI briefs (Claude or Ollama)

Four brief types — stock, options, day-trading, thesis-question — built in `ai/stock_brief.py`, routed through `ai/client.py`.

**Provider selection** (`AI_PROVIDER`):

- **`auto`** (default) — Claude if `ANTHROPIC_API_KEY` is set, otherwise Ollama
- **`claude`** — Claude only (`CLAUDE_MODEL` in `config/settings.py`)
- **`ollama`** — local server, no API key, no cost

**Per-brief model routing** — override via `OLLAMA_MODEL_STOCK_BRIEF`, `OLLAMA_MODEL_OPTIONS_BRIEF`, `OLLAMA_MODEL_DAYTRADING_BRIEF`, `OLLAMA_MODEL_THESIS` in `.env`; falls back to `OLLAMA_MODEL` when unset. Route judgment-heavy briefs to a bigger reasoning model (e.g. `deepseek-r1:32b`), keep templated ones fast (e.g. `llama3.2`) — no code changes needed.

Reasoning models need a generous token budget to finish their hidden `<think>` pass. `_ask_ollama` floors `num_predict` at 1500 and retries once at double budget if a response comes back empty.

### ML direction model

The Predictions tab (`analysis/ml_prediction.py`) trains an **XGBoost + Random Forest ensemble** per ticker on 18 technical features from daily price history (`data/feature_engineering.py`).

- **Auto-selects** the label horizon (3/5/10 trading days) and hyperparameters per ticker, scored via anchored walk-forward validation — no one-size-fits-all config.
- **Self-gates on quality.** Accepted only if mean walk-forward accuracy is ≥ 52% with std-dev ≤ 8% across folds. Below that, training reports the shortfall instead of saving a model that hasn't earned trust.

### Intraday direction model (15-min)

`analysis/intraday_prediction.py` is a **separate** model for intraday bars — not the daily model with a different interval.

**Why separate:** `ml_prediction.predict()` feeds two pages (Trading Desk, Research). Threading an interval parameter through it would put both at risk. The intraday module imports the daily module's walk-forward runner and model configs **read-only**, and writes only `{TICKER}_{interval}_*` files — a daily `SPY_xgb.pkl` is never touched.

**Three correctness fixes, not plumbing:**

- **Volatility-scaled labels.** A fixed ±0.5% band (calibrated for daily bars) labels 66–93% of 15m bars neutral — dropped before training, leaving a biased sample drawn only from high-volatility windows. The band is now `k × σ × √horizon`, with σ a *trailing* estimate over five sessions (a full-series σ would make each label depend on future bars).
- **Session-boundary masking.** Forward returns that would span the overnight gap are dropped — the model is never trained to predict a gap it can't see.
- **Cost-aware reporting.** At a 75-minute horizon the average move is ~0.35% — spread and commission eat most of any edge. The UI reports net edge after costs and breakeven accuracy next to raw accuracy.

Also: drops `day_of_week` (near-useless in a 60-day window), adds time-of-day, VWAP distance in ATRs, position in the session range, and position in the opening range.

**Caveat:** intraday direction prediction is a harder problem than daily — order-flow shops attack it with data this app doesn't have. Expect 50–53% accuracy, and expect costs to eat most of it.

### Horizon Cockpit — all five models at once

**The app has five direction models. Until this, comparing them meant changing a dropdown four times and holding the results in your head.**

The cockpit (top of Trading Desk → Predictions, `analysis/interval_consensus.py`) puts 5m / 15m / 30m / 1h / daily in one table: direction, probability, confidence, **live** accuracy paired with its sample size, when the signal expires, and whether it clears its own costs.

**Why it matters:** the horizon you're most likely to act on is often the one least able to pay for itself. The cockpit names those explicitly — "agree directionally but do **not** clear costs."

- **Read-only by construction.** `predict()` and `predict_intraday()` persist unless told otherwise, so a view that generated predictions on render would inflate the log and corrupt the live win rate Model Lab, the retrain triggers, and the scoreboard all read. One explicit **Refresh all horizons** button instead.
- **Alignment, not a score.** Unanimous / mixed / none, plus the tightest *tradeable* agreeing horizon. Expired signals don't vote.
- **Ask about these readings** — an optional AI Q&A grounded strictly on that table. No chart, no news feed. Ungrounded, an LLM narrates a VWAP reclaim that never happened.

### Every signal expires

**A prediction's horizon isn't a suggestion — it's the edge of the evidence.**

`resolve_intraday_predictions()` grades a 15m/5-bar call against the close exactly 75 minutes later. The model has validated evidence about that window and none about the bar after it. `analysis/horizon_clock.py` now says so:

- **Expiry in ET on every signal**, with a countdown, and an explicit expired state.
- **Bars are labeled at their start**, so the exit bar's close lands one interval past `horizon_minutes` alone. Getting this wrong puts expiry a full bar early.
- **Horizons crossing the 4:00 PM close are never gradeable** — forward returns spanning the overnight gap were excluded from training, so `resolve_intraday_predictions()` skips them permanently. That's a dead state, not a pending one.
- **Stops scaled to the horizon.** The Day Trading card's `1.5 × daily ATR` is a swing stop; against 75 minutes it would never be touched, making its R:R fiction. The intraday exit plan uses the average move over that signal's own horizon instead.

### Options cost model — the correction that mattered most

**`assess_tradeability()` prices costs in *underlying* percentage points. Right for shares. Wrong three ways for contracts.**

| Problem | Effect on the verdict |
|---|---|
| Payoff is leveraged ~19x at 30 DTE, **~100x at 0DTE** (`\|delta\| × S / P`) | Understates gross edge by 1–2 orders of magnitude |
| A 1–2¢ spread on a $3 contract is 0.5–1%, not 0.02% | Overstates it |
| No theta term at all | Overstates it, badly, near expiry |

Leverage helps, spread and theta hurt — so raising a constant couldn't fix it. `assess_options_tradeability()` models all three and returns each term separately.

**Why that separation matters:** `dominant_cost` names what actually kills the edge. Spread-dominated is fixable with better fills. **Theta-dominated means the expiry is wrong for the horizon** — a different action entirely.

- **The verdict flips across the ladder.** Leverage and theta both scale inversely with time to expiry, so they partly cancel. `sweep_expiries()` runs 0/2/7/30 DTE; Model Lab renders the full horizon×expiry grid. Expect a diagonal — shorter signals need longer expiries to outrun decay.
- **0DTE theta is guarded.** Black-Scholes theta diverges as `T → 0`, so under 1 DTE the model switches to an empirical sqrt-extrinsic decay and labels which method it used.
- **Vega is inverted, not forecast.** `iv_points_to_erase_edge` says how much IV crush would wipe out the edge. Predicting the IV move needs its own calibration study (`docs/ROADMAP.md`, Item 11) and is deliberately not faked.
- Ships **alongside** the shares verdict, labeled by `cost_model`, so the difference is visible rather than swapped in silently.

### Prediction Improvement Engine

Both models are wrapped in a closed loop that tracks, explains, and maintains their own accuracy over time — surfaced on the **Model Lab** page (`pages/model_lab.py`):

- **Tracking + performance dashboard** (`analysis/prediction_performance.py`) — precision/recall/F1/false-positive-rate/false-negative-rate per direction, avg profit per signal, holding time, and confidence calibration (does "high confidence" actually mean higher accuracy?), computed from every prediction Trading Desk has already logged and graded.
- **Failure analysis** (`analysis/prediction_errors.py`) — categorizes every *incorrect* prediction against the technical/volatility/earnings context it was made in (counter-trend, choppy market, volume anomaly, RSI divergence, elevated VIX regime, earnings window), so a run of misses points at a *reason*, not just a number.
- **Model comparison** (`compare_models()`/`compare_intraday_models()`) — an informational, read-only walk-forward bake-off of XGBoost, Random Forest, Logistic Regression, and Gradient Boosting, with a recommended softmax weighting. Never changes the deployed model on its own.
- **Ensemble weighting** — the XGBoost/RF blend is a **learned** softmax over each model's walk-forward accuracy (`_softmax_ensemble_weights()`), persisted per ticker and read back at inference. Falls back to the historical fixed 65/35 split for any model trained before this shipped.
- **Hyperparameter search** — both XGBoost and Random Forest configs are auto-selected per ticker from a small grid, scored via a reduced-fold walk-forward; always falls back to library defaults if nothing in the grid beats them.
- **Model versioning** — every retrain archives the model it replaces under `storage/versions/{TICKER}/`, with a rollback button in Model Lab that copies an older version's files back into place (a file copy, never a retrain).
- **Retrain triggers** (`analysis/retrain_triggers.py`) — staleness, a live-accuracy drop vs. the trained-in accuracy, and elevated VIX are checked on both Trading Desk's status badge and Model Lab, plus a standalone `scripts/scheduled_retrain.py` CLI for cron-driven sweeps.
- **Horizon Scoreboard** (`compare_horizons()`) — all five models ranked in one table. Verdict is an ordered rule chain, never a score, and it separates **Do not trade** (no edge) from **Uneconomic** (real edge, costs eat it) because those have different fixes. The `Calibrated` column asks whether HIGH confidence has actually beaten LOW — if `no`, ignore the badge and use raw accuracy.
- **Accuracy over time** (`rolling_accuracy()`) — rolling 20-prediction hit rate with retrain dates marked. A single number can't tell "steady at 55%" from "was 62%, now 48%", and those call for different actions.
- **Agreement backtest** (`backtest_agreement()`) — did waiting for a second horizon to confirm actually improve the hit rate? Buckets into agree / conflict / unconfirmed. Carries a permanent caveat: predictions exist only where you pressed a button, so it's a correlation in your own history, not a controlled backtest.
- **Similar setups** — `predict()` already found every historical bar where the model made the same call, to take the median as `expected_move_pct`. Now reports the full distribution too. In-sample, and labeled as such.

Full technical writeup, including the exact formulas and storage layout: `docs/ML_PREDICTION.md`. Design record for the cockpit work, including ideas explicitly rejected: `docs/ROADMAP.md`.

## Setup Environment Using Anaconda

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create the environment:
   ```bash
   conda create --name aether python=3.12
   ```
3. Activate it:
   ```bash
   conda activate aether
   ```
4. Install dependencies:
   ```bash
   cd aether
   pip install -r requirements.txt
   ```
5. Deactivate when finished:
   ```bash
   conda deactivate
   ```

## How to Run the Dashboard

1. **(Optional)** Start Ollama for free, local AI briefs:
   ```bash
   ollama pull llama3.2
   ollama serve
   ```
2. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `AI_PROVIDER` — see [AI & ML Model Overview](#ai--ml-model-overview)
4. Run it:
   ```bash
   streamlit run app.py
   ```
5. The sidebar shows a green **🤖 AI: ...** badge once a provider is connected.

## How to Use

### Dashboard (Home)

Landing page. No input required — just open the app.

- Regime banner (Bull/Uptrend/Sideways/Downtrend/Bear vs. the S&P's 200-day MA)
- Live index cards (SPY/QQQ/IWM/VIX)
- **Market Regime (Markov)** — a probabilistic second opinion on the same trend, reusing the Markov model Trading Desk runs per-ticker, applied to the S&P 500
- Sector performance
- Open positions (empty until logged)
- **Recent Activity** — the last 8 logged events across Trading Desk and Strategy Lab, newest first

**Example:** start every session here for a 10-second market read — if the regime banner says "Downtrend" and VIX is elevated, that's a cue to size down before opening Trading Desk. Once you've logged a few Trading Desk/Strategy Lab actions, Recent Activity doubles as a "what was I doing yesterday" scroll-back.

### Research

Enter a ticker + lookback period — loads automatically, no button to click.

- **Chart & Technicals** — candlestick + SMA/Bollinger/support-resistance/trendlines
- **Fundamentals** — Quality/Value/Growth 0–100 scores + red flags
- **Options** — IV Rank, ATM IV, IV/RV
- **News & Sentiment** — VADER-scored headlines, display-only
- **AI Brief** — one-click investment summary
- **ML Direction Signal** — runs automatically between the scorecard and the chart

**Example:** type `NVDA` in the sidebar's Quick Lookup (or directly in Research's ticker box), pick a 1-year lookback, and skim top-to-bottom: fundamentals scorecard first (does the business hold up?), then the ML Direction Signal (does the model see a near-term edge?), then click **Generate Stock Brief** to have the AI tie both together in plain English before you move to Trading Desk to act on it.

### Options Log

The trade journal — the only page where you log trades. Enter each options fill as your broker reports it; `portfolio/round_trips.py` FIFO-matches buys against sells into round trips with P&L and hold time.

- **Pattern-finding analytics** — a cumulative P&L equity curve, win rate by hold-time bucket, entry hour, option type, and day of week, and a per-ticker performance breakdown (total/avg P&L, win rate).
- **Model vs. Discretionary** — tag a fill with the signal that motivated it, and this compares round trips you opened on a model call against ones you didn't. **Every other number in the app grades the model; this grades the decision.** Stays silent until 40 round trips with 15 per arm — at 20 trades "I lose on countertrend setups" is noise that reads as self-knowledge. Leaving a fill untagged is a real data point, not a missing one.
- **Equity positions have no logging UI** — options fills only. Formerly "Portfolio," with Positions / Risk Analytics / Position Sizer tabs; those tracked equity positions with no UI to ever add one, and the Position Sizer duplicated Trading Desk's own Quick Risk Calculator, so all three were cut.

**Example:** after your broker fills a `SPY 15Feb25 590C` buy and, three days later, the matching sell, log both fills under the **Fill Ledger** as they happen. Once both sides are in, the FIFO matcher turns them into one round trip on the **Round Trips** table and folds it into **Win Rate** (e.g. "60% win rate on holds under 1 day") and the cumulative P&L equity curve — the picture of *your own* trading, not the model's.

### Trading Desk

Four tabs:

- **Day Trading** — VWAP deviation, momentum, volume ratio, trend alignment (all interval-aware except Trend Alignment, which stays on daily SMA20/50/200 + EMA50 by design), candlestick pattern detection, and Flag/Pennant continuation-pattern detection (`analysis/flag_pennant_detection.py` + `flag_pennant_scoring.py`) drawn directly on the chart with a 0–100 confidence score. Signals combine into a Suggested Entry/Stop/Target card via majority vote, plus an AI Day Trading Brief and a MACD-cross backtest.
- **Options** — IV Rank/Percentile, a GARCH(1,1) forward volatility forecast vs. ATM IV, the full chain, P&L diagrams, Black-Scholes Greeks, and an AI Options Brief.
- **News** — headline sentiment for the entered ticker, same VADER scoring as Research.
- **Predictions** — opens with the **Horizon Cockpit** (all five models side by side; see [above](#horizon-cockpit--all-five-models-at-once)). Below it, a **Prediction horizon** toggle switches between **Daily (swing)** — the original model, unchanged — and **Intraday (15-min bars)**, a separate model with its own features, labels, and storage. Both add an **exit plan**: when the signal expires and a stop scaled to that horizon.

**Examples:**
- **Start here.** Open Predictions and read the cockpit before anything else. If it says *"Agree directionally but do not clear costs: 5m, 15m"*, the setup that looks strongest is the one that can't pay for itself. Check **Price with live options quotes** to cost it as contracts rather than shares.
- **Day Trading** — enter `AAPL`, check the Suggested Entry/Stop/Target card; if it agrees with a Flag/Pennant pattern drawn on the chart at confidence ≥ 70, that's a stronger case than either alone. Sub-tabs hold Market Regime, Risk Calculator, Oscillators, the MACD backtest, and **Signal Weights** (fitted from your own logged Analyze clicks — blank until 30 have resolved, by design).
- **Options** — check IV Rank: above ~70 with GARCH forecasting lower forward vol than current ATM IV is the setup for *selling* premium, not buying it.
- **Predictions** — first time on a ticker, **Train / Update Model** (~10–20s), then **Generate Prediction**. A HIGH-confidence BULLISH call with a positive walk-forward delta is worth weighing; LOW or NEUTRAL means don't trade it today. Then read the exit plan — **flat by the expiry time**, because past it the model has no validated edge.

Day-by-day, week-by-week rhythm: `docs/workflow.md`.

### Strategy Lab

One live strategy — **ORBC**, with a Live Scanner and Backtest sub-tab — plus a read-only Intraday Predictions reference panel.

**ORBC (Opening Range)** — the first N minutes after the 9:30 ET open set a reference high/low. Waits for a **second consecutive close** outside that range before signalling — filters most post-open false breakouts. Logic: `analysis/orbc_strategy.py`.

- **Confirmation rule** — fires on the Nth consecutive close outside the range (default 2). A close back inside resets the count. If a filter blocks the Nth close, later closes can still fire up to `max_confirmation_closes` (default 3) — exactly one signal per breakout episode.
- **Configurable** — bar interval, opening-range duration, confirmation count, entry cutoff, three filters (volume vs. 20-bar avg, VWAP alignment, range vs. ATR), long/short enablement, stop method (range/ATR/percent), target method (R:R/ATR/range projection).
- **Scanner shows** — the opening-range band, every close outside it, filtered-out breakouts marked ✕ (hover for the reason), entry/stop/target for a confirmed signal, and a 0–100 confidence score built from volume thrust, VWAP alignment, breach decisiveness, range quality vs. ATR, and short-term EMA agreement. One click logs a confirmed signal.
- **Both directions supported** — `evaluate_orbc_trade()` is direction-aware, unlike this app's other long-only simulators. Positions always flatten at session close.
- **Small sample by design** — intraday bars cap at ~60 days and ORBC fires at most once per session, so a backtest yields a few dozen trades. Warns explicitly below 30.

**Example:** shortly after 9:30 ET, open the **Live Scanner** on `SPY` and watch the opening-range band form; when a close breaks it and a second consecutive close confirms in the same direction with no ✕ filter marks, check the confidence score — above ~60 with volume/VWAP agreement is the strongest version of this setup — then click **Log this ORBC signal**. Before trading it live, run the **Backtest** sub-tab on the same ticker/config to see the historical win rate (with the small-sample caveat in mind).

Full rule set + daily routine: `docs/ORBC_PLAYBOOK.md`.

**Intraday Predictions** — a read-only table of the latest saved Intraday Prediction per interval (5m/15m/30m/1h): direction, confidence, probability, model accuracy, and when it was generated. Training/refreshing those models stays on Trading Desk; this tab only reads what's already saved there. (This replaced an MTF strategy tab that wasn't seeing use — its logic still exists in `analysis/mtf_strategy.py` but isn't currently wired into any page.)

### Model Lab

Read-only track record for both ML models — the **Prediction Improvement Engine**'s dashboard. It never trains, predicts, or writes a prediction. Four opt-in exceptions, all read-only: Model Comparison fits throwaway models purely to score them, Version History's rollback copies archived files, Failure Analysis's "recompute legacy" checkbox re-fetches price history, and the Scoreboard's "live options quotes" checkbox fetches an expiry ladder.

**Start at the top.** The Horizon Scoreboard answers *which* horizon to trade; the tabs below answer *how one is doing*.

- **Horizon Scoreboard** — five rows: walk-forward accuracy, std, live accuracy with N, best expiry, net edge, dominant cost, IV crush to erase, calibrated?, verdict. Plus a **horizon × expiry grid** of net edge. Verdict is an ordered rule chain, not a score — and **Uneconomic** (real edge, costs eat it) is a different diagnosis from **Do not trade** (no edge).
- **Agreement backtest** — did a second horizon confirming actually raise the hit rate?
- **Performance dashboard** — accuracy, win rate, precision/recall/F1/FPR/FNR per direction with a confusion matrix, avg profit per signal, avg holding time, and confidence calibration.
- **Accuracy over time** — rolling 20-prediction hit rate, retrain dates marked.
- **Failure analysis ("Why the model was wrong")** — each miss tagged against its technical/volatility/earnings context (counter-trend, choppy, volume anomaly, RSI divergence, elevated VIX, earnings window).
- **Model comparison** — XGBoost vs RF vs LogReg vs GradientBoosting on the same walk-forward. Informational; never changes the deployed model.
- **Version history** — roll back any retrain in one click (archives the current model first, so nothing is discarded).
- **Retrain triggers** — staleness, live-accuracy drop, elevated VIX. Same three checks Trading Desk's badge uses.

**Example:** after a few weeks of live `SPY` predictions, open Model Lab. If the Scoreboard says 15m is **Uneconomic** while 30m is **Primary**, the 15m model isn't broken — its costs are, and the grid will show which expiry fixes it. If a horizon reads **Degraded, retrain**, check **Failure Analysis** first: misses clustering under `elevated_vol_regime` mean the market shifted, not the model, and retraining on fresher data is the fix. If that retrain makes things worse, **Version History** rolls it back.

## Architecture & Workflow

No central orchestrator. `app.py` sets page config, theme, and the sidebar, then `st.navigation()` routes between six independent pages (Dashboard, Research, Options Log, Trading Desk, Strategy Lab, Model Lab). Each page:

1. **Fetches** — price/fundamentals/options/news via `data/*.py`, cached per `config/settings.py` TTLs
2. **Computes** — indicators, scores, or the ML ensemble via `analysis/*.py`
3. **Renders** — Plotly charts and Streamlit widgets, inline
4. **Briefs (optional)** — a button click on Research or the Trading Desk routes the already-computed data through `ai/client.py`

**Persists across pages and reruns:**

- **`storage/journal.db`** (SQLite, via `portfolio/db.py`) — positions, activity log, options fills
- **`storage/{TICKER}_*`** — trained models, walk-forward accuracy, prediction history — one set per trained ticker
- **`storage/versions/{TICKER}/`** — archived prior model versions + an append-only rollback log; **`storage/retrain_log.jsonl`** and **`storage/alerts.jsonl`** — the two cron sweeps' logs

**One writer, one judge.** Trading Desk is the only page that logs predictions; Model Lab only reads and grades them. `predict()`/`predict_intraday()` persist unless passed `persist=False`, so any automated caller would otherwise inflate the live win rate that Model Lab, the retrain triggers, and the Horizon Scoreboard all read. `scripts/alert_sweep.py` is the one automated caller and passes it.

No request/response API layer — Streamlit's script-rerun model *is* the request cycle. `st.session_state` carries state (e.g. the quick-lookup ticker) across page switches.

## Project Structure

```
aether/
├── app.py                   # Entrypoint — page config, theme, sidebar, st.navigation()
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template (copy to .env)
├── config/
│   ├── settings.py          # API keys, AI provider + per-brief model routing, cache TTLs, risk defaults
│   └── tz.py                # US/Eastern time helpers — all user-facing timestamps are explicit ET
├── pages/
│   ├── home.py               # Dashboard — market overview, regime Markov, sector performance, positions, recent activity
│   ├── research.py           # Research page
│   ├── portfolio.py          # Options Log — the trade journal + pattern-finding analytics
│   ├── strategy_lab.py       # ORBC scanner+backtest + read-only Intraday Predictions reference panel
│   ├── trading.py            # Trading Desk — Day Trading / Options / News / Predictions tabs
│   └── model_lab.py          # Model Lab — read-only prediction performance dashboard (Prediction Improvement Engine)
├── analysis/
│   ├── indicators.py          # RSI, MACD, ADX, Bollinger Bands, etc.
│   ├── patterns.py            # Candlestick pattern detectors (Doji, Engulfing, Inside Bar, NR4)
│   ├── trendlines.py          # Fitted support/resistance trendlines + ATR-confirmed swing points
│   ├── flag_pennant_detection.py  # Flag/Pennant geometry: swing → pole → consolidation → confirmed breakout
│   ├── flag_pennant_scoring.py    # 0-100 confidence score for a detected pattern
│   ├── flag_pennant_backtest.py   # Entry/stop/target/R:R/MFE/MAE for a scored pattern
│   ├── mtf_strategy.py        # Strategy Lab's 4H/30m/5m setup: 4H resample, evaluate_setup, backtest_setup
│   ├── orbc_strategy.py       # Opening Range Breakout Confirmation: opening range, confirmation state machine, backtest
│   ├── volume_profile.py      # Volume-by-price profile — POC/value-area proxy used by mtf_strategy.py
│   ├── backtest.py            # Generic long-only backtest engine + MACD bullish-cross signal
│   ├── ml_prediction.py       # XGBoost + RF ensemble (daily): train, predict, evaluate, compare_models, versioning
│   ├── intraday_prediction.py # Separate intraday (15m) direction model — own features/labels/storage
│   ├── interval_consensus.py  # Horizon Cockpit: cross-horizon read + agreement backtest (read-only)
│   ├── horizon_clock.py       # When a signal expires; flags horizons that can never be graded
│   ├── signal_attribution.py  # Signal weights fitted from logged history (refuses under 30 events)
│   ├── trade_attribution.py   # Model-driven vs discretionary round trips (gated on sample size)
│   ├── prediction_performance.py  # Precision/recall/F1/calibration, compare_horizons, rolling_accuracy
│   ├── prediction_errors.py       # Categorizes incorrect predictions against their technical/vol/earnings context
│   ├── retrain_triggers.py        # Staleness, live-accuracy-drop, and elevated-VIX retrain checks
│   ├── price_projection.py    # Monte Carlo price-path simulation
│   ├── options_pricing.py     # Black-Scholes pricing, Greeks, IV solver, options cost model + expiry sweep
│   ├── volatility_forecast.py # GARCH(1,1) forward volatility forecast
│   ├── sentiment.py           # VADER headline sentiment scoring
│   ├── fundamental_score.py   # Quality/Value/Growth scoring engine
│   ├── regime.py              # Market regime detection (trend vs. 200-day MA)
│   ├── regime_markov.py       # Markov-chain regime model — persistence, forecast, stationary distribution
│   └── risk.py                # Position sizing, horizon-scaled stops, contract-level loss
├── data/
│   ├── price_data.py          # Price history and current price via yfinance
│   ├── fundamentals.py        # Balance sheet, income statement, FCF via yfinance
│   ├── options_data.py        # Options chain, IV Rank, Black-Scholes ATM Greeks
│   ├── macro_data.py          # Market overview, VIX, sector performance
│   ├── news_data.py           # Ticker headlines via Google News RSS
│   └── feature_engineering.py # Feature matrix and target labels for ML training
├── ai/
│   ├── client.py               # AI router — Claude or Ollama, per-brief model override
│   └── stock_brief.py          # Prompt builders for the four brief types
├── portfolio/
│   ├── db.py                   # SQLite schema/connection for storage/journal.db
│   ├── journal.py              # Position read access
│   ├── activity_log.py         # Records Day Trading / Options / Prediction view events
│   ├── option_fills.py         # Options fill ledger CRUD
│   └── round_trips.py          # FIFO buy/sell matcher → round trips with P&L, hold time
├── scripts/                 # Standalone CLIs — cron-driven, never imported by the app
│   ├── scheduled_retrain.py    # Retrain sweep
│   └── alert_sweep.py          # Alert conditions -> storage/alerts.jsonl (uses persist=False)
├── docs/
│   ├── workflow.md                    # Day-by-day and week-by-week usage workflow
│   ├── ORBC_PLAYBOOK.md               # ORBC rules, design decisions, and daily trading routine
│   ├── ML_PREDICTION.md               # Full technical writeup of the ML ensemble + Prediction Improvement Engine
│   ├── ROADMAP.md                     # Design record for the Horizon Cockpit work (+ ideas rejected, and why)
│   ├── Identifying-Chart-Patterns.md  # Flag/Pennant pattern reference
│   └── VERIFICATION_CHECKLIST.md      # Manual verification steps for a few past fixes
├── tests/
│   ├── conftest.py                    # Shared fixtures — synthetic daily + intraday OHLCV, isolated storage dir
│   ├── test_ml_prediction.py          # Regression suite for analysis/ml_prediction.py (see Reliability & Verification)
│   ├── test_orbc_strategy.py          # ORBC confirmation state machine, filters, stops/targets, direction-aware P&L
│   ├── test_intraday_prediction.py    # Intraday label masking, vol-scaled bands, storage isolation from daily
│   ├── test_prediction_performance.py # Precision/recall/F1/calibration metric math
│   ├── test_prediction_errors.py      # Failure-categorization rule set
│   ├── test_model_comparison.py       # 4-model walk-forward bake-off + learned ensemble-weight backward-compat
│   ├── test_model_versioning.py       # Archive/rollback file management, version history
│   ├── test_retrain_triggers.py       # Staleness/performance-drop/regime-change trigger logic
│   └── test_scheduled_retrain.py      # scripts/scheduled_retrain.py's discovery, dry-run, and failure isolation
└── storage/                    # Persisted ML models and prediction logs (auto-created)
    ├── {TICKER}_xgb.pkl
    ├── {TICKER}_rf.pkl
    ├── {TICKER}_accuracy.json
    ├── {TICKER}_predictions.jsonl
    ├── retrain_log.jsonl
    ├── alerts.jsonl
    └── versions/{TICKER}/        # Archived prior model versions + rollback history
```

## Extensibility & Customization

- **Add a technical indicator** — implement it in `analysis/indicators.py`'s `calculate_indicators()`, then reference the column wherever it should render.
- **Add an ML feature** — extend `data/feature_engineering.py`'s feature matrix. The daily ensemble picks up new columns on the next training run.
- **Add or retune an AI brief** — add a `generate_*` prompt builder in `ai/stock_brief.py`, and give it its own `OLLAMA_MODEL_*` override in `config/settings.py` if it deserves a different model.
- **Tune scoring or risk thresholds** — `config/settings.py` centralizes fundamental scoring cutoffs (ROIC, FCF yield, margin expansion), options thresholds (IVR high/low, IV/RV premium), and risk defaults (per-trade risk %, max position size).

## Future Ideas (Not Yet Built)

Things worth investigating later — noted here so they don't get re-litigated from scratch, not committed to:

- **Extended-hours (pre-market / after-hours) data.** Every fetch in `data/price_data.py` calls yfinance's `.history()` with no `prepost` argument, which defaults to regular-session-only (9:30–4:00 ET) bars — pre-market and after-hours prints are silently excluded everywhere (Day Trading, Predictions, ORBC). Passing `prepost=True` on intraday intervals would add them, and could help with two specific things: seeing an earnings-reaction move the moment it prints after-hours, and a pre-market read on Day Trading before 9:30. It's not a clean win, though — extended-hours bars are low-volume/wide-spread and would skew RSI/ADX/VWAP if fed into existing signals unchanged, and both ORBC's and the intraday ML model's session-boundary logic explicitly assume RTH-only bars (`config/tz.py`, `analysis/orbc_strategy.py`, `build_intraday_labels()`) — turning it on naively would break both. A safer scoped version: an opt-in "show pre/post market" toggle on the Day Trading chart for visual context only, never wired into VWAP/momentum/ORBC/the ML model. Needs its own investigation before building, not a small tweak. "24-hour" data isn't a real concept for stocks regardless (that's crypto-only); this is specifically about the pre-market/after-hours windows.

## Data Sources & Caching

| Source | What It Provides | API Key Required |
|--------|-------------------|-------------------|
| **yfinance** | Price history, fundamentals, options chains, earnings history | No |
| **Google News RSS** | Recent headlines for News & Sentiment scoring | No |
| **Ollama** | Local AI briefs — runs on your machine | No |
| **Claude API (Anthropic)** | Higher-quality AI briefs | Yes — `ANTHROPIC_API_KEY` |

Cache TTLs (`config/settings.py`): price data 5 min, fundamentals 1 hour, options chain 10 min, news headlines 15 min.

## Reliability & Verification

Reliability rests on three mechanisms:

**1. A `pytest` regression suite** (`tests/`, 436 tests, run with `pytest tests/ -q` — no network access needed):

- `test_ml_prediction.py` — the daily model end-to-end: an import-crash guard (the exact failure that silently killed the Predictions tab for 11 days), train/predict/evaluate on synthetic data, the reliability gate correctly rejecting a pure random walk, hyperparameter search, and the predict → save → history persistence round-trip.
- `test_orbc_strategy.py` — the ORBC confirmation state machine against hand-built sessions: a single breakout close never signals, a close back inside resets the count, filters fall through from the 2nd to the 3rd close, and short P&L carries the correct sign.
- `test_intraday_prediction.py` — storage isolation from the daily model, session-boundary label masking, trailing-sigma leak resistance, and naive/UTC index handling.
- `test_prediction_performance.py` / `test_prediction_errors.py` — the Model Lab metric math (precision/recall/F1/calibration) and the failure-categorization rule set, from the Prediction Improvement Engine.
- `test_model_comparison.py` — the 4-model walk-forward bake-off, and a bit-for-bit reproduction guard proving the learned ensemble weight never changes the original 65/35 XGB/RF blend's math.
- `test_model_versioning.py` / `test_retrain_triggers.py` / `test_scheduled_retrain.py` — model version archive/rollback, the three retrain triggers, and the standalone cron sweep script.
- `test_horizon_clock.py` / `test_options_tradeability.py` / `test_interval_consensus.py` / `test_horizon_scoreboard.py` / `test_horizon_stops.py` — the cockpit work: expiry math (including start-labeled bars and horizons that can never be graded), the options cost model's three terms plus the 0DTE theta guard and expiry sweep, cockpit alignment and its read-only guarantee, the scoreboard's verdict rule chain, and horizon-scaled stops.
- `test_agreement_backtest.py` / `test_accuracy_trend.py` / `test_signal_attribution.py` / `test_trade_attribution.py` — agreement buckets, rolling accuracy, and the two **refusal** paths: fitted signal weights return nothing under 30 events, and model-vs-discretionary stays silent under 40 round trips. Those refusals are tested as features.

Every clock test injects `now` explicitly — a test that read the wall clock would pass or fail depending on the hour it ran.

**Known gap:** anything touching a **live options chain** is unexercised. `get_expiry_ladder_quotes` has never seen a real chain — the math behind it is unit-tested with injected Greeks, but ladder-snapping and empty-book fallbacks want one session of real data. Check the scoreboard grid during market hours before trusting it.

**2. The ML model self-gates on quality.** Walk-forward validation must clear 52% mean directional accuracy with std-dev ≤ 8% across folds — a model that doesn't clear the bar is reported as such instead of silently saved.

**3. Manual verification checklists.** `docs/VERIFICATION_CHECKLIST.md` — timezone handling, activity logging, and the options FIFO round-trip matcher, checked against real fill data.

Next-highest-value coverage: `analysis/backtest.py`'s pure functions and `portfolio/round_trips.py`'s FIFO matcher — neither has a Streamlit dependency.

## Disclaimer & License

Personal research tool, for educational and informational purposes only. Not a registered investment advisor, broker-dealer, or financial planning service. Nothing here is financial advice, a recommendation to buy or sell any security, or a guarantee of future performance.

- **ML models** have a modest historical edge — typically 52–58% directional accuracy out-of-sample. They can't predict news, earnings surprises, or macro regime shifts. Past walk-forward accuracy doesn't guarantee future results.
- **All trading involves risk.** You may lose some or all of your capital. Always do your own due diligence.
- **No `LICENSE` file yet.** Treat the code as all-rights-reserved until one's added — include an MIT (or similar) license before making the repo public if you intend to allow reuse.

## Appendix: Helpful Commands

```bash
# Run the dashboard locally
streamlit run app.py

# Run Ollama for local, free AI briefs (keep the terminal open)
ollama serve
ollama pull llama3.2

# Install dependencies
pip install -r requirements.txt

# Run the regression test suite (436 tests, ~15 min — run subsets while iterating)
pytest tests/ -q
pytest tests/test_horizon_clock.py tests/test_options_tradeability.py -q   # fast, ~2s

# Cron-driven sweeps (standalone CLIs, never imported by the app)
python3 scripts/scheduled_retrain.py --dry-run
python3 scripts/alert_sweep.py --dry-run
python3 scripts/alert_sweep.py --tickers SPY --include-fresh   # uses persist=False

# Inspect the local database directly
sqlite3 storage/journal.db "select * from activity_log"
sqlite3 storage/journal.db "select filled_at, ticker, prediction_ref from option_fills"

# Alerts written by the sweep
tail -20 storage/alerts.jsonl
```
