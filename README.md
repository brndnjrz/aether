# Aether

> A modular Streamlit dashboard for independent traders — combining live technical analysis, fundamental scoring, an ensemble ML direction model, options analytics, and AI-generated briefs in one multi-page app. All data is live from yfinance. No mocks.

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
- [Data Sources & Caching](#data-sources--caching)
- [Reliability & Verification](#reliability--verification)
- [Disclaimer & License](#disclaimer--license)
- [Appendix: Helpful Commands](#appendix-helpful-commands)

## What It Is

Aether is a personal trading research tool built for self-directed traders who want institutional-grade analysis without paying for a Bloomberg terminal. It pulls live market data, runs a machine-learning direction model, scores company fundamentals, detects intraday chart patterns, and surfaces options strategies based on current implied volatility — then optionally has an LLM (Claude or a local Ollama model) synthesize the numbers into a plain-English brief.

It is not a brokerage, it does not execute trades, and it is not financial advice.

## Key Features

| Page | What It Does |
|------|--------------|
| **Dashboard** (`pages/home.py`) | Live market overview — index prices, VIX, S&P regime banner, sector performance, open positions summary |
| **Research** (`pages/research.py`) | Full single-stock deep dive: fundamental scorecard, technical chart, ML direction signal, options IV, news sentiment, and an AI investment brief |
| **Portfolio** (`pages/portfolio.py`) | Position tracking, correlation matrix, portfolio-level risk analytics, a risk-first position sizer, and an Options Log (manual fill entry → automatic FIFO round-trip P&L, hold-time/entry-hour win-rate analytics) |
| **Watchlist** (`pages/watchlist.py`) | Persistent weekly shortlist with a plan/thesis note per ticker, plus a Daily Watchlist Check (gap %, relative volume, ATR%, ML bias, today's session price envelope) scoped to just those tickers |
| **Screener** (`pages/screener.py`) | Runs empirically-backed screens (Quality + Momentum, Oversold Quality, High IV Rank, Small Account Options, Custom Watchlist) across a ~50-name large-cap sample |
| **Trading Desk** (`pages/trading.py`) | Four tabs in one page — **Day Trading** (market-status banner, intraday signals, candlestick pattern read, Flag/Pennant continuation-pattern detection with confidence scoring, suggested entry/stop/target, AI brief, MACD backtest), **Options** (chain, IV Rank, GARCH forward-vol forecast, Greeks, P&L diagrams, AI brief), **News** (headline sentiment), **Predictions** (ML direction signal + simulated price path) |
| **Strategy Lab** (`pages/strategy_lab.py`) | Two intraday strategies, each with a live scanner and a mechanical backtest — **MTF** (4H trend → 30m pullback into a demand zone → 5m structure shift → tape confirmation, in `analysis/mtf_strategy.py`) and **ORBC** (Opening Range Breakout Confirmation: requires a 2nd consecutive close outside the opening range before signalling, in `analysis/orbc_strategy.py`) |

Every Analyze click, options view, and prediction on the Trading Desk is written to a local activity log, later surfaced by the Portfolio Options Log's "what were you looking at" picker.

## Quick Start

**Prerequisites:** Python 3.12, conda (or any virtualenv manager)

```bash
conda create -n aether python=3.12 -y
conda activate aether
cd aether
pip install -r requirements.txt
cp .env.example .env
# Edit .env — see AI & ML Model Overview below for AI setup
streamlit run app.py
```

The app runs at **http://localhost:8501**

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

`fpdf2` and `Pillow` are also in `requirements.txt` but are not imported by any page — leftover from earlier iteration and safe to drop if trimming the dependency footprint. `narwhals` is listed explicitly even though nothing imports it directly: it's a real transitive dependency of `scikit-learn` and `plotly`, and a prior version mismatch in that chain took down the whole ML Predictions tab for 11 days before anyone noticed (see `tests/test_ml_prediction.py`) — it stays pinned on purpose.

## AI & ML Model Overview

### AI briefs (Claude or Ollama)

Four brief types — stock, options, day-trading, and thesis-question — are built in `ai/stock_brief.py` and routed through `ai/client.py` to whichever provider `AI_PROVIDER` selects:

- **`auto`** (default) — Claude if `ANTHROPIC_API_KEY` is set, otherwise falls back to Ollama
- **`claude`** — Claude only (`CLAUDE_MODEL` in `config/settings.py`)
- **`ollama`** — local Ollama server, no API key, no cost

Each brief type can be routed to a different Ollama model via `OLLAMA_MODEL_STOCK_BRIEF`, `OLLAMA_MODEL_OPTIONS_BRIEF`, `OLLAMA_MODEL_DAYTRADING_BRIEF`, and `OLLAMA_MODEL_THESIS` in `.env` — all fall back to `OLLAMA_MODEL` when unset. This lets you point judgment-heavy briefs at a larger reasoning model (e.g. `deepseek-r1:32b`) while keeping cheap/templated ones on something fast (e.g. `llama3.2`), without touching code. Reasoning models need a generous token budget to finish their hidden `<think>` pass before answering — `_ask_ollama` floors `num_predict` at 1500 and retries once at double budget if a response comes back empty.

### ML direction model

The Predictions tab (`analysis/ml_prediction.py`) trains an **XGBoost + Random Forest ensemble** per ticker on 18 technical features derived from daily price history (`data/feature_engineering.py`). Training auto-selects, per ticker, whichever label horizon (3, 5, or 10 trading days) and XGBoost hyperparameters the ensemble is most consistently accurate with, scored via anchored walk-forward validation — it doesn't assume one fixed configuration fits every ticker.

A model is only accepted if its mean walk-forward directional accuracy is ≥ 52% with a std-dev across folds ≤ 8%; otherwise training reports the shortfall instead of saving a model that hasn't earned trust.

### Intraday direction model (15-min)

`analysis/intraday_prediction.py` is a **separate** model for intraday bars, not the daily model with a different interval. It is separate on purpose: `ml_prediction.predict()` is consumed by four pages (Trading Desk, Research, Watchlist, Screener), so an interval parameter threaded through it would put all four at risk. The intraday module instead imports the daily module's walk-forward runner and model configs read-only, and writes only `{TICKER}_{interval}_*` files — a daily `SPY_xgb.pkl` and its prediction history are never opened for writing.

Three things differ from the daily model, all correctness rather than plumbing:

- **Volatility-scaled labels.** A fixed ±0.5% neutral band calibrated for daily bars labels 66–93% of 15m bars neutral, and neutral rows are dropped before training — leaving a small sample drawn only from high-volatility windows. The band is instead `k × σ × √horizon`, where σ is a *trailing* estimate over five sessions (a full-series σ would make each label depend on future bars).
- **Session-boundary masking.** `pct_change(n).shift(-n)` on an intraday index silently spans the overnight gap for the last n bars of every session. Those rows are dropped rather than training the model to predict a gap it cannot see. σ likewise excludes the gap return, which would otherwise inflate the band.
- **Cost-aware reporting.** At a 75-minute horizon the average move is ~0.35%, so spread plus commission consumes much of any edge. The UI reports net edge after costs and the breakeven accuracy alongside raw accuracy — a 53%-accurate intraday model can be statistically real and still lose money.

Intraday features drop `day_of_week` (near-useless inside a 60-day window) and add time of day, distance from session VWAP in ATRs, position in the session range, and position relative to the opening range.

Caveat worth stating plainly: intraday direction prediction is a much harder problem than daily, attacked elsewhere with order-flow data this app does not have. Expect accuracy nearer 50–53% and expect costs to consume most of it.

## Setup Environment Using Anaconda

1. Download and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create an environment:
   ```bash
   conda create --name aether python=3.12
   ```
3. Activate the environment:
   ```bash
   conda activate aether
   ```
4. Navigate to the project and install dependencies:
   ```bash
   cd aether
   pip install -r requirements.txt
   ```
5. Deactivate when finished:
   ```bash
   conda deactivate
   ```

## How to Run the Dashboard

1. (Optional) Start Ollama if you want free, local AI briefs instead of Claude:
   ```bash
   ollama pull llama3.2
   ollama serve
   ```
2. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `AI_PROVIDER` (see [AI & ML Model Overview](#ai--ml-model-overview))
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. The sidebar shows a green **🤖 AI: ...** badge once a provider is connected

## How to Use

### Dashboard (Home)

The landing page. No inputs required — shows the regime banner (Bull/Uptrend/Sideways/Downtrend/Bear Market vs. the S&P's 200-day MA), live index cards (SPY/QQQ/IWM/VIX), sector performance, and open positions (empty until the local database has any).

### Research

Enter a ticker and lookback period; the page loads automatically. Tabs: **Chart & Technicals** (candlestick + SMA/Bollinger/support-resistance/trendlines), **Fundamentals** (Quality/Value/Growth 0–100 scores with red flags), **Options** (IV Rank, ATM IV, IV/RV), **News & Sentiment** (VADER-scored headlines, display-only), **AI Brief** (one-click investment summary). The **ML Direction Signal** runs automatically between the scorecard and the chart.

### Portfolio

**Positions**, **Risk Analytics**, **Position Sizer**, and **Options Log** tabs. The Options Log is the one place in the app you actually log trades: enter each fill as your broker reports it, and `portfolio/round_trips.py` FIFO-matches buys against sells to produce round trips with P&L, hold time, and win-rate analytics by hold-time bucket and entry hour. Equity positions have no logging UI yet — only options fills are loggable today.

### Watchlist

A persistent weekly shortlist (ticker + target/stop + plan note) plus a **Daily Watchlist Check** that scans only your saved tickers for gap %, relative volume, ATR%, and ML bias — fast, since it's scoped to a handful of names instead of the full screener universe.

### Screener

Pick a screen (Quality + Momentum, Oversold Quality, High IV Rank, Small Account Options, or a pasted Custom Watchlist) and run it against a ~50-stock sample. Each run fetches live data per ticker, so expect 30–60 seconds.

### Trading Desk

Four tabs in one page:

- **Day Trading** — VWAP deviation, momentum, volume ratio, trend alignment (all interval-aware except Trend Alignment, which stays on daily SMA20/50/200 + EMA50 by design), candlestick pattern detection, and Flag/Pennant continuation-pattern detection (`analysis/flag_pennant_detection.py` + `flag_pennant_scoring.py`) drawn directly on the chart with a 0–100 confidence score. Signals combine into a Suggested Entry/Stop/Target card via majority vote, plus an AI Day Trading Brief and a MACD-cross backtest.
- **Options** — IV Rank/Percentile, a GARCH(1,1) forward volatility forecast compared against ATM IV, the full chain, P&L diagrams, Black-Scholes Greeks, and an AI Options Brief.
- **News** — headline sentiment for the entered ticker, same VADER scoring as Research.
- **Predictions** — train/retrain the ML ensemble and generate a direction signal with a simulated price path. A **Prediction horizon** toggle switches between **Daily (swing)** — the original model, unchanged — and **Intraday (15-min bars)**, a separate model in `analysis/intraday_prediction.py` with its own features, labels, and storage files. See [AI & ML Model Overview](#ai--ml-model-overview) and `docs/ML_PREDICTION.md` for the full technical writeup.

For the day-by-day and week-by-week rhythm this app is designed around, see `docs/workflow.md`.

### Strategy Lab

Two independent intraday strategies, each with its own Live Scanner and Backtest.

**MTF Scanner / MTF Backtest** — a single codified setup: 4H trend, pullback into a demand zone on the 30-minute chart, a market-structure shift on the 5-minute chart, then tape confirmation (price/volume proxy, not real order flow), with a target at the volume-profile point of control and a stop at the swing low. Logic in `analysis/mtf_strategy.py`.

**ORBC (Opening Range)** — Opening Range Breakout Confirmation. The first N minutes after the 9:30 ET open define a reference high/low; the strategy then waits for a **second consecutive close** outside that range before signalling, which filters most of the false breakouts that fire right after the open. Logic in `analysis/orbc_strategy.py`.

How the confirmation rule resolves: the signal fires on the Nth consecutive close outside the range (default 2). A close back inside resets the count — a breakout that round-trips starts over. If a filter (volume/VWAP/ATR) blocks the Nth close, later closes can still fire up to `max_confirmation_closes` (default 3). That's the "second OR third candle" rule: exactly one signal per breakout episode, but a filter miss on the second bar doesn't kill a setup that confirms on the third.

Configurable in the **Strategy parameters** panel: bar interval, opening-range duration, confirmation count, entry cutoff time, each of the three filters (volume vs. the 20-bar average, VWAP alignment, opening range vs. ATR), long/short enablement, stop method (opening range / ATR / percent), and target method (risk-reward / ATR / range projection).

The scanner overlays the opening-range band, marks every close outside it, marks filtered-out breakouts with an ✕ (hover for the reason), and draws entry/stop/target for a confirmed signal alongside a 0–100 confidence score built from volume thrust, VWAP alignment, breach decisiveness, range quality vs. ATR, and short-term EMA agreement. A confirmed signal can be written to the activity log with one click, the same way MTF setups are.

Both ORBC directions are supported — `evaluate_orbc_trade()` is direction-aware, unlike the long-only simulators elsewhere in the app. Positions are always flattened at the session close; the strategy never holds overnight. Because intraday bars are capped at ~60 days and ORBC fires at most once per session, the backtest yields a few dozen trades and warns explicitly when the sample is under 30.

See `docs/ORBC_PLAYBOOK.md` for the full rule set, the implementation decisions behind it, and a pre-open-to-weekly-review routine for trading it.

## Architecture & Workflow

Aether has no central orchestrator — `app.py` sets page config, theme, and the sidebar, then hands off to Streamlit's `st.navigation()` to route between seven independent pages. Each page owns its own data flow:

1. **Fetch** — pull price/fundamentals/options/news via `data/*.py` (all backed by yfinance or Google News RSS, cached per `config/settings.py` TTLs)
2. **Compute** — run indicators, scores, or the ML ensemble via `analysis/*.py`
3. **Render** — Plotly charts and Streamlit widgets, inline on the page
4. **AI brief (optional)** — on Research and the Trading Desk, a button click builds a prompt from the already-computed data and routes it through `ai/client.py` to Claude or Ollama

Two things persist across pages and reruns:

- **`storage/journal.db`** (SQLite, via `portfolio/db.py`) — equity positions, the activity log, options fills, and the watchlist
- **`storage/{TICKER}_*`** — trained ML model files (`_xgb.pkl`, `_rf.pkl`), walk-forward accuracy (`_accuracy.json`), and prediction history (`_predictions.jsonl`), one set per ticker you've trained

There's no request/response API layer — Streamlit's script-rerun model *is* the request cycle, and `st.session_state` carries state (like the quick-lookup ticker) across page switches.

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
│   ├── home.py               # Dashboard — market overview, sector performance, open positions
│   ├── research.py           # Research page
│   ├── portfolio.py          # Portfolio tracking, risk analytics, Options Log
│   ├── watchlist.py          # Weekly watchlist + daily gap/volume/ATR/ML check
│   ├── screener.py           # Multi-factor stock screener
│   ├── strategy_lab.py       # 4H/30m/5m multi-timeframe setup — live scanner + backtest
│   └── trading.py            # Trading Desk — Day Trading / Options / News / Predictions tabs
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
│   ├── ml_prediction.py       # XGBoost + RF ensemble (daily): train, predict, evaluate
│   ├── intraday_prediction.py # Separate intraday (15m) direction model — own features/labels/storage
│   ├── price_projection.py    # Monte Carlo price-path simulation
│   ├── options_pricing.py     # Black-Scholes pricing, Greeks, implied-vol solver
│   ├── volatility_forecast.py # GARCH(1,1) forward volatility forecast
│   ├── sentiment.py           # VADER headline sentiment scoring
│   ├── fundamental_score.py   # Quality/Value/Growth scoring engine
│   ├── regime.py              # Market regime detection (trend vs. 200-day MA)
│   ├── regime_markov.py       # Markov-chain regime model — persistence, forecast, stationary distribution
│   └── risk.py                # Portfolio risk metrics, position sizing, stress tests
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
│   ├── round_trips.py          # FIFO buy/sell matcher → round trips with P&L, hold time
│   └── watchlist.py            # Weekly watchlist persistence
├── docs/
│   ├── workflow.md                    # Day-by-day and week-by-week usage workflow
│   ├── ORBC_PLAYBOOK.md               # ORBC rules, design decisions, and daily trading routine
│   ├── ML_PREDICTION.md               # Full technical writeup of the ML ensemble
│   ├── Identifying-Chart-Patterns.md  # Flag/Pennant pattern reference
│   └── VERIFICATION_CHECKLIST.md      # Manual verification steps for a few past fixes
├── tests/
│   ├── conftest.py             # Shared fixtures — synthetic daily + intraday OHLCV, isolated storage dir
│   ├── test_ml_prediction.py   # Regression suite for analysis/ml_prediction.py (see Reliability & Verification)
│   ├── test_orbc_strategy.py   # ORBC confirmation state machine, filters, stops/targets, direction-aware P&L
│   └── test_intraday_prediction.py  # Intraday label masking, vol-scaled bands, storage isolation from daily
└── storage/                    # Persisted ML models and prediction logs (auto-created)
    ├── {TICKER}_xgb.pkl
    ├── {TICKER}_rf.pkl
    ├── {TICKER}_accuracy.json
    └── {TICKER}_predictions.jsonl
```

## Extensibility & Customization

- **Add a technical indicator** — implement it in `analysis/indicators.py`'s `calculate_indicators()`, then reference the new column wherever it should render (Research chart, Day Trading signal panel).
- **Add an ML feature** — extend `data/feature_engineering.py`'s feature matrix; the ensemble in `analysis/ml_prediction.py` picks up new columns automatically on the next training run.
- **Add a screener screen** — extend the screen dispatch in `pages/screener.py`'s `_run_screen()` / `_screen_ticker()` with the new filter logic and a label for the dropdown.
- **Add or retune an AI brief** — add a prompt builder function in `ai/stock_brief.py` following the existing `generate_*` pattern, and give it its own `OLLAMA_MODEL_*` override in `config/settings.py` if it deserves a different model than the defaults.
- **Tune scoring or risk thresholds** — `config/settings.py` centralizes fundamental scoring cutoffs (ROIC, FCF yield, margin expansion), options thresholds (IVR high/low, IV/RV premium), and risk defaults (per-trade risk %, max position size).

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

1. **A `pytest` regression suite** (`tests/`, 91 tests) covering two areas. `test_ml_prediction.py` exercises `analysis/ml_prediction.py` end-to-end — module import (the exact failure mode that silently killed the Predictions tab for 11 days when a dependency in the sklearn/xgboost/narwhals chain broke), train/predict/evaluate on synthetic OHLCV data, the reliability gate correctly rejecting a pure random walk, and the predict → save_prediction → get_prediction_history persistence round-trip. `test_orbc_strategy.py` pins the ORBC confirmation state machine against hand-laid-out intraday sessions: that a single breakout close never signals, that a close back inside resets the count, that filters fall through from the 2nd to the 3rd close, and that short P&L carries the correct sign. Run it with `pytest tests/ -q`. No network access required or used.
2. **The ML model self-gates on quality.** Training runs anchored walk-forward validation and only saves a model if it clears a 52% mean directional accuracy floor with std-dev ≤ 8% across folds; a model that doesn't clear the bar is reported as such instead of silently saved.
3. **Manual verification checklists.** `docs/VERIFICATION_CHECKLIST.md` documents the manual steps used to validate timezone handling, activity logging, and the options FIFO round-trip matcher against real fill data.

`analysis/backtest.py`'s pure functions and `portfolio/round_trips.py`'s FIFO matcher have no Streamlit dependency and are the next-highest-value places to add coverage.

## Disclaimer & License

Aether is a personal research tool for educational and informational purposes only. It is not a registered investment advisor, broker-dealer, or financial planning service. Nothing in this application constitutes financial advice, a recommendation to buy or sell any security, or a guarantee of future performance.

ML models in this application have a modest historical edge (typically 52–58% directional accuracy on out-of-sample data). They cannot predict news events, earnings surprises, or macro regime changes. Past walk-forward accuracy does not guarantee future results.

All trading involves risk. You may lose some or all of your capital. Always perform your own due diligence before making investment decisions.

No `LICENSE` file is included yet. Treat the code as all-rights-reserved until one is added — add an `MIT` or similar license file before making the repository public if you intend to allow reuse.

## Appendix: Helpful Commands

```bash
# Run the dashboard locally
streamlit run app.py

# Run Ollama for local, free AI briefs (keep the terminal open)
ollama serve
ollama pull llama3.2

# Install dependencies
pip install -r requirements.txt

# Run the regression test suite
pytest tests/ -q

# Inspect the local database directly
sqlite3 storage/journal.db "select * from activity_log"
```
