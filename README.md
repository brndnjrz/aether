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
| **Options Log** (`pages/portfolio.py`) | The trade journal: manual fill entry → automatic FIFO round-trip P&L, hold-time/entry-hour/ticker/option-type/day-of-week win-rate analytics, and a cumulative P&L equity curve |
| **Trading Desk** (`pages/trading.py`) | Four tabs in one page — **Day Trading** (market-status banner, intraday signals, candlestick pattern read, Flag/Pennant continuation-pattern detection with confidence scoring, suggested entry/stop/target, AI brief, MACD backtest), **Options** (chain, IV Rank, GARCH forward-vol forecast, Greeks, P&L diagrams, AI brief), **News** (headline sentiment), **Predictions** (ML direction signal + simulated price path) |
| **Strategy Lab** (`pages/strategy_lab.py`) | Two intraday strategies, each with a Live Scanner and Backtest sub-tab — **ORBC** first (Opening Range Breakout Confirmation: requires a 2nd consecutive close outside the opening range before signalling, in `analysis/orbc_strategy.py`), then **MTF** (4H trend → 30m pullback into a demand zone → 5m structure shift → tape confirmation, in `analysis/mtf_strategy.py`) |

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

Landing page. No input required.

- Regime banner (Bull/Uptrend/Sideways/Downtrend/Bear vs. the S&P's 200-day MA)
- Live index cards (SPY/QQQ/IWM/VIX)
- **Market Regime (Markov)** — a probabilistic second opinion on the same trend, reusing the Markov model Trading Desk runs per-ticker, applied to the S&P 500
- Sector performance
- Open positions (empty until logged)
- **Recent Activity** — the last 8 logged events across Trading Desk and Strategy Lab, newest first

### Research

Enter a ticker + lookback period — loads automatically.

- **Chart & Technicals** — candlestick + SMA/Bollinger/support-resistance/trendlines
- **Fundamentals** — Quality/Value/Growth 0–100 scores + red flags
- **Options** — IV Rank, ATM IV, IV/RV
- **News & Sentiment** — VADER-scored headlines, display-only
- **AI Brief** — one-click investment summary
- **ML Direction Signal** — runs automatically between the scorecard and the chart

### Options Log

The trade journal — the only page where you log trades. Enter each options fill as your broker reports it; `portfolio/round_trips.py` FIFO-matches buys against sells into round trips with P&L and hold time.

- **Pattern-finding analytics** — a cumulative P&L equity curve, win rate by hold-time bucket, entry hour, option type, and day of week, and a per-ticker performance breakdown (total/avg P&L, win rate).
- **Equity positions have no logging UI** — options fills only. Formerly "Portfolio," with Positions / Risk Analytics / Position Sizer tabs; those tracked equity positions with no UI to ever add one, and the Position Sizer duplicated Trading Desk's own Quick Risk Calculator, so all three were cut.

### Trading Desk

Four tabs:

- **Day Trading** — VWAP deviation, momentum, volume ratio, trend alignment (all interval-aware except Trend Alignment, which stays on daily SMA20/50/200 + EMA50 by design), candlestick pattern detection, and Flag/Pennant continuation-pattern detection (`analysis/flag_pennant_detection.py` + `flag_pennant_scoring.py`) drawn directly on the chart with a 0–100 confidence score. Signals combine into a Suggested Entry/Stop/Target card via majority vote, plus an AI Day Trading Brief and a MACD-cross backtest.
- **Options** — IV Rank/Percentile, a GARCH(1,1) forward volatility forecast vs. ATM IV, the full chain, P&L diagrams, Black-Scholes Greeks, and an AI Options Brief.
- **News** — headline sentiment for the entered ticker, same VADER scoring as Research.
- **Predictions** — train/retrain the ML ensemble, generate a direction signal + simulated price path. A **Prediction horizon** toggle switches between **Daily (swing)** — the original model, unchanged — and **Intraday (15-min bars)**, a separate model with its own features, labels, and storage. See [AI & ML Model Overview](#ai--ml-model-overview) and `docs/ML_PREDICTION.md`.

Day-by-day, week-by-week rhythm: `docs/workflow.md`.

### Strategy Lab

Two intraday strategies, each its own tab with a Live Scanner and a Backtest sub-tab, **ORBC first**.

**ORBC (Opening Range)** — the first N minutes after the 9:30 ET open set a reference high/low. Waits for a **second consecutive close** outside that range before signalling — filters most post-open false breakouts. Logic: `analysis/orbc_strategy.py`.

- **Confirmation rule** — fires on the Nth consecutive close outside the range (default 2). A close back inside resets the count. If a filter blocks the Nth close, later closes can still fire up to `max_confirmation_closes` (default 3) — exactly one signal per breakout episode.
- **Configurable** — bar interval, opening-range duration, confirmation count, entry cutoff, three filters (volume vs. 20-bar avg, VWAP alignment, range vs. ATR), long/short enablement, stop method (range/ATR/percent), target method (R:R/ATR/range projection).
- **Scanner shows** — the opening-range band, every close outside it, filtered-out breakouts marked ✕ (hover for the reason), entry/stop/target for a confirmed signal, and a 0–100 confidence score built from volume thrust, VWAP alignment, breach decisiveness, range quality vs. ATR, and short-term EMA agreement. One click logs a confirmed signal.
- **Both directions supported** — `evaluate_orbc_trade()` is direction-aware, unlike this app's other long-only simulators. Positions always flatten at session close.
- **Small sample by design** — intraday bars cap at ~60 days and ORBC fires at most once per session, so a backtest yields a few dozen trades. Warns explicitly below 30.

Full rule set + daily routine: `docs/ORBC_PLAYBOOK.md`.

**MTF** — 4H trend → 30m pullback into a demand zone → 5m structure shift → tape confirmation (price/volume proxy, not real order flow) → target at the volume-profile point of control, stop at the swing low. Logic: `analysis/mtf_strategy.py`.

## Architecture & Workflow

No central orchestrator. `app.py` sets page config, theme, and the sidebar, then `st.navigation()` routes between five independent pages. Each page:

1. **Fetches** — price/fundamentals/options/news via `data/*.py`, cached per `config/settings.py` TTLs
2. **Computes** — indicators, scores, or the ML ensemble via `analysis/*.py`
3. **Renders** — Plotly charts and Streamlit widgets, inline
4. **Briefs (optional)** — a button click on Research or the Trading Desk routes the already-computed data through `ai/client.py`

**Persists across pages and reruns:**

- **`storage/journal.db`** (SQLite, via `portfolio/db.py`) — positions, activity log, options fills
- **`storage/{TICKER}_*`** — trained models, walk-forward accuracy, prediction history — one set per trained ticker

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
│   ├── strategy_lab.py       # ORBC + MTF setups — each with a live scanner and backtest
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
│   └── round_trips.py          # FIFO buy/sell matcher → round trips with P&L, hold time
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

- **Add a technical indicator** — implement it in `analysis/indicators.py`'s `calculate_indicators()`, then reference the column wherever it should render.
- **Add an ML feature** — extend `data/feature_engineering.py`'s feature matrix. The daily ensemble picks up new columns on the next training run.
- **Add or retune an AI brief** — add a `generate_*` prompt builder in `ai/stock_brief.py`, and give it its own `OLLAMA_MODEL_*` override in `config/settings.py` if it deserves a different model.
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

**1. A `pytest` regression suite** (`tests/`, 91 tests, run with `pytest tests/ -q` — no network access needed):

- `test_ml_prediction.py` — the daily model end-to-end: an import-crash guard (the exact failure that silently killed the Predictions tab for 11 days), train/predict/evaluate on synthetic data, the reliability gate correctly rejecting a pure random walk, and the predict → save → history persistence round-trip.
- `test_orbc_strategy.py` — the ORBC confirmation state machine against hand-built sessions: a single breakout close never signals, a close back inside resets the count, filters fall through from the 2nd to the 3rd close, and short P&L carries the correct sign.
- `test_intraday_prediction.py` — storage isolation from the daily model, session-boundary label masking, trailing-sigma leak resistance, and naive/UTC index handling.

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

# Run the regression test suite
pytest tests/ -q

# Inspect the local database directly
sqlite3 storage/journal.db "select * from activity_log"
```
