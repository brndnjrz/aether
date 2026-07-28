# ML Prediction — Technical Reference

The **Predictions** tab (Trading Desk page, `pages/trading.py`) runs a machine-learning ensemble that predicts the 5-day price direction of a stock, and simulates a probabilistic 5-day open/close price band from the stock's own volatility. This document explains how to use the tab, how the model works, what it is good at, what it cannot do, and how to keep it performing well over time.

## Using the Predictions Tab

**Controls:**
- **Ticker Symbol** — any symbol; defaults to `AAPL` and persists across reruns in session state.
- **Train / Update Model** — fetches 2 years of daily bars, builds the 18-feature matrix, runs the 10-fold walk-forward validation described below, then fits the final model on all available directional history and saves it to `storage/`. Takes roughly 10-20 seconds. Requires at least 60 bars of history, and at least 50 directional (non-neutral) samples after the neutral-zone filter — tickers with too little history or an IPO within the lookback window will error out here.
- **Generate Prediction** — loads the saved model and runs inference on the latest bar. If no model exists yet for the ticker, this auto-trains one first.

**Status row** (above the buttons): model status (not trained / trained / overdue for refresh — flagged after 30 days), last-trained timestamp read from `{TICKER}_accuracy.json` (or the model file's mtime if that log is missing), and a "last prediction" timestamp for the current session.

Once you click **Generate Prediction**, five sections render in order:

1. **Direction Signal** — a gauge chart of the ensemble bull probability, deliberately windowed to 35-65% rather than 0-100% so the display never implies more confidence than the walk-forward accuracy supports. The 47-53% band is shaded as the neutral dead-band — inside it no directional call is issued. Alongside the gauge: a confidence badge (HIGH/MODERATE/LOW, see thresholds below), **Expected 5-Day Move** (median historical 5-day return in past setups where the model made the same call — a look-back statistic, not a forecast), and **Walk-Forward Accuracy** with a delta against the 50% coin-flip baseline. A callout below restates the accuracy as a plain-language "edge" (e.g. "+4.2% above coin-flip") and warns not to size a position larger than your standard allocation off this signal alone.
2. **5-Day Price Path (Simulated)** — a Monte Carlo band chart and table of simulated daily open/close prices for the next 5 trading days. See [5-Day Price Path Simulation](#5-day-price-path-simulation) below for how this is computed and why it is not the same thing as a price forecast.
3. **Feature Importance** — a horizontal bar chart of the top 8 of the 18 features by XGBoost gain (how much decision weight that feature contributed across the model's splits). This is what explains *why* the current prediction leans the way it does — e.g. `price_vs_sma20` and `rsi_norm` dominating points to a trend/momentum-driven call, while `vol_ratio` and `hl_range_pct` dominating points to a volatility-driven one. Rankings can shift between training runs; a large shift signals the model's regime sensitivity, not a bug.
4. **Model Performance** — the walk-forward metrics in full: a reliability verdict and reason string, directional accuracy, ROC-AUC, total out-of-sample validation predictions vs. total training samples used, in-sample Sharpe of the raw signal (explicitly labeled a sanity check only — not for live position sizing), accuracy standard deviation across folds, and the training label's class balance (bullish % vs bearish %, flagged if skewed past 35/65 since `scale_pos_weight` is auto-adjusted for that).
5. **Model Details** (expander) — the exact hyperparameters and walk-forward/retrain configuration in one place, for auditing the model rather than trusting the summary numbers.

Below that, **Prediction History** logs every prediction generated for the ticker (persisted to `storage/{TICKER}_predictions.jsonl`, so it survives across sessions): a time-series line of bull probability with markers colored by direction, plus a table of the last 15 predictions (direction, probability, confidence, expected move, model accuracy at that time) and a running count of bullish/bearish/neutral calls logged. `resolve_predictions()` auto-verifies pending predictions each time the history is loaded, back-filling `actual_outcome`/`correct` once a prediction's horizon has elapsed (comparing `price_at_prediction` to the close that many trading days later); predictions logged before this function existed, or still within their horizon, are left unresolved.

A disclaimer banner is shown at the bottom of the tab regardless of state, restating the 52-58% typical accuracy range, the inability to predict news/earnings/macro shocks, and that position sizing should come from the Quick Risk Calculator on the Trading Desk's Day Trading tab, not from this signal.

## How the ML Model Works

The model is a two-member ensemble:

| Model | Role | Weight |
|-------|------|--------|
| **XGBoost** (`binary:logistic`) | Primary classifier — gradient-boosted shallow trees optimized for AUC | 65% |
| **Random Forest** | Calibration member — provides diversity and prevents XGBoost from overconfident outputs | 35% |

The ensemble bull probability is:

```
P_bull = 0.65 × XGBoost_prob + 0.35 × RF_prob
```

The combined probability is then mapped to a direction:

- `P_bull > 0.55` → **BULLISH**
- `P_bull < 0.45` → **BEARISH**
- `0.45 ≤ P_bull ≤ 0.55` → **NEUTRAL** (dead-band; no directional call)

The display gauge is capped at 35–65% to prevent conveying false precision. Raw model output beyond these bounds does not meaningfully distinguish between different confidence levels given the amount of noise in financial data.

## 5-Day Price Path Simulation

Separately from the direction classifier above, the Predictions tab also simulates a **probabilistic band** of daily open/close prices for the next 5 trading days (`analysis/price_projection.py`, `simulate_price_path()`). This is not a second trained model and does not produce a single price target — it runs a Monte Carlo geometric random walk (3,000 simulated paths by default) and reports the 25th/median/75th percentile of simulated prices for each day.

**Inputs, reused from data already computed elsewhere in the app:**

- **Volatility (daily sigma)** — derived from `hv_21`, the 21-day annualized historical volatility already computed by `calculate_indicators()`, converted back to a daily figure (`hv_21 / 100 / sqrt(252)`). Falls back to a direct recomputation from `Close.pct_change()` if `hv_21` is unavailable.
- **Drift** — the classifier's own `expected_move_pct` (median historical 5-day return for the current setup), spread evenly across the 5 simulated days. If that figure is unavailable, drift falls back to a small tilt scaled off the raw bull `probability`.
- **Overnight gap** — each day's simulated open is the prior simulated close plus a gap term drawn from a reduced-variance version of the same daily sigma (40% of a full day's volatility). This is an explicit simplification, not a fitted overnight-gap distribution.

**Determinism:** the random generator is seeded from the last bar's timestamp, so repeated clicks of **Generate Prediction** against the same underlying data produce the same simulated numbers within a session — the numbers only change once new daily data arrives or the model is retrained (retraining clears the cached price path along with the direction signal).

**How to read it:** the band widens with each day out, which is the correct behavior for a random walk — uncertainty compounds. Roughly half of actual future outcomes should fall outside the shown 25th–75th percentile range; it is a range of plausible outcomes weighted by the stock's own volatility and the model's directional lean, not a forecast of where the price will actually be.

## Features Used

The model uses exactly 18 features, all computed from the stock's own daily OHLCV price history. No external data sources are used.

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `rsi_norm` | RSI-14 divided by 100 (0–1 range) | Measures overbought/oversold momentum |
| `rsi_5_norm` | 5-period RSI divided by 100 | Short-term momentum; faster signal than RSI-14 |
| `macd_hist_sign` | Sign of the MACD histogram (−1, 0, +1) | Momentum direction change |
| `adx_norm` | ADX divided by 100 | Trend strength; low ADX = choppy market |
| `atr_pct` | ATR as a percentage of closing price | Normalizes volatility across price regimes |
| `bb_pct` | Bollinger Band percent position (0 = lower band, 1 = upper band) | Mean-reversion positioning |
| `vol_ratio` | Current volume divided by 20-day average volume | Confirms or questions price moves |
| `price_vs_sma20` | (Close − SMA-20) / SMA-20 | Short-term trend deviation |
| `price_vs_sma50` | (Close − SMA-50) / SMA-50 | Medium-term trend deviation |
| `ret_1d` | 1-day return | Recent price momentum |
| `ret_5d` | 5-day lagged return (not the target) | Continuation vs reversal signal |
| `ret_10d` | 10-day lagged return | Intermediate momentum |
| `hv_ratio` | HV-10 divided by HV-21 (short vol / medium vol) | Volatility compression signal; low ratio precedes breakouts |
| `obv_slope` | Sign of the 5-bar OBV change | Volume-weighted trend confirmation |
| `stoch_k_norm` | Stochastic %K divided by 100 | Short-term overbought/oversold |
| `day_of_week` | Day of week as integer (0=Monday, 4=Friday) | Captures weekday seasonality effects |
| `above_200ma` | Binary 1 if price is above the 200-day SMA, 0 otherwise | Regime filter — bull market context |
| `hl_range_pct` | (High − Low) / Close | Daily bar range; measures intraday tension |

All features are computed using only data available at the time of prediction (no look-ahead). Continuous features are normalized or clipped to remove extreme outliers. Binary features (`above_200ma`, `macd_hist_sign`, `obv_slope`) are left as-is.

## Target Variable

The model predicts **5-day directional movement** based on the 5-trading-day forward return:

- **Bullish (label = 1)**: 5-day forward return > +0.5%
- **Bearish (label = -1)**: 5-day forward return < -0.5%
- **Neutral (label = 0)**: |5-day forward return| ≤ 0.5% — these rows are **excluded from training**

The neutral threshold of 0.5% serves as a noise filter. Rows where the stock barely moved are uninformative and would dilute the model's ability to learn real directional patterns. They are dropped before training begins, but displayed in the signal output as a "NEUTRAL" result when the model's own confidence falls in the 45–55% zone.

The target horizon of 5 days is a deliberate choice: it is long enough for a meaningful directional move but short enough that technical signals have predictive relevance. Longer horizons are dominated by macro regime and fundamentals; shorter horizons are dominated by noise.

## Training Approach

The model is trained with **anchored walk-forward validation** using `TimeSeriesSplit(n_splits=10, gap=5)` from scikit-learn.

```
Training configuration:
  Walk-forward splits: TimeSeriesSplit(n_splits=10, gap=5)
  Gap = 5 bars: prevents the 5-day forward return target from
                bleeding back into training features
  XGBoost: max_depth=4, min_child_weight=10, learning_rate=0.05,
           n_estimators=200, subsample=0.8, colsample_bytree=0.8
  Random Forest: n_estimators=100, max_depth=6, min_samples_leaf=20
  Minimum data: 60 bars required; 500+ bars recommended
```

Walk-forward validation works by training on the past and validating on the immediate future — the same way you would actually use the model in production. The key constraint is the `gap=5` parameter, which inserts a 5-bar gap between the last training bar and the first validation bar. Without this gap, the 5-day forward return target would overlap with training data and create lookahead bias.

The final production model is then retrained on **all available directional data** before being saved to disk. The walk-forward metrics are reported alongside predictions as an honest estimate of expected performance.

If the training dataset has a class imbalance (more bullish rows than bearish, or vice versa), XGBoost's `scale_pos_weight` parameter is automatically adjusted to compensate.

## How to Interpret Predictions

### Signal levels

```
BULLISH — bull probability > 55%
NEUTRAL — bull probability between 45% and 55% (no call)
BEARISH — bull probability < 45%
```

### Confidence levels

| Confidence | Bull Probability Range | Meaning |
|-----------|----------------------|---------|
| HIGH | < 35% or > 65% (clipped to display range) | Strong signal; model is well outside the neutral zone |
| MODERATE | 45–55% or 55–65% boundary | Some directional lean; treat as supporting evidence only |
| LOW | 47–53% | Near-neutral; signal is noise; do not trade on this alone |

In practice, most signals will be LOW or MODERATE confidence. HIGH confidence signals are rare, and that is expected — they represent setups where multiple technical features are aligned, which happens infrequently.

### Expected Move

The "Expected 5-Day Move" figure is **not a price forecast**. It is the median 5-day return observed historically in the training data for setups where the model predicted the same direction as it predicts now. It is a historical look-back statistic that answers: "In past setups like this one, what happened on average?"

Use it as a rough magnitude reference, not a target. Actual outcomes vary widely.

### Walk-Forward Accuracy

The accuracy shown in the prediction card is the mean directional accuracy across all 10 out-of-sample walk-forward folds. It is the closest thing to an honest forward-looking performance estimate you can compute from historical data.

The delta shown next to the accuracy figure is `accuracy − 50%`, which is the model's edge over a random coin flip. An edge of +5% means the model was correct 55% of the time vs 50% randomly.

## Model Accuracy

### What good accuracy looks like

| Directional Accuracy | Assessment |
|---------------------|------------|
| < 52% | Below the reliability threshold; signal should not be used |
| 52–55% | Reliable but modest edge; use as one of several inputs |
| 55–58% | Good edge for a financial ML model; most well-tuned models fall here |
| > 58% | Strong edge; validate carefully — this range may indicate overfit |

A model is considered **reliable** when:
- Mean walk-forward accuracy ≥ 52%
- Standard deviation of accuracy across folds ≤ 8%

The second condition matters as much as the first. A model that is 57% accurate on average but varies between 44% and 68% across folds is unstable — it is working in some market regimes and failing in others. You cannot know which regime you are in today.

### The ROC-AUC metric

AUC (Area Under the ROC Curve) measures the model's ability to rank bullish outcomes above bearish ones, regardless of the specific probability threshold. A random model scores 0.50. A perfect model scores 1.00. A well-functioning model on financial data typically scores 0.52–0.58.

### Baseline comparison

The 50% baseline is the "coin flip" — what you would achieve by predicting bullish (or bearish) for every single day. Any model below 50% is actively harmful. Any model above 52% with low variance across folds is generating real signal.

## Limitations

**The model cannot predict:**

- **News events** — earnings surprises, M&A announcements, FDA decisions, and geopolitical shocks are outside the feature set entirely. The model has no access to news data
- **Macro regime changes** — Fed rate decisions, recession signals, or sector rotations that unfold over weeks or months are poorly captured by 5-day technical features
- **Gap opens** — the model is trained on daily closes; a gap open caused by overnight news cannot be anticipated
- **Intraday moves** — the 5-day horizon and daily data make this unsuitable for intraday trading decisions
- **Low-liquidity stocks** — thinly traded stocks have erratic price action that technical indicators cannot characterize reliably
- **Very new stocks or recently-listed ETFs** — the model requires at least 60 bars of history; 500+ bars (roughly 2 years) produces the most stable results

**Structural limitations:**

- The model is trained on technical features only; fundamentals, earnings expectations, and sector momentum are not inputs
- Probabilities are clipped to 35–65% because the raw model output beyond these bounds does not reliably distinguish between different future outcomes given financial data's signal-to-noise ratio
- Walk-forward accuracy from the training period may not hold in the current market regime; accuracy degrades when market conditions shift significantly from the training window
- The 5-Day Price Path is a volatility simulation seeded from the stock's own historical volatility and the classifier's directional bias, not a separately backtested or validated forecasting model — it has no walk-forward accuracy figure of its own, and its overnight-gap modeling is a simplification, not a fitted gap distribution

## Retraining

### When to retrain

The app flags a model as **overdue** 30 days after the last training date. This is the default retrain schedule.

Retrain more frequently if:
- The stock has been through a major regime change (e.g., earnings blowout, sector rotation, acquisition announcement)
- Walk-forward accuracy was borderline reliable (52–54%) on the last training run — newer data may push it above or below the threshold
- The expected move estimate has been consistently wrong in recent predictions

### How to retrain

1. Navigate to the **AI Predictions** page
2. Enter the ticker
3. Click **Train / Update Model**
4. The new model replaces the old one in `storage/`; prediction history is preserved

Retraining fetches 2 years of fresh daily data each time. The walk-forward validation runs on that full 2-year window, so older data influences the early folds while more recent data dominates the later folds — which are the most predictive of near-term performance.

### Storage

Trained models are stored in the `storage/` directory at the project root:

```
storage/
├── {TICKER}_xgb.pkl         # XGBoost model
├── {TICKER}_rf.pkl          # Random Forest model
├── {TICKER}_accuracy.json   # Walk-forward metrics from last training run
└── {TICKER}_predictions.jsonl  # Timestamped prediction log (append-only)
```

Each ticker has its own model files. Training AAPL does not affect the NVDA model. The prediction log is append-only — predictions are never deleted, which allows you to review the model's signal history over time on the Prediction History chart.
