import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

CLAUDE_MODEL = "claude-sonnet-4-6"

# AI provider selection
# "auto"   → use Claude if ANTHROPIC_API_KEY is set, otherwise fall back to Ollama
# "claude" → Claude only (fails silently if key not set)
# "ollama" → Ollama only (fails silently if not running)
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Per-brief Ollama model overrides — fall back to OLLAMA_MODEL when unset.
# Lets you route judgment-heavy briefs to a larger reasoning model (e.g.
# deepseek-r1:32b) while keeping lighter tasks on something fast, without
# touching code.
OLLAMA_MODEL_STOCK_BRIEF = os.getenv("OLLAMA_MODEL_STOCK_BRIEF", OLLAMA_MODEL)
OLLAMA_MODEL_OPTIONS_BRIEF = os.getenv("OLLAMA_MODEL_OPTIONS_BRIEF", OLLAMA_MODEL)
OLLAMA_MODEL_DAYTRADING_BRIEF = os.getenv("OLLAMA_MODEL_DAYTRADING_BRIEF", OLLAMA_MODEL)
OLLAMA_MODEL_THESIS = os.getenv("OLLAMA_MODEL_THESIS", OLLAMA_MODEL)

# Cache TTL in seconds
PRICE_CACHE_TTL = 300       # 5 minutes
FUNDAMENTALS_CACHE_TTL = 3600   # 1 hour
OPTIONS_CACHE_TTL = 600     # 10 minutes
NEWS_CACHE_TTL = 900        # 15 minutes

# News sentiment
NEWS_MAX_ARTICLES = 8

# Risk defaults
DEFAULT_PORTFOLIO_SIZE = 100_000
DEFAULT_RISK_PER_TRADE = 0.01   # 1% of portfolio per trade
MAX_POSITION_PCT = 0.10         # Max 10% in any single position
RISK_FREE_RATE = 0.045          # ~4.5% (current T-bill approximation)

# Technical indicator defaults
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
ADX_PERIOD = 14

# Market indices for regime detection
SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"
NASDAQ_TICKER = "^IXIC"
RUSSELL_TICKER = "^RUT"

# Fundamental scoring thresholds
ROIC_EXCELLENT = 20
ROIC_GOOD = 12
FCF_YIELD_GOOD = 0.04
GROSS_MARGIN_EXPANDING = 0.005   # 50bps expansion = positive signal
NET_DEBT_EBITDA_WARNING = 3.0
NET_DEBT_EBITDA_DANGER = 5.0

# Options thresholds
IVR_HIGH = 50
IVR_LOW = 30
IV_RV_PREMIUM_THRESHOLD = 1.15  # IV 15% > RV = potentially rich premium

# Minimum accuracy a model must add over the naive "always predict the more
# common direction" baseline before it counts as reliable. The flat 52% floor
# alone is not a bar: once the neutral band drops small moves, bull-market drift
# puts the majority class at 54-60% on a trending ticker, so a 55% model can look
# reliable while losing to a constant guess. Read by
# ml_prediction._summarize_fold_scores.
MIN_EDGE_OVER_BASELINE = 0.02

# Fraction of the most recent history withheld from every search stage in
# train_model, used once at the end to produce an out-of-sample accuracy.
#
# Needed because train_model runs three sequential argmax searches (label scheme,
# XGB params, RF params) all scored on the same data with no holdout. The winner
# of many tries is not an unbiased estimate — measured at ~7 points on random
# walks with no signal at all. The walk-forward number is therefore an upper
# bound, not a forecast; holdout_accuracy is the honest one.
#
# 0.2 of two years is ~100 bars, ~70 after the neutral band, so the holdout
# estimate is unbiased but wide (roughly +/-12 points at 95%). Both the value and
# its sample size are reported so it cannot be read as precise.
HOLDOUT_FRACTION = 0.2

# Below this many holdout samples the estimate is too noisy to gate on, so
# is_reliable falls back to the walk-forward number and says so.
MIN_HOLDOUT_SAMPLES = 40

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")

# Model retraining triggers (Prediction Improvement Engine, Phase 8)
RETRAIN_STALENESS_DAYS = 30              # existing behavior, now centralized
RETRAIN_ACCURACY_DROP_THRESHOLD = 0.05   # trigger if live accuracy falls this
                                          # far below the trained-in accuracy
RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK = 20  # don't judge a drop on too few resolved predictions

# ── Options cost model (Roadmap Item 3) ──────────────────────────────────────
# analysis/intraday_prediction.assess_tradeability() prices costs in UNDERLYING
# percentage points, which is right for trading shares and wrong for contracts:
# it ignores delta leverage and has no theta term. assess_options_tradeability()
# in analysis/options_pricing.py is the contract-aware version; these are its
# defaults. See docs/ROADMAP.md for the derivation.
OPTIONS_COST_MODEL_DEFAULT = "options"    # "options" | "shares"

# Expiry ladder swept per signal horizon. Each rung is snapped to the nearest
# LISTED expiry at fetch time, so these are targets, not guarantees.
OPTIONS_EXPIRY_LADDER_DTE = [0, 2, 7, 30]

# Pro-rating theta: "trading" spreads a day's decay over the 390-minute session
# (most decay is realized during trading hours); "calendar" spreads it over
# 1440 minutes. "trading" is the more conservative intraday read.
OPTIONS_THETA_BASIS = "trading"

# Below this many days to expiry, closed-form Black-Scholes theta is unstable
# (it diverges as T -> 0), so the cost model switches to an empirical
# sqrt-of-time-remaining decay on the contract's extrinsic value instead.
OPTIONS_MIN_DTE_FOR_BS_THETA = 1.0

# Used only when no live chain is available (market closed, fetch failure).
# Expressed as a fraction of the contract's mid price for one round trip.
# Deliberately pessimistic — a fallback should never flatter the verdict.
OPTIONS_FALLBACK_SPREAD_PCT = 0.02        # 2% of mid

# Stop distance for an intraday signal, as a multiple of the average absolute
# move over that signal's own horizon. The daily-ATR stop (1.5 x ATR) stays in
# place for swing signals; see Roadmap Item 2B.
HORIZON_STOP_ATR_MULTIPLE = 1.25

