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

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
