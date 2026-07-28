"""
AI stock brief generator — produces a structured one-page investment summary.
Designed around the Financial Analyst's recommended format:
Quality / Value / Momentum scores, key metrics, thesis, stop suggestion.
"""
import logging
from typing import Dict, Any, Optional
from ai.client import ask_claude
from config.settings import (
    OLLAMA_MODEL_STOCK_BRIEF, OLLAMA_MODEL_OPTIONS_BRIEF,
    OLLAMA_MODEL_DAYTRADING_BRIEF, OLLAMA_MODEL_THESIS,
)

logger = logging.getLogger(__name__)

BRIEF_SYSTEM = """You are a senior equity research analyst producing concise investment briefs.
Your output is always structured, data-grounded, and actionable.
You do NOT predict stock prices. You synthesize data into investment-quality judgments.
Keep responses under 400 words unless analysis warrants more.
Be direct and opinionated — vague generalities are useless."""


def generate_stock_brief(
    ticker: str,
    fundamentals: Dict[str, Any],
    technicals: Dict[str, Any],
    regime: Dict[str, Any],
    scores: Dict[str, Any],
) -> Optional[str]:
    """
    Generate an AI investment brief combining fundamental + technical context.
    """
    # Build structured prompt with all the data
    f = fundamentals.get("raw", {})
    q_score = scores.get("quality", {}).get("score", "N/A")
    v_score = scores.get("value", {}).get("score", "N/A")
    g_score = scores.get("growth", {}).get("score", "N/A")
    composite = scores.get("composite_score", "N/A")
    flags = scores.get("red_flags", [])

    prompt = f"""Generate an investment brief for {ticker} ({f.get('name', ticker)}).

SECTOR: {f.get('sector', 'Unknown')} | {f.get('industry', 'Unknown')}

FUNDAMENTAL SCORES (0-100):
- Quality: {q_score}/100
- Value: {v_score}/100
- Growth: {g_score}/100
- Composite: {composite}/100

KEY METRICS:
- Revenue Growth (YoY): {_fmt_pct(f.get('revenue_growth_yoy') or f.get('revenue_growth'))}
- Gross Margin: {_fmt_pct(f.get('gross_margin_curr') or f.get('gross_margin'))}
- FCF Yield: {_fmt_pct(f.get('fcf_yield'))}
- ROIC Proxy: {_fmt_pct(f.get('roic_proxy'))}
- Debt/Equity: {f.get('debt_to_equity', 'N/A')}
- EV/EBITDA: {_fmt_num(f.get('ev_ebitda'))}x
- P/FCF: {_fmt_num(f.get('price_to_fcf'))}x
- PEG Ratio: {_fmt_num(f.get('peg_ratio'))}

TECHNICAL CONTEXT:
- Trend: {technicals.get('trend', 'N/A')} | Regime: {regime.get('regime', 'N/A')}
- Price vs 200MA: {'Above' if regime.get('above_200ma') else 'Below'}
- RSI: {technicals.get('rsi', 'N/A')}
- ADX: {regime.get('adx', 'N/A')} ({'Strong trend' if (regime.get('adx') or 0) > 25 else 'Weak trend'})
- 21-day return: {_fmt_pct_val(regime.get('price_21d_pct'))}

MARKET CONTEXT: {regime.get('market_regime', 'N/A')} | VIX: {regime.get('vix', 'N/A')}

RED FLAGS: {len(flags)} detected
{chr(10).join('- ' + f['flag'] + ': ' + f['detail'] for f in flags[:3]) if flags else 'None detected'}

Please produce a structured brief with:
1. ONE-SENTENCE THESIS (bull case)
2. KEY STRENGTHS (2-3 bullets)
3. KEY RISKS (2-3 bullets, be specific)
4. SETUP VERDICT: Rate the current setup as BUY / HOLD / AVOID / WATCH and explain in 1-2 sentences
5. KEY MONITORABLES: 2 specific metrics to watch that would confirm or break the thesis

Be direct. No fluff. If the data is weak, say so."""

    result = ask_claude(prompt, system=BRIEF_SYSTEM, max_tokens=600, ollama_model=OLLAMA_MODEL_STOCK_BRIEF)
    if result:
        logger.info(f"Stock brief generated for {ticker} ({len(result)} chars)")
    else:
        logger.warning(f"Stock brief generation returned no result for {ticker}")
    return result


def generate_options_brief(
    ticker: str,
    current_price: float,
    iv_metrics: Dict[str, Any],
    regime: Dict[str, Any],
) -> Optional[str]:
    """
    Generate an AI options strategy recommendation.
    """
    prompt = f"""Options strategy brief for {ticker} at ${current_price:.2f}.

IV METRICS:
- IV Rank: {iv_metrics.get('iv_rank', 'N/A')}th percentile
- IV Percentile: {iv_metrics.get('iv_percentile', 'N/A')}
- HV 21-day: {iv_metrics.get('hv_21', 'N/A')}%
- HV 10-day: {iv_metrics.get('hv_10', 'N/A')}%
- ATM IV: {iv_metrics.get('atm_iv', 'N/A')}%
- IV/RV Ratio: {iv_metrics.get('iv_rv_ratio', 'N/A')}
- Vol Regime: {iv_metrics.get('vol_regime', 'N/A')}
- Term Structure: {iv_metrics.get('term_structure', 'N/A')}

MARKET REGIME: {regime.get('regime', 'N/A')}
TREND: {regime.get('trend', 'N/A')}
RSI: {regime.get('rsi', 'N/A')}

Provide:
1. RECOMMENDED STRATEGY (name it specifically: e.g. Bull Put Spread, Iron Condor, Long Call, etc.)
2. RATIONALE (2 sentences — why this strategy fits the IV environment and trend)
3. SETUP PARAMETERS (approximate strikes, expiry timeframe, position sizing guidance)
4. KEY RISK (what would hurt this trade)

Be specific and concise."""

    result = ask_claude(prompt, system=BRIEF_SYSTEM, max_tokens=400, ollama_model=OLLAMA_MODEL_OPTIONS_BRIEF)
    if result:
        logger.info(f"Options brief generated for {ticker} ({len(result)} chars)")
    else:
        logger.warning(f"Options brief generation returned no result for {ticker}")
    return result


def generate_daytrading_brief(
    ticker: str,
    current_price: float,
    signals: Dict[str, Any],
    key_levels: Dict[str, Any],
) -> Optional[str]:
    """
    Generate an AI intraday read synthesizing the four rule-based Day Trading signals.
    """
    prompt = f"""Intraday trading brief for {ticker} at ${current_price:.2f}.

MARKET CONTEXT: {signals.get('market_status', 'N/A')} | VIX: {signals.get('vix', 'N/A')} ({signals.get('vix_regime', 'N/A')}) | S&P Regime: {signals.get('sp_regime', 'N/A')}

VWAP SIGNAL: {signals.get('vwap_note', 'N/A')} ({signals.get('vwap_dev', 0):+.1f}% vs VWAP)
MOMENTUM: {signals.get('mom_note', 'N/A')}
VOLUME: {signals.get('vol_note', 'N/A')}
TREND ALIGNMENT: {signals.get('ta_note', 'N/A')}

KEY LEVELS: Pivot ${key_levels.get('pivot', 0):.2f} | R1 ${key_levels.get('r1', 0):.2f} | S1 ${key_levels.get('s1', 0):.2f}
OPENING RANGE: {key_levels.get('or_note', 'Not available')}

Provide:
1. PLAIN-ENGLISH READ (2-3 sentences synthesizing the four signals into one intraday narrative — do they agree or conflict?)
2. BIAS FOR THE SESSION: LONG / SHORT / NEUTRAL, with the one signal that would change your mind
3. LEVELS TO WATCH (which key level matters most right now and why)
4. KEY RISK (what would invalidate this read intraday — e.g. VIX spike, volume drying up, VWAP reclaim)

Be direct and concise. This is intraday context, not a multi-day thesis."""

    result = ask_claude(prompt, system=BRIEF_SYSTEM, max_tokens=350, ollama_model=OLLAMA_MODEL_DAYTRADING_BRIEF)
    if result:
        logger.info(f"Day trading brief generated for {ticker} ({len(result)} chars)")
    else:
        logger.warning(f"Day trading brief generation returned no result for {ticker}")
    return result


def generate_thesis_prompt(ticker: str, name: str) -> Optional[str]:
    """
    Generate structured investment thesis prompts to guide user thinking.
    """
    prompt = f"""For {ticker} ({name}), generate 5 structured questions that an investor should answer before investing.

Format as a numbered list. Each question should:
- Be specific to the business (not generic)
- Target a key assumption that drives the investment thesis
- Have a clear "yes/no + evidence" format

Also include: What is the ONE number to watch quarterly that would confirm or invalidate a bull thesis?"""

    result = ask_claude(prompt, system=BRIEF_SYSTEM, max_tokens=350, ollama_model=OLLAMA_MODEL_THESIS)
    if result:
        logger.info(f"Thesis prompt generated for {ticker} ({len(result)} chars)")
    else:
        logger.warning(f"Thesis prompt generation returned no result for {ticker}")
    return result


def format_ai_markdown(text: str) -> str:
    """
    Escape literal '$' before handing brief text to st.markdown().
    Streamlit renders anything between two unescaped '$' as inline LaTeX —
    a brief that mentions two dollar prices in one paragraph (e.g. a pivot
    and a level to watch) gets its whole in-between text swallowed into one
    garbled math expression otherwise.
    """
    return text.replace("$", "\\$")


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return f"{f*100:.1f}%" if abs(f) < 1 else f"{f:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct_val(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return "N/A"
