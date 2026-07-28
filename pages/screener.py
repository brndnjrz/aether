"""
Stock Screener — empirically-backed screens:
Quality + Momentum, Earnings Revision + Momentum, 52w-Low Quality, Insider Buying proxy.
"""
import logging
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_price_history, get_financials, get_ticker_info
from analysis.indicators import calculate_indicators, get_signal_summary
from analysis.fundamental_score import full_fundamental_report

logger = logging.getLogger(__name__)

# ML prediction — only used when model already exists (never trains in screener loop)
try:
    from analysis.ml_prediction import predict as ml_predict
    from pathlib import Path
    from config.settings import STORAGE_DIR as _ML_STORAGE_DIR
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
    logger.debug("[screener] ML prediction module unavailable — ML Signal column will show placeholders")


SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "MRK",
    "AVGO", "CVX", "PEP", "ABBV", "COST", "ADBE", "CRM", "AMD", "NFLX",
    "TMO", "ACN", "DHR", "CSCO", "WMT", "DIS", "MCD", "NKE", "TXN",
    "QCOM", "BMY", "INTC", "NEE", "HON", "AMGN", "RTX", "LOW", "IBM",
    "SBUX", "GS", "BAC", "SPGI",
]

# Small Account Options screen — price/cap/liquidity band for options-affordable,
# options-liquid names. Same four filters as a manual TradingView screen:
# price band keeps contracts affordable, market cap + volume floors keep the
# underlying liquid enough that its options won't have punishing bid/ask spreads.
SMALL_ACCOUNT_PRICE_MIN = 30
SMALL_ACCOUNT_PRICE_MAX = 80
SMALL_ACCOUNT_MIN_MARKET_CAP = 2e9
SMALL_ACCOUNT_MIN_AVG_VOLUME = 2_000_000


def render():
    st.title("Stock Screener")
    st.caption("Empirically-backed screens. Quality filters prevent value traps.")

    screen_type = st.selectbox("Screen", [
        "Quality + Momentum (most robust combined factor)",
        "Oversold Quality (mean-reversion setup)",
        "High IV Rank (options premium selling candidates)",
        "Small Account Options (price $30-$80, cap $2B+, liquid enough to trade options)",
        "Custom Watchlist Screen",
    ])

    col1, col2 = st.columns([3, 1])
    with col2:
        custom_list = st.text_area("Custom tickers (one per line or comma-sep)", height=100)
        universe = [t.strip().upper() for t in custom_list.replace(",", "\n").split("\n") if t.strip()]

    with col1:
        if not universe:
            use_sample = st.checkbox("Use S&P 500 sample (50 stocks)", value=True)
            if use_sample:
                universe = SP500_SAMPLE
                st.caption(f"Screening {len(universe)} stocks...")

    if not universe:
        logger.warning("[screener] Run blocked: no tickers entered and sample universe not enabled")
        st.warning("Enter tickers or enable the sample universe")
        return

    if st.button("Run Screen", type="primary"):
        logger.info(f"[screener] 'Run Screen' button pressed: screen={screen_type!r} universe_size={len(universe)}")
        _run_screen(screen_type, universe)


def _run_screen(screen_type: str, universe: list):
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(universe):
        progress.progress((i + 1) / len(universe))
        status.text(f"Screening {ticker} ({i+1}/{len(universe)})...")
        try:
            row = _screen_ticker(ticker, screen_type)
            if row:
                results.append(row)
        except Exception as exc:
            logger.debug(f"[screener] Skipping {ticker} in screen — error during evaluation: {exc}")
            continue

    progress.empty()
    status.empty()

    if not results:
        logger.warning(f"[screener] Screen {screen_type!r} completed: no stocks passed out of {len(universe)} screened")
        st.warning("No stocks passed the screen criteria")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False)

    logger.info(f"[screener] Screen {screen_type!r} completed: {len(df)}/{len(universe)} stocks passed")
    st.success(f"✅ {len(df)} stocks passed the screen out of {len(universe)}")
    st.dataframe(df, hide_index=True, width="stretch")

    st.markdown("---")
    st.caption("""
    **Screen methodology:**
    - Quality + Momentum: ROIC > 12% + positive 3-month price momentum + ADX > 20
    - Oversold Quality: Quality score > 50 + RSI < 40 + above 200MA
    - High IV Rank: IV Rank > 50 (premium selling opportunity)
    - Small Account Options: price $30-$80 (affordable contracts on a <$20K account), market cap $2B+ (established names), 60-day avg volume 2M+ shares (liquid enough for tight options spreads) — sorted by liquidity, highest volume first
    - All screens require positive FCF or gross margin > 30%
    """)


def _screen_ticker(ticker: str, screen_type: str) -> dict:
    df_raw = get_price_history(ticker, period="1y")
    if df_raw is None or df_raw.empty or len(df_raw) < 30:
        return None

    df = calculate_indicators(df_raw)
    signals = get_signal_summary(df)
    f = get_financials(ticker)
    info = get_ticker_info(ticker)

    current = float(df["Close"].iloc[-1])
    price_3m = (current - float(df["Close"].iloc[-63])) / float(df["Close"].iloc[-63]) * 100 if len(df) >= 63 else 0
    price_1m = (current - float(df["Close"].iloc[-22])) / float(df["Close"].iloc[-22]) * 100 if len(df) >= 22 else 0

    rsi = signals.get("rsi", 50)
    adx = signals.get("adx", 15)
    above_200 = signals.get("above_200ma", False)
    roic = f.get("roic_proxy") or f.get("return_on_equity")
    fcf_yield = f.get("fcf_yield", 0) or 0
    gross_margin = (f.get("gross_margin_curr") or f.get("gross_margin") or 0)
    if gross_margin < 1:
        gross_margin *= 100
    rev_growth = (f.get("revenue_growth_yoy") or f.get("revenue_growth") or 0)
    if abs(rev_growth) < 1:
        rev_growth *= 100
    mkt_cap = f.get("market_cap")
    avg_vol_60d = float(df_raw["Volume"].tail(60).mean())

    # ── Quality gate: must pass basic quality filter ─────────────────────
    quality_ok = (
        (roic is not None and (roic * 100 if abs(roic) < 1 else roic) > 6)
        or gross_margin > 30
        or (fcf_yield > 0)
    )

    score = 0
    passes = False
    ivr = None

    if "Quality + Momentum" in screen_type:
        roic_num = (roic * 100 if roic and abs(roic) < 1 else roic) or 0
        if roic_num > 12 and price_3m > 0 and adx > 20 and quality_ok:
            passes = True
            score = roic_num * 0.4 + price_3m * 0.6

    elif "Oversold Quality" in screen_type:
        if quality_ok and rsi < 40 and above_200:
            passes = True
            score = (40 - rsi) + (gross_margin * 0.5)

    elif "High IV Rank" in screen_type:
        from data.options_data import calculate_iv_rank
        iv = calculate_iv_rank(ticker)
        ivr = iv.get("iv_rank", 50)
        if ivr > 50 and quality_ok:
            passes = True
            score = ivr

    elif "Small Account Options" in screen_type:
        price_ok = SMALL_ACCOUNT_PRICE_MIN <= current <= SMALL_ACCOUNT_PRICE_MAX
        cap_ok = mkt_cap is not None and mkt_cap >= SMALL_ACCOUNT_MIN_MARKET_CAP
        vol_ok = avg_vol_60d >= SMALL_ACCOUNT_MIN_AVG_VOLUME
        if price_ok and cap_ok and vol_ok and quality_ok:
            passes = True
            score = avg_vol_60d / 1e6
            from data.options_data import calculate_iv_rank
            try:
                ivr = calculate_iv_rank(ticker).get("iv_rank")
            except Exception:
                ivr = None

    elif "Custom" in screen_type:
        if quality_ok:
            passes = True
            score = (price_3m + price_1m) / 2

    if not passes:
        return None

    cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap and mkt_cap > 1e12 else (f"${mkt_cap/1e9:.1f}B" if mkt_cap else "—")

    # ── ML Signal — only if a pre-trained model exists (never train here) ──────
    ml_signal = "—"
    ml_prob = "—"
    if _ML_AVAILABLE:
        try:
            xgb_file = Path(_ML_STORAGE_DIR) / f"{ticker}_xgb.pkl"
            if xgb_file.exists():
                ml_result = ml_predict(ticker, df)
                if not ml_result.get("error"):
                    sig = ml_result.get("direction", "neutral").upper()
                    prob = ml_result.get("probability", 0.50)
                    ml_signal = sig
                    ml_prob = f"{prob * 100:.0f}%"
        except Exception:
            pass

    return {
        "Ticker": ticker,
        "Name": info.get("longName", ticker)[:25],
        "Sector": f.get("sector", "—")[:15],
        "Price": f"${current:.2f}",
        "1M %": f"{price_1m:+.1f}%",
        "3M %": f"{price_3m:+.1f}%",
        "RSI": round(rsi, 1),
        "ADX": round(adx, 1),
        "Gross Margin %": f"{gross_margin:.1f}%",
        "Rev Growth %": f"{rev_growth:+.1f}%",
        "FCF Yield %": f"{fcf_yield*100:.1f}%" if fcf_yield else "—",
        "Mkt Cap": cap_str,
        "Avg Vol (60d)": f"{avg_vol_60d/1e6:.1f}M",
        "IV Rank": f"{ivr:.0f}" if ivr is not None else "—",
        "Score": round(score, 1),
        "200MA": "✅" if above_200 else "❌",
        "ML Signal": ml_signal,
        "Bull Prob": ml_prob,
    }


render()
