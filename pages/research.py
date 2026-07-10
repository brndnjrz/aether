"""
Stock Research page — single stock deep dive.
Workflow enforced: Business context → Financials → Technical → AI Brief → Options
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_price_history, get_financials, get_ticker_info, get_earnings_history
from analysis import (
    calculate_indicators, get_signal_summary, detect_support_resistance, detect_regime, full_fundamental_report,
    detect_recent_trendlines, detect_swing_points,
)
from data.options_data import calculate_iv_rank
from ai.stock_brief import generate_stock_brief, generate_thesis_prompt, format_ai_markdown

# ML prediction — optional import so page still works if xgboost not installed
try:
    from analysis.ml_prediction import predict as ml_predict, train_model as ml_train
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

# News sentiment — optional import so page still works if feedparser/vaderSentiment not installed
try:
    from data.news_data import fetch_ticker_news
    from analysis.sentiment import analyze_ticker_sentiment
    _NEWS_AVAILABLE = True
except Exception:
    _NEWS_AVAILABLE = False


def render():
    st.title("Stock Research")
    st.caption("Workflow: Business → Financials → Technicals → AI Brief")

    if "quick_lookup_ticker" in st.session_state:
        st.session_state["research_ticker_input"] = st.session_state.pop("quick_lookup_ticker")

    col_input, col_period = st.columns([3, 1])
    with col_input:
        ticker = st.text_input(
            "Ticker", value="SPY", placeholder="e.g. AAPL, NVDA, MSFT",
            key="research_ticker_input",
        ).upper().strip()
    with col_period:
        period = st.selectbox("Period", ["1y", "2y", "6mo", "3mo"], index=0)

    if not ticker:
        return

    with st.spinner(f"Loading {ticker}..."):
        df_raw = get_price_history(ticker, period=period)
        info = get_ticker_info(ticker)
        fund_report = full_fundamental_report(ticker)

    if df_raw is None or df_raw.empty:
        st.error(f"No data found for {ticker}")
        return

    df = calculate_indicators(df_raw)
    signals = get_signal_summary(df)
    regime = detect_regime(df, ticker)
    current_price = float(df["Close"].iloc[-1])
    price_change_pct = float((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100)

    # ── Header ────────────────────────────────────────────────────────────
    name = info.get("longName") or fund_report.get("name", ticker)
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    mkt_cap = info.get("marketCap")
    cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap and mkt_cap > 1e12 else (f"${mkt_cap/1e9:.1f}B" if mkt_cap else "")

    st.markdown(f"## {name} ({ticker})")
    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
    hcol2.metric("Market Cap", cap_str or "N/A")
    hcol3.metric("Sector", sector or "N/A")
    hcol4.metric("Market Regime", regime.get("market_regime", "N/A"))

    # ── Scorecard ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Fundamental Scorecard")
    sc1, sc2, sc3, sc4 = st.columns(4)
    q = fund_report.get("quality", {}).get("score", 0)
    v = fund_report.get("value", {}).get("score", 0)
    g = fund_report.get("growth", {}).get("score", 0)
    comp = fund_report.get("composite_score", 0)
    verdict = fund_report.get("verdict", "")

    def score_color(s):
        if s >= 70: return "🟢"
        if s >= 50: return "🟡"
        if s >= 35: return "🟠"
        return "🔴"

    sc1.metric(f"{score_color(q)} Quality", f"{q}/100")
    sc2.metric(f"{score_color(v)} Value", f"{v}/100")
    sc3.metric(f"{score_color(g)} Growth", f"{g}/100")
    sc4.metric(f"{'🟢' if comp >= 60 else '🟡' if comp >= 40 else '🔴'} Composite", f"{comp}/100 — {verdict}")

    # Red flags
    flags = fund_report.get("red_flags", [])
    if flags:
        with st.expander(f"⚠️ {len(flags)} Red Flag(s) Detected", expanded=True):
            for flag in flags:
                color = "🔴" if flag["severity"] == "danger" else "🟡"
                st.markdown(f"{color} **{flag['flag']}** — {flag['detail']}")

    # ── ML Direction Signal ───────────────────────────────────────────────────
    if _ML_AVAILABLE:
        ml_cache_key = f"ml_signal_{ticker}_{period}"
        if ml_cache_key not in st.session_state:
            with st.spinner("Running ML direction model..."):
                try:
                    ml_result = ml_predict(ticker, df)
                    st.session_state[ml_cache_key] = ml_result
                except Exception as _ml_exc:
                    st.session_state[ml_cache_key] = {"error": str(_ml_exc)}

        ml_result = st.session_state.get(ml_cache_key, {})

        st.markdown("---")
        st.subheader(f"ML Direction Signal ({ml_result.get('horizon_days') or 5}-Day Forecast)")
        ml_cols = st.columns(4)

        if ml_result and not ml_result.get("error"):
            sig = ml_result.get("direction", "neutral").upper()
            prob = ml_result.get("probability", 0.50)
            conf = ml_result.get("confidence", "low").capitalize()
            acc = ml_result.get("model_accuracy") or 0.0
            exp_move = ml_result.get("expected_move_pct")

            sig_icon = "🟢" if sig == "BULLISH" else ("🔴" if sig == "BEARISH" else "⚪")

            ml_cols[0].metric(
                f"{sig_icon} ML Signal",
                sig,
                f"Confidence: {conf}",
            )
            ml_cols[1].metric(
                "Bull Probability",
                f"{prob * 100:.0f}%",
                f"Neutral zone: 45–55%",
            )
            ml_cols[2].metric(
                "Model Accuracy",
                f"{acc * 100:.1f}%" if acc else "N/A",
                "Walk-forward validation",
            )
            ml_cols[3].metric(
                "Expected Move",
                f"{exp_move:+.2f}%" if exp_move is not None else "N/A",
                "Median historical (similar setups)",
            )

            if ml_result.get("top_features"):
                with st.expander("Top Feature Drivers", expanded=False):
                    feat_items = sorted(
                        ml_result["top_features"].items(), key=lambda x: x[1], reverse=True
                    )
                    for feat_name, importance in feat_items[:5]:
                        bar = "█" * int(importance * 100) + "░" * (10 - int(importance * 100))
                        st.write(f"`{bar}` **{feat_name}**: {importance:.3f}")

            if not ml_result.get("last_trained"):
                st.caption("Model trained this session.")
            else:
                st.caption(f"Model trained: {ml_result['last_trained'][:10]}")

        elif ml_result.get("error"):
            ml_cols[0].warning(f"ML signal unavailable: {ml_result['error'][:80]}")
        else:
            ml_cols[0].info("ML model loading...")

    # ── Price Chart ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Price & Technical Analysis")

    ind_tab, fund_tab, options_tab, news_tab, ai_tab = st.tabs(
        ["📈 Chart & Technicals", "📊 Fundamentals", "🎯 Options", "📰 News & Sentiment", "🤖 AI Brief"]
    )

    with ind_tab:
        sr = detect_support_resistance(df)
        trendlines = detect_recent_trendlines(df)
        swings = detect_swing_points(df)
        _render_price_chart(df, ticker, sr, trendlines, swings)
        _render_indicator_panel(df, signals, regime)

    with fund_tab:
        _render_fundamentals(fund_report, df_raw)

    with options_tab:
        _render_options_tab(ticker, current_price, regime)

    with news_tab:
        _render_news_tab(ticker, info)

    with ai_tab:
        _render_ai_brief(ticker, fund_report, signals, regime)


def _render_price_chart(df: pd.DataFrame, ticker: str, sr: dict, trendlines: Optional[dict] = None, swings: Optional[list] = None):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Moving averages
    colors = {"SMA_20": "#ff9800", "SMA_50": "#2196f3", "SMA_200": "#9c27b0"}
    for ma, color in colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color, width=1)), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                                  line=dict(color="rgba(128,128,128,0.5)", dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                                  line=dict(color="rgba(128,128,128,0.5)", dash="dash"),
                                  fill="tonexty", fillcolor="rgba(128,128,128,0.1)", showlegend=False), row=1, col=1)

    # Support/Resistance
    for level in sr.get("support", []):
        fig.add_hline(y=level, line_dash="dot", line_color="#26a69a", line_width=1, row=1, col=1)
    for level in sr.get("resistance", []):
        fig.add_hline(y=level, line_dash="dot", line_color="#ef5350", line_width=1, row=1, col=1)

    # Trendlines (fitted support/resistance, projected one bar forward)
    if trendlines:
        fig.add_trace(go.Scatter(
            x=trendlines["index"], y=trendlines["support_line"], name="Support Line",
            line=dict(color="#00e676", width=2, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trendlines["index"], y=trendlines["resist_line"], name="Resistance Line",
            line=dict(color="#ff1744", width=2, dash="dash"),
        ), row=1, col=1)

    # Swing points
    if swings:
        highs = [s for s in swings if s["type"] == "high"]
        lows = [s for s in swings if s["type"] == "low"]
        if highs:
            fig.add_trace(go.Scatter(
                x=[s["timestamp"] for s in highs], y=[s["price"] for s in highs], name="Swing High",
                mode="markers", marker=dict(symbol="triangle-down", size=9, color="#ef5350"),
            ), row=1, col=1)
        if lows:
            fig.add_trace(go.Scatter(
                x=[s["timestamp"] for s in lows], y=[s["price"] for s in lows], name="Swing Low",
                mode="markers", marker=dict(symbol="triangle-up", size=9, color="#26a69a"),
            ), row=1, col=1)

    # Volume
    colors_vol = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors_vol, opacity=0.7), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff9800")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_indicator_panel(df: pd.DataFrame, signals: dict, regime: dict):
    cols = st.columns(4)
    last = df.iloc[-1]

    with cols[0]:
        st.markdown("**Trend**")
        st.write(f"Regime: **{regime.get('regime', 'N/A')}**")
        st.write(f"21d Return: {regime.get('price_21d_pct', 0):+.1f}%")
        st.write(f"200MA: {'✅ Above' if signals.get('above_200ma') else '❌ Below'}")
        st.write(f"50MA: {'✅ Above' if signals.get('above_50ma') else '❌ Below'}")

    with cols[1]:
        st.markdown("**Momentum**")
        rsi = signals.get("rsi", 50)
        rsi_zone = signals.get("rsi_zone", "neutral")
        rsi_color = "🔴" if rsi_zone == "overbought" else ("🟢" if rsi_zone == "oversold" else "⚪")
        st.write(f"RSI: {rsi_color} **{rsi}** ({rsi_zone})")
        adx = signals.get("adx", 0)
        st.write(f"ADX: **{adx}** ({'Strong' if adx > 25 else 'Weak'} trend)")
        st.write(f"MACD: {'📈 Bullish' if signals.get('macd_bullish') else '📉 Bearish'}")
        div = signals.get("rsi_divergence")
        if div:
            st.write(f"Divergence: ⚠️ **{div.replace('_', ' ').title()}**")

    with cols[2]:
        st.markdown("**Volatility**")
        atr_pct = signals.get("atr_pct", 0)
        st.write(f"ATR: **{atr_pct:.2f}%** of price")
        bb_pct = signals.get("bb_pct")
        if bb_pct is not None:
            bb_str = f"{bb_pct*100:.0f}%" if bb_pct <= 1 else ">100%"
            st.write(f"BB Position: **{bb_str}** (0%=bottom, 100%=top)")
        vol_regime = regime.get("vol_regime", "Normal")
        st.write(f"Vol Regime: **{vol_regime}**")

    with cols[3]:
        st.markdown("**Volume**")
        vol_surge = signals.get("volume_surge", False)
        st.write(f"Volume Surge: {'⚡ Yes' if vol_surge else 'No'}")
        if "vol_ratio" in df.columns:
            vr = float(df["vol_ratio"].iloc[-1])
            st.write(f"Vol Ratio: **{vr:.2f}x** avg")
        if "OBV" in df.columns:
            obv_trend = "Rising" if df["OBV"].iloc[-1] > df["OBV"].iloc[-10] else "Falling"
            st.write(f"OBV Trend: **{obv_trend}**")


def _render_fundamentals(report: dict, df: pd.DataFrame):
    f = report.get("raw", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profitability**")
        _metric_row("Gross Margin", f.get("gross_margin_curr") or f.get("gross_margin"), pct=True)
        _metric_row("Operating Margin", f.get("operating_margin"), pct=True)
        _metric_row("Net Margin", f.get("profit_margin"), pct=True)
        _metric_row("ROE", f.get("return_on_equity"), pct=True)
        _metric_row("ROIC (proxy)", f.get("roic_proxy"), pct=True)

    with col2:
        st.markdown("**Growth & FCF**")
        _metric_row("Revenue Growth YoY", f.get("revenue_growth_yoy") or f.get("revenue_growth"), pct=True)
        _metric_row("EPS Growth", f.get("earnings_growth"), pct=True)
        _metric_row("FCF Yield", f.get("fcf_yield"), pct=True)
        if f.get("fcf"):
            fcf_m = f["fcf"] / 1e6
            st.write(f"FCF: **${fcf_m:.0f}M**")

    with col3:
        st.markdown("**Valuation**")
        _metric_row("EV/EBITDA", f.get("ev_ebitda"), suffix="x")
        _metric_row("P/FCF", f.get("price_to_fcf"), suffix="x")
        _metric_row("P/E (TTM)", f.get("pe_ratio"), suffix="x")
        _metric_row("Forward P/E", f.get("forward_pe"), suffix="x")
        _metric_row("PEG Ratio", f.get("peg_ratio"), suffix="x")

    # Scorecard breakdown
    st.markdown("---")
    st.markdown("**Scoring Breakdown**")
    for category, data in [
        ("Quality", report.get("quality", {})),
        ("Value", report.get("value", {})),
        ("Growth", report.get("growth", {})),
    ]:
        breakdown = data.get("breakdown", {})
        if breakdown:
            score = data.get("score", 0)
            with st.expander(f"{category} Score: {score}/100"):
                for metric, details in breakdown.items():
                    pts = details.get("pts", 0)
                    max_pts = details.get("max", 0)
                    val = details.get("value")
                    bar = "█" * int(pts / max_pts * 10) + "░" * (10 - int(pts / max_pts * 10)) if max_pts > 0 else ""
                    val_str = f"{val:.1f}%" if isinstance(val, float) else str(val) if val is not None else "N/A"
                    st.write(f"`{bar}` {metric.replace('_', ' ').title()}: {val_str} — {pts}/{max_pts} pts")


def _render_options_tab(ticker: str, price: float, regime: dict):
    st.subheader(f"Options Analysis — {ticker}")
    with st.spinner("Fetching real options data..."):
        iv_metrics = calculate_iv_rank(ticker)

    if iv_metrics.get("status") == "insufficient_data":
        st.warning("Insufficient price history for IV calculation")
        return

    col1, col2, col3, col4 = st.columns(4)
    ivr = iv_metrics.get("iv_rank", 50)
    ivr_color = "🔴" if ivr > 60 else ("🟢" if ivr < 30 else "🟡")
    col1.metric(f"{ivr_color} IV Rank", f"{ivr:.0f}th %ile", help="High IVR > 60 = premium rich = prefer selling")
    col2.metric("IV Percentile", f"{iv_metrics.get('iv_percentile', 50):.0f}%")
    col3.metric("HV 21-day", f"{iv_metrics.get('hv_21', 0):.1f}%", help="30-day historical volatility")
    col4.metric("Vol Regime", iv_metrics.get("vol_regime", "N/A"))

    col5, col6 = st.columns(2)
    atm_iv = iv_metrics.get("atm_iv")
    iv_rv_ratio = iv_metrics.get("iv_rv_ratio")
    col5.metric("ATM IV (real)", f"{atm_iv:.1f}%" if atm_iv else "N/A", help="Real implied volatility from options chain")
    col6.metric("IV/RV Ratio", f"{iv_rv_ratio:.2f}x" if iv_rv_ratio else "N/A",
                help=">1.15 = options potentially overpriced (premium selling edge)")

    st.markdown("---")
    # Strategy recommendation based on IVR and trend
    trend = regime.get("trend", "Sideways")
    st.markdown("**Strategy Recommendation**")
    if ivr > 60 and trend == "Sideways":
        st.success("🏆 **Iron Condor** — High IV rank + range-bound = ideal premium selling conditions")
        st.write("Sell OTM call spread + OTM put spread. Target 1-2% portfolio risk. 30-45 DTE.")
    elif ivr > 60 and trend == "Uptrend":
        st.success("✅ **Bull Put Spread** — High IV rank + uptrend = sell put credit spread")
        st.write("Sell ATM put, buy further OTM put. Collect premium with directional edge.")
    elif ivr > 60 and trend == "Downtrend":
        st.success("✅ **Bear Call Spread** — High IV rank + downtrend = sell call credit spread")
    elif ivr < 30 and trend == "Uptrend":
        st.info("📈 **Long Call / Debit Spread** — Low IV = options cheap = favor buying premium")
        st.write("Buy ATM call or bull call spread. IV expansion can amplify gains.")
    elif ivr < 30 and trend == "Downtrend":
        st.info("📉 **Long Put / Debit Spread** — Low IV + downtrend = buy directional protection")
    else:
        st.warning(f"⚠️ **Mixed signals** — IV Rank {ivr:.0f} with {trend} trend. Consider covered call on existing position.")

    st.markdown("---")
    st.markdown("**IV Term Structure**")
    st.write(f"Term structure: **{iv_metrics.get('term_structure', 'Flat')}**")
    st.write(f"HV 10d: {iv_metrics.get('hv_10', 0):.1f}% | HV 21d: {iv_metrics.get('hv_21', 0):.1f}% | HV 63d: {iv_metrics.get('hv_63', 0):.1f}%")
    if iv_metrics.get("term_structure") == "Backwardation":
        st.warning("Short-term vol > long-term vol — market pricing in near-term event risk (earnings, news)")


def _render_news_tab(ticker: str, info: dict):
    st.subheader(f"News & Sentiment — {ticker}")
    if not _NEWS_AVAILABLE:
        st.info("News sentiment requires the `feedparser` and `vaderSentiment` packages. Run `pip install -r requirements.txt`.")
        return

    company_name = info.get("longName", "")
    with st.spinner("Fetching recent headlines..."):
        articles = fetch_ticker_news(ticker, company_name=company_name)
        sentiment = analyze_ticker_sentiment(articles)

    if sentiment["n_articles"] == 0:
        st.warning("No recent headlines found for this ticker.")
        return

    overall = sentiment["overall_label"]
    overall_icon = "🟢" if overall == "Bullish" else ("🔴" if overall == "Bearish" else "⚪")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{overall_icon} Overall Tone", overall, f"Compound: {sentiment['mean_compound']:+.3f}")
    col2.metric("Positive", f"{sentiment['positive_pct']:.0f}%")
    col3.metric("Neutral", f"{sentiment['neutral_pct']:.0f}%")
    col4.metric("Negative", f"{sentiment['negative_pct']:.0f}%")

    st.caption(
        f"Based on {sentiment['n_articles']} recent headlines from Google News. "
        "Headline-level sentiment only — not a trading signal on its own."
    )

    st.markdown("---")
    st.markdown("**Recent Headlines**")
    for article in sentiment["articles"]:
        label = article["label"]
        icon = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "⚪")
        st.markdown(f"{icon} [{article['title']}]({article['link']})")
        st.caption(f"{article.get('source', '')} · {article.get('published', '')} · compound {article['compound']:+.3f}")


def _render_ai_brief(ticker: str, fund_report: dict, signals: dict, regime: dict):
    st.subheader(f"AI Investment Brief — {ticker}")
    from config.settings import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        st.info("Add your ANTHROPIC_API_KEY to .env to enable AI features")
        st.code("ANTHROPIC_API_KEY=sk-ant-...")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Stock Brief", type="primary"):
            with st.spinner("Generating AI analysis..."):
                brief = generate_stock_brief(ticker, fund_report, signals, regime, fund_report)
            if brief:
                st.markdown(format_ai_markdown(brief))
            else:
                st.error("AI generation failed")

    with col2:
        if st.button("Generate Thesis Questions"):
            name = fund_report.get("name", ticker)
            with st.spinner("Generating thesis prompts..."):
                questions = generate_thesis_prompt(ticker, name)
            if questions:
                st.markdown(questions)


def _metric_row(label: str, value, pct: bool = False, suffix: str = ""):
    if value is None:
        st.write(f"{label}: **N/A**")
        return
    try:
        v = float(value)
        if pct:
            v_pct = v * 100 if abs(v) < 1 else v
            color = "green" if v_pct > 0 else "red"
            st.write(f"{label}: **:{color}[{v_pct:.1f}%]**")
        else:
            st.write(f"{label}: **{v:.2f}{suffix}**")
    except (TypeError, ValueError):
        st.write(f"{label}: **{value}**")


render()
