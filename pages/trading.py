"""
Trading Desk — Day Trading signals, Options analysis, and ML Predictions in one page.
"""
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.price_data import get_price_history, get_current_price
from analysis.indicators import calculate_indicators, get_signal_summary
from data.macro_data import get_vix_data, get_sp500_regime
from analysis.risk import position_size_from_stop, regime_kelly_multiplier
from analysis.regime_markov import analyze_regime_markov
from data.options_data import get_options_chain, calculate_iv_rank, get_atm_greeks, build_pnl_diagram
from analysis.regime import detect_regime
from ai.stock_brief import generate_options_brief, generate_daytrading_brief, format_ai_markdown
from ai.client import ai_available
from analysis.patterns import detect_candlestick_pattern
from analysis.trendlines import detect_recent_trendlines, detect_swing_points
from analysis.flag_pennant_detection import detect_flag_pennant_patterns
from analysis.backtest import macd_bullish_cross_signal, simulate_trades
from analysis.price_projection import simulate_price_path
from config.tz import now_et, utc_iso_to_et_str, MARKET_TZ
from portfolio.activity_log import log_activity

logger = logging.getLogger(__name__)

# ML prediction — optional import so page still works if scikit-learn/xgboost/narwhals not installed
try:
    from analysis.ml_prediction import predict, train_model, get_prediction_history, evaluate_model
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
    logger.debug("[trading] ML prediction module unavailable — Predictions tab disabled")

# News sentiment — optional import so page still works if feedparser/vaderSentiment not installed
try:
    from data.news_data import fetch_ticker_news
    from analysis.sentiment import analyze_ticker_sentiment
    _NEWS_AVAILABLE = True
except Exception:
    _NEWS_AVAILABLE = False
    logger.debug("[trading] News sentiment module unavailable — News tab disabled")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = _PROJECT_ROOT / "storage"

FEATURE_DISPLAY_NAMES = {
    "rsi_norm":        "RSI-14 (Normalized)",
    "rsi_5_norm":      "RSI-5 (Short Momentum)",
    "macd_hist_sign":  "MACD Histogram Sign",
    "adx_norm":        "ADX Trend Strength",
    "atr_pct":         "ATR % of Price",
    "bb_pct":          "Bollinger Band Position",
    "vol_ratio":       "Volume vs 20d Average",
    "price_vs_sma20":  "Price vs SMA-20",
    "price_vs_sma50":  "Price vs SMA-50",
    "ret_1d":          "1-Day Return",
    "ret_5d":          "5-Day Return (Lagged)",
    "ret_10d":         "10-Day Return",
    "hv_ratio":        "Volatility Compression (HV10/HV21)",
    "obv_slope":       "OBV Slope (5-Bar)",
    "stoch_k_norm":    "Stochastic %K",
    "day_of_week":     "Day of Week",
    "above_200ma":     "Above 200-Day MA",
    "hl_range_pct":    "Daily Bar Range %",
}

SIGNAL_CONFIG = {
    "BULLISH": {"color": "#26a69a", "bg_color": "rgba(38, 166, 154, 0.12)", "border": "#26a69a", "icon": "▲", "gauge_color": "#26a69a"},
    "BEARISH": {"color": "#ef5350", "bg_color": "rgba(239, 83, 80, 0.12)", "border": "#ef5350", "icon": "▼", "gauge_color": "#ef5350"},
    "NEUTRAL": {"color": "#9e9e9e", "bg_color": "rgba(158, 158, 158, 0.10)", "border": "#9e9e9e", "icon": "◆", "gauge_color": "#9e9e9e"},
}

CONFIDENCE_COLORS = {"HIGH": "#26a69a", "MODERATE": "#ff9800", "LOW": "#9e9e9e"}


# ══════════════════════════════════════════════════════════════════════════
# ── Day Trading ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _market_status() -> str:
    now = now_et()
    if now.weekday() >= 5:
        return "MARKET CLOSED (WEEKEND)"
    total = now.hour * 60 + now.minute
    if total < 570:
        return "PRE-MARKET"
    if total <= 960:
        return "MARKET OPEN"
    return "AFTER-HOURS"


def _signal_html(label: str, value: str, interpretation: str, direction: str) -> str:
    css = {"bull": "signal-bull", "bear": "signal-bear", "neutral": "signal-neutral"}.get(direction, "signal-neutral")
    icon = {"bull": "🟢", "bear": "🔴", "neutral": "🟡"}.get(direction, "⚪")
    return f"""
<div class="signal-card {css}">
  <strong>{icon} {label}</strong><br>
  <span class="signal-value">{value}</span><br>
  <span class="signal-note">{interpretation}</span>
</div>"""


@st.cache_data(ttl=60)
def _load_intraday(ticker: str, interval: str):
    period = "1d" if interval in ("5m", "15m") else "5d"
    df = get_price_history(ticker, period=period, interval=interval)
    if df is None or df.empty:
        logger.debug(f"[trading] Intraday load for {ticker} ({interval}) came back empty")
        return df
    df = calculate_indicators(df)
    logger.debug(f"[trading] Intraday load for {ticker} ({interval}) succeeded with {len(df)} rows")
    return df


@st.cache_data(ttl=300)
def _load_daily(ticker: str):
    df = get_price_history(ticker, period="1y", interval="1d")
    if df is None or df.empty:
        logger.debug(f"[trading] Daily load for {ticker} came back empty")
        return df
    df = calculate_indicators(df)
    logger.debug(f"[trading] Daily load for {ticker} succeeded with {len(df)} rows")
    return df


@st.cache_data(ttl=300)
def _load_backtest_df(ticker: str):
    df = get_price_history(ticker, period="2y", interval="1d")
    if df is None or df.empty:
        logger.debug(f"[trading] Backtest data load for {ticker} came back empty")
        return df
    df = calculate_indicators(df)
    logger.debug(f"[trading] Backtest data load for {ticker} succeeded with {len(df)} rows")
    return df


def _render_daytrading():
    st.subheader("Day Trading Dashboard")

    status = _market_status()
    status_class = "aeth-status-open" if status == "MARKET OPEN" else "aeth-status-closed"
    try:
        vix_data = get_vix_data()
        vix_val = vix_data.get("current", 20)
        vix_regime = vix_data.get("regime", "Normal")
        sp_regime = get_sp500_regime().get("regime", "Unknown")
    except Exception:
        vix_val, vix_regime, sp_regime = 20, "N/A", "N/A"

    st.markdown(
        f'<div class="aeth-status-strip">'
        f'<span class="{status_class}">{status}</span> &nbsp;|&nbsp; '
        f'VIX {vix_val:.1f} ({vix_regime}) &nbsp;|&nbsp; '
        f'S&P Regime: {sp_regime} &nbsp;|&nbsp; '
        f'⏰ {now_et().strftime("%I:%M %p ET")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "dt_lookup_ticker" in st.session_state:
        lookup = st.session_state.pop("dt_lookup_ticker")
        st.session_state["dt_ticker"] = lookup
        st.session_state[f"dt_{lookup}"] = True

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ticker = c1.text_input("Ticker", value="SPY", placeholder="AAPL, SPY, QQQ…", key="dt_ticker").upper().strip()
    interval = c2.selectbox("Interval", ["5m", "15m", "30m", "1h"], index=1, key="dt_interval")
    analyze = c3.button("Analyze", type="primary", width="stretch", key="dt_analyze")
    st.caption("Data refreshes every 60s for intraday intervals.")

    if not ticker:
        st.info("Enter a ticker above and click Analyze.")
        return

    if not analyze and f"dt_{ticker}" not in st.session_state:
        st.info(f"Click **Analyze** to load {ticker}.")
        return

    if analyze:
        logger.info(f"[trading] 'Analyze' button pressed for {ticker} ({interval})")

    with st.spinner(f"Loading {ticker}…"):
        intraday = _load_intraday(ticker, interval)
        daily = _load_daily(ticker)

    if daily is None or daily.empty:
        logger.warning(f"[trading] Day Trading: no daily data for {ticker} — showing error to user")
        st.error(f"No data for {ticker}. Check ticker symbol.")
        return

    st.session_state[f"dt_{ticker}"] = True

    has_intraday = intraday is not None and not intraday.empty
    last_d = daily.iloc[-1]
    prev_d = daily.iloc[-2] if len(daily) > 1 else last_d
    sig_row = intraday.iloc[-1] if has_intraday else last_d
    current_price = float(sig_row["Close"])

    st.markdown("### Signals")
    st.caption(
        f"VWAP / Momentum / Volume read from the {interval} bar (session-anchored VWAP)."
        if has_intraday else
        "VWAP / Momentum / Volume read from the daily bar — no intraday data available for this ticker/interval."
    )
    sig1, sig2, sig3, sig4, sig5 = st.columns(5)

    vwap = float(sig_row.get("VWAP", current_price))
    vwap_dev = (current_price - vwap) / vwap * 100
    if vwap_dev > 2:
        vwap_dir, vwap_note = "bear", "Extended above VWAP — mean reversion risk"
    elif vwap_dev < -2:
        vwap_dir, vwap_note = "bull", "Extended below VWAP — potential bounce zone"
    elif abs(vwap_dev) < 0.5:
        vwap_dir, vwap_note = "neutral", "At VWAP — watch for directional breakout"
    elif vwap_dev > 0:
        vwap_dir, vwap_note = "bull", "Above VWAP — buyers in control"
    else:
        vwap_dir, vwap_note = "bear", "Below VWAP — sellers in control"
    sig1.markdown(_signal_html("VWAP Signal", f"{vwap_dev:+.1f}% vs VWAP (${vwap:.2f})", vwap_note, vwap_dir), unsafe_allow_html=True)

    rsi = sig_row.get("RSI", 50)
    macd_bull = float(sig_row.get("MACD", 0)) > float(sig_row.get("MACD_signal", 0))
    ema20 = sig_row.get("EMA_20", current_price)
    above_ema20 = current_price > float(ema20)
    bull_count = sum([
        float(rsi) > 50 if rsi is not None else False,
        macd_bull,
        above_ema20,
    ])
    mom_dir = "bull" if bull_count >= 2 else ("bear" if bull_count == 0 else "neutral")
    mom_note = f"RSI {float(rsi):.0f} | MACD {'▲' if macd_bull else '▼'} | {'Above' if above_ema20 else 'Below'} EMA20"
    sig2.markdown(_signal_html("Momentum", f"{bull_count}/3 Bullish Signals", mom_note, mom_dir), unsafe_allow_html=True)

    vol_ratio = float(sig_row.get("vol_ratio", 1.0))
    if vol_ratio > 2.0:
        vol_dir, vol_note = "bull", "Strong institutional activity — trend is real"
    elif vol_ratio > 1.5:
        vol_dir, vol_note = "bull", "Above-average volume — move has conviction"
    elif vol_ratio < 0.5:
        vol_dir, vol_note = "bear", "Low volume — choppy, fade moves"
    else:
        vol_dir, vol_note = "neutral", "Average volume — no edge from volume alone"
    sig3.markdown(_signal_html("Volume", f"{vol_ratio:.1f}x Average", vol_note, vol_dir), unsafe_allow_html=True)

    sma20 = last_d.get("SMA_20", current_price)
    sma50 = last_d.get("SMA_50", current_price)
    sma200 = last_d.get("SMA_200", current_price)
    ema50 = last_d.get("EMA_50", current_price)
    alignment_checks = [
        current_price > float(sma20) if pd.notna(sma20) else False,
        current_price > float(sma50) if pd.notna(sma50) else False,
        current_price > float(sma200) if pd.notna(sma200) else False,
        current_price > float(ema50) if pd.notna(ema50) else False,
    ]
    score = sum(alignment_checks)
    if score == 4:
        ta_dir, ta_label = "bull", "All Timeframes Aligned ↑"
    elif score == 0:
        ta_dir, ta_label = "bear", "All Timeframes Aligned ↓"
    elif score >= 3:
        ta_dir, ta_label = "bull", f"{score}/4 Timeframes Bullish"
    elif score <= 1:
        ta_dir, ta_label = "bear", f"{score}/4 Timeframes Bullish"
    else:
        ta_dir, ta_label = "neutral", f"{score}/4 Mixed Alignment"
    labels = ["20MA", "50MA", "200MA", "EMA50"]
    ta_note = " | ".join(f"{'✅' if c else '❌'} {l}" for c, l in zip(alignment_checks, labels))
    sig4.markdown(_signal_html("Trend Alignment", ta_label, ta_note, ta_dir), unsafe_allow_html=True)
    st.caption("Trend Alignment always compares against daily MAs — it's the swing-trend backdrop regardless of the intraday interval selected above.")

    pattern_df = intraday if has_intraday else daily
    pattern = detect_candlestick_pattern(pattern_df)
    if pattern:
        sig5.markdown(_signal_html("Candlestick Pattern", pattern["name"], pattern["note"], pattern["direction"]), unsafe_allow_html=True)
    else:
        sig5.markdown(_signal_html("Candlestick Pattern", "No Clear Pattern", "No Doji/Engulfing/Inside Bar/NR4 on the current bar", "neutral"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Intraday Chart")

    chart_df = intraday if has_intraday else daily.tail(60)
    trendlines = detect_recent_trendlines(chart_df)
    swings = detect_swing_points(chart_df)

    st.markdown("### Chart Patterns")
    conf_col, _ = st.columns([1, 3])
    with conf_col:
        flag_confidence_threshold = st.slider(
            "Flag/Pennant confidence threshold", min_value=50, max_value=95, value=65, step=5,
            key="dt_flag_confidence_threshold",
            help="Minimum confidence score (0-100) for a detected continuation pattern to be shown.",
        )
    flag_result = detect_flag_pennant_patterns(chart_df, min_confidence=float(flag_confidence_threshold))
    flag_patterns = flag_result["patterns"]
    latest_flag_pattern = flag_result["latest_pattern"]

    if flag_patterns:
        shown_patterns = flag_patterns[-3:]
        chip_cols = st.columns(len(shown_patterns))
        for col, p in zip(chip_cols, shown_patterns):
            shape = "Pennant" if p["pennant"] else "Flag"
            label = f"{'Bull' if p['direction'] == 'bull' else 'Bear'} {shape}"
            rr = f"{p['reward'] / p['risk']:.1f}:1 R:R" if p.get("risk") else "breakout confirmed"
            col.markdown(
                _signal_html(label, f"{p['confidence_score']:.0f}/100", f"Breakout ${p['breakout_price']:.2f} · {rr}", p["direction"]),
                unsafe_allow_html=True,
            )
        if len(flag_patterns) > 3:
            with st.expander(f"{len(flag_patterns)} patterns detected — show all"):
                for p in flag_patterns:
                    shape = "Pennant" if p["pennant"] else "Flag"
                    st.write(f"{'Bull' if p['direction'] == 'bull' else 'Bear'} {shape} — confidence {p['confidence_score']:.0f}, breakout ${p['breakout_price']:.2f}")
    else:
        st.caption("No Flag/Pennant patterns detected above the confidence threshold in the current window.")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if "VWAP" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["VWAP"], name="VWAP", line=dict(color="#ff9800", width=2)), row=1, col=1)
    if "EMA_20" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_20"], name="EMA 20", line=dict(color="#5c6bc0", width=1.5)), row=1, col=1)
    if "BB_upper" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_upper"], name="BB Upper", line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_lower"], name="BB Lower", fill="tonexty", fillcolor="rgba(150,150,150,0.07)", line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)

    if trendlines:
        fig.add_trace(go.Scatter(
            x=trendlines["index"], y=trendlines["support_line"], name="Support Line",
            line=dict(color="#00e676", width=2, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trendlines["index"], y=trendlines["resist_line"], name="Resistance Line",
            line=dict(color="#ff1744", width=2, dash="dash"),
        ), row=1, col=1)

    if swings:
        swing_highs = [s for s in swings if s["type"] == "high"]
        swing_lows = [s for s in swings if s["type"] == "low"]
        if swing_highs:
            fig.add_trace(go.Scatter(
                x=[s["timestamp"] for s in swing_highs], y=[s["price"] for s in swing_highs], name="Swing High",
                mode="markers", marker=dict(symbol="triangle-down", size=9, color="#ef5350"),
            ), row=1, col=1)
        if swing_lows:
            fig.add_trace(go.Scatter(
                x=[s["timestamp"] for s in swing_lows], y=[s["price"] for s in swing_lows], name="Swing Low",
                mode="markers", marker=dict(symbol="triangle-up", size=9, color="#26a69a"),
            ), row=1, col=1)

    for p in flag_patterns:
        shape = "Pennant" if p["pennant"] else "Flag"
        if p["pennant"]:
            zone_color = "rgba(255,193,7,{a})"
        elif p["direction"] == "bull":
            zone_color = "rgba(38,166,154,{a})"
        else:
            zone_color = "rgba(239,83,80,{a})"
        alpha = 0.06 + p["confidence_score"] / 100 * 0.14
        border_color = "#26a69a" if p["direction"] == "bull" else "#ef5350"

        flag_idx = list(range(p["flag_start_index"], p["flag_end_index"] + 1))
        flag_ts = [chart_df.index[i] for i in flag_idx if i < len(chart_df)]
        x_offsets = np.arange(len(flag_idx))
        upper_vals = np.exp(p["upper_slope"] * x_offsets + p["upper_intercept"])
        lower_vals = np.exp(p["lower_slope"] * x_offsets + p["lower_intercept"])

        fig.add_trace(go.Scatter(
            x=flag_ts + flag_ts[::-1], y=list(upper_vals) + list(lower_vals[::-1]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=zone_color.format(a=alpha), name=f"{shape} zone", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=flag_ts, y=upper_vals, mode="lines", line=dict(color=border_color, width=1.5, dash="dash"),
            name=f"{shape} upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=flag_ts, y=lower_vals, mode="lines", line=dict(color=border_color, width=1.5, dash="dash"),
            name=f"{shape} lower", showlegend=False,
        ), row=1, col=1)

        if p["pole_base_index"] < len(chart_df) and p["pole_tip_index"] < len(chart_df):
            fig.add_trace(go.Scatter(
                x=[chart_df.index[p["pole_base_index"]], chart_df.index[p["pole_tip_index"]]],
                y=[p["pole_base_price"], p["pole_tip_price"]],
                mode="lines", line=dict(color=border_color, width=2, dash="dot"), name="Pole", showlegend=False,
            ), row=1, col=1)

        if p["breakout_index"] < len(chart_df):
            fig.add_trace(go.Scatter(
                x=[chart_df.index[p["breakout_index"]]], y=[p["breakout_price"]],
                mode="markers", marker=dict(
                    symbol="star" if p["confidence_score"] >= 80 else "diamond",
                    size=12, color=border_color, line=dict(color="white", width=1),
                ), name=f"{shape} Breakout", showlegend=False,
            ), row=1, col=1)


    prev_close = float(prev_d["Close"])
    fig.add_hline(y=prev_close, row=1, col=1, line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
                  annotation_text=f"Prev Close ${prev_close:.2f}", annotation_position="right")

    today_open = float(last_d["Open"]) if has_intraday else float(chart_df.iloc[0]["Open"])
    fig.add_hline(y=today_open, row=1, col=1, line=dict(color="rgba(255,235,59,0.5)", width=1, dash="dot"),
                  annotation_text=f"Open ${today_open:.2f}", annotation_position="right")

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume", marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=520, margin=dict(l=0, r=80, t=10, b=0),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    kl_col, or_col = st.columns([1, 1])

    with kl_col:
        st.markdown("### Key Levels")
        pivot = (float(prev_d["High"]) + float(prev_d["Low"]) + float(prev_d["Close"])) / 3
        r1 = 2 * pivot - float(prev_d["Low"])
        r2 = pivot + (float(prev_d["High"]) - float(prev_d["Low"]))
        s1 = 2 * pivot - float(prev_d["High"])
        s2 = pivot - (float(prev_d["High"]) - float(prev_d["Low"]))

        atr = float(last_d.get("ATR", 0))
        day_range_pct = (float(last_d["High"]) - float(last_d["Low"])) / max(float(last_d["Low"]), 0.01) * 100
        range_pos = ((current_price - float(last_d["Low"])) / max(float(last_d["High"]) - float(last_d["Low"]), 0.01)) * 100

        levels_data = [
            ("R2", f"${r2:.2f}", "aeth-badge--bear"),
            ("R1", f"${r1:.2f}", "aeth-badge--warn"),
            ("Pivot", f"${pivot:.2f}", "aeth-badge--neutral"),
            ("S1", f"${s1:.2f}", "aeth-badge--warn"),
            ("S2", f"${s2:.2f}", "aeth-badge--bull"),
        ]
        for name, val, badge_class in levels_data:
            arrow = " ← price" if abs(float(val[1:]) - current_price) == min(
                abs(float(v[1:]) - current_price) for _, v, _ in levels_data
            ) else ""
            st.markdown(f'<span class="aeth-badge {badge_class}">{name}</span> &nbsp; {val}{arrow}', unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"Prev High: ${float(prev_d['High']):.2f} | Prev Low: ${float(prev_d['Low']):.2f} | Prev Close: ${float(prev_d['Close']):.2f}")
        st.caption(f"Today's Range: {day_range_pct:.1f}% | Price at {range_pos:.0f}% of range | ATR: ${atr:.2f}")

    or_note = "Opening range data not available."
    with or_col:
        st.markdown("### Opening Range")
        if has_intraday:
            open_time = intraday.index[0]
            cutoff = open_time + pd.Timedelta(minutes=30)
            or_bars = intraday[intraday.index <= cutoff]
            if not or_bars.empty:
                or_high = float(or_bars["High"].max())
                or_low = float(or_bars["Low"].min())
                or_size = (or_high - or_low) / or_low * 100

                st.metric("OR High", f"${or_high:.2f}")
                st.metric("OR Low", f"${or_low:.2f}")
                st.metric("OR Size", f"{or_size:.2f}%")

                if current_price > or_high:
                    st.success(f"✅ Price broke **ABOVE** OR High (+{(current_price - or_high) / or_high * 100:.1f}%) — bullish breakout")
                    or_note = f"Broke above OR High ${or_high:.2f} (+{(current_price - or_high) / or_high * 100:.1f}%) — bullish breakout"
                elif current_price < or_low:
                    st.error(f"🔴 Price broke **BELOW** OR Low ({(current_price - or_low) / or_low * 100:.1f}%) — bearish breakdown")
                    or_note = f"Broke below OR Low ${or_low:.2f} ({(current_price - or_low) / or_low * 100:.1f}%) — bearish breakdown"
                else:
                    pct_in_range = (current_price - or_low) / (or_high - or_low) * 100 if or_high > or_low else 50.0
                    st.info(f"Price inside OR at {pct_in_range:.0f}% of range — awaiting breakout")
                    or_note = f"Inside opening range at {pct_in_range:.0f}% of range — awaiting breakout"
            else:
                st.info("Opening range forms in first 30 minutes of trading.")
        else:
            st.info("Opening range requires intraday data (5m, 15m, 30m, or 1h interval).")
            gap_pct = (float(daily.iloc[-1]["Open"]) - float(prev_d["Close"])) / float(prev_d["Close"]) * 100
            gap_dir = "▲ Gap Up" if gap_pct > 0 else "▼ Gap Down"
            gap_badge_class = "aeth-badge--bull" if gap_pct > 0 else "aeth-badge--bear"
            st.markdown(f'<span class="aeth-badge {gap_badge_class}">{gap_dir} {gap_pct:+.2f}% today\'s open vs prev close</span>', unsafe_allow_html=True)
            if abs(gap_pct) > 2:
                st.caption("Large gap (>2%) — watch for gap-fill tendency in first hour.")
            or_note = f"{gap_dir} {gap_pct:+.2f}% today's open vs prev close (no intraday OR available)"

    st.markdown("---")
    st.markdown("### Suggested Entry / Stop / Target")
    st.caption("Rule-based read of the signals above — a starting point to sanity-check, not a trade alert.")

    directional_signals = [vwap_dir, mom_dir, ta_dir]
    if pattern:
        directional_signals.append(pattern["direction"])
    if latest_flag_pattern:
        directional_signals.append(latest_flag_pattern["direction"])
    bull_votes = directional_signals.count("bull")
    bear_votes = directional_signals.count("bear")

    sug_entry = sug_stop = sug_target = direction_label = None
    if atr <= 0 or bull_votes == bear_votes or max(bull_votes, bear_votes) < 2:
        st.info(f"Signals conflict ({bull_votes} bullish vs {bear_votes} bearish of {len(directional_signals)}) — no high-conviction setup right now.")
    else:
        stop_distance = 1.5 * atr
        sug_entry = current_price
        if bull_votes > bear_votes:
            direction_label, dir_badge_class = "LONG", "aeth-badge--bull"
            sug_stop = sug_entry - stop_distance
            rr_to_r1 = (r1 - sug_entry) / stop_distance if stop_distance > 0 else 0
            if rr_to_r1 >= 1.5:
                sug_target, target_source = r1, "R1 pivot resistance"
            else:
                sug_target, target_source = sug_entry + 2 * stop_distance, "2:1 R:R (R1 too close for a clean target)"
        else:
            direction_label, dir_badge_class = "SHORT", "aeth-badge--bear"
            sug_stop = sug_entry + stop_distance
            rr_to_s1 = (sug_entry - s1) / stop_distance if stop_distance > 0 else 0
            if rr_to_s1 >= 1.5:
                sug_target, target_source = s1, "S1 pivot support"
            else:
                sug_target, target_source = sug_entry - 2 * stop_distance, "2:1 R:R (S1 too close for a clean target)"

        rr = abs(sug_target - sug_entry) / stop_distance if stop_distance > 0 else 0
        stop_pct = abs(sug_stop - sug_entry) / sug_entry * 100

        flag_note = ""
        if latest_flag_pattern and latest_flag_pattern["direction"] == ("bull" if direction_label == "LONG" else "bear"):
            flag_stop = latest_flag_pattern["stop_price"]
            tighter_stop = max(sug_stop, flag_stop) if direction_label == "LONG" else min(sug_stop, flag_stop)
            if tighter_stop != sug_stop:
                sug_stop = tighter_stop
                stop_pct = abs(sug_stop - sug_entry) / sug_entry * 100
                flag_note = f" Stop tightened to the confirmed {'Bull' if direction_label == 'LONG' else 'Bear'} {'Pennant' if latest_flag_pattern['pennant'] else 'Flag'}'s boundary (confidence {latest_flag_pattern['confidence_score']:.0f})."
            measured_move_target = latest_flag_pattern["target_price"]
            better_target = measured_move_target > sug_target if direction_label == "LONG" else measured_move_target < sug_target
            if better_target:
                sug_target, target_source = measured_move_target, f"Flag/Pennant measured move (confidence {latest_flag_pattern['confidence_score']:.0f})"
                rr = abs(sug_target - sug_entry) / stop_distance if stop_distance > 0 else 0

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.markdown(f'<span class="aeth-badge {dir_badge_class}">{direction_label}</span>', unsafe_allow_html=True)
        ec1.caption(f"{max(bull_votes, bear_votes)}/{len(directional_signals)} signals agree")
        ec2.metric("Entry", f"${sug_entry:.2f}")
        ec3.metric("Stop", f"${sug_stop:.2f}", f"-{stop_pct:.1f}%" if direction_label == "LONG" else f"+{stop_pct:.1f}%", delta_color="inverse")
        ec4.metric("Target", f"${sug_target:.2f}", f"{rr:.1f}:1 R:R")
        st.caption(f"Stop = 1.5× ATR (${atr:.2f}) from entry. Target = {target_source}. Position sizing not included — use the Quick Risk Calculator below.{flag_note}")

    if analyze:
        log_key = (ticker, interval, direction_label, sug_entry, sug_stop, sug_target)
        if st.session_state.get("_last_logged_dt") != log_key:
            log_activity("day_trading_analyze", ticker, {
                "interval": interval,
                "vwap_dev_pct": round(vwap_dev, 2),
                "vwap_direction": vwap_dir,
                "momentum_direction": mom_dir,
                "trend_direction": ta_dir,
                "suggested_direction": direction_label,
                "suggested_entry": sug_entry,
                "suggested_stop": sug_stop,
                "suggested_target": sug_target,
            })
            st.session_state["_last_logged_dt"] = log_key

    regime_markov = analyze_regime_markov(daily, ticker)

    st.markdown("---")
    st.markdown("### Market Regime (Markov)")
    st.caption(f"{ticker}'s own trend history, discretized into Bear/Neutral/Bull and fit to a first-order transition matrix — a probabilistic read, not a rule-based one.")
    if not regime_markov["available"]:
        st.info(regime_markov["reason"])
    else:
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Current State", regime_markov["current_state"])
        rm2.metric("Bull − Bear Signal", f"{regime_markov['signal']:+.2f}")
        rm3.metric("Confidence", f"{regime_markov['confidence'] * 100:.0f}%", help="Scales with how many times the current state has occurred historically (30+ = full confidence).")
        persist = regime_markov["persistence"]
        st.caption(
            f"Persistence — Bear: {persist['Bear'] * 100:.0f}% | Neutral: {persist['Neutral'] * 100:.0f}% | Bull: {persist['Bull'] * 100:.0f}% "
            f"(probability each regime repeats itself the next bar, from {regime_markov['n_bars']} bars of history)."
        )
        next_probs = regime_markov["next_step_probs"]
        st.caption(f"Next-bar odds from {regime_markov['current_state']}: Bear {next_probs['Bear'] * 100:.0f}% | Neutral {next_probs['Neutral'] * 100:.0f}% | Bull {next_probs['Bull'] * 100:.0f}%")

    st.markdown("---")
    with st.expander("Quick Risk Calculator", expanded=False):
        if st.session_state.get("dt_risk_calc_ticker") != ticker:
            st.session_state["dt_entry"] = round(current_price, 2)
            st.session_state["dt_stop"] = round(current_price * 0.98, 2)
            st.session_state["dt_risk_calc_ticker"] = ticker

        rc1, rc2 = st.columns(2)
        with rc1:
            port_val = st.number_input("Portfolio Size ($)", value=100_000, step=5_000, min_value=1_000, key="dt_port_val")
            entry = st.number_input("Entry Price ($)", min_value=0.01, key="dt_entry")
            stop = st.number_input("Stop Price ($)", min_value=0.01, key="dt_stop")
            risk_pct = st.slider("Risk per Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25, key="dt_risk_pct") / 100
            apply_regime = st.checkbox(
                "Scale size by regime signal", value=False,
                help="Multiplies the position size by the Bull-Bear regime signal above (0.5x-1.5x) — leans in when the regime favors the trade direction, pulls back when it doesn't.",
                key="dt_apply_regime",
            )

        with rc2:
            if stop < entry:
                result = position_size_from_stop(port_val, entry, stop, risk_pct)
                if "error" not in result:
                    shares, position_value, position_pct = result["shares"], result["position_value"], result["position_pct"]
                    if apply_regime and regime_markov["available"]:
                        multiplier = regime_kelly_multiplier(regime_markov["signal"], regime_markov["confidence"])
                        shares = round(shares * multiplier)
                        position_value = round(position_value * multiplier, 2)
                        position_pct = round(position_pct * multiplier, 2)
                        st.caption(f"Regime multiplier: {multiplier:.2f}x (signal {regime_markov['signal']:+.2f} × confidence {regime_markov['confidence'] * 100:.0f}%)")
                    st.metric("Shares", shares)
                    st.metric("Position Value", f"${position_value:,.0f} ({position_pct:.1f}%)")
                    st.metric("Dollar Risk", f"${result['dollar_risk']:,.0f}")
                    st.markdown(f"**2:1 Target:** ${result['risk_reward_2to1_target']:.2f}")
                    st.markdown(f"**3:1 Target:** ${result['risk_reward_3to1_target']:.2f}")
                    rr = (result['risk_reward_3to1_target'] - entry) / (entry - stop) if entry != stop else 0
                    if rr >= 3:
                        st.success(f"R:R = {rr:.1f}:1 — excellent setup")
                    elif rr >= 2:
                        st.info(f"R:R = {rr:.1f}:1 — acceptable")
                    else:
                        st.warning(f"R:R = {rr:.1f}:1 — poor risk/reward, skip or widen target")
                else:
                    st.warning(result["error"])
            else:
                st.warning("Stop must be below entry price.")

    if has_intraday and "RSI" in intraday.columns:
        with st.expander("Intraday RSI & MACD", expanded=False):
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)
            fig2.add_trace(go.Scatter(x=intraday.index, y=intraday["RSI"], name="RSI", line=dict(color="#7c4dff", width=1.5)), row=1, col=1)
            fig2.add_hline(y=70, row=1, col=1, line=dict(color="#ef5350", width=1, dash="dot"))
            fig2.add_hline(y=30, row=1, col=1, line=dict(color="#26a69a", width=1, dash="dot"))
            fig2.add_hline(y=50, row=1, col=1, line=dict(color="#888", width=0.8, dash="dot"))

            if "MACD_hist" in intraday.columns:
                hist = intraday["MACD_hist"]
                fig2.add_trace(go.Bar(x=intraday.index, y=hist, name="MACD Histogram",
                                      marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in hist.fillna(0)]), row=2, col=1)
                fig2.add_trace(go.Scatter(x=intraday.index, y=intraday["MACD"], name="MACD", line=dict(color="#42a5f5", width=1.2)), row=2, col=1)
                fig2.add_trace(go.Scatter(x=intraday.index, y=intraday["MACD_signal"], name="Signal", line=dict(color="#ff7043", width=1.2)), row=2, col=1)

            fig2.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=350, margin=dict(l=0, r=0, t=10, b=0))
            fig2.update_yaxes(range=[0, 100], row=1, col=1)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    with st.expander("Backtest: MACD Bullish Cross (2yr history)", expanded=False):
        st.caption(
            "Tests the momentum signal above against 2 years of history: buy when the MACD "
            "histogram turns up while MACD & Signal are still below zero and price is above "
            "the 200-day average, then exit on a 2% stop-loss or 3% take-profit — one trade "
            "at a time, long only. Close-to-close approximation, not a live-fill simulation."
        )
        if st.button("Run Backtest", key="dt_run_backtest"):
            logger.info(f"[trading] 'Run Backtest' button pressed for {ticker} (MACD bullish cross)")
            with st.spinner(f"Loading 2 years of {ticker} history…"):
                bt_df = _load_backtest_df(ticker)
            if bt_df is None or bt_df.empty or "MACD_hist" not in bt_df.columns:
                logger.warning(f"[trading] Backtest for {ticker} blocked: not enough history")
                st.warning("Not enough history to backtest this ticker.")
            else:
                signal = macd_bullish_cross_signal(bt_df)
                result = simulate_trades(bt_df, signal)
                logger.info(
                    f"[trading] Backtest for {ticker} completed: num_trades={result['num_trades']} "
                    f"win_rate={result['win_rate']:.1f}% total_return_pct={result['total_return_pct']:+.1f}%"
                )
                st.session_state[f"dt_backtest_{ticker}"] = result

        result = st.session_state.get(f"dt_backtest_{ticker}")
        if result:
            if result["num_trades"] == 0:
                st.info("No MACD bullish-cross signals fired for this ticker over the last 2 years.")
            else:
                if result["num_trades"] < 10:
                    st.warning(f"Only {result['num_trades']} trades in this window — too few to draw firm conclusions.")
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("Trades", result["num_trades"])
                bc2.metric("Win Rate", f"{result['win_rate']:.1f}%")
                bc3.metric("Total Return", f"{result['total_return_pct']:+.1f}%")
                bc4.metric("Final Value", f"${result['final_value']:,.0f}", help="Starting from $1,000")

                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    x=result["equity_curve"].index, y=result["equity_curve"].values,
                    line=dict(color="#9370DB", width=2), name="Equity",
                ))
                eq_fig.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=280, margin=dict(l=0, r=0, t=10, b=0),
                                      yaxis_title="Portfolio Value ($)")
                st.plotly_chart(eq_fig, use_container_width=True)

                with st.expander("Trade log", expanded=False):
                    st.dataframe(pd.DataFrame(result["trades"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    if ai_available():
        if st.button("Generate AI Day Trading Brief", key="dt_ai_brief"):
            logger.info(f"[trading] 'Generate AI Day Trading Brief' button pressed for {ticker}")
            signals = {
                "market_status": status,
                "vix": vix_val,
                "vix_regime": vix_regime,
                "sp_regime": sp_regime,
                "vwap_dev": vwap_dev,
                "vwap_note": vwap_note,
                "mom_note": mom_note,
                "vol_note": vol_note,
                "ta_note": ta_note,
            }
            key_levels = {"pivot": pivot, "r1": r1, "s1": s1, "or_note": or_note}
            with st.spinner("Generating AI day trading read..."):
                brief = generate_daytrading_brief(ticker, current_price, signals, key_levels)
            if brief:
                logger.info(f"[trading] AI day trading brief generated successfully for {ticker}")
                st.markdown(format_ai_markdown(brief))
            else:
                logger.warning(f"[trading] AI day trading brief generation failed for {ticker}")
                st.error("AI generation failed — check logs (Ollama model may be unreachable or unable to answer within its token budget).")

    st.caption("Data from yfinance · Refreshes every 60s · Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════
# ── Options ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_options_pnl_diagram(strategy: str, current_price: float):
    mid = round(current_price / 5) * 5
    premium = round(current_price * 0.03, 2)

    if strategy == "Long Call":
        result = build_pnl_diagram("Long Call", current_price, [mid], [premium], ["call"], [1])
    elif strategy == "Long Put":
        result = build_pnl_diagram("Long Put", current_price, [mid], [premium], ["put"], [1])
    elif strategy == "Covered Call":
        call_strike = mid + 5
        result = build_pnl_diagram("Covered Call", current_price, [call_strike], [premium * 0.5], ["call"], [-1])
    elif strategy == "Bull Put Spread":
        result = build_pnl_diagram("Bull Put Spread", current_price,
                                    [mid - 5, mid - 10], [premium * 0.7, premium * 0.3], ["put", "put"], [-1, 1])
    elif strategy == "Bear Call Spread":
        result = build_pnl_diagram("Bear Call Spread", current_price,
                                    [mid + 5, mid + 10], [premium * 0.7, premium * 0.3], ["call", "call"], [-1, 1])
    elif strategy == "Iron Condor":
        result = build_pnl_diagram("Iron Condor", current_price,
                                    [mid - 5, mid - 10, mid + 5, mid + 10],
                                    [premium * 0.6, premium * 0.3, premium * 0.6, premium * 0.3],
                                    ["put", "put", "call", "call"], [-1, 1, -1, 1])
    else:
        return

    fig = go.Figure()
    pnl = result["pnl"]
    prices = result["price_range"]

    fig.add_trace(go.Scatter(
        x=prices, y=pnl, fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
        line=dict(color="#26a69a", width=2), name="P&L at Expiry",
    ))
    fig.add_hline(y=0, line_color="white", line_dash="dash", line_width=1)
    fig.add_vline(x=current_price, line_color="yellow", line_dash="dot", annotation_text=f"Current ${current_price:.0f}")

    for be in result.get("breakevens", []):
        fig.add_vline(x=be, line_color="orange", line_dash="dot", annotation_text=f"BE ${be:.0f}")

    fig.update_layout(
        template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=350,
        title=f"{strategy} — Max Profit: ${result['max_profit']:.0f} | Max Loss: ${result['max_loss']:.0f}",
        xaxis_title="Stock Price at Expiry", yaxis_title="P&L ($)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Max Profit", f"${result['max_profit']:.0f}")
    cols[1].metric("Max Loss", f"${result['max_loss']:.0f}")
    cols[2].metric("Breakevens", ", ".join(f"${be:.2f}" for be in result["breakevens"]) or "N/A")
    st.caption("P&L shown per 1 contract (100 shares). Adjust for position size.")


def _render_options():
    st.subheader("Options Analysis")
    st.caption("Real options data from live market. No mock values.")

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker", value="SPY", placeholder="e.g. AAPL, SPY, QQQ", key="opt_ticker").upper().strip()

    if not ticker:
        return

    with st.spinner(f"Fetching live options data for {ticker}..."):
        iv_metrics = calculate_iv_rank(ticker)
        chain_data = get_options_chain(ticker)
        df_price = get_price_history(ticker, period="1y")
        atm_greeks = get_atm_greeks(ticker)

    if "error" in chain_data:
        logger.warning(f"[trading] Options tab: options not available for {ticker}: {chain_data.get('error', 'Unknown error')}")
        st.error(f"Options not available: {chain_data.get('error', 'Unknown error')}")
        st.info("Options data is only available for optionable stocks (not ETFs in some cases)")
        return

    current_price = chain_data.get("current_price", 0)
    expirations = chain_data.get("expirations", [])

    if st.session_state.get("_last_logged_opt_ticker") != ticker:
        logger.info(f"[trading] Options tab: ticker changed to {ticker}; logging options_view activity")
        log_activity("options_view", ticker, {
            "iv_rank": iv_metrics.get("iv_rank"),
            "iv_percentile": iv_metrics.get("iv_percentile"),
            "atm_iv": iv_metrics.get("atm_iv"),
            "term_structure": iv_metrics.get("term_structure"),
        })
        st.session_state["_last_logged_opt_ticker"] = ticker

    st.markdown("#### Implied Volatility Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    ivr = iv_metrics.get("iv_rank", 50)
    ivr_signal = "🔴 Sell Premium" if ivr > 60 else ("🟢 Buy Premium" if ivr < 30 else "🟡 Neutral")
    c1.metric("IV Rank", f"{ivr:.0f}", ivr_signal)
    c2.metric("IV Percentile", f"{iv_metrics.get('iv_percentile', 50):.0f}%")
    c3.metric("ATM IV (Live)", f"{iv_metrics.get('atm_iv', 0) or 0:.1f}%")
    c4.metric("HV 21-day", f"{iv_metrics.get('hv_21', 0):.1f}%")
    c5.metric("Term Structure", iv_metrics.get("term_structure", "Flat"))

    iv_rv = iv_metrics.get("iv_rv_ratio")
    if iv_rv and iv_rv > 1.15:
        st.warning(f"IV/RV Ratio {iv_rv:.2f}x — options premium is elevated vs realized volatility. Systematic edge in selling premium.")
    elif iv_rv and iv_rv < 0.85:
        st.success(f"IV/RV Ratio {iv_rv:.2f}x — options are cheap vs realized volatility. Better to buy than sell.")

    garch_vol = iv_metrics.get("garch_forecast_vol")
    iv_garch_ratio = iv_metrics.get("iv_vs_garch_ratio")
    if garch_vol:
        st.caption(f"GARCH forward vol forecast (horizon-matched to nearest expiry): {garch_vol:.1f}%")
        if iv_garch_ratio and iv_garch_ratio > 1.15:
            st.warning(f"IV/GARCH Ratio {iv_garch_ratio:.2f}x — IV is rich vs. forward-looking vol forecast, not just trailing HV.")
        elif iv_garch_ratio and iv_garch_ratio < 0.85:
            st.success(f"IV/GARCH Ratio {iv_garch_ratio:.2f}x — IV is cheap vs. forward-looking vol forecast.")

    if df_price is not None and not df_price.empty:
        df = calculate_indicators(df_price)
        if "hv_21" in df.columns:
            st.markdown("---")
            st.markdown("#### Historical Volatility History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["hv_21"], name="HV 21-day", line=dict(color="#2196f3")))
            if "hv_63" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["hv_63"], name="HV 63-day", line=dict(color="#ff9800", dash="dash")))
            if iv_metrics.get("atm_iv"):
                fig.add_hline(y=iv_metrics["atm_iv"], line_color="red", line_dash="dot", annotation_text=f"Current IV {iv_metrics['atm_iv']:.1f}%")
            fig.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=300, yaxis_title="Volatility (%)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Options Chain")
    if expirations:
        selected_expiry = st.selectbox("Expiration Date", expirations, index=0, key="opt_expiry")
        chain_selected = get_options_chain(ticker, selected_expiry)
        calls = chain_selected.get("calls", pd.DataFrame())
        puts = chain_selected.get("puts", pd.DataFrame())

        expiry_log_key = (ticker, selected_expiry)
        if st.session_state.get("_last_logged_opt_expiry") != expiry_log_key:
            log_activity("options_expiry_view", ticker, {"expiry": selected_expiry})
            st.session_state["_last_logged_opt_expiry"] = expiry_log_key

        chain_col1, chain_col2 = st.columns(2)
        with chain_col1:
            st.markdown("**Calls**")
            if not calls.empty:
                display_cols = ["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
                display_cols = [c for c in display_cols if c in calls.columns]
                calls_display = calls[display_cols].copy()
                if "impliedVolatility" in calls_display.columns:
                    calls_display["impliedVolatility"] = (calls_display["impliedVolatility"] * 100).round(1)
                    calls_display = calls_display.rename(columns={"impliedVolatility": "IV%"})
                if current_price:
                    calls_display["ATM"] = (calls["strike"] - current_price).abs() < (calls["strike"].diff().abs().median() / 2)
                st.dataframe(calls_display.head(15), hide_index=True, width="stretch")

        with chain_col2:
            st.markdown("**Puts**")
            if not puts.empty:
                display_cols = ["strike", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
                display_cols = [c for c in display_cols if c in puts.columns]
                puts_display = puts[display_cols].copy()
                if "impliedVolatility" in puts_display.columns:
                    puts_display["impliedVolatility"] = (puts_display["impliedVolatility"] * 100).round(1)
                    puts_display = puts_display.rename(columns={"impliedVolatility": "IV%"})
                st.dataframe(puts_display.head(15), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("#### P&L Diagram Builder")
    strategy = st.selectbox("Strategy", [
        "Long Call", "Long Put", "Covered Call",
        "Bull Put Spread", "Bear Call Spread", "Iron Condor",
    ], key="opt_strategy")

    if current_price:
        _render_options_pnl_diagram(strategy, current_price)

    if atm_greeks:
        st.markdown("---")
        st.markdown("#### ATM Greeks")

        def _fmt(value, decimals=2, prefix=""):
            if value is None:
                return "N/A"
            return f"{prefix}{value:.{decimals}f}"

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**ATM Call**")
            atm_call = atm_greeks.get("atm_call", {})
            st.write(f"Strike: **${atm_call.get('strike', 0):.2f}** | IV: **{atm_call.get('iv', 0):.1f}%**")
            st.write(f"Bid/Ask: **${atm_call.get('bid', 0):.2f} / ${atm_call.get('ask', 0):.2f}**")
            st.write(f"Volume: **{atm_call.get('volume', 0):,}** | OI: **{atm_call.get('open_interest', 0):,}**")
            st.write(f"Delta: **{_fmt(atm_call.get('delta'), 3)}** | Gamma: **{_fmt(atm_call.get('gamma'), 3)}**")
            st.write(f"Theta: **{_fmt(atm_call.get('theta'), 2, '$')}** | Vega: **{_fmt(atm_call.get('vega'), 2, '$')}** | Rho: **{_fmt(atm_call.get('rho'), 2, '$')}**")
        with g2:
            st.markdown("**ATM Put**")
            atm_put = atm_greeks.get("atm_put", {})
            st.write(f"Strike: **${atm_put.get('strike', 0):.2f}** | IV: **{atm_put.get('iv', 0):.1f}%**")
            st.write(f"Bid/Ask: **${atm_put.get('bid', 0):.2f} / ${atm_put.get('ask', 0):.2f}**")
            st.write(f"Volume: **{atm_put.get('volume', 0):,}** | OI: **{atm_put.get('open_interest', 0):,}**")
            st.write(f"Delta: **{_fmt(atm_put.get('delta'), 3)}** | Gamma: **{_fmt(atm_put.get('gamma'), 3)}**")
            st.write(f"Theta: **{_fmt(atm_put.get('theta'), 2, '$')}** | Vega: **{_fmt(atm_put.get('vega'), 2, '$')}** | Rho: **{_fmt(atm_put.get('rho'), 2, '$')}**")
        st.caption("Greeks are estimated via Black-Scholes using each option's own implied volatility (or a solved fallback), not observed market sensitivities.")

    st.markdown("---")
    if ai_available():
        if st.button("Generate AI Options Strategy Brief", key="opt_ai_brief"):
            logger.info(f"[trading] 'Generate AI Options Strategy Brief' button pressed for {ticker}")
            df_regime = calculate_indicators(df_price) if df_price is not None else None
            regime = detect_regime(df_regime, ticker) if df_regime is not None else {}
            with st.spinner("Generating AI options analysis..."):
                brief = generate_options_brief(ticker, current_price, iv_metrics, regime)
            if brief:
                logger.info(f"[trading] AI options strategy brief generated successfully for {ticker}")
                st.markdown(format_ai_markdown(brief))
            else:
                logger.warning(f"[trading] AI options strategy brief generation failed for {ticker}")
                st.error("AI generation failed — check logs (Ollama model may be unreachable or unable to answer within its token budget).")


# ══════════════════════════════════════════════════════════════════════════
# ── ML Predictions ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _model_exists(ticker: str) -> bool:
    xgb_ok = (STORAGE_DIR / f"{ticker.upper()}_xgb.pkl").exists()
    rf_ok = (STORAGE_DIR / f"{ticker.upper()}_rf.pkl").exists()
    return xgb_ok and rf_ok


def _trained_at_str(ticker: str) -> str | None:
    acc_path = STORAGE_DIR / f"{ticker.upper()}_accuracy.json"
    if not acc_path.exists():
        xgb_path = STORAGE_DIR / f"{ticker.upper()}_xgb.pkl"
        if not xgb_path.exists():
            return None
        try:
            mtime = xgb_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=MARKET_TZ)
            return dt.strftime("%Y-%m-%d %I:%M %p ET")
        except Exception:
            return None
    try:
        import json as _json
        with open(acc_path) as f:
            data = _json.load(f)
        ts = data.get("trained_at")
        if ts:
            return utc_iso_to_et_str(ts, "%Y-%m-%d %I:%M %p ET")
    except Exception:
        pass
    return None


def _retrain_overdue(ticker: str) -> bool:
    acc_path = STORAGE_DIR / f"{ticker.upper()}_accuracy.json"
    xgb_path = STORAGE_DIR / f"{ticker.upper()}_xgb.pkl"
    check_path = acc_path if acc_path.exists() else xgb_path
    if not check_path.exists():
        return False
    try:
        import time as _time
        mtime = check_path.stat().st_mtime
        age_days = (_time.time() - mtime) / 86400
        return age_days > 30
    except Exception:
        return False


def _load_pred_df(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    raw = get_price_history(ticker, period=period)
    if raw is None or raw.empty:
        return None
    return calculate_indicators(raw)


def _cache_key(ticker: str) -> str:
    return f"ml_prediction_{ticker.upper()}"


def _eval_cache_key(ticker: str) -> str:
    return f"ml_eval_{ticker.upper()}"


def _price_path_cache_key(ticker: str) -> str:
    return f"ml_price_path_{ticker.upper()}"


def _render_prediction_card(result: dict):
    direction = result.get("direction", "neutral").upper()
    bull_prob = result.get("probability", 0.50)
    confidence = result.get("confidence", "low").upper()
    val_acc = result.get("model_accuracy") or 0.0
    exp_move = result.get("expected_move_pct")
    horizon_days = result.get("horizon_days") or 5
    retrain = result.get("retrain_overdue", False)

    if confidence == "MEDIUM":
        confidence = "MODERATE"

    cfg = SIGNAL_CONFIG.get(direction, SIGNAL_CONFIG["NEUTRAL"])
    icon = cfg["icon"]

    if retrain:
        st.warning("Model is more than 30 days old. Click **Train / Update Model** to refresh.", icon="⏰")

    gauge_pct = bull_prob * 100

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(gauge_pct, 1),
        number={"suffix": "%", "font": {"size": 40, "color": cfg["color"]}},
        delta={"reference": 50, "relative": False, "increasing": {"color": "#26a69a"}, "decreasing": {"color": "#ef5350"}, "font": {"size": 18}},
        gauge={
            "axis": {
                "range": [35, 65],
                "tickvals": [35, 40, 47, 50, 53, 60, 65],
                "ticktext": ["35%", "40%", "47%", "50%", "53%", "60%", "65%"],
                "tickfont": {"size": 11, "color": "#aaa"},
            },
            "bar": {"color": cfg["gauge_color"], "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [35, 40], "color": "rgba(239,83,80,0.35)"},
                {"range": [40, 47], "color": "rgba(239,83,80,0.15)"},
                {"range": [47, 53], "color": "rgba(158,158,158,0.15)"},
                {"range": [53, 60], "color": "rgba(38,166,154,0.15)"},
                {"range": [60, 65], "color": "rgba(38,166,154,0.35)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.75, "value": gauge_pct},
        },
        title={
            "text": f"<b>{icon} {direction}</b><br><span style='font-size:14px;color:#aaa'>Bullish Probability</span>",
            "font": {"size": 22, "color": cfg["color"]},
        },
    ))
    fig_gauge.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    col_gauge, col_stats = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("Neutral zone: 47–53% — no directional call issued. Display range capped at 35–65% to prevent false precision.")

    with col_stats:
        st.markdown(f"#### Signal Details")
        conf_badge_class = {"HIGH": "aeth-badge--bull", "MODERATE": "aeth-badge--warn", "LOW": "aeth-badge--neutral"}.get(confidence, "aeth-badge--neutral")
        st.markdown(
            f'<span class="aeth-badge {conf_badge_class}">Confidence: {confidence}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if exp_move is not None:
            move_sign = "+" if exp_move >= 0 else ""
            st.metric(label=f"Expected {horizon_days}-Day Move", value=f"{move_sign}{exp_move:.1f}%",
                     help=f"Median {horizon_days}-day return observed in historical setups where model also predicted this direction. Not a price target.")
        else:
            st.metric(f"Expected {horizon_days}-Day Move", "N/A", help="Requires sufficient validation data.")

        st.markdown("---")
        st.metric(label="Walk-Forward Accuracy", value=f"{val_acc * 100:.1f}%",
                 delta=f"{(val_acc - 0.50) * 100:+.1f}% vs coin flip",
                 help="Mean directional accuracy across out-of-sample walk-forward folds.")

        trained_at = result.get("last_trained")
        if trained_at:
            try:
                st.caption(f"Model trained: {utc_iso_to_et_str(trained_at, '%Y-%m-%d %I:%M %p ET')}")
            except Exception:
                st.caption(f"Model trained: {trained_at}")

    edge_pct = (val_acc - 0.50) * 100
    st.info(
        f"Model edge: **+{edge_pct:.1f}%** above the coin-flip baseline "
        f"({val_acc * 100:.1f}% accuracy vs 50.0% random). "
        "This signal has a small, real edge — not a certainty. "
        "Do not size this position larger than your standard allocation.",
        icon="📊",
    )


def _render_price_path(price_path: dict):
    days = price_path["days"]
    current_price = price_path["current_price"]
    daily_vol = price_path["daily_volatility_pct"]
    n_sims = price_path["n_sims"]

    x_days = [0] + [d["day"] for d in days]
    close_median = [current_price] + [d["close_median"] for d in days]
    close_p25 = [current_price] + [d["close_p25"] for d in days]
    close_p75 = [current_price] + [d["close_p75"] for d in days]
    open_median = [None] + [d["open_median"] for d in days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_days + x_days[::-1], y=close_p75 + close_p25[::-1],
        fill="toself", fillcolor="rgba(38,166,154,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name="25th–75th percentile (Close)",
    ))
    fig.add_trace(go.Scatter(
        x=x_days, y=close_median, mode="lines+markers", name="Median Close",
        line=dict(color="#26a69a", width=2), marker=dict(size=6),
        hovertemplate="Day %{x}<br>Median close: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_days, y=open_median, mode="markers", name="Median Open",
        marker=dict(size=8, symbol="diamond", color="#ffb74d"),
        hovertemplate="Day %{x}<br>Median open: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Trading days ahead", dtick=1, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", y=1.15, font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.6)" if st.context.theme.type == "dark" else "rgba(255,255,255,0.6)",
    )
    st.plotly_chart(fig, use_container_width=True)

    table_df = pd.DataFrame([
        {
            "Day": f"+{d['day']}",
            "Open (25–75%)": f"${d['open_p25']:.2f} – ${d['open_p75']:.2f}",
            "Open (median)": f"${d['open_median']:.2f}",
            "Close (25–75%)": f"${d['close_p25']:.2f} – ${d['close_p75']:.2f}",
            "Close (median)": f"${d['close_median']:.2f}",
        }
        for d in days
    ])
    st.dataframe(table_df, hide_index=True, width="stretch")

    st.caption(
        f"Simulated via {n_sims:,} Monte Carlo paths seeded with this stock's own 21-day "
        f"realized volatility ({daily_vol:.2f}% daily) and the direction model's own bias as "
        "drift. This is a probability band, not a price forecast — a meaningful share of actual "
        "outcomes should land outside the shown 25th–75th percentile range. Overnight opens use "
        "a simplified reduced-volatility gap assumption, not a fitted gap distribution."
    )


def _render_feature_importance(top_features):
    if not top_features:
        st.caption("Feature importance not available for this prediction.")
        return

    if isinstance(top_features, dict):
        df_feat = pd.DataFrame([{"feature": k, "importance": v} for k, v in top_features.items()])
    else:
        df_feat = pd.DataFrame(top_features)

    if "feature" not in df_feat.columns or "importance" not in df_feat.columns:
        st.caption("Feature importance data format unexpected.")
        return

    df_feat["display_name"] = df_feat["feature"].map(FEATURE_DISPLAY_NAMES).fillna(df_feat["feature"])
    df_feat = df_feat.sort_values("importance", ascending=True)

    max_imp = df_feat["importance"].max()
    bar_colors = [
        "#26a69a" if v >= max_imp * 0.7 else ("#2196f3" if v >= max_imp * 0.4 else "#78909c")
        for v in df_feat["importance"]
    ]

    fig = go.Figure(go.Bar(
        x=df_feat["importance"], y=df_feat["display_name"], orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in df_feat["importance"]],
        textposition="outside", textfont={"size": 11, "color": "#ccc"},
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="What Drove This Prediction", font=dict(size=16, color="#e0e0e0")),
        template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=max(280, len(df_feat) * 44),
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(title="Importance Score (XGBoost Gain)", title_font=dict(size=12), gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="", tickfont=dict(size=12), gridcolor="rgba(255,255,255,0.04)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.6)" if st.context.theme.type == "dark" else "rgba(255,255,255,0.6)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Importance = average gain when a feature is used in a split. "
        "Higher means that feature contributed more decision weight to this prediction. "
        "Feature rankings can shift across training runs — large changes indicate regime sensitivity."
    )


def _render_model_performance(eval_result: dict, ticker: str):
    if not eval_result or eval_result.get("error"):
        err = eval_result.get("error", "Evaluation not available.") if eval_result else "Evaluation not available."
        st.warning(f"Model evaluation unavailable: {err}")
        return

    dir_acc = eval_result.get("directional_accuracy") or 0.0
    acc_std = eval_result.get("accuracy_std") or 0.0
    auc = eval_result.get("mean_auc") or 0.0
    sharpe = eval_result.get("sharpe_of_signals")
    n_preds = eval_result.get("n_validation_samples") or 0
    n_samples = eval_result.get("n_training_samples") or 0
    reliable = eval_result.get("is_reliable", False)
    reason = eval_result.get("reliability_reason", "")
    balance = eval_result.get("class_balance", {})

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Performance Metrics")
        rel_icon = "✅" if reliable else "⚠️"
        rel_badge_class = "aeth-badge--bull" if reliable else "aeth-badge--warn"
        st.markdown(
            f'<span class="aeth-badge {rel_badge_class}">{rel_icon} {"Reliable" if reliable else "Unreliable"}</span>',
            unsafe_allow_html=True,
        )
        if reason:
            st.caption(reason)
        st.markdown("")

        m1, m2 = st.columns(2)
        m1.metric("Directional Accuracy", f"{dir_acc * 100:.1f}%", delta=f"{(dir_acc - 0.50) * 100:+.1f}% edge")
        m2.metric("ROC-AUC", f"{auc:.3f}", delta=f"{(auc - 0.5):.3f} above random")
        m3, m4 = st.columns(2)
        m3.metric("Total Predictions", f"{n_preds:,}", help="Out-of-sample events across all WF folds")
        m4.metric("Training Samples", f"{n_samples:,}", help="Labelled rows after neutral zone removal")
        if sharpe is not None:
            m5, m6 = st.columns(2)
            m5.metric("Signal Sharpe (IS)", f"{sharpe:.2f}",
                     delta=(f"{sharpe:+.2f}" if abs(sharpe) < 5 else "capped"),
                     help="In-sample Sharpe — optimistic. Use for sanity check only, not for live sizing.")
            m6.metric("Accuracy Std", f"{acc_std * 100:.1f}%", help="< 8% = stable across regimes")

        if balance:
            st.markdown("---")
            st.markdown("**Class Balance**")
            bull_pct = balance.get("bull_pct", 50)
            bear_pct = balance.get("bear_pct", 50)
            n_bull = balance.get("n_bullish", 0)
            n_bear = balance.get("n_bearish", 0)
            if balance.get("imbalance_flag", False):
                st.warning(f"Imbalanced classes detected: {bull_pct:.0f}% bullish / {bear_pct:.0f}% bearish. scale_pos_weight has been adjusted.")
            else:
                st.caption(f"Balanced: {bull_pct:.0f}% bullish ({n_bull:,}) / {bear_pct:.0f}% bearish ({n_bear:,})")

    with col2:
        st.markdown("#### Fold-by-Fold Accuracy")
        st.caption(
            "Per-fold accuracy chart requires running Generate Prediction first. "
            "Walk-forward mean accuracy and std are shown in the Performance Metrics panel."
        )


def _render_prediction_history(ticker: str):
    hist_df = get_prediction_history(ticker)

    if hist_df.empty:
        st.info("No prediction history yet for this ticker. Generate a prediction to start tracking.")
        return

    hist_df = hist_df.dropna(subset=["date"])
    if hist_df.empty:
        st.info("No prediction history yet for this ticker. Generate a prediction to start tracking.")
        return

    # Year/month filter — defaults to the current month. Plotly's autorange on
    # a date axis with sparse/clustered points (a new ticker with only a
    # handful of logged predictions) can blow out to a multi-decade span, so
    # we always pick an explicit window rather than relying on autorange.
    dates_et = hist_df["date"].dt.tz_convert(MARKET_TZ)
    today_et = now_et()
    years_available = sorted(set(dates_et.dt.year.tolist()) | {today_et.year}, reverse=True)
    month_names = ["All Months", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    col_year, col_month = st.columns(2)
    with col_year:
        year_sel = st.selectbox(
            "Year", years_available, index=years_available.index(today_et.year), key=f"pred_hist_year_{ticker}"
        )
    with col_month:
        month_sel = st.selectbox(
            "Month", month_names, index=today_et.month, key=f"pred_hist_month_{ticker}"
        )

    mask = dates_et.dt.year == year_sel
    if month_sel != "All Months":
        mask &= dates_et.dt.month == month_names.index(month_sel)
    window_df = hist_df[mask].sort_values("date", ascending=False)
    window_label = f"{month_sel} {year_sel}" if month_sel != "All Months" else str(year_sel)

    if window_df.empty:
        st.caption(f"No predictions logged in {window_label}.")
        return

    st.markdown("#### Prediction Signal History")
    if len(window_df) >= 2:
        plot_df = window_df.sort_values("date")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=plot_df["date"], y=(plot_df["probability"] * 100).round(1),
            mode="lines+markers", name="Bull Probability %",
            line=dict(color="#2196f3", width=2), marker=dict(size=5),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Bull prob: %{y:.1f}%<extra></extra>",
        ))

        signal_colors = {"bullish": "#26a69a", "bearish": "#ef5350", "neutral": "#9e9e9e"}
        for sig, grp in plot_df.groupby("direction"):
            fig_hist.add_trace(go.Scatter(
                x=grp["date"], y=(grp["probability"] * 100).round(1), mode="markers", name=sig.upper(),
                marker=dict(size=10, color=signal_colors.get(sig, "#9e9e9e"), symbol="circle", line=dict(width=1, color="white")),
                hovertemplate=f"<b>{sig.upper()}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>Prob: %{{y:.1f}}%<extra></extra>",
                showlegend=True,
            ))

        fig_hist.add_hline(y=50, line_dash="dot", line_color="rgba(158,158,158,0.5)")
        fig_hist.add_hrect(y0=45, y1=55, fillcolor="rgba(158,158,158,0.08)", line_width=0, annotation_text="Neutral zone")

        if month_sel != "All Months":
            range_start = pd.Timestamp(year=year_sel, month=month_names.index(month_sel), day=1, tz=MARKET_TZ)
        else:
            range_start = pd.Timestamp(year=year_sel, month=1, day=1, tz=MARKET_TZ)
        range_end = range_start + pd.DateOffset(months=1 if month_sel != "All Months" else 12) - pd.Timedelta(seconds=1)

        fig_hist.update_layout(
            template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", height=260, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Bull Probability %", range=[33, 67], gridcolor="rgba(255,255,255,0.07)"),
            xaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)", range=[range_start, range_end]),
            legend=dict(orientation="h", y=1.1, font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,17,27,0.6)" if st.context.theme.type == "dark" else "rgba(255,255,255,0.6)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.caption(f"Only 1 prediction logged in {window_label} — not enough points yet to chart.")

    st.markdown(f"#### Predictions — {window_label}")

    display_df = window_df.head(15).copy()

    if "date" in display_df.columns:
        display_df["Date"] = (
            pd.to_datetime(display_df["date"], utc=True)
            .dt.tz_convert(MARKET_TZ)
            .dt.strftime("%Y-%m-%d %I:%M %p ET")
        )
    else:
        display_df["Date"] = "N/A"

    if "probability" in display_df.columns:
        display_df["Probability"] = (display_df["probability"] * 100).map("{:.1f}%".format)
    else:
        display_df["Probability"] = "N/A"

    if "model_accuracy" in display_df.columns:
        display_df["Model Acc"] = display_df["model_accuracy"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A")
    else:
        display_df["Model Acc"] = "N/A"

    if "expected_move_pct" in display_df.columns:
        display_df["Exp Move"] = display_df["expected_move_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) and v is not None else "—")
    else:
        display_df["Exp Move"] = "—"

    if "direction" in display_df.columns:
        display_df["direction_display"] = display_df["direction"].str.upper()
    else:
        display_df["direction_display"] = "N/A"

    cols_to_show = ["Date", "direction_display", "Probability", "confidence", "Exp Move", "Model Acc"]
    available_cols = [c for c in cols_to_show if c in display_df.columns]
    table_df = display_df[available_cols].rename(columns={"direction_display": "Direction", "confidence": "Confidence"})

    def _style_signal_row(row):
        direction = row.get("Direction", "NEUTRAL")
        if direction == "BULLISH":
            bg = "background-color: rgba(38, 166, 154, 0.10)"
        elif direction == "BEARISH":
            bg = "background-color: rgba(239, 83, 80, 0.10)"
        else:
            bg = "background-color: rgba(158, 158, 158, 0.06)"
        return [bg] * len(row)

    styled = table_df.style.apply(_style_signal_row, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    total = len(window_df)
    bull_count = (window_df["direction"].str.lower() == "bullish").sum() if "direction" in window_df else 0
    bear_count = (window_df["direction"].str.lower() == "bearish").sum() if "direction" in window_df else 0
    neut_count = total - bull_count - bear_count
    st.caption(
        f"Predictions logged in {window_label}: **{total}** — "
        f"Bullish: **{bull_count}** | Bearish: **{bear_count}** | Neutral: **{neut_count}**"
    )


def _render_predictions_disclaimer():
    st.markdown("---")
    st.markdown(
        """
        <div class="aeth-disclaimer">
        <strong class="aeth-disclaimer__title">Important Disclaimer</strong>
        This ML signal predicts price direction over an automatically-selected horizon
        (3, 5, or 10 trading days, chosen per ticker for the best walk-forward accuracy) with a
        modest edge (typically 52–58% accuracy).
        It cannot predict news events, earnings surprises, or macro regime changes.
        Past walk-forward accuracy is <b>not</b> a guarantee of future performance.
        Probability values are capped at 35–65% — any raw model output beyond these bounds
        is clipped to prevent conveying false precision.<br><br>
        <b>This output is for research purposes only.</b>
        Position sizing decisions should use the Risk page — not this signal alone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_intraday_predictions():
    """
    Intraday (15-min) direction model. Entirely separate from the daily model:
    its own module, its own feature set, its own {TICKER}_{interval}_* storage
    files. Nothing here reads or writes a daily model.
    """
    try:
        from analysis.intraday_prediction import (
            INTERVAL_SPECS,
            get_intraday_prediction_history,
            load_metadata,
            model_exists as intraday_model_exists,
            predict_intraday,
            train_intraday_model,
        )
    except Exception as exc:
        logger.warning(f"[trading] intraday prediction module unavailable: {exc}")
        st.info("Intraday predictions require the ML dependencies. Run `pip install -r requirements.txt`.")
        return

    st.caption(
        "Separate model from the daily predictor — intraday features (time of day, "
        "distance from session VWAP, position in the opening range), volatility-scaled "
        "labels, and forward returns that never cross the overnight gap."
    )
    st.warning("For research only. Not financial advice.", icon="⚠️")

    c_ticker, c_interval, c_train, c_predict = st.columns([2, 1, 1, 1])
    with c_ticker:
        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("intraday_pred_ticker", "SPY"),
            key="intraday_pred_ticker_input",
        ).upper().strip()
    with c_interval:
        intervals = list(INTERVAL_SPECS)
        interval = st.selectbox(
            "Bar interval", intervals, index=intervals.index("15m"),
            key="intraday_pred_interval",
        )
    with c_train:
        train_btn = st.button(
            "Train / Update", type="secondary", width="stretch", key="intraday_train_btn",
            help="Search label horizon and threshold, then validate with 10-fold walk-forward.",
        )
    with c_predict:
        predict_btn = st.button(
            "Generate Prediction", type="primary", width="stretch", key="intraday_predict_btn",
        )

    if not ticker:
        st.info("Enter a ticker symbol to get started.")
        return
    st.session_state["intraday_pred_ticker"] = ticker

    meta = load_metadata(ticker, interval)
    has_model = intraday_model_exists(ticker, interval)

    s1, s2 = st.columns(2)
    with s1:
        if has_model:
            st.markdown(f"✅ **Model status:** Trained ({interval})")
        else:
            st.markdown(f"❌ **Model status:** Not trained for {interval}")
    with s2:
        if meta.get("trained_at"):
            try:
                st.caption(f"Last trained: **{utc_iso_to_et_str(meta['trained_at'], '%Y-%m-%d %I:%M %p ET')}**")
            except Exception:
                pass
        if meta.get("horizon_minutes"):
            st.caption(f"Horizon: **{meta['horizon_bars']} bars ({meta['horizon_minutes']} min)**")

    st.info(
        f"Intraday history is capped at ~{INTERVAL_SPECS[interval]['max_period']} by the data "
        "provider, so this model sees far less regime variety than the daily one and goes "
        "stale faster. Retrain every few days.",
        icon="ℹ️",
    )
    st.markdown("---")

    if train_btn:
        logger.info(f"[trading] intraday train pressed for {ticker} {interval}")
        with st.spinner(f"Training {interval} model for {ticker}..."):
            result = train_intraday_model(ticker, interval)

        if result.get("error"):
            logger.warning(f"[trading] intraday training failed for {ticker} {interval}: {result['error']}")
            st.error(f"Training failed: {result['error']}")
            return

        acc = result["directional_accuracy"]
        trade = result["tradeability"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Directional Accuracy", f"{acc * 100:.1f}%", f"±{result['accuracy_std'] * 100:.1f}%")
        m2.metric("Horizon", f"{result['horizon_bars']} bars", f"{result['horizon_minutes']} min")
        m3.metric("Neutral Band", f"±{result['threshold_pct']:.3f}%")
        m4.metric("Training Samples", f"{result['n_train']:,}")

        if result["is_reliable"]:
            st.success(f"Walk-forward: {result['reliability_reason']}")
        else:
            st.warning(f"Walk-forward: {result['reliability_reason']}")

        # Accuracy alone is misleading at intraday horizons — show the cost math.
        if trade["is_tradeable"]:
            st.success(
                f"**After costs:** net edge {trade['net_edge_pct']:+.4f}% per trade "
                f"(avg move {trade['avg_move_pct']:.3f}%, breakeven accuracy "
                f"{trade['breakeven_accuracy'] * 100:.1f}%)."
            )
        else:
            st.error(
                f"**Not tradeable after costs.** Net edge {trade['net_edge_pct']:+.4f}% per trade. "
                f"This model needs {trade['breakeven_accuracy'] * 100:.1f}% accuracy to break even "
                f"against a {trade['round_trip_cost_pct']:.2f}% round trip, and it scores "
                f"{acc * 100:.1f}%. A statistically real edge can still lose money."
            )

        if result.get("session_mask_dropped"):
            st.caption(
                f"{result['session_mask_dropped']:,} bars excluded because their forward "
                "window crossed the session close."
            )
        with st.expander("Label search results", expanded=False):
            st.dataframe(pd.DataFrame(result["label_search"]), hide_index=True, width="stretch")

    if predict_btn:
        logger.info(f"[trading] intraday predict pressed for {ticker} {interval}")
        with st.spinner(f"Predicting {interval} direction for {ticker}..."):
            result = predict_intraday(ticker, interval)

        if result.get("error"):
            logger.warning(f"[trading] intraday predict failed for {ticker} {interval}: {result['error']}")
            st.error(result["error"])
            return

        direction = result["direction"]
        icon = {"bullish": "🟢", "bearish": "🔴"}.get(direction, "⚪")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(f"{icon} Direction", direction.upper(), f"Confidence: {result['confidence'].title()}")
        p2.metric("Bull Probability", f"{result['probability'] * 100:.0f}%", "Neutral zone: 45–55%")
        p3.metric("Horizon", f"{result['horizon_minutes']} min", f"{result['horizon_bars']} bars")
        p4.metric("Model Accuracy", f"{result['model_accuracy'] * 100:.1f}%")

        trade = result["tradeability"]
        if not trade["is_tradeable"]:
            st.warning(
                f"This model's edge does not survive costs (net {trade['net_edge_pct']:+.4f}% "
                f"per trade). Treat the signal as information, not a trade.",
                icon="⚠️",
            )
        st.caption(
            f"Signal from the {result['bar_timestamp']} bar at "
            f"{result['price_at_prediction']:.2f} · neutral band ±{result['threshold_pct']:.3f}%"
        )
        log_activity("intraday_prediction_generated", ticker, result)

    st.markdown("---")
    st.markdown(f"#### Prediction History — {ticker} ({interval})")
    hist = get_intraday_prediction_history(ticker, interval)
    if hist.empty:
        st.info("No intraday predictions logged yet for this ticker and interval.")
    else:
        resolved = hist[hist["correct"].notna()]
        if not resolved.empty:
            hit_rate = resolved["correct"].astype(bool).mean() * 100
            h1, h2 = st.columns(2)
            h1.metric("Logged Predictions", len(hist))
            h2.metric("Resolved Hit Rate", f"{hit_rate:.1f}%", f"{len(resolved)} resolved")
        else:
            st.caption(
                f"{len(hist)} logged · none resolved yet — a prediction resolves once its "
                "horizon elapses within the same session."
            )
        display = hist.copy()
        display["date"] = display["date"].dt.tz_convert(MARKET_TZ).dt.strftime("%Y-%m-%d %I:%M %p ET")
        st.dataframe(display, hide_index=True, width="stretch")


def _render_predictions():
    """
    Dispatcher for the Predictions tab. Defaults to Daily, which renders the
    original code path unchanged — the intraday model lives in its own module
    (analysis/intraday_prediction.py) and its own storage files, so selecting it
    cannot affect daily models or the Research page.
    """
    st.subheader("AI Price Predictions")

    if not _ML_AVAILABLE:
        st.info("ML predictions require the `scikit-learn`, `xgboost`, and `narwhals` packages. Run `pip install -r requirements.txt`.")
        return

    horizon_mode = st.radio(
        "Prediction horizon",
        ["Daily (swing)", "Intraday (15-min bars)"],
        index=0,
        horizontal=True,
        key="pred_horizon_mode",
        help=(
            "Daily predicts direction several trading days out. Intraday predicts "
            "direction a few bars (minutes) out using a separate model with its own "
            "features, labels, and stored files."
        ),
    )

    if horizon_mode.startswith("Daily"):
        _render_daily_predictions()
    else:
        _render_intraday_predictions()


def _render_daily_predictions():
    if not _ML_AVAILABLE:
        return
    st.caption("ML ensemble model — XGBoost + Random Forest — trained on 18 technical features derived from historical price action.")
    st.warning("For research only. Not financial advice. Accuracy varies by market conditions.", icon="⚠️")
    st.markdown("")

    col_ticker, col_train, col_predict = st.columns([2, 1, 1])

    with col_ticker:
        ticker = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get("predictions_ticker", "SPY"),
            placeholder="e.g. AAPL, NVDA, MSFT",
            key="predictions_ticker_input",
        ).upper().strip()

    with col_train:
        train_btn = st.button("Train / Update Model", type="secondary", width="stretch",
                              help="Train or refresh the XGBoost + RF ensemble for this ticker, searching for the best label horizon and hyperparameters. Takes 20–45 seconds.",
                              key="pred_train_btn")

    with col_predict:
        predict_btn = st.button("Generate Prediction", type="primary", width="stretch",
                                help="Run the trained model on the latest bar and output a direction signal over its auto-selected horizon.",
                                key="pred_predict_btn")

    if not ticker:
        st.info("Enter a ticker symbol above to get started.")
        return

    st.session_state["predictions_ticker"] = ticker

    status_col1, status_col2, status_col3 = st.columns(3)
    model_exists = _model_exists(ticker)
    trained_at = _trained_at_str(ticker) if model_exists else None
    overdue = _retrain_overdue(ticker) if model_exists else False

    with status_col1:
        if model_exists:
            icon = "✅" if not overdue else "⏰"
            st.markdown(f"{icon} **Model status:** {'Trained' if not overdue else 'Overdue for refresh'}")
        else:
            st.markdown("❌ **Model status:** Not trained")

    with status_col2:
        if trained_at:
            st.caption(f"Last trained: **{trained_at}**")
        else:
            st.caption("No model on disk for this ticker.")

    with status_col3:
        cache_key = _cache_key(ticker)
        if cache_key in st.session_state:
            pred_ts = st.session_state[cache_key].get("last_trained", "")
            if pred_ts:
                try:
                    st.caption(f"Last prediction: **{utc_iso_to_et_str(pred_ts, '%I:%M %p ET')}** (this session)")
                except Exception:
                    pass

    st.markdown("---")

    if train_btn:
        logger.info(f"[trading] 'Train / Update Model' button pressed for {ticker}")
        with st.spinner(f"Loading price data for {ticker}..."):
            df = _load_pred_df(ticker, period="2y")

        if df is None or df.empty:
            logger.warning(f"[trading] ML training for {ticker} blocked: could not load price data")
            st.error(f"Could not load price data for **{ticker}**. Check the ticker symbol.")
            return

        n_bars = len(df)
        if n_bars < 60:
            logger.warning(f"[trading] ML training for {ticker} blocked: only {n_bars} bars of history (need 60+)")
            st.error(f"Only {n_bars} bars of history found for {ticker}. ML training requires at least 60 bars (2+ years recommended).")
            return

        st.info(f"Training on **{n_bars} bars** of {ticker} daily data. Running 10-fold walk-forward validation — this takes ~10–20 seconds...", icon="⚙️")

        try:
            with st.spinner("Training XGBoost + Random Forest ensemble..."):
                train_result = train_model(ticker, df)

            if train_result.get("error"):
                logger.warning(f"[trading] ML training failed for {ticker}: {train_result['error']}")
                st.error(f"Training failed: {train_result['error']}")
                return

            mean_acc = train_result.get("directional_accuracy") or 0.0
            acc_std = train_result.get("accuracy_std") or 0.0
            reliable = train_result.get("is_reliable", False)
            reliable_icon = "✅" if reliable else "⚠️"
            logger.info(
                f"[trading] ML model trained for {ticker}: accuracy={mean_acc * 100:.1f}% "
                f"± {acc_std * 100:.1f}% reliable={reliable}"
            )
            st.success(f"Model trained successfully for **{ticker}**. {reliable_icon} Walk-forward accuracy: **{mean_acc * 100:.1f}% ± {acc_std * 100:.1f}%**")

            st.session_state.pop(_cache_key(ticker), None)
            st.session_state.pop(_eval_cache_key(ticker), None)
            st.session_state.pop(_price_path_cache_key(ticker), None)

        except ValueError as exc:
            logger.warning(f"[trading] ML training for {ticker} failed — insufficient data: {exc}")
            st.error(f"Training failed — insufficient data: {exc}")
            return
        except Exception as exc:
            logger.error(f"[trading] ML training for {ticker} raised an unexpected error: {exc}", exc_info=True)
            st.error(f"Training error: {exc}")
            return

    if predict_btn:
        logger.info(f"[trading] 'Generate Prediction' button pressed for {ticker}")
        if not _model_exists(ticker):
            logger.warning(f"[trading] Prediction blocked for {ticker}: no trained model found")
            st.warning(f"No trained model found for **{ticker}**. Click **Train / Update Model** first.", icon="⚠️")
        else:
            with st.spinner(f"Loading price data and running inference for {ticker}..."):
                df = _load_pred_df(ticker, period="2y")

            if df is None or df.empty:
                logger.warning(f"[trading] Prediction for {ticker} blocked: could not load price data")
                st.error(f"Could not load price data for **{ticker}**.")
            else:
                try:
                    with st.spinner("Running ML inference..."):
                        result = predict(ticker, df)

                    if result.get("error"):
                        logger.warning(f"[trading] Prediction failed for {ticker}: {result['error']}")
                        st.error(f"Prediction failed: {result['error']}")
                    else:
                        logger.info(
                            f"[trading] Prediction generated for {ticker}: direction={result.get('direction')} "
                            f"probability={result.get('probability')} confidence={result.get('confidence')}"
                        )
                        st.session_state[_cache_key(ticker)] = result
                        log_activity("prediction_generated", ticker, {
                            "direction": result.get("direction"),
                            "probability": result.get("probability"),
                            "confidence": result.get("confidence"),
                            "horizon_days": result.get("horizon_days"),
                            "expected_move_pct": result.get("expected_move_pct"),
                        })
                        with st.spinner("Running model evaluation..."):
                            eval_result = evaluate_model(ticker, df)
                        st.session_state[_eval_cache_key(ticker)] = eval_result

                        price_path = simulate_price_path(
                            df,
                            bull_probability=result.get("probability", 0.5),
                            expected_move_pct=result.get("expected_move_pct"),
                            n_days=result.get("horizon_days") or 5,
                        )
                        st.session_state[_price_path_cache_key(ticker)] = price_path

                except Exception as exc:
                    logger.error(f"[trading] Unexpected prediction error for {ticker}: {exc}", exc_info=True)
                    st.error(f"Unexpected prediction error: {exc}")

    cached_result = st.session_state.get(_cache_key(ticker))
    cached_eval = st.session_state.get(_eval_cache_key(ticker))
    cached_price_path = st.session_state.get(_price_path_cache_key(ticker))

    if cached_result and not cached_result.get("error"):
        st.markdown("## Direction Signal")
        _render_prediction_card(cached_result)

        st.markdown("---")
        st.markdown(f"## {cached_result.get('horizon_days') or 5}-Day Price Path (Simulated)")
        if cached_price_path:
            _render_price_path(cached_price_path)
        else:
            st.info("Run **Generate Prediction** to load the simulated price path.")

        st.markdown("---")
        st.markdown("## Feature Importance")
        top_features = cached_result.get("top_features", [])
        if top_features:
            _render_feature_importance(top_features)
        else:
            st.caption("No feature importance data available for this prediction.")

        st.markdown("---")
        st.markdown("## Model Performance")
        if cached_eval:
            _render_model_performance(cached_eval, ticker)
        else:
            st.info("Run **Generate Prediction** to load model performance metrics.")

        with st.expander("Model Details — Walk-Forward Configuration", expanded=False):
            _horizon = cached_result.get("horizon_days") or 5
            _neutral_pct = cached_result.get("neutral_threshold_pct")
            _neutral_str = f"{_neutral_pct}%" if _neutral_pct is not None else "0.5%"
            _hp_overrides = cached_result.get("hyperparam_overrides") or {}
            _hp_str = ", ".join(f"`{k}={v}`" for k, v in _hp_overrides.items()) or "defaults (`max_depth=4`, `learning_rate=0.05`, `n_estimators=200`)"
            st.markdown(
                f"""
                **Training configuration:**
                - Model: XGBoost (`binary:logistic`) + Random Forest (calibration ensemble)
                - Ensemble weights: 65% XGBoost + 35% RF
                - Label horizon: **{_horizon} trading days** — auto-selected per ticker from a
                  {{3, 5, 10}}-day grid by comparing reduced-fold walk-forward accuracy
                - XGBoost hyperparameters: {_hp_str} — auto-tuned per ticker via the same
                  reduced-fold walk-forward search
                - Walk-forward splits: `TimeSeriesSplit(n_splits=10, gap={_horizon})`
                - Gap = {_horizon} bars: prevents the {_horizon}-day forward return from bleeding
                  into training features
                - Reliability threshold: mean accuracy ≥ 52% AND std ≤ 8% across folds
                - Neutral dead-band: bull probability in [0.45, 0.55] → signal = NEUTRAL
                - Probability display range: capped to [35%, 65%]

                **Feature engineering:**
                - 18 features derived from `calculate_indicators()` output (no external data sources)
                - Neutral zone removed from training: |{_horizon}-day forward return| < {_neutral_str} rows dropped
                - Minimum bars required: 60 (60–250 bars produces low accuracy; 500+ recommended)

                **Retrain policy:**
                - Bundle expires 30 days after training
                - Retrain when `retrain_overdue = True` to capture recent price action
                """
            )

        st.markdown("---")
        st.markdown("## Prediction History")
        _render_prediction_history(ticker)

    elif _model_exists(ticker):
        st.info(f"Model exists for **{ticker}**. Click **Generate Prediction** to produce a signal.", icon="ℹ️")
        hist_df = get_prediction_history(ticker)
        if not hist_df.empty:
            st.markdown("---")
            st.markdown("## Prediction History")
            _render_prediction_history(ticker)

    else:
        st.markdown("---")
        st.markdown(
            """
            ### Getting Started

            1. Enter a ticker symbol (e.g. **AAPL**, **NVDA**, **SPY**)
            2. Click **Train / Update Model** to train the XGBoost + RF ensemble on 2 years of daily data
            3. Click **Generate Prediction** to see the directional signal

            **What the model predicts:**
            - Direction: BULLISH / NEUTRAL / BEARISH over an automatically-selected horizon
              (3, 5, or 10 trading days — whichever the ensemble is most consistently accurate
              at for this ticker)
            - Probability: Calibrated bull probability, capped to 35–65% to prevent false precision
            - A simulated open/close price band over that same horizon (Monte Carlo, seeded from
              the stock's own volatility and the model's bias) — a probability range, not a point forecast
            - Feature importance: Which technical indicators drove this prediction
            - Walk-forward accuracy: Out-of-sample performance across 10 validation folds

            **What the model does not predict:**
            - A single exact price target — the classifier itself only outputs direction/probability;
              the price band shown is a separate volatility simulation, not a regression output
            - News events, earnings surprises, or macro regime changes
            - Intraday moves (use the Day Trading tab for intraday signals)
            """
        )

    _render_predictions_disclaimer()


# ══════════════════════════════════════════════════════════════════════════
# ── News & Sentiment ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_news():
    st.subheader("News & Sentiment")
    st.caption("Headline-level VADER sentiment from Google News. Display-only — not a trading signal on its own.")

    if not _NEWS_AVAILABLE:
        st.info("News sentiment requires the `feedparser` and `vaderSentiment` packages. Run `pip install -r requirements.txt`.")
        return

    ticker = st.text_input("Ticker", value="SPY", placeholder="e.g. AAPL, SPY, QQQ", key="news_ticker").upper().strip()
    if not ticker:
        return

    with st.spinner(f"Fetching recent headlines for {ticker}..."):
        articles = fetch_ticker_news(ticker)
        sentiment = analyze_ticker_sentiment(articles)

    if sentiment["n_articles"] == 0:
        logger.warning(f"[trading] News tab: no recent headlines found for {ticker}")
        st.warning("No recent headlines found for this ticker.")
        return

    overall = sentiment["overall_label"]
    direction = "bull" if overall == "Bullish" else ("bear" if overall == "Bearish" else "neutral")

    st.markdown(
        _signal_html(
            "Overall Sentiment",
            overall,
            f"Mean compound score {sentiment['mean_compound']:+.3f} across {sentiment['n_articles']} headlines",
            direction,
        ),
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positive", f"{sentiment['positive_pct']:.0f}%")
    col2.metric("⚪ Neutral", f"{sentiment['neutral_pct']:.0f}%")
    col3.metric("🔴 Negative", f"{sentiment['negative_pct']:.0f}%")

    fig = px.bar(
        x=["Positive", "Neutral", "Negative"],
        y=[sentiment["positive_pct"], sentiment["neutral_pct"], sentiment["negative_pct"]],
        color=["Positive", "Neutral", "Negative"],
        color_discrete_map={"Positive": "#26a69a", "Neutral": "#9e9e9e", "Negative": "#ef5350"},
        labels={"x": "", "y": "% of headlines"},
    )
    fig.update_layout(template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white", showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Recent Headlines**")
    for article in sentiment["articles"]:
        label = article["label"]
        icon = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "⚪")
        st.markdown(f"{icon} [{article['title']}]({article['link']})")
        st.caption(f"{article.get('source', '')} · {article.get('published', '')} · compound {article['compound']:+.3f}")


# ══════════════════════════════════════════════════════════════════════════
# ── Page entry point ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def render():
    st.title("Trading Desk")
    st.caption("Day Trading signals, Options analysis, News sentiment, and ML Predictions in one place.")

    tab_dt, tab_opt, tab_news, tab_pred = st.tabs(["📉 Day Trading", "🎯 Options", "📰 News", "🤖 Predictions"])

    with tab_dt:
        _render_daytrading()

    with tab_opt:
        _render_options()

    with tab_news:
        _render_news()

    with tab_pred:
        _render_predictions()


render()
