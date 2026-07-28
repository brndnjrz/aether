"""
Strategy Lab — two independent intraday strategies, each with a live scanner
and a mechanical backtest over recent history.

1. ORBC (Opening Range Breakout Confirmation) — define the opening range from
   the first N minutes after the 9:30 ET open, then require a second (or
   third) consecutive close outside that range before signalling, which
   filters the false breakouts common right after the open. Logic in
   analysis/orbc_strategy.py.

2. MTF setup — identify trend on the 4-hour chart, wait for a pullback into a
   demand zone on the 30-minute chart, confirm a market-structure shift on the
   5-minute chart, read the tape for absorption then buyers taking control,
   enter targeting the VAP with a stop at the swing low. Logic in
   analysis/mtf_strategy.py.

This page is display only — all detection lives in the analysis modules.
"""
import logging
import os
import sys
from datetime import time as dtime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.price_data import get_price_history
from analysis.indicators import calculate_indicators
from analysis.mtf_strategy import resample_to_4h, evaluate_setup, backtest_setup
from analysis.orbc_strategy import (
    ORBCConfig,
    backtest_orbc,
    latest_session_state,
    to_market_tz,
)
from portfolio.activity_log import log_activity

logger = logging.getLogger(__name__)

TAPE_PROXY_NOTICE = (
    "Tape reading here is a price/volume proxy — shrinking range and volume "
    "decline (absorption), then a high-volume reversal candle that reclaims "
    "VWAP/short structure (buyers-in-control proxy). This app has no Level 2 "
    "or order-flow data, so it is not literal absorption of passive offers."
)


@st.cache_data(ttl=60)
def _load_5m(ticker: str):
    df = get_price_history(ticker, period="5d", interval="5m")
    if df is None or df.empty:
        logger.debug(f"[strategy_lab] 5m load for {ticker} came back empty")
        return None
    logger.debug(f"[strategy_lab] 5m load for {ticker} succeeded with {len(df)} rows")
    return calculate_indicators(df)


@st.cache_data(ttl=300)
def _load_30m(ticker: str):
    df = get_price_history(ticker, period="60d", interval="30m")
    if df is None or df.empty:
        logger.debug(f"[strategy_lab] 30m load for {ticker} came back empty")
        return None
    logger.debug(f"[strategy_lab] 30m load for {ticker} succeeded with {len(df)} rows")
    return calculate_indicators(df)


@st.cache_data(ttl=300)
def _load_4h(ticker: str):
    df_1h = get_price_history(ticker, period="60d", interval="1h")
    if df_1h is None or df_1h.empty:
        logger.debug(f"[strategy_lab] 4h load for {ticker} came back empty (no 1h source data)")
        return None
    df_4h = resample_to_4h(df_1h)
    if df_4h is None or df_4h.empty:
        logger.debug(f"[strategy_lab] 4h load for {ticker} came back empty after resampling")
        return None
    logger.debug(f"[strategy_lab] 4h load for {ticker} succeeded with {len(df_4h)} rows")
    return calculate_indicators(df_4h)


def _stage_card(label: str, stage: dict) -> str:
    css = "signal-bull" if stage["ok"] else "signal-neutral"
    icon = "🟢" if stage["ok"] else "⚪"
    value = "Confirmed" if stage["ok"] else "Not yet"
    return f"""
<div class="signal-card {css}">
  <strong>{icon} {label}</strong><br>
  <span class="signal-value">{value}</span><br>
  <span class="signal-note">{stage['reason']}</span>
</div>"""


def _candlestick_row(fig: go.Figure, df: pd.DataFrame, row: int, name: str):
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=name, showlegend=False,
        ),
        row=row, col=1,
    )


def _render_scanner(ticker: str):
    df_4h = _load_4h(ticker)
    df_30m = _load_30m(ticker)
    df_5m = _load_5m(ticker)

    if df_4h is None or df_30m is None or df_5m is None:
        logger.warning(
            f"[strategy_lab] Scanner blocked for {ticker}: missing data "
            f"(4h={'ok' if df_4h is not None else 'missing'}, "
            f"30m={'ok' if df_30m is not None else 'missing'}, "
            f"5m={'ok' if df_5m is not None else 'missing'})"
        )
        st.warning(f"Not enough intraday data available for {ticker} yet.")
        return

    setup = evaluate_setup(df_4h, df_30m, df_5m)

    st.caption(TAPE_PROXY_NOTICE)

    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("4H Trend", "trend_4h"),
        ("30m Pullback", "pullback_30m"),
        ("5m Structure Shift", "structure_shift_5m"),
        ("Tape Proxy", "tape_proxy_5m"),
    ]):
        col.markdown(_stage_card(label, setup[key]), unsafe_allow_html=True)

    theme = "plotly_dark" if st.context.theme.type == "dark" else "plotly_white"
    fig = make_subplots(rows=3, cols=1, subplot_titles=("4-Hour", "30-Minute", "5-Minute"), vertical_spacing=0.08)
    _candlestick_row(fig, df_4h.tail(60), 1, "4H")
    _candlestick_row(fig, df_30m.tail(80), 2, "30m")
    _candlestick_row(fig, df_5m.tail(120), 3, "5m")

    if setup["setup_valid"]:
        for price, label, color in [
            (setup["entry_price"], "Entry", "#2196f3"),
            (setup["target_price"], "Target (VAP proxy)", "#26a69a"),
            (setup["stop_price"], "Stop (swing low)", "#ef5350"),
        ]:
            fig.add_hline(y=price, row=3, col=1, line_dash="dot", line_color=color,
                           annotation_text=f"{label} {price:.2f}", annotation_position="right")

    fig.update_layout(template=theme, height=800, margin=dict(l=0, r=0, t=40, b=0), xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if setup["setup_valid"]:
        rr = setup["rr_ratio"]
        logger.info(
            f"[strategy_lab] Setup confirmed for {ticker}: entry={setup['entry_price']:.2f} "
            f"target={setup['target_price']:.2f} stop={setup['stop_price']:.2f} rr={rr:.2f}"
        )
        st.success(
            f"**Setup confirmed** — Entry {setup['entry_price']:.2f} · "
            f"Target {setup['target_price']:.2f} · Stop {setup['stop_price']:.2f} · R:R {rr:.2f}"
        )
        if setup.get("target_is_vwap_fallback"):
            st.caption("No distinct high-volume node found above entry — target fell back to session VWAP.")
        if st.button(f"Log this {ticker} setup", key="log_setup_btn"):
            logger.info(f"[strategy_lab] 'Log this setup' button pressed for {ticker}")
            log_activity("strategy_lab_setup", ticker, setup)
            logger.info(f"[strategy_lab] Setup for {ticker} logged to activity log")
            st.toast("Setup logged.")
    else:
        st.info("Setup not confirmed yet — waiting on the checklist above.")


def _render_backtest(ticker: str):
    st.caption(
        "Intraday history is limited by the data provider to roughly the trailing "
        "60 days. The backtest window below reflects what was actually fetched, "
        "bounded by the shortest of the 4H/30m/5m series."
    )
    st.caption(TAPE_PROXY_NOTICE)
    if st.button(f"Run backtest on {ticker}", key="run_backtest_btn"):
        logger.info(f"[strategy_lab] 'Run backtest' button pressed for {ticker}")
        with st.spinner(f"Scanning {ticker} history for setups..."):
            result = backtest_setup(ticker)

        if result.get("error"):
            logger.warning(f"[strategy_lab] Backtest for {ticker} returned an error: {result['error']}")
            st.warning(result["error"])
            return

        logger.info(
            f"[strategy_lab] Backtest for {ticker} completed: "
            f"num_trades={result['num_trades']} win_rate={result['win_rate']:.1f}% "
            f"avg_rr={result['avg_rr']:.2f} total_return_pct={result['total_return_pct']:+.1f}%"
        )

        st.caption(f"Window: {result['window_start']:%Y-%m-%d %H:%M} → {result['window_end']:%Y-%m-%d %H:%M}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", result["num_trades"])
        c2.metric("Win Rate", f"{result['win_rate']:.1f}%")
        c3.metric("Avg R:R", f"{result['avg_rr']:.2f}")
        c4.metric("Total Return", f"{result['total_return_pct']:+.1f}%")

        if result["num_trades"] == 0:
            st.info("No qualifying setups fired in the available history.")
            return

        st.line_chart(result["equity_curve"])

        trades_df = pd.DataFrame([
            {
                "Entry Date": t["entry_date"],
                "Entry": t["entry_price"],
                "Stop": t["stop_price"],
                "Target": t["target_price"],
                "Exit": t["exit_price"],
                "Exit Reason": t["exit_reason"],
                "Return %": t["return_pct"],
                "Bars Held": t["holding_period_bars"],
                "R:R": t["rr_ratio"],
            }
            for t in result["trades"]
        ])
        st.dataframe(trades_df, hide_index=True, width="stretch")


def _render_mtf(ticker: str):
    st.markdown(
        "4H trend → 30m pullback into a demand zone → 5m structure shift → "
        "tape confirmation → entry targeting the VAP with a stop at the swing low."
    )
    sub_scan, sub_backtest = st.tabs(["Live Scanner", "Backtest"])
    with sub_scan:
        _render_scanner(ticker)
    with sub_backtest:
        _render_backtest(ticker)


# ══════════════════════════════════════════════════════════════════════════
# ── ORBC — Opening Range Breakout Confirmation ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════

ORBC_INTERVALS = ["1m", "2m", "5m", "15m"]
ORBC_STOP_LABELS = {
    "Opening range (opposite side)": "opening_range",
    "ATR multiple": "atr",
    "Percent of entry": "percent",
}
ORBC_TARGET_LABELS = {
    "Risk/reward multiple": "risk_reward",
    "ATR multiple": "atr",
    "Opening range projection": "opening_range_extension",
}


@st.cache_data(ttl=60)
def _load_orbc_intraday(ticker: str, interval: str):
    """
    60 days of intraday bars — enough for the backtest, and the same frame
    feeds the live scanner so ATR/VWAP/vol_sma_20 are fully warmed up rather
    than NaN for the first 20 bars of the session (which would silently skip
    every filter on today's signal).
    """
    df = get_price_history(ticker, period="60d", interval=interval)
    if df is None or df.empty:
        logger.debug(f"[strategy_lab] ORBC {interval} load for {ticker} came back empty")
        return None
    logger.debug(f"[strategy_lab] ORBC {interval} load for {ticker}: {len(df)} rows")
    return calculate_indicators(df)


def _orbc_config_controls(key_prefix: str) -> tuple:
    """Renders the ORBC parameter panel and returns (ORBCConfig, interval)."""
    with st.expander("Strategy parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        interval = c1.selectbox(
            "Bar interval", ORBC_INTERVALS, index=ORBC_INTERVALS.index("5m"),
            key=f"{key_prefix}_interval",
            help="Bar size used for both the opening range and the confirmation closes.",
        )
        opening_range_minutes = c2.number_input(
            "Opening range (minutes)", min_value=1, max_value=120, value=15, step=5,
            key=f"{key_prefix}_or_minutes",
            help="Window after 9:30 ET that defines the reference high/low.",
        )
        confirmation_closes = c3.number_input(
            "Confirm on close #", min_value=1, max_value=5, value=2, step=1,
            key=f"{key_prefix}_confirm",
            help="Number of consecutive closes outside the range required to signal.",
        )

        c4, c5, c6 = st.columns(3)
        max_confirmation_closes = c4.number_input(
            "Give up after close #", min_value=1, max_value=8, value=3, step=1,
            key=f"{key_prefix}_max_confirm",
            help=(
                "If a filter blocks the signal on the confirming close, later closes "
                "can still fire up to this count. The spec's 'second OR third candle'."
            ),
        )
        cutoff_hour = c5.number_input(
            "Entry cutoff hour (ET)", min_value=10, max_value=16, value=11, step=1,
            key=f"{key_prefix}_cutoff_h",
        )
        cutoff_minute = c6.number_input(
            "Entry cutoff minute", min_value=0, max_value=59, value=0, step=15,
            key=f"{key_prefix}_cutoff_m",
        )

        st.markdown("**Filters**")
        f1, f2, f3 = st.columns(3)
        require_volume = f1.checkbox("Volume confirmation", value=True, key=f"{key_prefix}_f_vol")
        volume_multiple = f1.number_input(
            "× 20-bar avg volume", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key=f"{key_prefix}_vol_mult", disabled=not require_volume,
        )
        require_vwap = f2.checkbox("VWAP alignment", value=True, key=f"{key_prefix}_f_vwap",
                                   help="Longs only above VWAP, shorts only below.")
        require_atr = f3.checkbox("ATR range filter", value=True, key=f"{key_prefix}_f_atr")
        atr_multiple = f3.number_input(
            "Range > × ATR", min_value=0.1, max_value=5.0, value=0.5, step=0.1,
            key=f"{key_prefix}_atr_mult", disabled=not require_atr,
        )

        st.markdown("**Direction & exits**")
        d1, d2 = st.columns(2)
        allow_long = d1.checkbox("Allow longs", value=True, key=f"{key_prefix}_long")
        allow_short = d2.checkbox("Allow shorts", value=True, key=f"{key_prefix}_short")
        if not (allow_long or allow_short):
            st.warning("At least one direction must be enabled — re-enabling longs.")
            allow_long = True

        e1, e2 = st.columns(2)
        stop_label = e1.selectbox("Stop loss method", list(ORBC_STOP_LABELS), key=f"{key_prefix}_stop_m")
        stop_method = ORBC_STOP_LABELS[stop_label]
        stop_atr_multiple = e1.number_input(
            "Stop ATR ×", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key=f"{key_prefix}_stop_atr", disabled=stop_method != "atr",
        )
        stop_percent = e1.number_input(
            "Stop %", min_value=0.05, max_value=10.0, value=0.5, step=0.05,
            key=f"{key_prefix}_stop_pct", disabled=stop_method != "percent",
        )

        target_label = e2.selectbox("Profit target method", list(ORBC_TARGET_LABELS), key=f"{key_prefix}_tgt_m")
        target_method = ORBC_TARGET_LABELS[target_label]
        target_rr = e2.number_input(
            "Target R:R", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            key=f"{key_prefix}_tgt_rr", disabled=target_method != "risk_reward",
        )
        target_atr_multiple = e2.number_input(
            "Target ATR ×", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            key=f"{key_prefix}_tgt_atr", disabled=target_method != "atr",
        )

        one_signal_per_session = st.checkbox(
            "One signal per session", value=True, key=f"{key_prefix}_one_sig",
            help=(
                "On: the first confirmed signal ends the day's scan, including a "
                "reversal. Off: allows re-entry after a fresh breakout episode."
            ),
        )

    config = ORBCConfig(
        opening_range_minutes=int(opening_range_minutes),
        confirmation_closes=int(confirmation_closes),
        max_confirmation_closes=max(int(max_confirmation_closes), int(confirmation_closes)),
        entry_cutoff=dtime(int(cutoff_hour), int(cutoff_minute)),
        require_volume=require_volume,
        volume_multiple=float(volume_multiple),
        require_vwap=require_vwap,
        require_atr=require_atr,
        atr_multiple=float(atr_multiple),
        allow_long=allow_long,
        allow_short=allow_short,
        stop_method=stop_method,
        stop_atr_multiple=float(stop_atr_multiple),
        stop_percent=float(stop_percent),
        target_method=target_method,
        target_rr=float(target_rr),
        target_atr_multiple=float(target_atr_multiple),
        one_signal_per_session=one_signal_per_session,
    )
    return config, interval


def _orbc_signal_chart(session_df: pd.DataFrame, orange: dict, signals: list, rejections: list, config: ORBCConfig):
    theme = "plotly_dark" if st.context.theme.type == "dark" else "plotly_white"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.75, 0.25], subplot_titles=("Session", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=session_df.index, open=session_df["Open"], high=session_df["High"],
            low=session_df["Low"], close=session_df["Close"], name="Price", showlegend=False,
        ),
        row=1, col=1,
    )

    if "VWAP" in session_df.columns:
        fig.add_trace(
            go.Scatter(x=session_df.index, y=session_df["VWAP"], name="VWAP",
                       line=dict(color="#ff9800", width=1.5)),
            row=1, col=1,
        )

    # Opening range band + edges
    fig.add_hrect(
        y0=orange["opening_low"], y1=orange["opening_high"], row=1, col=1,
        fillcolor="#5c6bc0", opacity=0.12, line_width=0,
    )
    for price, label in [(orange["opening_high"], "Opening High"), (orange["opening_low"], "Opening Low")]:
        fig.add_hline(
            y=price, row=1, col=1, line_dash="dash", line_color="#5c6bc0",
            annotation_text=f"{label} {price:.2f}", annotation_position="left",
        )
    fig.add_vline(x=orange["range_end"], row=1, col=1, line_dash="dot", line_color="#9e9e9e")

    # Breakout closes (informational) — every close outside the range.
    oh, ol = orange["opening_high"], orange["opening_low"]
    scan = session_df[session_df.index > orange["range_end"]]
    outside = scan[(scan["Close"] > oh) | (scan["Close"] < ol)]
    if not outside.empty:
        fig.add_trace(
            go.Scatter(
                x=outside.index, y=outside["Close"], mode="markers", name="Close outside range",
                marker=dict(symbol="circle-open", size=8, color="#9e9e9e", line=dict(width=1.5)),
            ),
            row=1, col=1,
        )

    for rej in rejections:
        fig.add_trace(
            go.Scatter(
                x=[rej["timestamp"]], y=[float(session_df.loc[rej["timestamp"], "Close"])],
                mode="markers", name=f"Filtered (#{rej['confirmation_count']})",
                marker=dict(symbol="x", size=11, color="#ff9800"),
                hovertext="<br>".join(rej["reasons"]), hoverinfo="text",
            ),
            row=1, col=1,
        )

    for sig in signals:
        is_long = sig["direction"] == "long"
        fig.add_trace(
            go.Scatter(
                x=[sig["timestamp"]], y=[sig["entry_price"]], mode="markers",
                name=f"{sig['action']} signal",
                marker=dict(symbol="triangle-up" if is_long else "triangle-down",
                            size=16, color="#26a69a" if is_long else "#ef5350"),
            ),
            row=1, col=1,
        )
        for price, label, color in [
            (sig["entry_price"], "Entry", "#2196f3"),
            (sig["target_price"], "Target", "#26a69a"),
            (sig["stop_price"], "Stop", "#ef5350"),
        ]:
            fig.add_hline(
                y=price, row=1, col=1, line_dash="dot", line_color=color,
                annotation_text=f"{label} {price:.2f}", annotation_position="right",
            )

    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(session_df["Close"], session_df["Open"])]
    fig.add_trace(
        go.Bar(x=session_df.index, y=session_df["Volume"], marker_color=vol_colors,
               name="Volume", showlegend=False),
        row=2, col=1,
    )
    if "vol_sma_20" in session_df.columns:
        fig.add_trace(
            go.Scatter(x=session_df.index, y=session_df["vol_sma_20"], name="Vol 20-bar avg",
                       line=dict(color="#ff9800", width=1.5)),
            row=2, col=1,
        )

    fig.update_layout(
        template=theme, height=640, margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06),
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_orbc_scanner(ticker: str, config: ORBCConfig, interval: str):
    df = _load_orbc_intraday(ticker, interval)
    if df is None:
        st.warning(f"No {interval} intraday data available for {ticker}.")
        return

    state = latest_session_state(df, config)
    session_df = state.get("session_df")
    orange = state.get("opening_range")

    if orange is None or session_df is None:
        st.info(state.get("skip_reason") or "No opening range could be established for the latest session.")
        return

    st.caption(
        f"Session {state['session_date']:%A, %B %d, %Y} · {interval} bars · "
        f"opening range from {orange['bar_count']} bar(s) "
        f"({orange['range_start']:%H:%M}–{orange['range_end']:%H:%M} ET)"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Opening High", f"{orange['opening_high']:.2f}")
    m2.metric("Opening Low", f"{orange['opening_low']:.2f}")
    m3.metric("Range Size", f"{orange['range_size']:.2f}")
    last_close = float(session_df["Close"].iloc[-1])
    if last_close > orange["opening_high"]:
        posn = "Above range"
    elif last_close < orange["opening_low"]:
        posn = "Below range"
    else:
        posn = "Inside range"
    m4.metric("Last Close", f"{last_close:.2f}", posn)

    signals = state["signals"]
    rejections = state["rejections"]

    _orbc_signal_chart(session_df, orange, signals, rejections, config)

    st.markdown("---")

    if signals:
        for sig in signals:
            is_long = sig["direction"] == "long"
            box = st.success if is_long else st.error
            box(
                f"**{sig['action']} signal** at {sig['timestamp']:%H:%M ET} — "
                f"confirmed on close #{sig['confirmation_count']} outside the range · "
                f"Entry {sig['entry_price']:.2f} · Stop {sig['stop_price']:.2f} · "
                f"Target {sig['target_price']:.2f} · R:R {sig['rr_ratio']:.2f} · "
                f"Confidence {sig['confidence_score']:.0f}/100"
            )
            with st.expander("Confidence breakdown & filter readings", expanded=False):
                comp_df = pd.DataFrame([
                    {"Component": k.replace("_", " ").title(), "Score (0-1)": v}
                    for k, v in sig["score_components"].items()
                ])
                st.dataframe(comp_df, hide_index=True, width="stretch")
                st.json(sig["filters"])

            if st.button(f"Log this {ticker} ORBC signal", key=f"log_orbc_{sig['timestamp']:%H%M}"):
                logger.info(f"[strategy_lab] Logging ORBC signal for {ticker} at {sig['timestamp']}")
                log_activity("orbc_signal", ticker, sig)
                st.toast("ORBC signal logged.")
    else:
        st.info(
            "No confirmed ORBC signal for this session yet — "
            "waiting on a second consecutive close outside the opening range."
        )

    if rejections:
        with st.expander(f"{len(rejections)} breakout(s) reached the confirmation count but were filtered out", expanded=False):
            for rej in rejections:
                st.markdown(
                    f"**{rej['timestamp']:%H:%M ET}** · {rej['direction']} · "
                    f"close #{rej['confirmation_count']}"
                )
                for reason in rej["reasons"]:
                    st.caption(f"— {reason}")


def _render_orbc_backtest(ticker: str, config: ORBCConfig, interval: str):
    st.caption(
        "Intraday history is capped by the data provider at roughly the trailing 60 "
        "days, and ORBC fires at most once or twice per session — so expect a few "
        "dozen trades. Entries are the confirming bar's close, exits are evaluated "
        "close-to-close, and every position is flattened at the session close."
    )

    if not st.button(f"Run ORBC backtest on {ticker}", key="run_orbc_backtest_btn"):
        return

    logger.info(f"[strategy_lab] 'Run ORBC backtest' button pressed for {ticker}")
    with st.spinner(f"Scanning {ticker} sessions for ORBC signals..."):
        df = _load_orbc_intraday(ticker, interval)
        if df is None:
            st.warning(f"No {interval} intraday data available for {ticker}.")
            return
        result = backtest_orbc(ticker, config, interval=interval, df=df)

    if result.get("error"):
        logger.warning(f"[strategy_lab] ORBC backtest for {ticker} error: {result['error']}")
        st.warning(result["error"])
        return

    logger.info(
        f"[strategy_lab] ORBC backtest {ticker}: trades={result['num_trades']} "
        f"win_rate={result['win_rate']}% total_return={result['total_return_pct']}%"
    )

    st.caption(
        f"Window: {result['window_start']:%Y-%m-%d %H:%M} → {result['window_end']:%Y-%m-%d %H:%M} · "
        f"{result['num_sessions']} sessions ({result['sessions_with_range']} with a usable opening range)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", result["num_trades"])
    c2.metric("Win Rate", f"{result['win_rate']:.1f}%")
    c3.metric("Avg R:R", f"{result['avg_rr']:.2f}")
    c4.metric("Total Return", f"{result['total_return_pct']:+.2f}%")

    if result["num_trades"] == 0:
        st.info("No qualifying ORBC signals fired in the available history. Try loosening the filters.")
        return

    if result["num_trades"] < 30:
        st.warning(
            f"Only {result['num_trades']} trades across {result['num_sessions']} sessions — "
            "too small a sample to read the win rate as predictive. Treat it as indicative only."
        )

    st.line_chart(result["equity_curve"])

    if result["by_direction"]:
        st.markdown("**By direction**")
        st.dataframe(
            pd.DataFrame([
                {"Direction": d.title(), "Trades": v["trades"],
                 "Win Rate %": v["win_rate"], "Avg Return %": v["avg_return_pct"]}
                for d, v in result["by_direction"].items()
            ]),
            hide_index=True, width="stretch",
        )

    st.markdown("**Exit reasons**")
    st.dataframe(
        pd.DataFrame([{"Exit Reason": k, "Count": v} for k, v in result["exit_breakdown"].items()]),
        hide_index=True, width="stretch",
    )

    st.markdown("**Trades**")
    trades_df = pd.DataFrame([
        {
            "Signal Time": t["timestamp"],
            "Dir": t["action"],
            "Conf #": t["confirmation_count"],
            "Score": t["confidence_score"],
            "Entry": t["entry_price"],
            "Stop": t["stop_price"],
            "Target": t["target_price"],
            "Exit": t["exit_price"],
            "Exit Reason": t["exit_reason"],
            "Return %": t["return_pct"],
            "Bars Held": t["holding_period_bars"],
        }
        for t in result["trades"]
    ])
    st.dataframe(trades_df, hide_index=True, width="stretch")


def _render_orbc(ticker: str):
    st.markdown(
        "Define the opening range from the first minutes after the 9:30 ET open, then "
        "require a **second consecutive close** outside that range before signalling — "
        "the confirmation step that filters most of the false breakouts right after the open."
    )
    config, interval = _orbc_config_controls("orbc")

    sub_scan, sub_backtest = st.tabs(["Live Scanner", "Backtest"])
    with sub_scan:
        _render_orbc_scanner(ticker, config, interval)
    with sub_backtest:
        _render_orbc_backtest(ticker, config, interval)


def render():
    st.markdown("# Strategy Lab")
    st.caption("Two intraday strategies, each with a live scanner and a mechanical backtest.")

    ticker = st.text_input("Ticker", value=st.session_state.get("quick_lookup_ticker", "SPY")).upper().strip()
    if not ticker:
        st.info("Enter a ticker to get started.")
        return

    if st.session_state.get("_strategy_lab_last_ticker") != ticker:
        logger.info(f"[strategy_lab] Ticker changed to {ticker}; loading Strategy Lab data")
        st.session_state["_strategy_lab_last_ticker"] = ticker

    tab_orbc, tab_mtf = st.tabs(["ORBC (Opening Range)", "MTF"])
    with tab_orbc:
        _render_orbc(ticker)
    with tab_mtf:
        _render_mtf(ticker)

    st.markdown("---")
    st.caption("Aether • Data from yfinance • For personal use only. Not financial advice.")


render()
