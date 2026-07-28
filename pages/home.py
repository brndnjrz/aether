"""
Market Dashboard — landing page with market overview, sector performance, and open positions.
"""
import logging
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.macro_data import get_market_overview, get_sp500_regime, get_sector_performance, get_vix_data

logger = logging.getLogger(__name__)

_EVENT_LABELS = {
    "day_trading_analyze": "Day Trading — Analyze",
    "options_view": "Options — Chain viewed",
    "options_expiry_view": "Options — Expiry picked",
    "prediction_generated": "Predictions — Daily signal",
    "intraday_prediction_generated": "Predictions — Intraday signal",
    "strategy_lab_setup": "Strategy Lab — MTF setup logged",
    "orbc_signal": "Strategy Lab — ORBC signal logged",
}


def render():
    st.markdown("# Market Dashboard")
    logger.debug("[home] Rendering Market Dashboard")

    with st.spinner("Loading market data..."):
        overview = get_market_overview()
        regime = get_sp500_regime()
        vix = get_vix_data()
        sectors = get_sector_performance()

    if not overview:
        logger.warning("[home] Market overview data came back empty")
    else:
        logger.debug(f"[home] Market overview loaded with {len(overview)} indices")
    if not sectors:
        logger.debug("[home] Sector performance data came back empty")

    # Regime banner
    regime_label = regime.get("regime", "Unknown")
    regime_colors = {
        "Bull Market": "success",
        "Uptrend": "success",
        "Sideways / Choppy": "warning",
        "Downtrend": "warning",
        "Bear Market": "error",
    }
    banner_type = regime_colors.get(regime_label, "info")
    if banner_type == "success":
        st.success(f"🟢 Market Regime: **{regime_label}** | S&P 500 {regime.get('pct_from_200ma', 0):+.1f}% vs 200MA | VIX: {vix.get('current', 20):.1f} ({vix.get('regime', 'N/A')})")
    elif banner_type == "warning":
        st.warning(f"🟡 Market Regime: **{regime_label}** | S&P 500 {regime.get('pct_from_200ma', 0):+.1f}% vs 200MA | VIX: {vix.get('current', 20):.1f} ({vix.get('regime', 'N/A')})")
    elif banner_type == "error":
        st.error(f"🔴 Market Regime: **{regime_label}** | S&P 500 {regime.get('pct_from_200ma', 0):+.1f}% vs 200MA | VIX: {vix.get('current', 20):.1f} ({vix.get('regime', 'N/A')})")

    # ── Index Cards ───────────────────────────────────────────────────────
    st.markdown("---")
    cols = st.columns(len(overview) + 1)
    for i, (name, data) in enumerate(overview.items()):
        change = data.get("change_pct", 0)
        cols[i].metric(name, f"{data['price']:,.0f}", f"{change:+.2f}%")

    vix_change = vix.get("current", 20) - vix.get("week_ago", 20)
    cols[-1].metric("VIX", f"{vix.get('current', 20):.2f}", f"{vix_change:+.2f}")

    # ── Regime Markov ─────────────────────────────────────────────────────
    # Reuses the same Markov model Trading Desk runs per-ticker, applied to the
    # S&P 500 — a probabilistic second opinion on the trend-following regime
    # banner above, not a replacement for it.
    st.markdown("---")
    st.subheader("Market Regime (Markov)")
    with st.spinner("Fitting regime model..."):
        from data.price_data import get_price_history
        from analysis.indicators import calculate_indicators
        from analysis.regime_markov import analyze_regime_markov

        sp500_df = get_price_history("^GSPC", period="2y", interval="1d")
        regime_markov = (
            analyze_regime_markov(calculate_indicators(sp500_df), "S&P 500")
            if sp500_df is not None and not sp500_df.empty
            else {"available": False, "reason": "No S&P 500 price history available."}
        )

    if not regime_markov["available"]:
        logger.debug(f"[home] Regime Markov unavailable: {regime_markov['reason']}")
        st.info(regime_markov["reason"])
    else:
        st.caption("S&P 500 trend history, discretized into Bear/Neutral/Bull and fit to a first-order transition matrix — a probabilistic read, not a rule-based one.")
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Current State", regime_markov["current_state"])
        rm2.metric("Bull − Bear Signal", f"{regime_markov['signal']:+.2f}")
        rm3.metric("Confidence", f"{regime_markov['confidence'] * 100:.0f}%", help="Scales with how many times the current state has occurred historically (30+ = full confidence).")
        persist = regime_markov["persistence"]
        st.caption(
            f"Persistence — Bear: {persist['Bear'] * 100:.0f}% | Neutral: {persist['Neutral'] * 100:.0f}% | Bull: {persist['Bull'] * 100:.0f}% "
            f"(probability each regime repeats itself the next day, from {regime_markov['n_bars']} days of history)."
        )

    # ── Sector Performance ────────────────────────────────────────────────
    if sectors:
        st.markdown("---")
        st.subheader("Sector Performance")
        import plotly.graph_objects as go

        sector_names = list(sectors.keys())
        m1_returns = [sectors[s]["1m_pct"] for s in sector_names]
        m3_returns = [sectors[s]["3m_pct"] for s in sector_names]

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sector_names, y=m1_returns, name="1 Month",
                marker_color=["#26a69a" if r > 0 else "#ef5350" for r in m1_returns],
            ))
            fig.add_trace(go.Bar(
                x=sector_names, y=m3_returns, name="3 Month",
                marker_color=["rgba(38,166,154,0.4)" if r > 0 else "rgba(239,83,80,0.4)" for r in m3_returns],
            ))
            fig.update_layout(
                template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white",
                height=300, barmode="group",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            sector_df = pd.DataFrame([
                {"Sector": s, "1M %": f"{d['1m_pct']:+.1f}%", "3M %": f"{d['3m_pct']:+.1f}%", "ETF": d["etf"]}
                for s, d in sectors.items()
            ])
            sector_df = sector_df.sort_values("1M %", ascending=False)
            st.dataframe(sector_df, hide_index=True, width="stretch")

    # ── Quick Portfolio Summary ───────────────────────────────────────────
    from portfolio.journal import get_open_positions
    open_positions = get_open_positions()
    if open_positions:
        logger.debug(f"[home] {len(open_positions)} open positions loaded for summary")
        st.markdown("---")
        st.subheader(f"Open Positions ({len(open_positions)})")
        rows = []
        for p in open_positions[:5]:
            from data.price_data import get_current_price
            curr = get_current_price(p["ticker"])
            pnl = ((curr - p["entry_price"]) / p["entry_price"] * 100) if curr else None
            rows.append({
                "Ticker": p["ticker"],
                "Entry": f"${p['entry_price']:.2f}",
                "Current": f"${curr:.2f}" if curr else "—",
                "P&L": f"{pnl:+.1f}%" if pnl is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(open_positions) > 5:
            st.caption(f"+ {len(open_positions) - 5} more logged.")
    else:
        logger.debug("[home] No open positions to display")
        st.markdown("---")
        st.info("No open positions yet.")

    # ── Recent Activity ────────────────────────────────────────────────────
    from portfolio.activity_log import get_recent_activity

    recent = get_recent_activity(limit=8)
    if recent:
        logger.debug(f"[home] {len(recent)} recent activity rows loaded")
        st.markdown("---")
        st.subheader("Recent Activity")
        st.caption("What you've been looking at across Trading Desk and Strategy Lab — a pulse on the week, not a to-do list.")
        rows = [{
            "Time": pd.Timestamp(a["logged_at"]).strftime("%m/%d %I:%M %p ET"),
            "Ticker": a["ticker"],
            "Event": _EVENT_LABELS.get(a["event_type"], a["event_type"]),
        } for a in recent]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Aether • Data from yfinance • AI by Claude • For personal use only. Not financial advice.")


render()
