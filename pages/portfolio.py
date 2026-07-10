"""
Portfolio page — position tracking, risk analytics, correlation, stress tests.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, time as dt_time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_price_history, get_current_price
from portfolio.journal import get_open_positions, get_closed_performance
from portfolio.option_fills import add_fill, get_fills, remove_fill, update_fill
from portfolio.round_trips import compute_round_trips
from portfolio.activity_log import get_activity_in_window
from config.tz import MARKET_TZ, now_et
from analysis.risk import (
    portfolio_correlation_matrix, calculate_portfolio_metrics,
    stress_test_portfolio, position_size_from_stop,
)


def render():
    st.title("Portfolio")

    tab_overview, tab_risk, tab_sizer, tab_options_log = st.tabs(
        ["📊 Positions", "⚡ Risk Analytics", "🎯 Position Sizer", "📝 Options Log"]
    )

    with tab_overview:
        _render_positions()

    with tab_risk:
        _render_risk_analytics()

    with tab_sizer:
        _render_position_sizer()

    with tab_options_log:
        _render_options_log()


def _render_positions():
    positions = get_open_positions()
    if not positions:
        st.info("No open positions.")
        _render_closed_stats()
        return

    # Build enriched table with live prices
    rows = []
    total_value = 0
    for p in positions:
        current = get_current_price(p["ticker"])
        if current and p["entry_price"] and p["shares"]:
            pnl_pct = (current - p["entry_price"]) / p["entry_price"] * 100
            market_val = current * p["shares"]
            total_value += market_val
            rows.append({
                "Ticker": p["ticker"],
                "Shares": p["shares"],
                "Entry": f"${p['entry_price']:.2f}",
                "Current": f"${current:.2f}",
                "P&L %": f"{pnl_pct:+.1f}%",
                "Mkt Value": f"${market_val:,.0f}",
                "Stop": f"${p['stop_price']:.2f}" if p.get("stop_price") else "—",
                "Target": f"${p['target_price']:.2f}" if p.get("target_price") else "—",
                "Conviction": "⭐" * (p.get("conviction") or 3),
            })
        else:
            rows.append({
                "Ticker": p["ticker"], "Shares": p.get("shares", 0),
                "Entry": f"${p['entry_price']:.2f}", "Current": "—",
                "P&L %": "—", "Mkt Value": "—", "Stop": "—", "Target": "—",
                "Conviction": "⭐" * (p.get("conviction") or 3),
            })

    st.markdown(f"**{len(positions)} open positions** | Total value: **${total_value:,.0f}**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Sector allocation pie (from ticker info)
    _render_closed_stats()


def _render_closed_stats():
    perf = get_closed_performance()
    if perf.get("trades", 0) == 0:
        return

    st.markdown("---")
    st.subheader("Closed Trades Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
    c2.metric("Avg Winner", f"{perf.get('avg_win', 0):+.1f}%")
    c3.metric("Avg Loser", f"{perf.get('avg_loss', 0):.1f}%")
    c4.metric("Avg Return", f"{perf.get('avg_return', 0):+.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Trades", perf.get("trades", 0))
    c6.metric("Best Trade", f"{perf.get('best_trade', 0):+.1f}%")
    c7.metric("Worst Trade", f"{perf.get('worst_trade', 0):.1f}%")


def _render_risk_analytics():
    st.subheader("Portfolio Risk Analytics")

    positions = get_open_positions()
    tickers = [p["ticker"] for p in positions if p.get("ticker")]

    if len(tickers) < 2:
        st.info("Add at least 2 positions to see correlation analysis.")
        _render_stress_test_standalone()
        return

    # Fetch price history for all
    price_data = {}
    weights = {}
    total_val = 0
    for p in positions:
        df = get_price_history(p["ticker"], period="1y")
        if df is not None:
            price_data[p["ticker"]] = df
            val = (p.get("shares") or 1) * (p.get("entry_price") or 1)
            weights[p["ticker"]] = val
            total_val += val

    if total_val > 0:
        weights = {k: v / total_val for k, v in weights.items()}

    # Correlation matrix
    corr = portfolio_correlation_matrix(price_data)
    if not corr.empty:
        st.subheader("Correlation Matrix (1-year returns)")
        fig = px.imshow(
            corr,
            color_continuous_scale="RdYlGn_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Warn on high correlation
        high_corr_pairs = []
        for i, t1 in enumerate(corr.columns):
            for j, t2 in enumerate(corr.columns):
                if i < j and corr.loc[t1, t2] > 0.8:
                    high_corr_pairs.append((t1, t2, corr.loc[t1, t2]))
        if high_corr_pairs:
            st.warning(f"⚠️ High correlation detected: " + ", ".join(f"{a}/{b} ({c:.2f})" for a, b, c in high_corr_pairs))

    # Portfolio returns
    combined_returns = []
    for ticker, df in price_data.items():
        w = weights.get(ticker, 0)
        if "Close" in df.columns and w > 0:
            r = df["Close"].pct_change().dropna()
            combined_returns.append(r * w)

    if combined_returns:
        # Keep the DatetimeIndex (don't collapse to a list) so drawdown can be
        # attributed to an actual date and the wealth index can be charted.
        port_returns = pd.concat(combined_returns, axis=1).sum(axis=1).dropna()
        metrics = calculate_portfolio_metrics(port_returns)

        st.subheader("Portfolio Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        sharpe = metrics.get("sharpe", 0)
        m1.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted returns. >1 = good, >2 = excellent")
        dd_date = metrics.get("max_drawdown_date")
        m2.metric(
            "Max Drawdown", f"{metrics.get('max_drawdown', 0):.1f}%",
            help=f"Trough on {dd_date.date()}" if dd_date is not None else None,
        )
        m3.metric("Ann. Return (CAGR)", f"{metrics.get('annualized_return', 0):.1f}%")
        m4.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")

        wealth_index = metrics.get("wealth_index")
        drawdown_series = metrics.get("drawdown_series")
        if wealth_index is not None and not wealth_index.empty:
            st.subheader("Growth of $1")
            fig_wealth = px.line(x=wealth_index.index, y=wealth_index.values, template="plotly_dark")
            fig_wealth.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="$")
            st.plotly_chart(fig_wealth, use_container_width=True)

            st.subheader("Drawdown")
            fig_dd = px.area(x=drawdown_series.index, y=drawdown_series.values * 100, template="plotly_dark")
            fig_dd.update_traces(line_color="#ef5350", fillcolor="rgba(239,83,80,0.3)")
            fig_dd.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="%")
            st.plotly_chart(fig_dd, use_container_width=True)

    # Stress tests
    st.markdown("---")
    _render_stress_test_standalone(weights)


def _render_stress_test_standalone(weights: dict = None):
    st.subheader("Portfolio Stress Test Scenarios")
    if not weights:
        weights = {"Portfolio": 1.0}
    results = stress_test_portfolio(weights)
    df_stress = pd.DataFrame(list(results.items()), columns=["Scenario", "Estimated Return (%)"])
    df_stress["Color"] = df_stress["Estimated Return (%)"].apply(lambda x: "Loss" if x < 0 else "Gain")
    fig = px.bar(
        df_stress, x="Estimated Return (%)", y="Scenario",
        orientation="h",
        color="Color",
        color_discrete_map={"Loss": "#ef5350", "Gain": "#26a69a"},
        template="plotly_dark",
    )
    fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Stress test applies market scenario returns assuming beta ≈ 1. Beta-adjust manually for more precision.")


def _render_position_sizer():
    st.subheader("Position Sizer (Risk-First)")
    st.info("Define your stop FIRST, then let risk % determine your position size. This is the correct approach.")

    col1, col2 = st.columns(2)
    with col1:
        port_val = st.number_input("Portfolio Value ($)", value=100_000, step=1000)
        entry = st.number_input("Entry Price ($)", value=150.0, step=0.5)
        stop = st.number_input("Stop Price ($)", value=142.0, step=0.5)
        risk_pct = st.slider("Risk per trade (% of portfolio)", 0.5, 3.0, 1.0, step=0.25)

    result = position_size_from_stop(port_val, entry, stop, risk_pct / 100)

    with col2:
        if "error" not in result:
            st.markdown("**Position Sizing Result**")
            st.metric("Shares to Buy", f"{result['shares']:,}")
            st.metric("Position Value", f"${result['position_value']:,.0f}")
            st.metric("% of Portfolio", f"{result['position_pct']:.1f}%")
            st.metric("Dollar Risk (1R)", f"${result['dollar_risk']:,.0f}")
            st.markdown("---")
            st.markdown("**Price Targets**")
            st.write(f"2:1 R/R Target: **${result['risk_reward_2to1_target']:.2f}**")
            st.write(f"3:1 R/R Target: **${result['risk_reward_3to1_target']:.2f}**")
            st.write(f"Risk per share: **${result['risk_per_share']:.2f}**")
        else:
            st.error(result["error"])


@st.cache_data(ttl=30)
def _cached_round_trips(fills: list) -> list:
    return compute_round_trips(fills)


def _render_options_log():
    st.subheader("Options Trade Log")
    st.caption("Log each fill as your broker reports it (one row per buy or sell) — round trips and P&L are computed automatically.")

    _render_add_fill_form()
    st.markdown("---")

    fills = get_fills()
    _render_fill_ledger(fills)
    st.markdown("---")

    round_trips = _cached_round_trips(fills)
    _render_round_trips_table(round_trips)
    st.markdown("---")

    _render_round_trip_analytics(round_trips)


def _render_add_fill_form():
    st.markdown("**Log a fill**")
    with st.form("add_fill_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        ticker = col1.text_input("Ticker").upper().strip()
        strike = col2.number_input("Strike ($)", value=0.0, step=1.0)
        option_type = col3.selectbox("Type", ["call", "put"])
        expiry_date = col4.date_input("Expiry date", value=now_et().date())

        col5, col6, col7 = st.columns(3)
        side = col5.selectbox("Side", ["buy", "sell"])
        qty = col6.number_input("Contracts", value=1, step=1, min_value=1)
        price = col7.number_input("Price per contract ($)", value=0.0, step=0.01)

        col8, col9, col10 = st.columns(3)
        fill_date = col8.date_input("Fill date", value=now_et().date())
        now = now_et()
        fill_hour = col9.number_input("Fill hour (0-23, ET)", value=now.hour, min_value=0, max_value=23, step=1)
        fill_minute = col10.number_input("Fill minute (ET)", value=now.minute, min_value=0, max_value=59, step=1)
        fill_time = dt_time(int(fill_hour), int(fill_minute))

        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Log Fill", type="primary")

    if submitted:
        if not ticker:
            st.warning("Enter a ticker first.")
            return
        if not strike:
            st.warning("Enter a strike price.")
            return
        if price < 0:
            st.warning("Price cannot be negative.")
            return
        filled_at = datetime.combine(fill_date, fill_time, tzinfo=MARKET_TZ).isoformat()
        add_fill(ticker, strike, option_type, expiry_date.isoformat(), side, int(qty), price, filled_at, notes)
        st.success(f"Logged {side} {qty}x {ticker} ${strike:g} {option_type} {expiry_date.isoformat()} @ ${price:.2f}.")
        st.rerun()


_LEDGER_COL_WIDTHS = [0.8, 0.9, 0.7, 0.7, 1.0, 0.6, 0.8, 1.5, 0.7, 0.7]


def _render_fill_ledger(fills: list):
    st.markdown("**Fill ledger**")
    if not fills:
        st.info("No fills logged yet.")
        return

    header = st.columns(_LEDGER_COL_WIDTHS)
    for col, label in zip(header, ["Side", "Ticker", "Strike", "Type", "Expiry", "Qty", "Price", "Filled (ET)", "", ""]):
        col.markdown(f"**{label}**")

    for f in fills:
        st.markdown("---")
        cols = st.columns(_LEDGER_COL_WIDTHS)
        side_dot = "🟢" if f["side"] == "buy" else "🔴"
        cols[0].write(f"{side_dot} {f['side'].upper()}")
        cols[1].write(f["ticker"])
        cols[2].write(f"${f['strike']:g}")
        cols[3].write(f["option_type"])
        cols[4].write(f["expiry_date"])
        cols[5].write(str(f["qty"]))
        cols[6].write(f"${f['price']:.2f}")
        filled_str = pd.Timestamp(f["filled_at"]).strftime("%m/%d/%y %I:%M %p")
        cols[7].write(filled_str)

        editing = st.session_state.get("editing_fill_id") == f["id"]
        if cols[8].button("Edit", key=f"edit_fill_{f['id']}"):
            st.session_state["editing_fill_id"] = None if editing else f["id"]
            st.rerun()
        if cols[9].button("Delete", key=f"del_fill_{f['id']}"):
            remove_fill(f["id"])
            st.rerun()

        if f.get("notes"):
            st.caption(f["notes"])

        if editing:
            _render_edit_fill_form(f)


def _render_edit_fill_form(f: dict):
    filled_ts = pd.Timestamp(f["filled_at"])
    with st.container(border=True):
        st.markdown(f"**Edit fill #{f['id']}**")
        with st.form(f"edit_fill_form_{f['id']}"):
            col1, col2, col3, col4 = st.columns(4)
            ticker = col1.text_input("Ticker", value=f["ticker"]).upper().strip()
            strike = col2.number_input("Strike ($)", value=float(f["strike"]), step=1.0)
            option_type = col3.selectbox("Type", ["call", "put"], index=["call", "put"].index(f["option_type"]))
            expiry_date = col4.date_input("Expiry date", value=pd.Timestamp(f["expiry_date"]).date())

            col5, col6, col7 = st.columns(3)
            side = col5.selectbox("Side", ["buy", "sell"], index=["buy", "sell"].index(f["side"]))
            qty = col6.number_input("Contracts", value=int(f["qty"]), step=1, min_value=1)
            price = col7.number_input("Price per contract ($)", value=float(f["price"]), step=0.01)

            col8, col9, col10 = st.columns(3)
            fill_date = col8.date_input("Fill date", value=filled_ts.date())
            fill_hour = col9.number_input("Fill hour (0-23, ET)", value=int(filled_ts.hour), min_value=0, max_value=23, step=1)
            fill_minute = col10.number_input("Fill minute (ET)", value=int(filled_ts.minute), min_value=0, max_value=59, step=1)
            fill_time = dt_time(int(fill_hour), int(fill_minute))

            notes = st.text_area("Notes (optional)", value=f.get("notes") or "")

            csave, ccancel = st.columns(2)
            saved = csave.form_submit_button("Save changes", type="primary")
            cancelled = ccancel.form_submit_button("Cancel")

        if saved:
            filled_at = datetime.combine(fill_date, fill_time, tzinfo=MARKET_TZ).isoformat()
            update_fill(f["id"], ticker, strike, option_type, expiry_date.isoformat(), side, int(qty), price, filled_at, notes)
            st.session_state["editing_fill_id"] = None
            st.success("Fill updated.")
            st.rerun()
        if cancelled:
            st.session_state["editing_fill_id"] = None
            st.rerun()


def _render_round_trips_table(round_trips: list):
    st.markdown("**Round trips**")
    if not round_trips:
        st.info("No completed round trips yet — log both an opening and closing fill for a contract.")
        return

    rows = []
    for rt in round_trips:
        rows.append({
            "Ticker": rt["ticker"],
            "Strike": rt["strike"],
            "Type": rt["option_type"],
            "Expiry": rt["expiry_date"],
            "Entry Time": pd.Timestamp(rt["entry_time"]).strftime("%Y-%m-%d %I:%M %p ET"),
            "Exit Time": pd.Timestamp(rt["exit_time"]).strftime("%Y-%m-%d %I:%M %p ET"),
            "Hold": _format_hold(rt["hold_time_minutes"]),
            "Qty": rt["qty"],
            "Entry $": f"${rt['entry_price']:.2f}",
            "Exit $": f"${rt['exit_price']:.2f}",
            "P&L $": f"${rt['pnl_dollars']:+,.2f}",
            "P&L %": f"{rt['pnl_pct']:+.1f}%" if rt["pnl_pct"] is not None else "N/A",
            "Win/Loss": "Win" if rt["win"] else "Loss",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _format_hold(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        h = int(hours)
        m = int(minutes - h * 60)
        return f"{h}h{m}m"
    return f"{hours / 24:.1f}d"


def _render_round_trip_analytics(round_trips: list):
    st.markdown("**Analytics**")
    if not round_trips:
        st.info("Analytics will appear once you have completed round trips.")
        return

    df = pd.DataFrame(round_trips)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Win rate by hold time")
        bucket_order = ["<30min", "30min-2h", "2h-24h (overnight)", ">24h"]
        grp = df.groupby("hold_bucket")["win"].agg(["mean", "count"]).reindex(bucket_order).dropna()
        if not grp.empty:
            grp["win_rate"] = grp["mean"] * 100
            fig = px.bar(
                grp.reset_index(), x="hold_bucket", y="win_rate", text="count",
                template="plotly_dark", labels={"hold_bucket": "Hold time", "win_rate": "Win rate (%)"},
            )
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("Win rate by entry hour (ET)")
        df["entry_hour"] = pd.to_datetime(df["entry_time"]).dt.hour
        grp_hour = df.groupby("entry_hour")["win"].agg(["mean", "count"])
        if not grp_hour.empty:
            grp_hour["win_rate"] = grp_hour["mean"] * 100
            fig2 = px.bar(
                grp_hour.reset_index(), x="entry_hour", y="win_rate", text="count",
                template="plotly_dark", labels={"entry_hour": "Entry hour (ET)", "win_rate": "Win rate (%)"},
            )
            fig2.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**What were you looking at during this trade?**")
    options = {
        f"{rt['contract_key']} — entry {pd.Timestamp(rt['entry_time']).strftime('%m/%d %I:%M %p')}": rt
        for rt in round_trips
    }
    selected_label = st.selectbox("Round trip", list(options.keys()))
    if selected_label:
        rt = options[selected_label]
        activity = get_activity_in_window(rt["entry_time"], rt["exit_time"], rt["ticker"])
        if not activity:
            st.info("No logged Day Trading / Options / Predictions activity for this ticker in this window.")
        else:
            act_rows = [{
                "Time": pd.Timestamp(a["logged_at"]).strftime("%I:%M %p ET"),
                "Event": a["event_type"],
                "Detail": a["detail_json"],
            } for a in activity]
            st.dataframe(pd.DataFrame(act_rows), hide_index=True, width="stretch")


render()
