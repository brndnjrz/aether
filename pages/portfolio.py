"""
Options Log — the trade journal. Log each options fill as your broker reports
it; round trips, P&L, and pattern-finding analytics are computed automatically.

Formerly "Portfolio," with Positions / Risk Analytics / Position Sizer tabs.
Those tracked equity positions, but the app has no UI to log an equity
position — they always rendered empty. Position Sizer is also redundant with
Trading Desk's Quick Risk Calculator, which does the same math plus
regime-based scaling. Trimmed to the one tab that has real data behind it.
"""
import logging
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, time as dt_time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.option_fills import add_fill, get_fills, remove_fill, update_fill
from portfolio.round_trips import compute_round_trips
from portfolio.activity_log import get_activity_in_window
from config.tz import MARKET_TZ, now_et

logger = logging.getLogger(__name__)


def render():
    st.title("Options Log")
    st.caption("Log each fill as your broker reports it (one row per buy or sell) — round trips, P&L, and pattern analytics are computed automatically.")

    _render_add_fill_form()
    st.markdown("---")

    fills = get_fills()
    _render_fill_ledger(fills)
    st.markdown("---")

    round_trips = _cached_round_trips(fills)
    _render_round_trips_table(round_trips)
    st.markdown("---")

    _render_round_trip_analytics(round_trips)


@st.cache_data(ttl=30)
def _cached_round_trips(fills: list) -> list:
    result = compute_round_trips(fills)
    logger.debug(f"[portfolio] Computed {len(result)} round trips from {len(fills)} fills")
    return result


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
        logger.info(f"[portfolio] 'Log Fill' form submitted for {ticker or '(no ticker)'}")
        if not ticker:
            logger.warning("[portfolio] Log Fill submission rejected: no ticker entered")
            st.warning("Enter a ticker first.")
            return
        if not strike:
            logger.warning(f"[portfolio] Log Fill submission for {ticker} rejected: no strike entered")
            st.warning("Enter a strike price.")
            return
        if price < 0:
            logger.warning(f"[portfolio] Log Fill submission for {ticker} rejected: negative price {price}")
            st.warning("Price cannot be negative.")
            return
        filled_at = datetime.combine(fill_date, fill_time, tzinfo=MARKET_TZ).isoformat()
        add_fill(ticker, strike, option_type, expiry_date.isoformat(), side, int(qty), price, filled_at, notes)
        logger.info(f"[portfolio] Fill logged: {side} {qty}x {ticker} ${strike:g} {option_type} @ ${price:.2f}")
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
            logger.info(f"[portfolio] 'Edit' fill button pressed for fill #{f['id']} ({f['ticker']})")
            st.session_state["editing_fill_id"] = None if editing else f["id"]
            st.rerun()
        if cols[9].button("Delete", key=f"del_fill_{f['id']}"):
            logger.info(f"[portfolio] 'Delete' fill button pressed for fill #{f['id']} ({f['ticker']})")
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
            logger.info(f"[portfolio] Fill #{f['id']} ({ticker}) updated")
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


def _theme() -> str:
    return "plotly_dark" if st.context.theme.type == "dark" else "plotly_white"


def _render_round_trip_analytics(round_trips: list):
    st.markdown("**Analytics**")
    st.caption("Slice your own trade history to find what's actually working — and what to stop doing.")
    if not round_trips:
        st.info("Analytics will appear once you have completed round trips.")
        return

    df = pd.DataFrame(round_trips)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    _render_equity_curve(df)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _render_win_rate_by(df, "hold_bucket", "Win rate by hold time",
                            order=["<30min", "30min-2h", "2h-24h (overnight)", ">24h"])
    with col2:
        df["entry_hour"] = df["entry_time"].dt.hour
        _render_win_rate_by(df, "entry_hour", "Win rate by entry hour (ET)")

    col3, col4 = st.columns(2)
    with col3:
        _render_win_rate_by(df, "option_type", "Win rate by option type")
    with col4:
        df["entry_weekday"] = df["entry_time"].dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        _render_win_rate_by(df, "entry_weekday", "Win rate by day of week", order=weekday_order)

    st.markdown("---")
    _render_ticker_breakdown(df)

    st.markdown("---")
    _render_activity_lookup(round_trips)


def _render_equity_curve(df: pd.DataFrame):
    """Cumulative P&L over time — the single-glance answer to 'is this working?'"""
    st.markdown("Cumulative P&L")
    ordered = df.sort_values("exit_time").copy()
    ordered["cumulative_pnl"] = ordered["pnl_dollars"].cumsum()

    fig = px.area(
        ordered, x="exit_time", y="cumulative_pnl", template=_theme(),
        labels={"exit_time": "", "cumulative_pnl": "Cumulative P&L ($)"},
    )
    is_positive = ordered["cumulative_pnl"].iloc[-1] >= 0
    fig.update_traces(
        line_color="#26a69a" if is_positive else "#ef5350",
        fillcolor="rgba(38,166,154,0.15)" if is_positive else "rgba(239,83,80,0.15)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    total = ordered["cumulative_pnl"].iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total P&L", f"${total:+,.2f}")
    c2.metric("Total Round Trips", len(ordered))
    c3.metric("Win Rate", f"{(ordered['win'].mean() * 100):.1f}%")


def _render_win_rate_by(df: pd.DataFrame, group_col: str, title: str, order: list = None):
    st.markdown(title)
    grp = df.groupby(group_col)["win"].agg(["mean", "count"])
    if order:
        grp = grp.reindex(order).dropna()
    if grp.empty:
        st.caption("Not enough data yet.")
        return
    grp["win_rate"] = grp["mean"] * 100
    fig = px.bar(
        grp.reset_index(), x=group_col, y="win_rate", text="count",
        template=_theme(), labels={group_col: "", "win_rate": "Win rate (%)"},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)


def _render_ticker_breakdown(df: pd.DataFrame):
    st.markdown("Performance by ticker")
    grp = df.groupby("ticker").agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        total_pnl=("pnl_dollars", "sum"),
        avg_pnl=("pnl_dollars", "mean"),
    ).sort_values("total_pnl", ascending=False)
    grp["win_rate"] = (grp["win_rate"] * 100).round(1)
    grp["total_pnl"] = grp["total_pnl"].round(2)
    grp["avg_pnl"] = grp["avg_pnl"].round(2)
    st.dataframe(
        grp.reset_index().rename(columns={
            "ticker": "Ticker", "trades": "Trades", "win_rate": "Win Rate %",
            "total_pnl": "Total P&L $", "avg_pnl": "Avg P&L $",
        }),
        hide_index=True, width="stretch",
    )


def _render_activity_lookup(round_trips: list):
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
            st.info("No logged Day Trading / Options / Predictions / Strategy Lab activity for this ticker in this window.")
        else:
            act_rows = [{
                "Time": pd.Timestamp(a["logged_at"]).strftime("%I:%M %p ET"),
                "Event": a["event_type"],
                "Detail": a["detail_json"],
            } for a in activity]
            st.dataframe(pd.DataFrame(act_rows), hide_index=True, width="stretch")


render()
