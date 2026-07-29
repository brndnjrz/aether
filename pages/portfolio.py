"""
Options Log — the trade journal. Log each options fill as your broker reports
it; round trips, P&L, and pattern-finding analytics are computed automatically.

Page order is deliberately top-down from insight to input: Analytics (is this
working, overall?) -> Win Rate (which slices work?) -> Round Trips (the raw
matched trades) -> Performance by Ticker -> Fill Ledger (where you actually
log fills) -> "what were you looking at" (drill into one trade's context).
Everything downstream of the fill ledger's add-fill form is a read of the
same round_trips list, computed once per render via _cached_round_trips().

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
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.option_fills import add_fill, get_fills, remove_fill, update_fill
from portfolio.round_trips import compute_round_trips
from portfolio.activity_log import get_activity_in_window, EVENT_LABELS
from config.tz import MARKET_TZ, now_et

logger = logging.getLogger(__name__)


def render():
    st.title("Options Log")
    st.caption("Log each fill as your broker reports it (one row per buy or sell) — round trips, P&L, and pattern analytics are computed automatically.")

    fills = get_fills()
    round_trips = _cached_round_trips(fills)

    st.markdown("## Analytics")
    st.caption("Is this working, overall? Cumulative P&L and whether your edge is trending up or down.")
    _render_analytics_section(round_trips)
    st.markdown("---")

    st.markdown("## Win Rate")
    st.caption("Which slices of your trading actually work — and which to stop doing.")
    _render_win_rate_section(round_trips)
    st.markdown("---")

    st.markdown("## Round Trips")
    _render_round_trips_table(round_trips)
    st.markdown("---")

    st.markdown("## Performance by Ticker")
    _render_ticker_breakdown_section(round_trips)
    st.markdown("---")

    st.markdown("## Fill Ledger")
    _render_add_fill_form()
    _render_fill_ledger(fills)
    st.markdown("---")

    st.markdown("## What Were You Looking At During This Trade?")
    _render_activity_lookup(round_trips)


@st.cache_data(ttl=30)
def _cached_round_trips(fills: list) -> list:
    result = compute_round_trips(fills)
    logger.debug(f"[portfolio] Computed {len(result)} round trips from {len(fills)} fills")
    return result


def _theme() -> str:
    return "plotly_dark" if st.context.theme.type == "dark" else "plotly_white"


def _round_trips_df(round_trips: list) -> pd.DataFrame:
    df = pd.DataFrame(round_trips)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df


# ══════════════════════════════════════════════════════════════════════════
# ── Analytics ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_analytics_section(round_trips: list):
    if not round_trips:
        st.info("Analytics will appear once you have completed round trips.")
        return

    df = _round_trips_df(round_trips)
    _render_equity_curve(df)
    st.markdown("")
    _render_rolling_win_rate(df)


def _render_equity_curve(df: pd.DataFrame):
    """Cumulative P&L over time — the single-glance answer to 'is this working?'"""
    st.markdown("**Cumulative P&L**")
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


def _render_rolling_win_rate(df: pd.DataFrame, window: int = 10):
    """
    A single aggregate win rate hides direction — it can't tell you whether
    you're improving or sliding. This is the trend the headline number
    can't show: a trailing window over your last N trades, in order.
    """
    ordered = df.sort_values("exit_time").copy()
    if len(ordered) < window:
        st.caption(f"Need at least {window} round trips for a rolling win rate — logged: {len(ordered)}.")
        return

    st.markdown(f"**Rolling Win Rate (last {window} trades)**")
    ordered["rolling_win_rate"] = ordered["win"].rolling(window, min_periods=window).mean() * 100
    fig = px.line(
        ordered, x="exit_time", y="rolling_win_rate", template=_theme(),
        labels={"exit_time": "", "rolling_win_rate": "Win rate (%)"},
    )
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1)
    fig.update_traces(line_color="#2196f3", mode="lines+markers", marker=dict(size=5))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ── Win Rate ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_win_rate_section(round_trips: list):
    if not round_trips:
        st.info("Win-rate breakdowns will appear once you have completed round trips.")
        return

    df = _round_trips_df(round_trips)

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

    col5, col6 = st.columns(2)
    with col5:
        _render_pnl_distribution(df)
    with col6:
        _render_hold_time_vs_return(df)


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


def _render_pnl_distribution(df: pd.DataFrame):
    """
    The shape of your returns matters as much as the average. A few large
    winners carrying many small losers looks very different from a pile of
    consistent small edges, even at the same average P&L — this is the chart
    that tells those two apart.
    """
    st.markdown("P&L distribution")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["win"].map({True: "Win", False: "Loss"})
    fig = px.histogram(
        plot_df, x="pnl_dollars", color="Outcome", nbins=30, template=_theme(),
        color_discrete_map={"Win": "#26a69a", "Loss": "#ef5350"},
        labels={"pnl_dollars": "P&L ($)"},
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


def _render_hold_time_vs_return(df: pd.DataFrame):
    """
    Surfaces a habit the aggregate win rate can't: cutting winners short, or
    holding losers hoping they turn around. Log-scale x-axis since hold
    times span minutes to multi-day — a linear axis would crush the
    intraday trades into an unreadable cluster near zero.
    """
    st.markdown("Hold time vs. return")
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["win"].map({True: "Win", False: "Loss"})
    fig = px.scatter(
        plot_df, x="hold_time_minutes", y="pnl_pct", color="Outcome", template=_theme(),
        color_discrete_map={"Win": "#26a69a", "Loss": "#ef5350"},
        labels={"hold_time_minutes": "Hold time (minutes)", "pnl_pct": "Return (%)"},
        hover_data=["ticker", "strike", "option_type"],
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.update_xaxes(type="log")
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ── Round Trips ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_round_trips_table(round_trips: list):
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


# ══════════════════════════════════════════════════════════════════════════
# ── Performance by Ticker ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_ticker_breakdown_section(round_trips: list):
    if not round_trips:
        st.info("Per-ticker performance will appear once you have completed round trips.")
        return
    _render_ticker_breakdown(_round_trips_df(round_trips))


def _render_ticker_breakdown(df: pd.DataFrame):
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


# ══════════════════════════════════════════════════════════════════════════
# ── Fill Ledger ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

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

        col8, col9 = st.columns(2)
        fill_date = col8.date_input("Fill date", value=now_et().date())
        fill_time = col9.time_input("Fill time (ET)", value=now_et().time(), step=timedelta(minutes=1))

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


def _render_fill_ledger(fills: list):
    """
    A proper table (sortable, evenly aligned) rather than a hand-built column
    grid — the fixed relative column widths in the old version drifted out
    of alignment with real data (tickers/prices of different lengths), which
    is what made it hard to scan.

    Edit/Delete act on whichever row is clicked, via st.dataframe's native
    row selection, rather than a button in every row — Streamlit's table
    widget can't embed buttons in cells, so this is the closest equivalent
    that doesn't reintroduce the alignment problem the grid had. The
    DataFrame's index is set to each fill's id (hidden from view) so the
    selected row survives sorting without needing a separate lookup table.
    The click itself is captured into session_state right away rather than
    re-read from the widget on every rerun — see the comment below.
    """
    st.markdown("**Fill ledger**")
    if not fills:
        st.info("No fills logged yet.")
        return

    rows, ids = [], []
    for f in fills:
        ids.append(f["id"])
        rows.append({
            "Side": f"{'🟢' if f['side'] == 'buy' else '🔴'} {f['side'].upper()}",
            "Ticker": f["ticker"],
            "Strike": f["strike"],
            "Type": f["option_type"].title(),
            "Expiry": f["expiry_date"],
            "Qty": f["qty"],
            "Price": f["price"],
            # Kept as a real Timestamp (not pre-formatted to a string) so both
            # the sort below and the interactive column-header sort in the
            # dataframe are chronological, not lexicographic on "hh:mm AM/PM".
            "Filled (ET)": pd.Timestamp(f["filled_at"]),
            "Notes": f.get("notes") or "",
        })
    display_df = pd.DataFrame(rows, index=ids).sort_values("Filled (ET)", ascending=False)

    st.caption("Click a row to edit or delete it.")
    state = st.dataframe(
        display_df, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="fill_ledger_table",
        column_config={
            "Strike": st.column_config.NumberColumn(format="$%.2f"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Filled (ET)": st.column_config.DatetimeColumn(format="MM/DD/YY hh:mm A"),
        },
    )

    # The dataframe only reports a selection on the rerun triggered by the
    # click itself — clicking Edit/Save afterward reruns the script again
    # without a new click on the table, and that later rerun can come back
    # with an empty selection. Capture the click into our own session_state
    # immediately, and use that (not the widget's live selection) for
    # everything below, so "which fill am I acting on" survives every
    # subsequent rerun regardless of what the table reports on it.
    selected_rows = state["selection"]["rows"]
    if selected_rows:
        st.session_state["selected_fill_id"] = display_df.index[selected_rows[0]]

    fill_id = st.session_state.get("selected_fill_id")
    if fill_id is None or fill_id not in display_df.index:
        return

    f = next(fill for fill in fills if fill["id"] == fill_id)
    editing = st.session_state.get("editing_fill_id") == f["id"]

    ecol, dcol = st.columns(2)
    if ecol.button("Close edit" if editing else "Edit", key=f"edit_fill_{f['id']}"):
        logger.info(f"[portfolio] 'Edit' fill button pressed for fill #{f['id']} ({f['ticker']})")
        st.session_state["editing_fill_id"] = None if editing else f["id"]
        st.rerun()
    if dcol.button("Delete", key=f"del_fill_{f['id']}"):
        logger.info(f"[portfolio] 'Delete' fill button pressed for fill #{f['id']} ({f['ticker']})")
        remove_fill(f["id"])
        st.session_state["editing_fill_id"] = None
        st.session_state["selected_fill_id"] = None
        st.session_state.pop("fill_ledger_table", None)  # clear the now-stale row selection
        st.rerun()

    if editing:
        _render_edit_fill_form(f)


def _render_edit_fill_form(f: dict):
    """
    Every widget below is keyed on f['id'] explicitly. Streamlit widgets
    without an explicit key derive one from position/label alone, which is
    fine while only one fill is ever being edited at a time — but it's the
    kind of thing that's cheap to get wrong and expensive to debug (a stale
    value silently surviving from whichever fill was edited previously), so
    it's pinned here rather than left implicit.
    """
    filled_ts = pd.Timestamp(f["filled_at"])
    fid = f["id"]
    with st.container(border=True):
        st.markdown(f"**Edit fill #{fid}**")
        with st.form(f"edit_fill_form_{fid}"):
            col1, col2, col3, col4 = st.columns(4)
            ticker = col1.text_input("Ticker", value=f["ticker"], key=f"edit_ticker_{fid}").upper().strip()
            strike = col2.number_input("Strike ($)", value=float(f["strike"]), step=1.0, key=f"edit_strike_{fid}")
            option_type = col3.selectbox("Type", ["call", "put"], index=["call", "put"].index(f["option_type"]), key=f"edit_type_{fid}")
            expiry_date = col4.date_input("Expiry date", value=pd.Timestamp(f["expiry_date"]).date(), key=f"edit_expiry_{fid}")

            col5, col6, col7 = st.columns(3)
            side = col5.selectbox("Side", ["buy", "sell"], index=["buy", "sell"].index(f["side"]), key=f"edit_side_{fid}")
            qty = col6.number_input("Contracts", value=int(f["qty"]), step=1, min_value=1, key=f"edit_qty_{fid}")
            price = col7.number_input("Price per contract ($)", value=float(f["price"]), step=0.01, key=f"edit_price_{fid}")

            col8, col9 = st.columns(2)
            fill_date = col8.date_input("Fill date", value=filled_ts.date(), key=f"edit_filldate_{fid}")
            fill_time = col9.time_input(
                "Fill time (ET)", value=filled_ts.time(), step=timedelta(minutes=1), key=f"edit_filltime_{fid}"
            )

            notes = st.text_area("Notes (optional)", value=f.get("notes") or "", key=f"edit_notes_{fid}")

            csave, ccancel = st.columns(2)
            saved = csave.form_submit_button("Save changes", type="primary")
            cancelled = ccancel.form_submit_button("Cancel")

        if saved:
            filled_at = datetime.combine(fill_date, fill_time, tzinfo=MARKET_TZ).isoformat()
            update_fill(fid, ticker, strike, option_type, expiry_date.isoformat(), side, int(qty), price, filled_at, notes)
            logger.info(f"[portfolio] Fill #{fid} ({ticker}) updated")
            st.session_state["editing_fill_id"] = None
            # A changed fill time can move this row to a new sort position —
            # clear the ledger's row selection rather than leave it pointing
            # at whatever fill now lands in the old position.
            st.session_state.pop("fill_ledger_table", None)
            st.success("Fill updated.")
            st.rerun()
        if cancelled:
            st.session_state["editing_fill_id"] = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ── Activity Lookup ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def _render_activity_lookup(round_trips: list):
    if not round_trips:
        st.info("This will let you look up what was on screen during a trade once you have completed round trips.")
        return

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
                "Event": EVENT_LABELS.get(a["event_type"], a["event_type"]),
                "Detail": a["detail_json"],
            } for a in activity]
            st.dataframe(pd.DataFrame(act_rows), hide_index=True, width="stretch")


render()
