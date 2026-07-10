"""
Watchlist — persistent weekly shortlist + a fast daily scan limited to just
those tickers (gap %, relative volume, ATR%, ML bias, and a today's-session
price envelope), so the weekly plan and the day-of check live in one place
instead of re-running the Screener or re-typing tickers every morning.
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_price_history
from analysis.indicators import calculate_indicators
from analysis.price_projection import simulate_intraday_path
from portfolio.watchlist import add_to_watchlist, get_watchlist, remove_from_watchlist

try:
    from analysis.ml_prediction import predict as ml_predict
    from pathlib import Path
    from config.settings import STORAGE_DIR as _ML_STORAGE_DIR
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False


def render():
    st.title("Watchlist")
    st.caption("Build your plan for the week, then run the daily check each morning to see which names actually deserve attention today.")

    _render_add_form()
    st.markdown("---")
    items = get_watchlist()
    _render_list(items)
    st.markdown("---")
    _render_daily_check(items)


def _render_add_form():
    st.subheader("Add to this week's watchlist")
    with st.form("add_watchlist_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        ticker = col1.text_input("Ticker").upper().strip()
        target = col2.number_input("Target price (optional)", value=0.0, step=0.5)
        stop = col3.number_input("Stop price (optional)", value=0.0, step=0.5)
        notes = st.text_area("Plan / thesis for the week", placeholder="e.g. Watching for a breakout above the swing high near $195 on volume confirmation.")
        submitted = st.form_submit_button("Add to Watchlist", type="primary")

    if submitted:
        if not ticker:
            st.warning("Enter a ticker first.")
            return
        existing = {i["ticker"] for i in get_watchlist()}
        if ticker in existing:
            st.warning(f"{ticker} is already on your watchlist.")
            return
        add_to_watchlist(ticker, notes, target or None, stop or None)
        st.success(f"Added {ticker} to your watchlist.")
        st.rerun()


def _render_list(items: list):
    st.subheader(f"This week's list ({len(items)})")
    if not items:
        st.info("Your watchlist is empty. Add tickers above, or pull candidates from the Screener first.")
        return

    for item in items:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                st.markdown(f"**{item['ticker']}**")
                if item["target_price"]:
                    st.caption(f"Target: ${item['target_price']:.2f}")
                if item["stop_price"]:
                    st.caption(f"Stop: ${item['stop_price']:.2f}")
            with c2:
                st.write(item["plan_notes"] or "—")
            with c3:
                if st.button("Open in Trading Desk", key=f"open_{item['id']}"):
                    st.session_state["dt_lookup_ticker"] = item["ticker"]
                    st.switch_page("pages/trading.py")
                if st.button("Remove", key=f"remove_{item['id']}"):
                    remove_from_watchlist(item["id"])
                    st.rerun()


def _render_daily_check(items: list):
    st.subheader("Daily Watchlist Check")
    st.caption("Scans only the tickers above — gap %, volume vs. 20-day average, ATR%, and (if a model exists) the ML directional bias.")

    if not items:
        st.info("Add tickers to your watchlist to run the daily check.")
        return

    if not st.button("Run Daily Check", type="primary"):
        return

    rows = []
    progress = st.progress(0)
    tickers = [i["ticker"] for i in items]
    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / len(tickers))
        try:
            row = _check_ticker(ticker)
            if row:
                rows.append(row)
        except Exception:
            continue
    progress.empty()

    if not rows:
        st.warning("Couldn't fetch data for any watchlist ticker right now.")
        return

    df = pd.DataFrame(rows).sort_values("Vol Ratio", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("""
    **Reading this table:** Vol Ratio > 1.5x and a non-trivial Gap % are the names showing something unusual today — open those on the Trading Desk for the actual entry/exit read (VWAP, trend, pattern, trendline/swing overlay).
    Today's Proj. Range is a statistical volatility envelope for the rest of the session (25th-75th percentile), drift-tilted by the ML model's own daily bias where a trained model exists — it is NOT the ML model producing hourly predictions; the model itself only classifies multi-day direction.
    """)


def _check_ticker(ticker: str) -> dict:
    daily = get_price_history(ticker, period="6mo", interval="1d")
    if daily is None or daily.empty or len(daily) < 25:
        return None
    daily = calculate_indicators(daily)

    current = float(daily["Close"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-2])
    today_open = float(daily["Open"].iloc[-1])
    gap_pct = (today_open - prev_close) / prev_close * 100

    vol_ratio = float(daily["vol_ratio"].iloc[-1]) if "vol_ratio" in daily.columns and pd.notna(daily["vol_ratio"].iloc[-1]) else 1.0
    atr_pct = float(daily["ATR_pct"].iloc[-1]) if "ATR_pct" in daily.columns and pd.notna(daily["ATR_pct"].iloc[-1]) else None

    ml_signal, ml_prob, expected_move = "—", None, None
    if _ML_AVAILABLE:
        try:
            xgb_file = Path(_ML_STORAGE_DIR) / f"{ticker}_xgb.pkl"
            if xgb_file.exists():
                result = ml_predict(ticker, daily)
                if not result.get("error"):
                    ml_signal = result.get("direction", "neutral").upper()
                    ml_prob = result.get("probability", 0.50)
                    expected_move = result.get("expected_move_pct")
        except Exception:
            pass

    proj_range = "—"
    try:
        intraday = get_price_history(ticker, period="5d", interval="1h")
        if intraday is not None and not intraday.empty:
            path = simulate_intraday_path(
                intraday,
                bull_probability=ml_prob if ml_prob is not None else 0.5,
                expected_move_pct=expected_move,
            )
            if "error" not in path:
                proj_range = f"${path['session_close_p25']:.2f} - ${path['session_close_p75']:.2f}"
    except Exception:
        pass

    return {
        "Ticker": ticker,
        "Price": f"${current:.2f}",
        "Gap %": f"{gap_pct:+.1f}%",
        "Vol Ratio": round(vol_ratio, 2),
        "ATR %": f"{atr_pct:.1f}%" if atr_pct is not None else "—",
        "ML Signal": ml_signal,
        "Bull Prob": f"{ml_prob*100:.0f}%" if ml_prob is not None else "—",
        "Today's Proj. Range": proj_range,
    }


render()
