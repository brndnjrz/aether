"""
Model Lab — read-only prediction performance dashboard across the daily and
intraday models. Reads what Trading Desk has already logged and graded
(via analysis.prediction_performance); trains or generates nothing itself.
"""
import logging
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.ml_prediction import _xgb_path as _daily_xgb_path, _rf_path as _daily_rf_path
from analysis.intraday_prediction import INTERVAL_SPECS, model_exists as intraday_model_exists
from analysis.prediction_performance import compute_daily_prediction_metrics, compute_intraday_prediction_metrics

logger = logging.getLogger(__name__)


def _daily_model_exists(ticker: str) -> bool:
    return _daily_xgb_path(ticker).exists() and _daily_rf_path(ticker).exists()


def _pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _render_precision_recall(metrics: dict):
    st.markdown("**Precision / Recall / F1 by direction**")
    rows = [
        {
            "Class": cls.title(),
            "Precision": _pct(metrics["precision"][cls]),
            "Recall": _pct(metrics["recall"][cls]),
            "F1": _pct(metrics["f1"][cls]),
            "False Positive Rate": _pct(metrics["false_positive_rate"][cls]),
            "False Negative Rate": _pct(metrics["false_negative_rate"][cls]),
        }
        for cls in ("bullish", "bearish", "macro")
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Confusion matrix (resolved predictions only)", expanded=False):
        cc = metrics["confusion_counts"]
        st.dataframe(
            pd.DataFrame([
                {"Direction": "Bullish", "Correct": cc["bullish"]["correct"], "Incorrect": cc["bullish"]["incorrect"]},
                {"Direction": "Bearish", "Correct": cc["bearish"]["correct"], "Incorrect": cc["bearish"]["incorrect"]},
            ]),
            hide_index=True, width="stretch",
        )


def _render_calibration(metrics: dict):
    st.markdown("**Confidence calibration**")
    by_conf = metrics["calibration"]["by_confidence"]
    st.dataframe(
        pd.DataFrame([
            {
                "Confidence": level.title(), "N": by_conf[level]["n"],
                "Avg Predicted Prob.": _pct(by_conf[level]["avg_predicted_probability"]),
                "Realized Accuracy": _pct(by_conf[level]["realized_accuracy"]),
            }
            for level in ("high", "medium", "low")
        ]),
        hide_index=True, width="stretch",
    )

    deciles = metrics["calibration"]["by_probability_decile"]
    if deciles:
        st.caption(
            "Predicted probability vs. realized accuracy, bucketed by directional "
            "confidence (P(call actually made), not raw P(bullish))."
        )
        chart_df = pd.DataFrame([
            {"Bucket": d["bucket"], "Predicted": d["avg_predicted_probability"], "Realized": d["realized_accuracy"]}
            for d in deciles
        ]).set_index("Bucket")
        st.bar_chart(chart_df)
    elif metrics["calibration"]["note"]:
        st.caption(metrics["calibration"]["note"])


def _render_dashboard(metrics: dict, ticker: str, model_label: str):
    if metrics["n_total"] == 0:
        st.info(f"No {model_label} predictions logged yet for {ticker}.")
        return
    if metrics["n_resolved"] == 0:
        st.info(
            f"{metrics['n_total']} {model_label} prediction(s) logged for {ticker}, "
            "none resolved yet — check back once their horizon has elapsed."
        )
        return

    st.caption(f"{metrics['n_resolved']} resolved / {metrics['n_total']} logged {model_label} prediction(s).")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", _pct(metrics["accuracy"]))
    m2.metric("Win Rate", _pct(metrics["win_rate"]))
    m3.metric("Precision (macro)", _pct(metrics["precision"]["macro"]))
    m4.metric("Recall (macro)", _pct(metrics["recall"]["macro"]))
    m5.metric("F1 (macro)", _pct(metrics["f1"]["macro"]))

    m6, m7, m8, m9 = st.columns(4)
    m6.metric("False Positive Rate", _pct(metrics["false_positive_rate"]["macro"]))
    m7.metric("False Negative Rate", _pct(metrics["false_negative_rate"]["macro"]))
    avg_profit = metrics["avg_profit_per_signal"]
    m8.metric("Avg Profit / Signal", f"{avg_profit:+.2f}%" if avg_profit is not None else "—")
    holding = metrics["holding_time"]
    unit = holding["unit"] or ""
    m9.metric("Avg Holding Time", f"{holding['mean']:.1f} {unit}" if holding["mean"] is not None else "—")

    st.markdown("---")
    _render_precision_recall(metrics)
    st.markdown("---")
    _render_calibration(metrics)

    if metrics["warnings"]:
        with st.expander(f"{len(metrics['warnings'])} note(s)", expanded=False):
            for w in metrics["warnings"]:
                st.caption(f"— {w}")


def _render_daily_performance_dashboard(ticker: str):
    if not _daily_model_exists(ticker):
        st.info(f"No daily model trained yet for {ticker} — train it from Trading Desk → Predictions.")
        return
    metrics = compute_daily_prediction_metrics(ticker)
    _render_dashboard(metrics, ticker, "daily")


def _render_intraday_performance_dashboard(ticker: str, interval: str):
    if not intraday_model_exists(ticker, interval):
        st.info(
            f"No {interval} model trained yet for {ticker} — train it from "
            "Trading Desk → Predictions → Intraday."
        )
        return
    metrics = compute_intraday_prediction_metrics(ticker, interval)
    _render_dashboard(metrics, ticker, f"{interval} intraday")


def render():
    st.markdown("# Model Lab")
    st.caption(
        "Live prediction track record — reads what Trading Desk has already logged, "
        "doesn't train or generate anything itself."
    )

    ticker = st.text_input("Ticker", value=st.session_state.get("quick_lookup_ticker", "SPY")).upper().strip()
    if not ticker:
        st.info("Enter a ticker to get started.")
        return

    tab_daily, tab_intraday = st.tabs(["Daily Model", "Intraday Model"])
    with tab_daily:
        _render_daily_performance_dashboard(ticker)
    with tab_intraday:
        intervals = list(INTERVAL_SPECS)
        interval = st.selectbox(
            "Interval", intervals, index=intervals.index("15m"), key="model_lab_interval",
        )
        _render_intraday_performance_dashboard(ticker, interval)

    st.markdown("---")
    st.caption("Aether • Data from yfinance • For personal use only. Not financial advice.")


render()
