"""
Model Lab — read-only prediction performance dashboard across the daily and
intraday models. Reads what Trading Desk has already logged and graded
(via analysis.prediction_performance); trains or generates nothing itself,
with one deliberate exception — Version History's rollback button, which
copies previously-archived model files back into place (never a retrain).
"""
import logging
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.ml_prediction import (
    _xgb_path as _daily_xgb_path, _rf_path as _daily_rf_path,
    get_prediction_history, compare_models,
    get_version_history as get_daily_version_history,
    rollback_to_version as rollback_daily_version,
)
from analysis.intraday_prediction import (
    INTERVAL_SPECS, model_exists as intraday_model_exists,
    get_intraday_prediction_history, compare_intraday_models,
    get_version_history as get_intraday_version_history,
    rollback_to_version as rollback_intraday_version,
)
from analysis.prediction_performance import compute_daily_prediction_metrics, compute_intraday_prediction_metrics
from analysis.prediction_errors import categorize_incorrect_predictions, aggregate_failure_categories

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


def _render_failure_analysis(ticker: str, model_label: str, history: pd.DataFrame, interval: str = None):
    st.markdown("**Why the model was wrong**")
    recompute = st.checkbox(
        "Recompute categories for legacy predictions (fetches network data)",
        value=False, key=f"model_lab_recompute_{model_label}",
        help=(
            "Predictions logged before this feature shipped have no saved indicator "
            "snapshot. Checking this re-fetches price history and recomputes "
            "indicators for those rows only — predictions with a saved snapshot "
            "are categorized either way, no network needed."
        ),
    )

    price_history_fetcher = None
    if recompute:
        from data.price_data import get_price_history

        if interval is None:
            price_history_fetcher = lambda t: get_price_history(t, period="2y")
        else:
            max_period = INTERVAL_SPECS[interval]["max_period"]
            price_history_fetcher = lambda t: get_price_history(t, period=max_period, interval=interval)

    from data.fundamentals import get_earnings_history

    categorized = categorize_incorrect_predictions(
        history, price_history_fetcher=price_history_fetcher, ticker=ticker,
        earnings_fetcher=get_earnings_history,
    )
    if categorized.empty:
        st.caption("No incorrect predictions to analyze yet.")
        return

    agg = aggregate_failure_categories(categorized)
    top = (agg["top_category"] or "—").replace("_", " ")
    st.caption(f"{agg['n_incorrect']} incorrect prediction(s) — most common reason: {top}.")

    counts_df = pd.DataFrame([
        {"Category": c.replace("_", " ").title(), "Count": v}
        for c, v in agg["category_counts"].items() if v > 0
    ])
    if not counts_df.empty:
        st.bar_chart(counts_df.set_index("Category"))

    with st.expander(f"{len(categorized)} incorrect prediction(s) — detail", expanded=False):
        detail_df = pd.DataFrame([
            {
                "Date": row["date"],
                "Direction": row["direction"],
                "Categories": ", ".join(c.replace("_", " ") for c in row["failure_categories"]),
                "Snapshot": row["snapshot_source"],
            }
            for _, row in categorized.iterrows()
        ])
        st.dataframe(detail_df, hide_index=True, width="stretch")


def _render_model_comparison(ticker: str, model_label: str, comparison_fn, button_key: str):
    st.markdown("**Model comparison**")
    st.caption(
        "Scores XGBoost, RandomForest, LogisticRegression, and GradientBoosting "
        "against each other through the same walk-forward validation training "
        "uses. Read-only — doesn't retrain or change the deployed model."
    )
    if not st.button(f"Run comparison for {ticker}", key=button_key):
        return

    with st.spinner("Running walk-forward validation for 4 models..."):
        result = comparison_fn(ticker)

    if result.get("error"):
        st.warning(result["error"])
        return

    rows = [
        {
            "Model": name.title(),
            "Accuracy": _pct(m["mean_directional_accuracy"]),
            "Std": _pct(m["std_directional_accuracy"]),
            "AUC": f"{m['mean_auc']:.3f}",
            "Folds": m["n_folds"],
            "Recommended Weight": _pct(result["ensemble_weights"].get(name)),
        }
        for name, m in result["models"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Best single model: **{result['best_single_model'].title()}**. "
        f"Recommended weight is a softmax over each model's accuracy — informational, "
        f"not wired into the deployed 2-model (XGB/RF) ensemble."
    )


def _render_version_history(ticker: str, model_label: str, get_history_fn, rollback_fn):
    st.markdown("**Version history**")
    history = get_history_fn(ticker)
    if history.empty:
        st.caption("No prior versions yet — versions are created starting from a model's second training run.")
        return

    display_df = history.copy()
    display_df["Accuracy"] = display_df["directional_accuracy"].apply(_pct)
    display_df["Event"] = display_df["rolled_back_from_latest"].apply(
        lambda v: "Rolled back to" if v else "Archived (replaced by retrain)"
    )
    st.dataframe(
        display_df[["version", "Event", "archived_at", "Accuracy", "is_reliable"]].rename(
            columns={"version": "Version", "archived_at": "When", "is_reliable": "Was Reliable"}
        ),
        hide_index=True, width="stretch",
    )

    archived_versions = sorted(history[~history["rolled_back_from_latest"]]["version"].unique(), reverse=True)
    if not archived_versions:
        return
    with st.expander("Rollback to a prior version", expanded=False):
        st.caption(
            "Copies that version's files back into place — a straight file "
            "copy, not a retrain. The current model is archived first, so "
            "rolling back never discards it."
        )
        target = st.selectbox(
            "Version", archived_versions, key=f"model_lab_rollback_select_{model_label}",
        )
        if st.button(f"Rollback {ticker} ({model_label}) to v{target}", key=f"model_lab_rollback_btn_{model_label}"):
            result = rollback_fn(ticker, target)
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Rolled back to v{target}.")
                st.rerun()


def _render_daily_performance_dashboard(ticker: str):
    if not _daily_model_exists(ticker):
        st.info(f"No daily model trained yet for {ticker} — train it from Trading Desk → Predictions.")
        return
    history = get_prediction_history(ticker)
    metrics = compute_daily_prediction_metrics(ticker, history=history)
    _render_dashboard(metrics, ticker, "daily")
    if metrics["n_resolved"] > 0:
        st.markdown("---")
        _render_failure_analysis(ticker, "daily", history)
    st.markdown("---")
    _render_model_comparison(ticker, "daily", compare_models, "model_lab_compare_daily")
    st.markdown("---")
    _render_version_history(ticker, "daily", get_daily_version_history, rollback_daily_version)


def _render_intraday_performance_dashboard(ticker: str, interval: str):
    if not intraday_model_exists(ticker, interval):
        st.info(
            f"No {interval} model trained yet for {ticker} — train it from "
            "Trading Desk → Predictions → Intraday."
        )
        return
    history = get_intraday_prediction_history(ticker, interval)
    metrics = compute_intraday_prediction_metrics(ticker, interval, history=history)
    _render_dashboard(metrics, ticker, f"{interval} intraday")
    if metrics["n_resolved"] > 0:
        st.markdown("---")
        _render_failure_analysis(ticker, f"{interval}_intraday", history, interval=interval)
    st.markdown("---")
    _render_model_comparison(
        ticker, f"{interval} intraday",
        lambda t: compare_intraday_models(t, interval),
        f"model_lab_compare_intraday_{interval}",
    )
    st.markdown("---")
    _render_version_history(
        ticker, f"{interval}_intraday",
        lambda t: get_intraday_version_history(t, interval),
        lambda t, v: rollback_intraday_version(t, interval, v),
    )


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
