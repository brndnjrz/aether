"""
Model Lab — read-only prediction performance dashboard across the daily and
intraday models. Reads what Trading Desk has already logged and graded
(via analysis.prediction_performance); never persists a new or changed model.

Deliberate exceptions to "read-only", kept named here so the claim cannot go
stale as panels are added:

1. **Model Comparison** fits four models in-memory purely to score them —
   nothing is written to storage/.
2. **Version History's rollback** copies previously-archived model files back
   into place (a file copy, never a retrain).
3. **Failure Analysis's "recompute legacy" checkbox** re-fetches price history
   for rows logged before indicator snapshots existed. Opt-in, network.
4. **Horizon Scoreboard's "price with live options quotes" checkbox** fetches an
   ATM expiry ladder so costs can be priced as options rather than shares.
   Opt-in, network, cached — reads only, writes nothing.

Nothing here trains, predicts, or appends to a prediction log. That matters
beyond tidiness: predict()/predict_intraday() persist unconditionally, so a page
that generated predictions would inflate the very win rate this page reports.
"""
import logging
import os
import sys

import pandas as pd
import plotly.graph_objects as go
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
from analysis.prediction_performance import (
    MIN_N_PER_CONFIDENCE_BUCKET,
    MIN_USEFUL_ACCURACY,
    compare_horizons,
    compute_daily_prediction_metrics,
    compute_intraday_prediction_metrics,
)
from analysis.prediction_errors import categorize_incorrect_predictions, aggregate_failure_categories
from analysis.retrain_triggers import check_all_retrain_triggers
from config.settings import (
    RETRAIN_ACCURACY_DROP_THRESHOLD,
    RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK,
)
from config.tz import MARKET_TZ

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
    # "Win Rate" was here as a second tile, but prediction_performance aliases it
    # to accuracy — the same number shown twice reads as independent
    # corroboration. Sample size takes that slot instead, since it is what the
    # accuracy actually needs to be interpreted.
    m1.metric("Accuracy", _pct(metrics["accuracy"]),
              help="Also reported as 'win rate' — they are the same number here: "
                   "the share of resolved directional calls that were right.")
    m2.metric("Resolved n", f"{metrics['n_resolved']:,}",
              help="Accuracy over fewer than ~30 resolved predictions carries a "
                   "margin of error wider than most edges you would act on.")
    m3.metric("Precision (macro)", _pct(metrics["precision"]["macro"]))
    m4.metric("Recall (macro)", _pct(metrics["recall"]["macro"]))
    m5.metric("F1 (macro)", _pct(metrics["f1"]["macro"]))

    m6, m7, m8, m9 = st.columns(4)
    m6.metric("False Positive Rate", _pct(metrics["false_positive_rate"]["macro"]))
    m7.metric("False Negative Rate", _pct(metrics["false_negative_rate"]["macro"]))
    avg_profit = metrics["avg_profit_per_signal"]
    m8.metric("Avg Profit / Signal", f"{avg_profit:+.2f}%" if avg_profit is not None else "—",
              help="Gross of costs: no spread, commission, delta leverage, or theta. "
                   "For a signal whose average move is ~1%, a 6bp round-trip spread "
                   "consumes most of the edge at accuracies near 52%. It also averages "
                   "across mixed horizons.")
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


def _render_retrain_triggers(ticker: str, interval: str = None):
    st.markdown("**Retrain triggers**")
    check = check_all_retrain_triggers(ticker, interval=interval)
    triggers = check["triggers"]

    if check["should_retrain"]:
        st.warning("Retrain recommended — see below for which trigger(s) fired.")
    else:
        st.caption("No retrain trigger currently active.")

    rows = [
        {"Trigger": name.replace("_", " ").title(), "Fired": "Yes" if t["triggered"] else "No", "Detail": t["reason"]}
        for name, t in triggers.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_daily_performance_dashboard(ticker: str):
    if not _daily_model_exists(ticker):
        st.info(f"No daily model trained yet for {ticker} — train it from Trading Desk → Predictions.")
        return
    # resolve=False: this page is read-only. Resolving fetches price history
    # and rewrites the prediction log, which the page docstring promises it
    # does not do. Trading Desk and the cron sweep own resolution.
    history = get_prediction_history(ticker, resolve=False)
    metrics = compute_daily_prediction_metrics(ticker, history=history)
    _render_dashboard(metrics, ticker, "daily")
    if metrics["n_resolved"] > 0:
        st.markdown("---")
        _render_accuracy_trend(
            ticker, history, "daily", version_history=get_daily_version_history(ticker),
        )
        st.markdown("---")
        _render_failure_analysis(ticker, "daily", history)
    st.markdown("---")
    _render_model_comparison(ticker, "daily", compare_models, "model_lab_compare_daily")
    st.markdown("---")
    _render_version_history(ticker, "daily", get_daily_version_history, rollback_daily_version)
    st.markdown("---")
    _render_retrain_triggers(ticker)


def _render_intraday_performance_dashboard(ticker: str, interval: str):
    if not intraday_model_exists(ticker, interval):
        st.info(
            f"No {interval} model trained yet for {ticker} — train it from "
            "Trading Desk → Predictions → Intraday."
        )
        return
    history = get_intraday_prediction_history(ticker, interval, resolve=False)
    metrics = compute_intraday_prediction_metrics(ticker, interval, history=history)
    _render_dashboard(metrics, ticker, f"{interval} intraday")
    if metrics["n_resolved"] > 0:
        st.markdown("---")
        _render_accuracy_trend(
            ticker, history, f"{interval}_intraday",
            version_history=get_intraday_version_history(ticker, interval),
        )
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
    st.markdown("---")
    _render_retrain_triggers(ticker, interval=interval)


def _render_accuracy_trend(ticker: str, history: pd.DataFrame, model_label: str,
                           version_history: pd.DataFrame = None):
    """
    Rolling hit rate over time, with retrain dates marked (Roadmap Item 8).

    A single accuracy number cannot tell "steady at 55%" from "was 62%, now 48%",
    and those call for different actions. Marking retrains makes each one's effect
    visible rather than inferred.
    """
    from analysis.prediction_performance import DEFAULT_ROLLING_WINDOW, rolling_accuracy

    st.markdown("**Accuracy over time**")
    trend = rolling_accuracy(history)
    if trend.empty:
        st.caption("No resolved predictions yet — nothing to trend.")
        return
    if len(trend) < 3:
        st.caption(f"Only {len(trend)} resolved prediction(s) — not enough points to trend yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["date"], y=(trend["accuracy"] * 100).round(1),
        mode="lines+markers", name=f"Rolling {DEFAULT_ROLLING_WINDOW}-prediction accuracy",
        line=dict(color="#42a5f5", width=2), marker=dict(size=5),
        customdata=trend["n_window"],
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Accuracy: %{y:.1f}%<br>Window: %{customdata:.0f} preds<extra></extra>",
    ))
    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(158,158,158,0.6)",
        annotation_text="coin flip", annotation_position="right",
    )

    if version_history is not None and not version_history.empty:
        for _, v in version_history.iterrows():
            when = v.get("archived_at")
            if not when:
                continue
            try:
                ts = pd.Timestamp(when)
            except (ValueError, TypeError):
                continue
            if ts.tz is None:
                ts = ts.tz_localize(MARKET_TZ)
            fig.add_vline(
                x=ts.tz_convert("UTC"), line_dash="dot", line_width=1,
                line_color="rgba(255,152,0,0.7)",
            )

    fig.update_layout(
        template="plotly_dark" if st.context.theme.type == "dark" else "plotly_white",
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Hit rate %", range=[0, 100]),
        xaxis=dict(title=""),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Rolling window of {DEFAULT_ROLLING_WINDOW} resolved predictions (fewer early on — "
        "hover for the window size at each point). Orange dotted lines mark retrains, so a "
        "retrain that made things worse is visible rather than inferred."
    )


def _render_agreement_backtest(ticker: str):
    """
    Does waiting for a second horizon to confirm actually improve the hit rate?
    (Roadmap Item 7.)

    Click-to-run: it reads two prediction logs and joins them, which is cheap, but
    the result is only meaningful once both logs have accumulated, so there is no
    point computing it on every page load.
    """
    from analysis.interval_consensus import backtest_agreement

    st.markdown("**Does two-horizon confirmation help?**")
    st.caption(
        "Compares the base horizon's hit rate when a second horizon agreed against its "
        "hit rate overall. If confirmation measurably helps, \"wait for a second "
        "horizon\" becomes a measured rule rather than a hunch."
    )

    intervals = list(INTERVAL_SPECS)
    c1, c2, c3 = st.columns([1, 1, 1])
    base = c1.selectbox("Base horizon", intervals, index=intervals.index("15m"),
                        key="agree_base")
    confirm_options = [i for i in intervals if i != base]
    confirm = c2.selectbox("Confirmed by", confirm_options,
                           index=min(confirm_options.index("30m") if "30m" in confirm_options else 0,
                                     len(confirm_options) - 1),
                           key="agree_confirm")
    tolerance = c3.number_input(
        "Join tolerance (min)", min_value=1, max_value=240, value=30, step=5,
        key="agree_tolerance",
        help=(
            "Two predictions count as concurrent within this many minutes. Predictions "
            "are logged by hand, so they rarely line up exactly."
        ),
    )

    if not st.button(f"Run agreement backtest for {ticker}", key="agree_run"):
        return

    result = backtest_agreement(
        ticker, base_interval=base, confirm_interval=confirm,
        tolerance_minutes=float(tolerance),
    )

    baseline = result["baseline_accuracy"]
    if baseline is None:
        st.info("Not enough resolved predictions on the base horizon yet.")
        for w in result["warnings"]:
            st.caption(f"— {w}")
        return

    m1, m2 = st.columns(2)
    m1.metric(f"{base} alone", _pct(baseline), f"{result['n_base_resolved']} resolved")
    lift = result["agreement_lift"]
    m2.metric(
        f"{base} when {confirm} agreed",
        _pct(result["buckets"]["agree"]["accuracy"]),
        f"{lift * 100:+.1f} pts vs alone" if lift is not None else "not enough pairs",
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Bucket": name.title(),
                "N": b["n"],
                "Hit rate": _pct(b["accuracy"]),
            }
            for name, b in result["buckets"].items()
        ]),
        hide_index=True, width="stretch",
    )

    if lift is not None:
        if lift > 0.02:
            st.success(
                f"Confirmation helped by {lift * 100:+.1f} points on this sample. Waiting for "
                f"{confirm} costs only theta, so this may be worth the delay.",
            )
        elif lift < -0.02:
            st.warning(
                f"Confirmation *hurt* by {lift * 100:+.1f} points on this sample — waiting "
                f"filtered out more winners than losers.",
            )
        else:
            st.info("No meaningful difference on this sample.")

    with st.expander("Why this is not a clean backtest", expanded=False):
        for w in result["warnings"]:
            st.caption(f"— {w}")


def _render_horizon_scoreboard(ticker: str):
    """
    Which horizon should I trade, and which expiry should I buy to trade it?

    Every column here already existed somewhere in the app — the walk-forward
    numbers in each model's accuracy file, live accuracy in the per-interval
    dashboards, costs in the tradeability record. What was missing was seeing
    them in one place, because "how is the 15m model doing" is a different
    question from "is 15m the horizon I should be using at all."
    """
    st.markdown("### Horizon Scoreboard")
    st.caption(
        "All five models ranked against each other. Walk-forward is what the model "
        "trained with; Live is what it has actually done since. Verdicts follow a "
        "fixed rule chain, not a score — see the expander below."
    )

    use_options = st.checkbox(
        "Price with live options quotes",
        value=False,
        key="scoreboard_use_options",
        help=(
            "Off: net edge comes from the stored shares model (2 bps round trip), "
            "which ignores delta leverage and has no theta term — wrong for "
            "contracts. On: fetches an ATM expiry ladder and prices every horizon "
            "as an option across 0/2/7/30 DTE, filling the grid below. Network, "
            "cached 10 minutes."
        ),
    )

    ladder_quotes = None
    underlying_price = None
    if use_options:
        from data.options_data import get_expiry_ladder_quotes

        with st.spinner("Fetching ATM expiry ladder…"):
            ladder = get_expiry_ladder_quotes(ticker)
        if ladder.get("error"):
            st.warning(f"Options quotes unavailable ({ladder['error']}) — showing the shares cost model.")
        else:
            ladder_quotes = ladder.get("quotes") or None
            underlying_price = ladder.get("underlying_price")

    result = compare_horizons(
        ticker, ladder_quotes=ladder_quotes, underlying_price=underlying_price,
    )

    rows = []
    for r in result["rows"]:
        net = r["net_edge_pct"]
        rows.append({
            "Horizon": r["horizon"],
            "Walk-fwd": _pct(r["trained_accuracy"]),
            "Std": _pct(r["accuracy_std"]),
            "Live acc": _pct(r["live_accuracy"]),
            "N": r["n_resolved"] or 0,
            "Best expiry": (
                f"{r['best_dte']:g} DTE" if r["best_dte"] is not None else "—"
            ),
            "Net edge": f"{net:+.3f}%" if net is not None else "—",
            "Dominant cost": r["dominant_cost"] or "—",
            "IV crush to erase": (
                f"{r['iv_points_to_erase_edge']:.1f} vol pts"
                if r.get("iv_points_to_erase_edge") is not None else "—"
            ),
            "Calibrated": r["calibrated"],
            "Verdict": r["verdict"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if result["grid"]:
        with st.expander("Net edge % — every horizon x expiry combination", expanded=True):
            ladder = result["ladder"]
            grid_rows = []
            for r in result["rows"]:
                cells = result["grid"].get(r["horizon"])
                if not cells:
                    continue
                row = {"Horizon": r["horizon"]}
                for dte in ladder:
                    v = cells.get(dte)
                    row[f"{dte:g} DTE"] = f"{v:+.2f}%" if v is not None else "—"
                grid_rows.append(row)
            if grid_rows:
                st.dataframe(pd.DataFrame(grid_rows), hide_index=True, width="stretch")
                st.caption(
                    "Leverage and theta both scale inversely with time to expiry, so "
                    "they partly cancel — which is why the sign can flip across a row. "
                    "Expect a diagonal: the shorter the signal horizon, the more time "
                    "you have to buy to outrun decay."
                )
            else:
                st.caption("No horizon had both a trained model and a volatility anchor to price against.")

    with st.expander("How the verdict is decided", expanded=False):
        st.markdown(
            "Evaluated in order, first match wins:\n\n"
            "1. **Not trained** — no model on disk.\n"
            f"2. **Insufficient data** — fewer than {RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK} "
            "resolved predictions. Economics can't be judged on a handful.\n"
            "3. **Do not trade** — negative net edge *and* walk-forward accuracy at or "
            f"below {MIN_USEFUL_ACCURACY * 100:.0f}% (the reliability gate). No real edge, "
            "and costs fail too.\n"
            "4. **Uneconomic** — negative net edge but accuracy *above* the gate. The "
            "model works; the instrument is wrong. Usually fixed by buying more time "
            "(a longer expiry), not by retraining.\n"
            f"5. **Degraded, retrain** — live accuracy has fallen "
            f"{RETRAIN_ACCURACY_DROP_THRESHOLD * 100:.0f}+ points below trained-in.\n"
            "6. **Primary / Secondary** — survivors, ranked by net edge.\n"
            "7. **Swing context** — the daily model, never ranked against intraday "
            "horizons; a 5-day call and a 75-minute call are not substitutes.\n\n"
            "**Calibrated** asks whether HIGH confidence has actually landed more often "
            "than LOW for that horizon. Confidence here is *distance from the neutral "
            "band*, not a probability of being right — if this column says `no`, ignore "
            f"the badge and use raw accuracy. `unknown (n)` means under "
            f"{MIN_N_PER_CONFIDENCE_BUCKET} rows in one of the buckets."
        )

    if result["warnings"]:
        with st.expander(f"{len(result['warnings'])} note(s)", expanded=False):
            for w in result["warnings"]:
                st.caption(f"— {w}")


def render():
    st.markdown("# Model Lab")
    st.caption(
        "Live prediction track record — reads what Trading Desk has already logged, "
        "never persists a new or changed model itself."
    )

    with st.expander("How to read this page", expanded=False):
        st.markdown(
            "- **Horizon Scoreboard** (top) — the panel that answers *which* horizon to "
            "trade, rather than how one horizon is doing. Read the Verdict column first: "
            "`Uneconomic` means the model works but the instrument/expiry is wrong, which "
            "is a different fix from `Do not trade`.\n"
            "- **Accuracy / Win Rate** — % of graded predictions that were correct. Compare to "
            "the accuracy the model *trained with* (Trading Desk's Predictions tab) — if live "
            "is meaningfully lower, check Retrain Triggers below.\n"
            "- **Precision (per direction)** — of every call in that direction, how often it was "
            "right. **Recall** — of every real move in that direction, how many it caught. Low "
            "recall on one side means it's missing those moves, not that it's \"biased wrong.\"\n"
            "- **Confidence calibration** — the one worth checking regularly: compare \"Avg "
            "Predicted Prob.\" to \"Realized Accuracy\" per bucket. If HIGH isn't meaningfully "
            "more accurate than LOW, don't trust the confidence badge for this ticker — trust "
            "the raw accuracy number instead.\n"
            "- **Why the model was wrong** — every miss tagged with a reason. A pile-up in one "
            "category (e.g. `elevated_vol_regime`) usually means the market shifted, not that "
            "the model is broken — a retrain on fresher data is the usual fix. `uncategorized` "
            "misses are the genuine model errors.\n"
            "- **Model comparison** — informational only, never changes the deployed model. Run "
            "it out of curiosity, not before every trade.\n"
            "- **Version history** — if a retrain makes things worse, roll back with one click; "
            "the current model is archived first, so nothing is ever discarded.\n"
            "- **Retrain triggers** — a nudge, not an alarm. Check Failure Analysis first to "
            "understand *why* before retraining on Trading Desk.\n\n"
            "Full writeup: `docs/ML_PREDICTION.md` → \"How to Read Model Lab\"."
        )

    ticker = st.text_input("Ticker", value=st.session_state.get("quick_lookup_ticker", "SPY")).upper().strip()
    if not ticker:
        st.info("Enter a ticker to get started.")
        return

    # Spans both models, so it sits above the tabs rather than inside either.
    _render_horizon_scoreboard(ticker)
    st.markdown("---")
    _render_agreement_backtest(ticker)
    st.markdown("---")

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
