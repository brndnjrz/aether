"""
Signal attribution (Roadmap Item 10) — which Day Trading signals actually predict
direction, fitted from this user's own logged history.

Why this exists in this shape: a weighting like
`Trend 35% / Momentum 20% / Volume 15% / Pattern 20% / News 10%` looks measured,
but nothing produces those numbers. Once on screen they read as evidence. The
honest alternatives are to fit weights from real logged outcomes or to show the
vote count the trade card already shows — so this module fits them, and refuses
to return anything when the sample is too small to fit (see MIN_EVENTS_TO_FIT).

The awkward part, stated plainly: `activity_log` records what the signals *said*
but never what happened next. There is no label in the data. So
`resolve_signal_events()` has to grade each logged event against subsequent price
action, and the horizon it uses is a convention this module picks (1 trading day
close-to-close by default), not something the log knows. That makes these weights
"which signals led price over the next day, on the days you happened to look" —
not a clean experiment. Once Roadmap Item 12 links real fills, the labels get
better and this can be re-fit against actual trades.

Streamlit-free. The price fetcher is injected so tests need no network.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNAL_EVENT_TYPE = "day_trading_analyze"

# The three directional signals the trade card votes with. Volume is recorded but
# excluded here for the same reason it is excluded from the card's vote: it is a
# conviction read, not a direction.
SIGNAL_FIELDS = ["vwap_direction", "momentum_direction", "trend_direction"]

# bull/bear/neutral -> ordinal, so one fitted coefficient per signal is directly
# interpretable as "how much this signal moves the odds" and coefficients are
# comparable to each other without scaling.
_DIRECTION_VALUES = {
    "bull": 1.0, "bullish": 1.0,
    "neutral": 0.0, None: 0.0, "": 0.0,
    "bear": -1.0, "bearish": -1.0,
}

# A logistic fit over 3 features needs a real sample. Below this, return nothing
# rather than coefficients that would be noise wearing a lab coat.
MIN_EVENTS_TO_FIT = 30

DEFAULT_HORIZON_DAYS = 1


def parse_signal_events(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Turn raw `activity_log` rows into a tidy frame of signal states.

    Accepts the full mixed activity log and filters to SIGNAL_EVENT_TYPE. Rows
    with unparseable `detail_json`, or with none of SIGNAL_FIELDS present, are
    dropped. Returns columns: logged_at, ticker, interval, the SIGNAL_FIELDS as
    ordinals, and suggested_direction.

    Never raises — a corrupt row is skipped, not fatal.
    """
    cols = ["logged_at", "ticker", "interval", *SIGNAL_FIELDS, "suggested_direction"]
    if not rows:
        return pd.DataFrame(columns=cols)

    records = []
    for row in rows:
        if row.get("event_type") != SIGNAL_EVENT_TYPE:
            continue
        raw = row.get("detail_json")
        try:
            detail = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            logger.debug("parse_signal_events: unparseable detail_json, skipping row")
            continue
        if not isinstance(detail, dict):
            continue
        if not any(f in detail for f in SIGNAL_FIELDS):
            continue

        rec = {
            "logged_at": row.get("logged_at"),
            "ticker": (row.get("ticker") or "").upper(),
            "interval": detail.get("interval"),
            "suggested_direction": detail.get("suggested_direction"),
        }
        for field in SIGNAL_FIELDS:
            rec[field] = _DIRECTION_VALUES.get(detail.get(field), 0.0)
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records)
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True, format="mixed", errors="coerce")
    df = df[df["logged_at"].notna()]
    return df[cols].sort_values("logged_at").reset_index(drop=True)


def resolve_signal_events(
    events: pd.DataFrame,
    price_fetcher: Callable[[str], pd.DataFrame],
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Attach a realized label to each logged signal event.

    `activity_log` has no outcome column, so the label is derived: close-to-close
    over `horizon_days` trading bars after the event, matching
    `resolve_predictions()`'s convention of taking bars strictly after the
    timestamp. `went_up` is 1 when that return is positive.

    `horizon_days` is this module's choice, not the log's — the trade card is an
    intraday-to-one-day setup, so 1 is the default. Changing it changes what the
    fitted weights mean, which is why it is explicit rather than buried.

    Adds `forward_return_pct` and `went_up`; drops events whose horizon has not
    elapsed. Never raises — a ticker whose prices cannot be fetched is dropped
    with a debug log.
    """
    out_cols = list(events.columns) + ["forward_return_pct", "went_up"]
    if events is None or events.empty:
        return pd.DataFrame(columns=out_cols)

    frames = []
    for ticker, group in events.groupby("ticker"):
        try:
            prices = price_fetcher(ticker)
        except Exception as exc:
            logger.debug("resolve_signal_events: price fetch failed for %s: %s", ticker, exc)
            continue
        if prices is None or prices.empty or "Close" not in prices:
            continue

        idx = pd.DatetimeIndex(prices.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        closes = pd.Series(prices["Close"].values, index=idx)

        rets, ups = [], []
        for _, ev in group.iterrows():
            future = closes[closes.index > ev["logged_at"]]
            if len(future) < horizon_days + 1:
                rets.append(np.nan)
                ups.append(np.nan)
                continue
            entry = float(future.iloc[0])
            exit_price = float(future.iloc[horizon_days])
            ret = (exit_price - entry) / entry * 100 if entry else np.nan
            rets.append(round(ret, 4) if ret == ret else np.nan)
            ups.append(1.0 if ret == ret and ret > 0 else (0.0 if ret == ret else np.nan))

        g = group.copy()
        g["forward_return_pct"] = rets
        g["went_up"] = ups
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=out_cols)

    resolved = pd.concat(frames, ignore_index=True)
    resolved = resolved[resolved["went_up"].notna()]
    return resolved.sort_values("logged_at").reset_index(drop=True)


def fit_signal_weights(resolved: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit one coefficient per signal, from the user's own resolved history.

    Logistic regression of `went_up` on the ordinal signal states. Because every
    feature is in {-1, 0, +1}, the coefficients are directly comparable and need
    no standardization: a coefficient of +0.4 on `trend_direction` means a bullish
    trend read shifts the log-odds of an up-move by +0.4.

    Returns
    -------
    {"fitted": bool, "n": int, "coefficients": {field: float}, "intercept": float,
     "train_accuracy": float, "base_rate": float, "reason": str | None}

    `fitted` is False — with `coefficients` empty and `reason` set — whenever the
    sample is under MIN_EVENTS_TO_FIT, the label has only one class, or sklearn is
    unavailable. Callers must fall back to the vote count in that case rather than
    displaying anything weight-shaped.
    """
    out: Dict[str, Any] = {
        "fitted": False, "n": 0, "coefficients": {}, "intercept": None,
        "train_accuracy": None, "base_rate": None, "reason": None,
    }
    if resolved is None or resolved.empty:
        out["reason"] = "No resolved signal events yet."
        return out

    usable = resolved.dropna(subset=["went_up", *SIGNAL_FIELDS])
    out["n"] = int(len(usable))
    if out["n"] < MIN_EVENTS_TO_FIT:
        out["reason"] = (
            f"Only {out['n']} resolved signal event(s) — need at least "
            f"{MIN_EVENTS_TO_FIT} before fitted weights mean anything. Use the "
            f"trade card's vote count instead."
        )
        return out

    y = usable["went_up"].astype(int).values
    out["base_rate"] = round(float(y.mean()), 4)
    if len(np.unique(y)) < 2:
        out["reason"] = (
            "Every resolved event went the same way, so there is nothing for a "
            "classifier to separate."
        )
        return out

    X = usable[SIGNAL_FIELDS].astype(float).values
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        out["reason"] = "scikit-learn is not installed."
        return out

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
    except Exception as exc:
        logger.warning("fit_signal_weights: fit failed: %s", exc)
        out["reason"] = f"Fit failed: {exc}"
        return out

    out.update({
        "fitted": True,
        "coefficients": {
            f: round(float(c), 4) for f, c in zip(SIGNAL_FIELDS, model.coef_[0])
        },
        "intercept": round(float(model.intercept_[0]), 4),
        # In-sample by construction — the same caveat the app's other in-sample
        # numbers carry. It is reported so a suspiciously perfect fit on a small
        # sample is visible rather than reassuring.
        "train_accuracy": round(float(model.score(X, y)), 4),
    })
    return out


def signal_contributions(
    weights: Dict[str, Any], signal_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply fitted weights to a current reading: contribution per signal, plus each
    one's share of the total absolute movement.

    `contribution = coefficient x signal_value`, in log-odds. `share_pct` is that
    contribution's fraction of the total absolute contribution — useful for "what
    drove this call", and deliberately computed over absolute values so an
    opposing signal shows up as a real share rather than cancelling silently.

    Returns {} when `weights` was not fitted; callers must not fabricate a
    breakdown from an unfitted model.
    """
    if not weights.get("fitted"):
        return {}

    contribs = {}
    for field, coef in weights["coefficients"].items():
        value = _DIRECTION_VALUES.get(signal_state.get(field), 0.0)
        contribs[field] = round(float(coef) * value, 4)

    total_abs = sum(abs(v) for v in contribs.values())
    return {
        "contributions": contribs,
        "shares_pct": {
            f: (round(abs(v) / total_abs * 100, 1) if total_abs > 0 else 0.0)
            for f, v in contribs.items()
        },
        "net_log_odds": round(sum(contribs.values()) + (weights["intercept"] or 0.0), 4),
        "total_abs": round(total_abs, 4),
    }
