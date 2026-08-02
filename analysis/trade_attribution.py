"""
Trade attribution (Roadmap Item 12) — did following the model actually make money?

Every other performance number in this app grades the *model*: was the direction
right. This grades the *decision*: when you followed a signal versus when you
overrode it, which produced better round trips. Those are different questions, and
only real fills can answer the second one.

**The N gate is the feature, not a limitation.** With 20-40 round trips, "you lose
on countertrend setups" is noise that reads as self-knowledge, and acting on it is
worse than acting on nothing. This module therefore refuses to report a comparison
below MIN_ROUND_TRIPS_FOR_COMPARISON, matching the bar the app already holds
elsewhere (RETRAIN_MIN_RESOLVED_FOR_DROP_CHECK = 20, ORBC's sub-30-trade warning).
Log from day one; read the comparison much later.

Streamlit-free. Operates on the round trips portfolio/round_trips.py already
derives, so P&L is never recomputed here and cannot drift from the Options Log.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Below this, no comparison is reported. Deliberately higher than the app's other
# floors: this splits the sample in two before measuring, so each side needs to be
# non-trivial on its own.
MIN_ROUND_TRIPS_FOR_COMPARISON = 40

# And each arm needs its own minimum, or a 38/2 split would clear the total gate
# while telling you nothing about the 2.
MIN_PER_ARM = 15


def parse_prediction_ref(ref: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Split a `prediction_ref` into its parts.

    Format is "<horizon>|<prediction ISO timestamp>", e.g.
    "15m|2026-08-03T10:30:00-04:00". Returns {"horizon", "predicted_at"} with
    None values for anything unparseable — a malformed ref must degrade to
    "discretionary" rather than raise.
    """
    if not ref or not isinstance(ref, str) or "|" not in ref:
        return {"horizon": None, "predicted_at": None}
    horizon, _, predicted_at = ref.partition("|")
    return {
        "horizon": horizon.strip() or None,
        "predicted_at": predicted_at.strip() or None,
    }


def _summarize(trips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Win rate and P&L for one arm. None (not 0.0) on an empty arm."""
    n = len(trips)
    if n == 0:
        return {
            "n": 0, "win_rate": None, "avg_pnl_pct": None,
            "median_pnl_pct": None, "total_pnl_dollars": None,
        }
    pnl_pct = np.array(
        [t.get("pnl_pct") for t in trips if t.get("pnl_pct") is not None], dtype=float,
    )
    wins = [bool(t.get("win")) for t in trips if t.get("win") is not None]
    dollars = [
        float(t["pnl_dollars"]) for t in trips if t.get("pnl_dollars") is not None
    ]
    return {
        "n": n,
        "win_rate": round(float(np.mean(wins)), 4) if wins else None,
        "avg_pnl_pct": round(float(pnl_pct.mean()), 3) if len(pnl_pct) else None,
        "median_pnl_pct": round(float(np.median(pnl_pct)), 3) if len(pnl_pct) else None,
        "total_pnl_dollars": round(sum(dollars), 2) if dollars else None,
    }


def split_by_attribution(
    round_trips: List[Dict[str, Any]], fills: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split round trips into "followed" (entry fill carried a prediction_ref) and
    "discretionary".

    Attribution is keyed off the **entry** fill: a round trip is model-driven if
    the decision to open it was, regardless of how it was closed. Matching is by
    (ticker, contract_key, entry_time) against the fills ledger, because
    round_trips.py does not carry fill ids through.
    """
    refs = {}
    for f in fills or []:
        if not f.get("prediction_ref"):
            continue
        key = (
            (f.get("ticker") or "").upper(),
            float(f.get("strike") or 0),
            (f.get("option_type") or "").lower(),
            f.get("expiry_date"),
            f.get("filled_at"),
        )
        refs[key] = f["prediction_ref"]

    followed, discretionary = [], []
    for t in round_trips or []:
        key = (
            (t.get("ticker") or "").upper(),
            float(t.get("strike") or 0),
            (t.get("option_type") or "").lower(),
            t.get("expiry_date"),
            t.get("entry_time"),
        )
        ref = refs.get(key)
        if ref:
            enriched = {**t, "prediction_ref": ref, **parse_prediction_ref(ref)}
            followed.append(enriched)
        else:
            discretionary.append(t)
    return {"followed": followed, "discretionary": discretionary}


def compare_followed_vs_discretionary(
    round_trips: List[Dict[str, Any]], fills: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare model-driven round trips against discretionary ones.

    Returns
    -------
    {
      "n_total": int,
      "reportable": bool,
      "followed": {n, win_rate, avg_pnl_pct, median_pnl_pct, total_pnl_dollars},
      "discretionary": {...same...},
      "by_horizon": {horizon: {...same...}},
      "reason": str | None,
    }

    `reportable` is False — with `reason` explaining — until both arms clear their
    minimums. Callers must not render a comparison in that case; showing one would
    dress noise as a finding, which is exactly what this gate exists to prevent.
    """
    split = split_by_attribution(round_trips, fills)
    followed, discretionary = split["followed"], split["discretionary"]
    n_total = len(followed) + len(discretionary)

    out: Dict[str, Any] = {
        "n_total": n_total,
        "reportable": False,
        "followed": _summarize(followed),
        "discretionary": _summarize(discretionary),
        "by_horizon": {},
        "reason": None,
    }

    if n_total < MIN_ROUND_TRIPS_FOR_COMPARISON:
        out["reason"] = (
            f"{n_total} round trip(s) logged — need at least "
            f"{MIN_ROUND_TRIPS_FOR_COMPARISON} before this comparison means anything. "
            f"Keep tagging fills; the answer improves with the sample."
        )
        return out
    if out["followed"]["n"] < MIN_PER_ARM or out["discretionary"]["n"] < MIN_PER_ARM:
        out["reason"] = (
            f"Split is {out['followed']['n']} model-driven / "
            f"{out['discretionary']['n']} discretionary — each side needs at least "
            f"{MIN_PER_ARM}. A lopsided split clears the total but says nothing about "
            f"the smaller arm."
        )
        return out

    out["reportable"] = True
    by_horizon: Dict[str, List[Dict[str, Any]]] = {}
    for trip in followed:
        by_horizon.setdefault(trip.get("horizon") or "unknown", []).append(trip)
    # Per-horizon arms are thinner still, so they carry their own N and are
    # reported without a verdict attached.
    out["by_horizon"] = {h: _summarize(trips) for h, trips in sorted(by_horizon.items())}
    return out
