#!/usr/bin/env python3
"""
A/B harness for the uniqueness-weighting experiment. NOT imported by the app.

    python3 scripts/ab_uniqueness_weights.py --tickers SPY,QQQ,AAPL
    python3 scripts/ab_uniqueness_weights.py --synthetic     # no network needed

Question
-------
Row t's label is the forward return over bars (t, t+horizon], so consecutive rows
share horizon-1 bars of outcome. Because _filter_directional drops neutral rows
before training, the retained rows are unevenly spaced: a row inside a volatile
cluster overlaps many other retained rows, while an isolated row's outcome is
entirely its own. Weighting by average uniqueness (Lopez de Prado, AFML ch. 4)
should stop volatile stretches carrying more of the fit than they earn.

Should. Whether it does is an empirical question, and the honest answer so far is
no. Measured across eight synthetic series -- identical split, identical
configuration, only the fit weights differing -- the mean holdout delta was -0.5
points, the median 0.0, with 3 wins in 8 against a 2.5-point run-to-run spread.
The weights were live (0.82 to 3.7), so the mechanism was doing something; it just
was not doing anything useful.

That result is not decisive. Synthetic random walks contain no signal, and no
improvement to *how* a model fits can recover signal that does not exist. This
script exists so the same comparison can be made on real tickers, where it might.

Reading the output
------------------
Judge the mean delta against the spread, not against zero. A +1-point mean with a
2.5-point sd across 8 series is noise. Adding tickers is the only way to tighten
it -- and one ticker's result is worth nothing at all, which is why --tickers
takes a list.

Nothing is written: this calls _evaluate_on_holdout directly and persists no
models, no metadata, and no predictions.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_uniqueness")

SYNTHETIC_SEEDS = (3, 7, 11, 13, 21, 33, 41, 55)


def _synthetic(seed: int, n: int = 620, drift: float = 0.0004, vol: float = 0.012):
    from analysis.indicators import calculate_indicators
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-06-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return calculate_indicators(pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.008, "Low": close * 0.992,
        "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx))


def _real(ticker: str):
    from data.price_data import get_price_history
    from analysis.indicators import calculate_indicators
    raw = get_price_history(ticker, period="2y")
    if raw is None or raw.empty:
        return None
    return calculate_indicators(raw)


def _compare(label: str, df, horizon: int, threshold: float):
    """One paired comparison. Everything is held fixed except the fit weights."""
    import analysis.ml_prediction as ml

    if df is None or len(df) < 400:
        return {"label": label, "skip": "under 400 bars"}

    xgb_cfg = ml._xgb_config(scale_pos_weight=1.0)
    rf_cfg = ml._rf_config(n_features=len(ml.FEATURE_NAMES))
    weights = {"xgb": 0.65, "rf": 0.35}
    search_end = df.index[int(len(df) * (1 - ml.HOLDOUT_FRACTION))]

    plain = ml._evaluate_on_holdout(
        df, label, horizon, threshold, xgb_cfg, rf_cfg, weights,
        search_end, uniqueness_weights=False,
    )
    weighted = ml._evaluate_on_holdout(
        df, label, horizon, threshold, xgb_cfg, rf_cfg, weights,
        search_end, uniqueness_weights=True,
    )
    if plain["accuracy"] is None or weighted["accuracy"] is None:
        return {"label": label, "skip": plain["reason"] or weighted["reason"]}

    try:
        X, y = ml.build_features(df, ticker=label, forward_bars=horizon,
                                 neutral_threshold=threshold)
        X_dir, y_dir = ml._filter_directional(X, y)
        train_mask = X_dir.index <= search_end
        w = ml._average_uniqueness(X_dir.index[train_mask], df.index, horizon)
        w_min, w_max = float(w.min()), float(w.max())
    except Exception:
        w_min = w_max = float("nan")

    return {
        "label": label, "skip": None, "n": plain["n"],
        "plain": plain["accuracy"], "weighted": weighted["accuracy"],
        "delta": weighted["accuracy"] - plain["accuracy"],
        "w_min": w_min, "w_max": w_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated real tickers, e.g. SPY,QQQ,AAPL")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random walks instead (no network).")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()

    if not args.tickers and not args.synthetic:
        parser.error("pass --tickers SPY,QQQ or --synthetic")

    cases = []
    if args.synthetic:
        cases += [(f"RW{s}", _synthetic(s)) for s in SYNTHETIC_SEEDS]
    if args.tickers:
        for t in (x.strip().upper() for x in args.tickers.split(",") if x.strip()):
            df = _real(t)
            if df is None:
                print(f"{t}: no price data, skipping")
                continue
            cases.append((t, df))

    if not cases:
        print("Nothing to compare.")
        return 1

    print(f"\nhorizon={args.horizon} bars, neutral band={args.threshold:.3%}, "
          f"paired on identical splits\n")
    print(f"{'case':>8}{'n':>6}{'plain':>9}{'weighted':>10}{'delta':>9}{'w.min':>8}{'w.max':>8}")
    print("-" * 58)

    deltas = []
    for label, df in cases:
        r = _compare(label, df, args.horizon, args.threshold)
        if r.get("skip"):
            print(f"{r['label']:>8}  skipped: {r['skip']}")
            continue
        deltas.append(r["delta"])
        print(f"{r['label']:>8}{r['n']:>6}{r['plain']:>9.4f}{r['weighted']:>10.4f}"
              f"{r['delta']:>+9.4f}{r['w_min']:>8.2f}{r['w_max']:>8.2f}")

    if not deltas:
        print("\nNo usable comparisons.")
        return 1

    d = np.array(deltas)
    print("-" * 58)
    print(f"mean {d.mean():+.4f}   median {np.median(d):+.4f}   "
          f"wins {int((d > 0).sum())}/{len(d)}   sd {d.std():.4f}")
    print()
    if len(d) < 5:
        print("Fewer than 5 cases — not enough to conclude anything either way.")
    elif abs(d.mean()) < d.std() / np.sqrt(len(d)) * 1.96:
        print("Mean delta is inside its own 95% interval: no measurable effect.")
        print("Leave uniqueness_weights=False.")
    elif d.mean() > 0:
        print("Mean delta is positive beyond noise. Worth enabling — pass")
        print("uniqueness_weights=True to train_model and re-check on new tickers.")
    else:
        print("Mean delta is negative beyond noise. Keep uniqueness_weights=False.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
