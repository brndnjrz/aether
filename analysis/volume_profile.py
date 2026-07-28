"""
Volume-by-price profile — bins traded volume into price buckets over a
lookback window so a "point of control" / high-volume node / value area
can be read off bars alone, with no order-book or tick data. Used by
analysis/mtf_strategy.py as the demand-zone and VAP-target proxy.
"""
from typing import Any, Dict, List, Optional

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 60
DEFAULT_NUM_BINS = 24
VALUE_AREA_PCT = 0.70


def build_volume_profile(
    df: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    num_bins: int = DEFAULT_NUM_BINS,
) -> Optional[Dict[str, Any]]:
    """
    Bin each bar's typical price ((High+Low+Close)/3) into `num_bins` equal
    price buckets spanning the last `lookback` bars, summing Volume per
    bucket. Returns None if there isn't enough range/history to bin.

    Returns {"bin_edges", "bin_centers", "volume_by_bin", "poc_price",
    "value_area_high", "value_area_low", "hvns": List[float]}.
    """
    if df is None or len(df) < 5:
        logger.debug("build_volume_profile: not enough history to build a profile.")
        return None

    window = df.tail(lookback)
    lo, hi = float(window["Low"].min()), float(window["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        logger.debug(f"build_volume_profile: no usable price range (lo={lo}, hi={hi}).")
        return None

    typical_price = (window["High"] + window["Low"] + window["Close"]) / 3
    bin_edges = np.linspace(lo, hi, num_bins + 1)
    bin_idx = np.clip(np.digitize(typical_price.to_numpy(), bin_edges) - 1, 0, num_bins - 1)
    volume_by_bin = np.bincount(bin_idx, weights=window["Volume"].to_numpy(), minlength=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_volume = float(volume_by_bin.sum())
    if total_volume <= 0:
        logger.debug("build_volume_profile: total volume in window is zero.")
        return None

    poc_idx = int(np.argmax(volume_by_bin))

    # Value area: expand outward from the POC, each step adding whichever
    # adjacent bin (below or above the current included range) carries more
    # volume, until the included volume reaches VALUE_AREA_PCT of the total.
    lo_idx, hi_idx = poc_idx, poc_idx
    included = float(volume_by_bin[poc_idx])
    target = total_volume * VALUE_AREA_PCT
    while included < target and (lo_idx > 0 or hi_idx < num_bins - 1):
        vol_below = volume_by_bin[lo_idx - 1] if lo_idx > 0 else -1.0
        vol_above = volume_by_bin[hi_idx + 1] if hi_idx < num_bins - 1 else -1.0
        if vol_above >= vol_below:
            hi_idx += 1
            included += volume_by_bin[hi_idx]
        else:
            lo_idx -= 1
            included += volume_by_bin[lo_idx]

    # High-volume nodes: bins that are local maxima and carry above-average volume.
    mean_vol = total_volume / num_bins
    hvns: List[float] = []
    for i in range(num_bins):
        left = volume_by_bin[i - 1] if i > 0 else -1.0
        right = volume_by_bin[i + 1] if i < num_bins - 1 else -1.0
        if volume_by_bin[i] > mean_vol and volume_by_bin[i] >= left and volume_by_bin[i] >= right:
            hvns.append(float(bin_centers[i]))

    logger.debug(f"build_volume_profile: poc_price={bin_centers[poc_idx]:.4f} hvns_found={len(hvns)}")
    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "volume_by_bin": volume_by_bin,
        "poc_price": float(bin_centers[poc_idx]),
        "value_area_high": float(bin_centers[hi_idx]),
        "value_area_low": float(bin_centers[lo_idx]),
        "hvns": sorted(hvns),
    }


def nearest_hvn_above(profile: Optional[Dict[str, Any]], price: float) -> Optional[float]:
    """First high-volume-node price >= `price` — the VAP target candidate."""
    if not profile:
        return None
    candidates = [p for p in profile["hvns"] if p >= price]
    return min(candidates) if candidates else None


def nearest_hvn_below(profile: Optional[Dict[str, Any]], price: float) -> Optional[float]:
    """Highest high-volume-node price <= `price` — helps validate a demand zone."""
    if not profile:
        return None
    candidates = [p for p in profile["hvns"] if p <= price]
    return max(candidates) if candidates else None
