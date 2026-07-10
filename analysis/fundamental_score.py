"""
Fundamental quality scoring — translates raw financial data into actionable scores.
Follows the Financial Analyst's framework: Quality, Value, Growth, Red Flags.
"""
import logging
from typing import Dict, Any, Optional, List
from data.fundamentals import get_financials

logger = logging.getLogger(__name__)


def score_quality(f: dict) -> Dict[str, Any]:
    """
    Quality Score (0-100):
    - ROIC proxy (30 pts)
    - Gross margin stability/expansion (25 pts)
    - FCF conversion (25 pts)
    - Balance sheet strength (20 pts)
    """
    score = 0
    breakdown = {}

    # ROIC proxy (higher = better)
    roic = f.get("roic_proxy") or f.get("return_on_equity")
    if roic is not None:
        roic_pct = roic * 100 if roic < 1 else roic
        if roic_pct > 20:
            pts = 30
        elif roic_pct > 12:
            pts = 22
        elif roic_pct > 6:
            pts = 14
        elif roic_pct > 0:
            pts = 7
        else:
            pts = 0
        score += pts
        breakdown["roic"] = {"value": round(roic_pct, 1), "pts": pts, "max": 30}
    else:
        breakdown["roic"] = {"value": None, "pts": 0, "max": 30}

    # Gross margin stability/expansion
    gm_curr = f.get("gross_margin_curr") or f.get("gross_margin")
    gm_prev = f.get("gross_margin_prev")
    if gm_curr is not None:
        gm_pct = gm_curr * 100 if gm_curr < 1 else gm_curr
        base_pts = min(20, max(0, int(gm_pct / 5)))
        expansion = (gm_curr - gm_prev > 0.002) if gm_prev else None
        pts = base_pts + (5 if expansion else 0)
        pts = min(pts, 25)
        score += pts
        breakdown["gross_margin"] = {"value": round(gm_pct, 1), "expanding": expansion, "pts": pts, "max": 25}
    else:
        breakdown["gross_margin"] = {"value": None, "pts": 0, "max": 25}

    # FCF yield / conversion
    fcf_yield = f.get("fcf_yield")
    if fcf_yield is not None:
        fy_pct = fcf_yield * 100 if fcf_yield < 1 else fcf_yield
        if fy_pct > 6:
            pts = 25
        elif fy_pct > 3:
            pts = 18
        elif fy_pct > 0:
            pts = 10
        else:
            pts = 0
        score += pts
        breakdown["fcf_yield"] = {"value": round(fy_pct, 2), "pts": pts, "max": 25}
    else:
        breakdown["fcf_yield"] = {"value": None, "pts": 0, "max": 25}

    # Balance sheet (net debt / EBITDA proxy via debt_to_equity)
    dte = f.get("debt_to_equity")
    if dte is not None:
        if dte < 30:
            pts = 20
        elif dte < 80:
            pts = 14
        elif dte < 150:
            pts = 8
        elif dte < 300:
            pts = 3
        else:
            pts = 0
        score += pts
        breakdown["balance_sheet"] = {"debt_to_equity": round(dte, 1), "pts": pts, "max": 20}
    else:
        breakdown["balance_sheet"] = {"value": None, "pts": 0, "max": 20}

    return {"score": min(100, score), "breakdown": breakdown}


def score_value(f: dict, sector: str = "") -> Dict[str, Any]:
    """
    Value Score (0-100):
    - EV/EBITDA vs typical sector range (40 pts)
    - P/FCF (30 pts)
    - PEG ratio (30 pts)
    """
    score = 0
    breakdown = {}

    # EV/EBITDA
    ev_ebitda = f.get("ev_ebitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 8:
            pts = 40
        elif ev_ebitda < 15:
            pts = 28
        elif ev_ebitda < 25:
            pts = 18
        elif ev_ebitda < 40:
            pts = 8
        else:
            pts = 0
        score += pts
        breakdown["ev_ebitda"] = {"value": round(ev_ebitda, 1), "pts": pts, "max": 40}
    else:
        breakdown["ev_ebitda"] = {"value": None, "pts": 0, "max": 40}

    # P/FCF
    pfcf = f.get("price_to_fcf")
    if pfcf is not None and pfcf > 0:
        if pfcf < 12:
            pts = 30
        elif pfcf < 20:
            pts = 22
        elif pfcf < 30:
            pts = 14
        elif pfcf < 50:
            pts = 6
        else:
            pts = 0
        score += pts
        breakdown["price_to_fcf"] = {"value": round(pfcf, 1), "pts": pts, "max": 30}
    else:
        breakdown["price_to_fcf"] = {"value": None, "pts": 0, "max": 30}

    # PEG ratio
    peg = f.get("peg_ratio")
    if peg is not None and peg > 0:
        if peg < 1.0:
            pts = 30
        elif peg < 1.5:
            pts = 22
        elif peg < 2.5:
            pts = 12
        elif peg < 4.0:
            pts = 5
        else:
            pts = 0
        score += pts
        breakdown["peg_ratio"] = {"value": round(peg, 2), "pts": pts, "max": 30}
    else:
        breakdown["peg_ratio"] = {"value": None, "pts": 0, "max": 30}

    return {"score": min(100, score), "breakdown": breakdown}


def score_growth(f: dict) -> Dict[str, Any]:
    """
    Growth Score (0-100):
    - Revenue growth (50 pts)
    - Earnings growth (50 pts)
    """
    score = 0
    breakdown = {}

    rev_growth = f.get("revenue_growth_yoy") or f.get("revenue_growth")
    if rev_growth is not None:
        rg = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth
        if rg > 25:
            pts = 50
        elif rg > 15:
            pts = 38
        elif rg > 8:
            pts = 25
        elif rg > 0:
            pts = 12
        else:
            pts = 0
        score += pts
        breakdown["revenue_growth"] = {"value": round(rg, 1), "pts": pts, "max": 50}
    else:
        breakdown["revenue_growth"] = {"value": None, "pts": 0, "max": 50}

    eps_growth = f.get("earnings_growth")
    if eps_growth is not None:
        eg = eps_growth * 100 if abs(eps_growth) < 1 else eps_growth
        if eg > 25:
            pts = 50
        elif eg > 15:
            pts = 38
        elif eg > 5:
            pts = 22
        elif eg > 0:
            pts = 10
        else:
            pts = 0
        score += pts
        breakdown["eps_growth"] = {"value": round(eg, 1), "pts": pts, "max": 50}
    else:
        breakdown["eps_growth"] = {"value": None, "pts": 0, "max": 50}

    return {"score": min(100, score), "breakdown": breakdown}


def detect_red_flags(f: dict) -> List[Dict[str, str]]:
    """
    Automated red flag scanner.
    Returns list of {flag, severity, detail} dicts.
    Severity: "warning" | "danger"
    """
    flags = []

    # DSO trend — rising DSO = potentially aggressive revenue recognition
    dso_trend = f.get("dso_trend")
    if dso_trend is not None and dso_trend > 10:
        flags.append({
            "flag": "Rising Days Sales Outstanding",
            "severity": "warning" if dso_trend < 25 else "danger",
            "detail": f"DSO has increased by {dso_trend:.0f} days YoY — customers taking longer to pay",
        })

    # Gross margin compression
    gm_curr = f.get("gross_margin_curr") or f.get("gross_margin")
    gm_prev = f.get("gross_margin_prev")
    if gm_curr and gm_prev and (gm_curr - gm_prev) < -0.03:
        delta_bps = (gm_curr - gm_prev) * 10000
        flags.append({
            "flag": "Gross Margin Compression",
            "severity": "danger" if delta_bps < -300 else "warning",
            "detail": f"Gross margin declined {abs(delta_bps):.0f}bps YoY — potential pricing power erosion",
        })

    # High leverage
    dte = f.get("debt_to_equity")
    if dte is not None and dte > 200:
        flags.append({
            "flag": "High Financial Leverage",
            "severity": "danger" if dte > 400 else "warning",
            "detail": f"Debt/Equity ratio is {dte:.0f}% — elevated financial risk",
        })

    # Negative FCF
    fcf = f.get("fcf")
    if fcf is not None and fcf < 0:
        flags.append({
            "flag": "Negative Free Cash Flow",
            "severity": "warning",
            "detail": f"FCF is ${fcf/1e6:.0f}M — company is consuming cash",
        })

    # Revenue growth deceleration
    rev_growth = f.get("revenue_growth_yoy") or f.get("revenue_growth")
    if rev_growth is not None:
        rg = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth
        if rg < -5:
            flags.append({
                "flag": "Revenue Contraction",
                "severity": "danger",
                "detail": f"Revenue declining at {abs(rg):.1f}% YoY",
            })

    # Negative ROE
    roe = f.get("return_on_equity")
    if roe is not None and roe < -0.05:
        flags.append({
            "flag": "Negative Return on Equity",
            "severity": "warning",
            "detail": f"ROE is {roe*100:.1f}% — company destroying shareholder value",
        })

    # Very high short ratio (potential stress signal)
    short_ratio = f.get("short_ratio")
    if short_ratio is not None and short_ratio > 10:
        flags.append({
            "flag": "Elevated Short Interest",
            "severity": "warning",
            "detail": f"Short ratio of {short_ratio:.1f} days — significant bearish conviction from market",
        })

    return flags


def full_fundamental_report(ticker: str) -> Dict[str, Any]:
    """
    Full fundamental analysis for a ticker.
    Returns quality, value, growth scores + red flags + raw metrics.
    """
    f = get_financials(ticker)
    if not f:
        return {"error": "Could not fetch fundamental data"}

    quality = score_quality(f)
    value = score_value(f, f.get("sector", ""))
    growth = score_growth(f)
    flags = detect_red_flags(f)

    composite = round(
        quality["score"] * 0.40 + value["score"] * 0.30 + growth["score"] * 0.30, 1
    )

    verdict = "Strong" if composite > 70 else ("Solid" if composite > 50 else ("Mixed" if composite > 35 else "Weak"))

    return {
        "ticker": ticker,
        "name": f.get("name", ticker),
        "sector": f.get("sector", "Unknown"),
        "quality": quality,
        "value": value,
        "growth": growth,
        "composite_score": composite,
        "verdict": verdict,
        "red_flags": flags,
        "raw": f,
    }
