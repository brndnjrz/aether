"""
Fundamental data — pulls financials from yfinance and computes derived ratios.
All numbers are validated; missing data returns None rather than crashing.
"""
import time
import logging
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
_cache: Dict[str, Dict] = {}


def _fresh(entry: dict, ttl: int) -> bool:
    return (time.time() - entry["ts"]) < ttl


def _safe(val, default=None):
    """Return default if val is None, NaN, or non-finite."""
    try:
        if val is None:
            return default
        f = float(val)
        import math
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def get_financials(ticker: str, ttl: int = 3600) -> Dict[str, Any]:
    """
    Returns a flat dict of fundamental metrics ready for scoring.
    Keys: revenue_growth, gross_margin, operating_margin, fcf_yield,
          roic, net_debt_ebitda, eps_growth, dso_trend, sbc_pct_revenue,
          current_ratio, pe_ratio, forward_pe, ev_ebitda, peg_ratio,
          market_cap, enterprise_value, dividend_yield, beta
    """
    key = f"fundamentals_{ticker}"
    if key in _cache and _fresh(_cache[key], ttl):
        logger.debug(f"get_financials cache hit for {ticker}")
        return _cache[key]["data"]

    try:
        t = yf.Ticker(ticker)
        info = t.info

        income = t.financials          # annual P&L, columns = dates
        balance = t.balance_sheet      # annual balance sheet
        cashflow = t.cashflow          # annual cash flow

        result: Dict[str, Any] = {}

        # ── Info-derived metrics ──────────────────────────────────────────
        result["market_cap"] = _safe(info.get("marketCap"))
        result["enterprise_value"] = _safe(info.get("enterpriseValue"))
        result["pe_ratio"] = _safe(info.get("trailingPE"))
        result["forward_pe"] = _safe(info.get("forwardPE"))
        result["peg_ratio"] = _safe(info.get("trailingPegRatio"))
        result["ev_ebitda"] = _safe(info.get("enterpriseToEbitda"))
        result["ev_revenue"] = _safe(info.get("enterpriseToRevenue"))
        result["price_to_book"] = _safe(info.get("priceToBook"))
        result["price_to_fcf"] = _safe(info.get("priceToFreeCashflow"))
        result["dividend_yield"] = _safe(info.get("dividendYield"))
        result["beta"] = _safe(info.get("beta"))
        result["gross_margin"] = _safe(info.get("grossMargins"))
        result["operating_margin"] = _safe(info.get("operatingMargins"))
        result["profit_margin"] = _safe(info.get("profitMargins"))
        result["revenue_growth"] = _safe(info.get("revenueGrowth"))
        result["earnings_growth"] = _safe(info.get("earningsGrowth"))
        result["return_on_equity"] = _safe(info.get("returnOnEquity"))
        result["return_on_assets"] = _safe(info.get("returnOnAssets"))
        result["current_ratio"] = _safe(info.get("currentRatio"))
        result["debt_to_equity"] = _safe(info.get("debtToEquity"))
        result["short_ratio"] = _safe(info.get("shortRatio"))
        result["shares_outstanding"] = _safe(info.get("sharesOutstanding"))
        result["float_shares"] = _safe(info.get("floatShares"))
        result["52w_high"] = _safe(info.get("fiftyTwoWeekHigh"))
        result["52w_low"] = _safe(info.get("fiftyTwoWeekLow"))
        result["name"] = info.get("longName", ticker)
        result["sector"] = info.get("sector", "Unknown")
        result["industry"] = info.get("industry", "Unknown")
        result["description"] = info.get("longBusinessSummary", "")

        # ── Computed from financial statements ─────────────────────────
        try:
            if income is not None and not income.empty and len(income.columns) >= 2:
                rev_curr = _safe(income.loc["Total Revenue", income.columns[0]]) if "Total Revenue" in income.index else None
                rev_prev = _safe(income.loc["Total Revenue", income.columns[1]]) if "Total Revenue" in income.index else None
                if rev_curr and rev_prev and rev_prev != 0:
                    result["revenue_growth_yoy"] = (rev_curr - rev_prev) / abs(rev_prev)
                else:
                    result["revenue_growth_yoy"] = result.get("revenue_growth")

                gp_curr = _safe(income.loc["Gross Profit", income.columns[0]]) if "Gross Profit" in income.index else None
                gp_prev = _safe(income.loc["Gross Profit", income.columns[1]]) if "Gross Profit" in income.index else None
                if gp_curr and rev_curr and rev_curr != 0:
                    result["gross_margin_curr"] = gp_curr / rev_curr
                if gp_prev and rev_prev and rev_prev != 0:
                    result["gross_margin_prev"] = gp_prev / rev_prev

                ebit = _safe(income.loc["EBIT", income.columns[0]]) if "EBIT" in income.index else None
                result["ebit"] = ebit
        except Exception as e:
            logger.debug(f"Income statement parsing: {e}")

        # ── FCF Yield ───────────────────────────────────────────────────
        try:
            if cashflow is not None and not cashflow.empty:
                ocf = _safe(cashflow.loc["Operating Cash Flow", cashflow.columns[0]]) if "Operating Cash Flow" in cashflow.index else None
                capex = _safe(cashflow.loc["Capital Expenditure", cashflow.columns[0]]) if "Capital Expenditure" in cashflow.index else None
                if ocf is not None and capex is not None:
                    fcf = ocf + capex  # capex is typically negative in yfinance
                    result["fcf"] = fcf
                    if result["market_cap"] and result["market_cap"] > 0:
                        result["fcf_yield"] = fcf / result["market_cap"]
                    if result["enterprise_value"] and result["enterprise_value"] > 0:
                        result["fcf_ev_yield"] = fcf / result["enterprise_value"]

                # SBC as % of revenue
                sbc = _safe(cashflow.loc["Stock Based Compensation", cashflow.columns[0]]) if "Stock Based Compensation" in cashflow.index else None
                rev_curr = result.get("ebit")  # reuse if set earlier
                if sbc is not None:
                    result["sbc"] = sbc
        except Exception as e:
            logger.debug(f"Cashflow parsing: {e}")

        # ── Net Debt / EBITDA ───────────────────────────────────────────
        try:
            if balance is not None and not balance.empty:
                cash = _safe(balance.loc["Cash And Cash Equivalents", balance.columns[0]]) if "Cash And Cash Equivalents" in balance.index else None
                total_debt = _safe(balance.loc["Total Debt", balance.columns[0]]) if "Total Debt" in balance.index else None
                if cash is not None and total_debt is not None:
                    result["net_debt"] = total_debt - cash

                # DSO (Days Sales Outstanding) — compare 2 years
                receivables_curr = _safe(balance.loc["Net Receivables", balance.columns[0]]) if "Net Receivables" in balance.index else None
                receivables_prev = _safe(balance.loc["Net Receivables", balance.columns[1]]) if ("Net Receivables" in balance.index and len(balance.columns) > 1) else None
                rev_curr_stmt = None
                if "revenue_growth_yoy" in result:
                    pass  # already have income
                try:
                    if income is not None and not income.empty:
                        rc = _safe(income.loc["Total Revenue", income.columns[0]]) if "Total Revenue" in income.index else None
                        if rc and receivables_curr:
                            result["dso"] = (receivables_curr / rc) * 365
                        if rc and receivables_prev and len(income.columns) > 1:
                            rc_prev = _safe(income.loc["Total Revenue", income.columns[1]])
                            if rc_prev:
                                dso_prev = (receivables_prev / rc_prev) * 365
                                dso_curr = result.get("dso")
                                if dso_curr and dso_prev:
                                    result["dso_trend"] = dso_curr - dso_prev  # positive = worsening
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Balance sheet parsing: {e}")

        # ── ROIC (approx) ───────────────────────────────────────────────
        try:
            roe = result.get("return_on_equity")
            dte = result.get("debt_to_equity")
            if roe is not None and dte is not None and dte >= 0:
                # Simplified ROIC proxy: ROE / (1 + D/E)
                result["roic_proxy"] = roe / (1 + dte / 100) if dte > 0 else roe
        except Exception:
            pass

        # ── EV/EBITDA fallback ─────────────────────────────────────────
        if result.get("ev_ebitda") is None:
            ev = result.get("enterprise_value")
            ebit = result.get("ebit")
            if ev and ebit and ebit > 0:
                result["ev_ebitda"] = ev / ebit  # rough proxy with EBIT

        _cache[key] = {"data": result, "ts": time.time()}
        logger.info(f"Fetched fundamentals for {ticker}: {len(result)} fields populated")
        return result

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {}


def get_earnings_history(ticker: str, ttl: int = 3600) -> pd.DataFrame:
    """Returns earnings history with surprise data if available."""
    key = f"earnings_{ticker}"
    if key in _cache and _fresh(_cache[key], ttl):
        logger.debug(f"get_earnings_history cache hit for {ticker}")
        return _cache[key]["data"]
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            logger.warning(f"No earnings history available for {ticker}")
            hist = pd.DataFrame()
        else:
            logger.debug(f"Fetched earnings history for {ticker}: {len(hist)} rows")
        _cache[key] = {"data": hist, "ts": time.time()}
        return hist
    except Exception as e:
        logger.debug(f"Earnings history for {ticker}: {e}")
        return pd.DataFrame()


def get_analyst_recommendations(ticker: str, ttl: int = 3600) -> pd.DataFrame:
    key = f"recs_{ticker}"
    if key in _cache and _fresh(_cache[key], ttl):
        logger.debug(f"get_analyst_recommendations cache hit for {ticker}")
        return _cache[key]["data"]
    try:
        t = yf.Ticker(ticker)
        recs = t.recommendations
        if recs is None or recs.empty:
            logger.warning(f"No analyst recommendations available for {ticker}")
            recs = pd.DataFrame()
        else:
            logger.debug(f"Fetched analyst recommendations for {ticker}: {len(recs)} rows")
        _cache[key] = {"data": recs, "ts": time.time()}
        return recs
    except Exception as e:
        logger.debug(f"Analyst recommendations for {ticker}: {e}")
        return pd.DataFrame()
