"""
GARCH conditional-volatility forecasting.
Fits on historical returns, forecasts forward — kept separate from
options_pricing.py, which is closed-form math only (no fitting, no lookahead).
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

MIN_OBS = 250  # ~1y of daily returns needed for a stable GARCH(1,1) fit


def garch_forecast_vol(returns: pd.Series, horizon_days: int) -> Dict:
    """
    Fit GARCH(1,1) on daily returns and forecast annualized volatility
    over horizon_days. Returns a forward vol curve plus a single
    horizon-averaged scalar comparable to an option's implied volatility
    over that same horizon.
    """
    clean = returns.dropna() * 100  # arch_model expects returns scaled to percent
    if len(clean) < MIN_OBS:
        logger.warning(f"garch_forecast_vol: insufficient return history ({len(clean)} obs, need {MIN_OBS}) — skipping fit.")
        return {"status": "insufficient_data"}

    horizon_days = max(int(horizon_days), 1)

    try:
        model = arch_model(clean, vol="Garch", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")

        forecast = fit.forecast(horizon=horizon_days, reindex=False)
        daily_variance = forecast.variance.values[-1]  # shape (horizon_days,), in %^2
        daily_vol_pct = np.sqrt(daily_variance)         # daily vol, in %

        annualized_path = daily_vol_pct * np.sqrt(252)   # annualized vol, in %
        horizon_avg_vol = float(np.mean(annualized_path))

        alpha = float(fit.params["alpha[1]"])
        beta = float(fit.params["beta[1]"])

        logger.info(
            f"garch_forecast_vol: horizon_days={horizon_days} garch_vol_horizon={round(horizon_avg_vol, 2)} "
            f"persistence={round(alpha + beta, 4)}"
        )
        return {
            "status": "ok",
            "garch_vol_path": annualized_path.tolist(),
            "garch_vol_horizon": round(horizon_avg_vol, 2),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "persistence": round(alpha + beta, 4),
        }
    except Exception as e:
        logger.error(f"garch_forecast_vol: GARCH fit/forecast failed: {e}")
        return {"status": "error", "error": str(e)}
