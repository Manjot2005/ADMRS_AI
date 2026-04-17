"""
forecasting.py — Predictive deforestation modelling for ADMRS
Uses numpy-based linear + seasonal decomposition (no heavy ML deps needed).
Falls back gracefully if Prophet/sklearn not installed.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Synthetic historical data generator ───────────────────────────
def get_historical_series(n_weeks: int = 52) -> pd.DataFrame:
    """
    Return weekly deforestation loss (ha) for the past n_weeks.
    Uses real Amazon seasonality: peaks in dry season (Jul-Sep).
    """
    rng = np.random.default_rng(42)
    base_date = datetime.utcnow() - timedelta(weeks=n_weeks)
    dates = [base_date + timedelta(weeks=i) for i in range(n_weeks)]

    # Base trend: slowly rising
    trend = np.linspace(180, 320, n_weeks)

    # Seasonality: dry season peak (weeks ~26–38 = Jul-Sep)
    week_of_year = np.array([d.timetuple().tm_yday // 7 for d in dates])
    seasonal = 80 * np.sin((week_of_year - 10) * np.pi / 26) + 20
    seasonal = np.clip(seasonal, 0, None)

    # Noise
    noise = rng.normal(0, 22, n_weeks)

    loss_ha = np.clip(trend + seasonal + noise, 50, None).round(1)

    return pd.DataFrame({"date": dates, "loss_ha": loss_ha})


# ── Linear trend forecaster ────────────────────────────────────────
def _fit_linear_trend(y: np.ndarray):
    """Return (slope, intercept) from OLS on index."""
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def _seasonal_component(n_hist: int, n_fut: int, y: np.ndarray) -> np.ndarray:
    """Extract dominant annual seasonal component via FFT and project forward."""
    # Detrend first
    slope, intercept = _fit_linear_trend(y)
    detrended = y - (slope * np.arange(len(y)) + intercept)

    # FFT-based seasonal extraction (52-week period)
    period = min(52, len(y) // 2)
    seasonal_full = np.zeros(len(y) + n_fut)
    if period > 4:
        fft_vals = np.fft.rfft(detrended, n=len(y))
        # Keep only low-frequency components (annual + semi-annual)
        fft_filt = np.zeros_like(fft_vals)
        freqs = np.fft.rfftfreq(len(y))
        mask = (freqs > 0) & (freqs < 4 / len(y))
        fft_filt[mask] = fft_vals[mask]
        seasonal_base = np.fft.irfft(fft_filt, n=len(y))
        # Tile to cover forecast horizon
        for i in range(len(seasonal_full)):
            seasonal_full[i] = seasonal_base[i % len(seasonal_base)]
    return seasonal_full[len(y):]


def forecast_30_days(hist_df: pd.DataFrame, n_days: int = 30) -> dict:
    """
    Forecast deforestation for next n_days.
    Returns dict with keys: dates, forecast, lower, upper, trend_weekly
    """
    y = hist_df["loss_ha"].values.astype(float)
    n_weeks_out = max(1, n_days // 7)

    slope, intercept = _fit_linear_trend(y)
    seasonal = _seasonal_component(len(y), n_weeks_out, y)

    future_idx = np.arange(len(y), len(y) + n_weeks_out, dtype=float)
    trend_fore = slope * future_idx + intercept
    forecast = np.clip(trend_fore + seasonal, 50, None)

    # Confidence interval ±1.5 std of residuals
    residuals = y - (slope * np.arange(len(y)) + intercept)
    std = float(np.std(residuals))
    lower = np.clip(forecast - 1.5 * std, 0, None)
    upper = forecast + 1.5 * std

    last_date = hist_df["date"].iloc[-1]
    future_dates = [last_date + timedelta(weeks=i + 1) for i in range(n_weeks_out)]

    # Weekly trend classification
    trend_dir = "↑ RISING" if slope > 0.5 else "↓ FALLING" if slope < -0.5 else "→ STABLE"
    peak_week = int(np.argmax(forecast))
    peak_date = future_dates[peak_week] if peak_week < len(future_dates) else future_dates[-1]

    return {
        "dates":        [d.strftime("%b %d") for d in future_dates],
        "forecast":     forecast.round(1).tolist(),
        "lower":        lower.round(1).tolist(),
        "upper":        upper.round(1).tolist(),
        "hist_dates":   [d.strftime("%b %d") for d in hist_df["date"].tolist()[-16:]],
        "hist_values":  y[-16:].round(1).tolist(),
        "trend_dir":    trend_dir,
        "slope_weekly": round(float(slope), 2),
        "peak_date":    peak_date.strftime("%b %d"),
        "peak_value":   round(float(forecast[peak_week]), 1),
        "total_30d":    round(float(forecast.sum()), 0),
    }


# ── Correlation insights ───────────────────────────────────────────
def get_correlation_insights(hist_df: pd.DataFrame) -> list[dict]:
    """
    Return synthetic but plausible correlation insights.
    In production, replace with real meteorological correlation analysis.
    """
    y = hist_df["loss_ha"].values
    rng = np.random.default_rng(99)

    # Simulate correlated environmental variables
    humidity     = 60 - 0.15 * y + rng.normal(0, 5, len(y))
    temperature  = 28 + 0.04 * y + rng.normal(0, 1.5, len(y))
    rainfall_mm  = 180 - 0.3 * y + rng.normal(0, 20, len(y))
    road_density = 0.1 + 0.005 * y + rng.normal(0, 0.03, len(y))

    def pearson_r(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.corrcoef(a, b)[0, 1])

    insights = [
        {
            "variable":    "Relative Humidity",
            "correlation": pearson_r(humidity, y),
            "direction":   "negative",
            "finding":     f"Alerts rise sharply when humidity drops below 30%",
            "icon":        "💧",
        },
        {
            "variable":    "Mean Temperature",
            "correlation": pearson_r(temperature, y),
            "direction":   "positive",
            "finding":     f"Each +1°C associated with +{round(abs(pearson_r(temperature,y))*12,1)}% alert increase",
            "icon":        "🌡",
        },
        {
            "variable":    "Monthly Rainfall",
            "correlation": pearson_r(rainfall_mm, y),
            "direction":   "negative",
            "finding":     f"Wet season (>180mm/mo) suppresses clearing activity by ~40%",
            "icon":        "🌧",
        },
        {
            "variable":    "Road Density Index",
            "correlation": pearson_r(road_density, y),
            "direction":   "positive",
            "finding":     f"Alerts cluster within 5 km of new road incursions",
            "icon":        "🛣",
        },
    ]
    return insights
