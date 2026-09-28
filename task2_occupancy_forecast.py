"""
UC2 – SDI Stress-Test Engine
Task 2: Occupancy Rate Forecasting Model
Fits SARIMA per building type, outputs 15-year forecasts with confidence intervals.
"""

import warnings
import numpy as np
import pandas as pd
import json
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

DATA_PATH   = "data/mef_building_stock.csv"
OUTPUT_DIR  = "data"
HORIZON     = 15          # forecast years
FUTURE_YRS  = list(range(2025, 2025 + HORIZON))
BUILDING_TYPES = ["ministry_hq", "regional_office", "annex"]


# ── 1. Load & prepare data ────────────────────────────────────────────────────

def load_and_aggregate(path: str) -> dict[str, pd.Series]:
    """
    Aggregate occupancy by building type × year.
    Returns mean occupancy rate per type as annual time series.
    """
    df = pd.read_csv(path)

    series_by_type = {}
    for btype in BUILDING_TYPES:
        sub = (
            df[df["building_type"] == btype]
            .groupby("year")["occupancy_rate"]
            .mean()
        )
        sub.index = pd.PeriodIndex(sub.index, freq="Y")
        series_by_type[btype] = sub

    return series_by_type


# ── 2. Stationarity check ─────────────────────────────────────────────────────

def check_stationarity(series: pd.Series, name: str) -> int:
    """ADF test — returns recommended differencing order d."""
    result = adfuller(series.values, autolag="AIC")
    pval   = result[1]
    d      = 0 if pval < 0.05 else 1
    print(f"  ADF p-value [{name}]: {pval:.4f}  →  d={d} "
          f"({'stationary' if d == 0 else 'needs differencing'})")
    return d


# ── 3. Auto-select SARIMA order ───────────────────────────────────────────────

def find_best_order(series: pd.Series, d: int) -> tuple:
    """
    Use auto_arima to find best (p,d,q) — no seasonal component
    because we have annual data (s=1 makes no sense).
    Returns (p, d, q).
    """
    model = auto_arima(
        series.values,
        d=d,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        seasonal=False,
        information_criterion="aic",
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
    )
    p, _, q = model.order
    return p, d, q


# ── 4. Fit SARIMA & forecast ──────────────────────────────────────────────────

def fit_and_forecast(
    series: pd.Series,
    order: tuple,
    horizon: int,
    btype: str,
) -> pd.DataFrame:
    """
    Fit SARIMAX with chosen (p,d,q), produce horizon-step forecast.
    Returns DataFrame with columns: year, mean, lower_80, upper_80, lower_95, upper_95.
    """
    model  = SARIMAX(series.values, order=order, trend="c",
                     enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=200)

    forecast   = result.get_forecast(steps=horizon)
    mean_fc    = forecast.predicted_mean
    conf_80    = forecast.conf_int(alpha=0.20)   # 80% CI
    conf_95    = forecast.conf_int(alpha=0.05)   # 95% CI

    df_fc = pd.DataFrame({
        "year":      FUTURE_YRS,
        "mean":      np.clip(mean_fc, 0.15, 0.98),
        "lower_80":  np.clip(conf_80[:, 0], 0.10, 0.98),
        "upper_80":  np.clip(conf_80[:, 1], 0.10, 0.98),
        "lower_95":  np.clip(conf_95[:, 0], 0.05, 0.99),
        "upper_95":  np.clip(conf_95[:, 1], 0.05, 0.99),
        "building_type": btype,
    })

    aic  = round(result.aic, 2)
    bic  = round(result.bic, 2)
    return df_fc, result, aic, bic


# ── 5. In-sample evaluation ───────────────────────────────────────────────────

def evaluate_insample(series: pd.Series, result) -> dict:
    """MAE and RMSE on the fitted (in-sample) values."""
    fitted = result.fittedvalues
    resid  = series.values - fitted
    mae    = float(np.mean(np.abs(resid)))
    rmse   = float(np.sqrt(np.mean(resid ** 2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


# ── 6. Main pipeline ──────────────────────────────────────────────────────────

def run_occupancy_forecasting() -> pd.DataFrame:
    print("=" * 56)
    print("  Task 2 — Occupancy Rate Forecasting")
    print("=" * 56)

    series_dict = load_and_aggregate(DATA_PATH)
    all_forecasts = []
    model_summary = {}

    for btype in BUILDING_TYPES:
        print(f"\n── {btype.replace('_', ' ').upper()} ──")
        series = series_dict[btype]
        print(f"  Historical mean: {series.mean():.3f}  "
              f"std: {series.std():.3f}  "
              f"n={len(series)}")

        # stationarity
        d = check_stationarity(series, btype)

        # order selection
        p, d, q = find_best_order(series, d)
        print(f"  Best order via auto_arima: ARIMA({p},{d},{q})")

        # fit & forecast
        df_fc, result, aic, bic = fit_and_forecast(series, (p, d, q), HORIZON, btype)
        metrics = evaluate_insample(series, result)

        print(f"  AIC={aic}  BIC={bic}  "
              f"MAE={metrics['mae']}  RMSE={metrics['rmse']}")
        print(f"  Forecast 2025–2039:")
        print(f"    2025: {df_fc['mean'].iloc[0]:.3f}  "
              f"[{df_fc['lower_95'].iloc[0]:.3f} – {df_fc['upper_95'].iloc[0]:.3f}]")
        print(f"    2032: {df_fc['mean'].iloc[7]:.3f}  "
              f"[{df_fc['lower_95'].iloc[7]:.3f} – {df_fc['upper_95'].iloc[7]:.3f}]")
        print(f"    2039: {df_fc['mean'].iloc[-1]:.3f}  "
              f"[{df_fc['lower_95'].iloc[-1]:.3f} – {df_fc['upper_95'].iloc[-1]:.3f}]")

        all_forecasts.append(df_fc)
        model_summary[btype] = {
            "order":   (p, d, q),
            "aic":     aic,
            "bic":     bic,
            "mae":     metrics["mae"],
            "rmse":    metrics["rmse"],
            "hist_mean": round(float(series.mean()), 4),
            "hist_std":  round(float(series.std()), 4),
        }

    # combine all building type forecasts
    df_all = pd.concat(all_forecasts, ignore_index=True)

    # also attach historical series for plotting
    hist_records = []
    for btype, series in series_dict.items():
        for yr, val in series.items():
            hist_records.append({
                "year":          int(str(yr)),
                "mean":          round(float(val), 4),
                "lower_80":      None,
                "upper_80":      None,
                "lower_95":      None,
                "upper_95":      None,
                "building_type": btype,
                "split":         "historical",
            })
    df_all["split"] = "forecast"
    df_hist = pd.DataFrame(hist_records)
    df_combined = pd.concat([df_hist, df_all], ignore_index=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    fc_path   = f"{OUTPUT_DIR}/occupancy_forecasts.csv"
    meta_path = "model_meta/occupancy_model_meta.json"

    df_combined.to_csv(fc_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(model_summary, f, indent=2)

    print("\n" + "=" * 56)
    print("  Validation summary")
    print("=" * 56)
    for btype, meta in model_summary.items():
        print(f"  {btype:<20}  ARIMA{meta['order']}  "
              f"AIC={meta['aic']:>8}  MAE={meta['mae']:.4f}")
    print()
    print("  Files saved:")
    print(f"    occupancy_forecasts.csv    — historical + forecast, all types")
    print(f"    occupancy_model_meta.json  — orders, AIC, MAE per type")
    print("=" * 56)

    return df_combined, model_summary


if __name__ == "__main__":
    df_combined, model_summary = run_occupancy_forecasting()
    print("\nDone. Ready for Task 3 — Workforce demand model.")
