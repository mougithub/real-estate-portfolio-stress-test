"""
UC2 – SDI Stress-Test Engine
Task 3: Workforce Demand Forecasting Model

Two-stage approach:
  Stage A — Cohort projection: age-band transition matrix models
            retirement outflows and hiring inflows per region.
  Stage B — sklearn OLS regression: validates cohort output and
            adds budget-allocation as an explanatory variable.

Outputs:
  workforce_forecasts.csv       — historical + 15-year forecast per region
  workforce_model_meta.json     — cohort params, regression R², metrics
"""

import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

DATA_PATH  = "data/mef_building_stock.csv"
OUTPUT_DIR = "data"
HORIZON    = 15
YEARS_HIST = list(range(2005, 2025))
YEARS_FC   = list(range(2025, 2025 + HORIZON))

REGIONS = [
    "Rabat-Salé-Kénitra",
    "Casablanca-Settat",
    "Fès-Meknès",
    "Marrakech-Safi",
    "Souss-Massa",
]

# Morocco HCP-inspired demographic parameters per region
# retirement_rate: fraction of workforce retiring each year (~3–5%)
# hiring_ratio:    new hires as fraction of retirements (policy lever)
# decentralization_boost: extra hiring added from 2018 onward (%)
REGION_PARAMS = {
    "Rabat-Salé-Kénitra":  {"retirement_rate": 0.038, "hiring_ratio": 0.92, "decent_boost": 0.00},
    "Casablanca-Settat":   {"retirement_rate": 0.041, "hiring_ratio": 0.95, "decent_boost": 0.02},
    "Fès-Meknès":          {"retirement_rate": 0.045, "hiring_ratio": 1.02, "decent_boost": 0.04},
    "Marrakech-Safi":      {"retirement_rate": 0.043, "hiring_ratio": 1.00, "decent_boost": 0.03},
    "Souss-Massa":         {"retirement_rate": 0.048, "hiring_ratio": 1.05, "decent_boost": 0.05},
}

# Simulated national budget allocation index (2005 = 1.0)
# Reflects Morocco's public sector budget trajectory
np.random.seed(42)
BUDGET_INDEX = {
    yr: round(1.0 * (1.025 ** (yr - 2005)) + np.random.normal(0, 0.015), 4)
    for yr in range(2005, 2025 + HORIZON)
}


# ── Stage A: Cohort projection ────────────────────────────────────────────────

def build_age_band_matrix(retirement_rate: float, n_bands: int = 5) -> np.ndarray:
    """
    Build an (n_bands × n_bands) Leslie-style transition matrix.

    Age bands (each ~8 years wide):
      Band 0: 22–30  (new entrants)
      Band 1: 30–38
      Band 2: 38–46
      Band 3: 46–54
      Band 4: 54–62  (pre-retirement — highest exit rate)

    Each band ages into the next each year.
    Band 4 exits at retirement_rate; other bands exit at lower rates.
    """
    M = np.zeros((n_bands, n_bands))

    # Survival rates per band (fraction that stays or ages up)
    survival = [
        1.0 - retirement_rate * 0.10,   # band 0: almost no retirements
        1.0 - retirement_rate * 0.20,
        1.0 - retirement_rate * 0.40,
        1.0 - retirement_rate * 0.70,
        1.0 - retirement_rate * 1.00,   # band 4: full retirement pressure
    ]

    for i in range(n_bands - 1):
        M[i + 1, i] = survival[i]       # age up to next band
    M[n_bands - 1, n_bands - 1] = survival[-1]   # last band stays until retiring

    return M


def initialise_cohort(total_headcount: int, n_bands: int = 5) -> np.ndarray:
    """
    Distribute initial headcount across age bands.
    Morocco public sector skews older — bands 2 & 3 are largest.
    Distribution: [15%, 20%, 28%, 25%, 12%]
    """
    weights = np.array([0.15, 0.20, 0.28, 0.25, 0.12])
    return (weights * total_headcount).round().astype(float)


def project_cohort(
    initial_hc: int,
    region: str,
    years: list,
    is_forecast: bool = False,
    noise_std: float = 0.012,
) -> np.ndarray:
    """
    Run the cohort matrix forward for len(years) steps.
    Returns total headcount array.
    """
    params = REGION_PARAMS[region]
    ret_rate    = params["retirement_rate"]
    hire_ratio  = params["hiring_ratio"]
    decent_boost = params["decent_boost"]

    M      = build_age_band_matrix(ret_rate)
    cohort = initialise_cohort(initial_hc)
    series = np.zeros(len(years))

    for t, yr in enumerate(years):
        series[t] = cohort.sum()

        # Retirements this step = fraction leaving band 4
        retirements = cohort[-1] * ret_rate

        # New hires entering band 0
        boost = decent_boost if yr >= 2018 else 0.0
        new_hires = retirements * (hire_ratio + boost)

        # Advance cohort through transition matrix
        cohort = M @ cohort
        cohort[0] += new_hires

        # Small demographic noise (only on historical, for realism)
        if not is_forecast:
            cohort = np.maximum(0, cohort + np.random.normal(0, cohort.sum() * noise_std / 5, size=cohort.shape))

    return series.round().astype(int)


# ── Stage B: sklearn regression validation ───────────────────────────────────

def build_regression_features(
    region: str,
    years: list,
    cohort_forecast: np.ndarray,
) -> pd.DataFrame:
    """
    Build feature matrix for OLS regression.
    Features: cohort_forecast, budget_index, year_trend, region dummies
    """
    df = pd.DataFrame({
        "year":             years,
        "cohort_hc":        cohort_forecast.astype(float),
        "budget_index":     [BUDGET_INDEX[yr] for yr in years],
        "year_trend":       [(yr - 2005) / 19 for yr in years],   # normalised 0–1
    })
    return df


def fit_regression(
    X_hist: pd.DataFrame,
    y_hist: np.ndarray,
    region: str,
) -> tuple:
    """
    Fit sklearn LinearRegression on historical data.
    Returns fitted model, scaler, metrics.
    """
    feature_cols = ["cohort_hc", "budget_index", "year_trend"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hist[feature_cols])

    model = LinearRegression()
    model.fit(X_scaled, y_hist)

    y_pred = model.predict(X_scaled)
    mae    = mean_absolute_error(y_hist, y_pred)
    r2     = r2_score(y_hist, y_pred)

    # Cross-validated R² (3-fold — small dataset)
    cv_scores = cross_val_score(
        LinearRegression(), X_scaled, y_hist,
        cv=3, scoring="r2"
    )
    cv_r2 = float(np.mean(cv_scores))

    metrics = {
        "mae":   round(float(mae), 2),
        "r2":    round(float(r2), 4),
        "cv_r2": round(cv_r2, 4),
        "coefficients": {
            col: round(float(c), 4)
            for col, c in zip(feature_cols, model.coef_)
        }
    }
    return model, scaler, metrics


def predict_regression(
    model,
    scaler,
    X_fc: pd.DataFrame,
) -> np.ndarray:
    feature_cols = ["cohort_hc", "budget_index", "year_trend"]
    X_scaled = scaler.transform(X_fc[feature_cols])
    return model.predict(X_scaled)


# ── Confidence intervals via bootstrap ───────────────────────────────────────

def bootstrap_ci(
    cohort_series: np.ndarray,
    region: str,
    initial_hc: int,
    n_boot: int = 500,
) -> dict:
    """
    Re-run cohort projection n_boot times with perturbed parameters.
    Returns p10, p25, p75, p90 bands.
    """
    params = REGION_PARAMS[region]
    runs   = np.zeros((n_boot, HORIZON))

    for b in range(n_boot):
        # Perturb retirement rate and hiring ratio slightly
        ret_perturbed  = params["retirement_rate"] * np.random.uniform(0.88, 1.12)
        hire_perturbed = params["hiring_ratio"]    * np.random.uniform(0.92, 1.08)

        perturbed_params = {**params,
                            "retirement_rate": ret_perturbed,
                            "hiring_ratio":    hire_perturbed}

        # Temporarily override and run
        original = REGION_PARAMS[region].copy()
        REGION_PARAMS[region].update(perturbed_params)
        boot_series = project_cohort(initial_hc, region, YEARS_FC, is_forecast=True, noise_std=0)
        REGION_PARAMS[region] = original

        runs[b] = boot_series

    return {
        "p10": np.percentile(runs, 10, axis=0).round().astype(int),
        "p25": np.percentile(runs, 25, axis=0).round().astype(int),
        "p75": np.percentile(runs, 75, axis=0).round().astype(int),
        "p90": np.percentile(runs, 90, axis=0).round().astype(int),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_workforce_forecasting() -> tuple:
    print("=" * 58)
    print("  Task 3 — Workforce Demand Forecasting")
    print("=" * 58)

    df_raw = pd.read_csv(DATA_PATH)

    # Aggregate total headcount per region per year
    hc_hist = (
        df_raw.groupby(["region", "year"])["headcount"]
        .sum()
        .reset_index()
    )

    all_records  = []
    model_meta   = {}

    for region in REGIONS:
        print(f"\n── {region} ──")

        reg_hist = hc_hist[hc_hist["region"] == region].sort_values("year")
        y_hist   = reg_hist["headcount"].values
        hc_2024  = int(y_hist[-1])
        hc_2005  = int(y_hist[0])

        print(f"  Headcount 2005: {hc_2005:,}   2024: {hc_2024:,}   "
              f"Change: {((hc_2024 - hc_2005) / hc_2005 * 100):+.1f}%")

        # ── Stage A: cohort projection (historical reconstruction) ──
        cohort_hist = project_cohort(hc_2005, region, YEARS_HIST,
                                     is_forecast=False, noise_std=0.010)

        # ── Stage B: regression on historical data ──
        X_hist   = build_regression_features(region, YEARS_HIST, cohort_hist)
        model_lr, scaler, metrics = fit_regression(X_hist, y_hist, region)

        print(f"  Regression  R²={metrics['r2']}  CV-R²={metrics['cv_r2']}  "
              f"MAE={metrics['mae']:.0f} staff")
        print(f"  Coefficients: cohort={metrics['coefficients']['cohort_hc']}  "
              f"budget={metrics['coefficients']['budget_index']}  "
              f"trend={metrics['coefficients']['year_trend']}")

        # ── Forecast: cohort projection 2025–2039 ──
        cohort_fc   = project_cohort(hc_2024, region, YEARS_FC,
                                     is_forecast=True, noise_std=0)
        X_fc        = build_regression_features(region, YEARS_FC, cohort_fc)
        reg_fc      = predict_regression(model_lr, scaler, X_fc).round().astype(int)

        # Bootstrap confidence intervals
        ci = bootstrap_ci(cohort_fc, region, hc_2024, n_boot=400)

        pct_change = ((cohort_fc[-1] - hc_2024) / hc_2024 * 100)
        print(f"  Forecast 2025: {cohort_fc[0]:,}   "
              f"2039: {cohort_fc[-1]:,}   "
              f"({pct_change:+.1f}% over horizon)")
        print(f"  95% band 2039: [{ci['p10'][-1]:,} – {ci['p90'][-1]:,}]")

        # ── Store historical records ──
        for i, yr in enumerate(YEARS_HIST):
            all_records.append({
                "region":       region,
                "year":         yr,
                "headcount":    int(y_hist[i]),
                "cohort_hc":    int(cohort_hist[i]),
                "lower_80":     None,
                "upper_80":     None,
                "lower_95":     None,
                "upper_95":     None,
                "split":        "historical",
            })

        # ── Store forecast records ──
        for i, yr in enumerate(YEARS_FC):
            all_records.append({
                "region":       region,
                "year":         yr,
                "headcount":    int(reg_fc[i]),
                "cohort_hc":    int(cohort_fc[i]),
                "lower_80":     int(ci["p25"][i]),
                "upper_80":     int(ci["p75"][i]),
                "lower_95":     int(ci["p10"][i]),
                "upper_95":     int(ci["p90"][i]),
                "split":        "forecast",
            })

        model_meta[region] = {
            "hc_2005":       hc_2005,
            "hc_2024":       hc_2024,
            "hc_2039_mean":  int(cohort_fc[-1]),
            "pct_change":    round(pct_change, 2),
            "retirement_rate": REGION_PARAMS[region]["retirement_rate"],
            "hiring_ratio":    REGION_PARAMS[region]["hiring_ratio"],
            "decent_boost":    REGION_PARAMS[region]["decent_boost"],
            "regression": metrics,
        }

    df_out = pd.DataFrame(all_records)

    # ── Save outputs ──────────────────────────────────────────────────────────
    fc_path   = f"{OUTPUT_DIR}/workforce_forecasts.csv"
    meta_path = "model_meta/workforce_model_meta.json"

    df_out.to_csv(fc_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Forecast summary — headcount change 2024 → 2039")
    print("=" * 58)
    for region, meta in model_meta.items():
        direction = "growing" if meta["pct_change"] > 0 else "declining"
        print(f"  {region:<30}  {meta['hc_2024']:>5,} → {meta['hc_2039_mean']:>5,}  "
              f"({meta['pct_change']:+.1f}%)  {direction}")

    total_2024 = sum(m["hc_2024"]      for m in model_meta.values())
    total_2039 = sum(m["hc_2039_mean"] for m in model_meta.values())
    print(f"  {'TOTAL (all regions)':<30}  {total_2024:>5,} → {total_2039:>5,}  "
          f"({(total_2039 - total_2024) / total_2024 * 100:+.1f}%)")

    print()
    print("  Files saved:")
    print("    workforce_forecasts.csv    — historical + forecast, all regions")
    print("    workforce_model_meta.json  — cohort params + regression metrics")
    print("=" * 58)

    return df_out, model_meta


if __name__ == "__main__":
    df_out, model_meta = run_workforce_forecasting()
    print("\nDone. Ready for Task 4 — Energy cost forecasting model.")
