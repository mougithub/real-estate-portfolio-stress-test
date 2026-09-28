"""
UC2 – SDI Stress-Test Engine
Task 4: Energy Cost Forecasting Model

Approach:
  - OLS regression (statsmodels) on building-level panel data
  - Features: surface_m2, building_age, climate_zone (dummies),
              occupancy_rate, ONEE price index, building_type dummy
  - VIF check to detect multicollinearity
  - 15-year forecast per building, aggregated to portfolio level
  - Confidence intervals via prediction intervals + Monte Carlo age drift

Outputs:
  energy_forecasts.csv        — historical + forecast per building
  energy_portfolio_forecast.csv — portfolio totals (MAD M/yr)
  energy_model_meta.json      — OLS coefficients, R², VIF, diagnostics
"""

import warnings
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

DATA_PATH  = "data/mef_building_stock.csv"
OUTPUT_DIR = "data"
HORIZON    = 15
YEARS_HIST = list(range(2005, 2025))
YEARS_FC   = list(range(2025, 2025 + HORIZON))

# Morocco ONEE electricity tariff index (2005=1.0, ~3.8%/yr growth)
# Source: simulated from known ONEE tariff trajectory
np.random.seed(42)
_base_idx = {yr: 1.0 * (1.038 ** (yr - 2005)) for yr in range(2005, 2025 + HORIZON)}
ONEE_INDEX = {
    yr: round(_base_idx[yr] + (np.random.normal(0, 0.012) if yr < 2025 else 0), 4)
    for yr in range(2005, 2025 + HORIZON)
}

# Climate future projections — arid zones heat up faster under climate shock
# Baseline assumes modest annual energy intensity increase per climate zone
CLIMATE_INTENSITY_DRIFT = {
    "coastal":  0.004,   # +0.4% per year energy intensity increase
    "mountain": 0.006,
    "arid":     0.009,   # highest — cooling demand rises with temperature
}


# ── 1. Feature engineering ────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regression-ready feature matrix from the building panel.
    Adds: ONEE price index, climate dummies, log-transform of surface.
    """
    feat = df.copy()

    # ONEE price index
    feat["onee_index"] = feat["year"].map(ONEE_INDEX)

    # Log surface — reduces skew from the 0.95 correlation
    feat["log_surface"] = np.log(feat["surface_m2"])

    # Climate zone dummies (coastal = baseline)
    feat["is_arid"]     = (feat["climate_zone"] == "arid").astype(int)
    feat["is_mountain"] = (feat["climate_zone"] == "mountain").astype(int)

    # Building type dummies (annex = baseline)
    feat["is_hq"]       = (feat["building_type"] == "ministry_hq").astype(int)
    feat["is_regional"] = (feat["building_type"] == "regional_office").astype(int)

    # Interaction: age × climate (old buildings in hot climates cost more)
    feat["age_x_arid"]  = feat["building_age"] * feat["is_arid"]

    return feat


# ── 2. VIF check ──────────────────────────────────────────────────────────────

def check_vif(X: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.
    VIF > 10 signals problematic multicollinearity.
    """
    X_vif = X[feature_cols].copy()
    X_vif = sm.add_constant(X_vif)
    vif_data = pd.DataFrame({
        "feature": feature_cols,
        "VIF": [
            variance_inflation_factor(X_vif.values, i + 1)
            for i in range(len(feature_cols))
        ]
    }).sort_values("VIF", ascending=False)
    return vif_data


# ── 3. OLS regression with statsmodels ───────────────────────────────────────

FEATURE_COLS = [
    "log_surface",
    "building_age",
    "age_x_arid",
    "occupancy_rate",
    "onee_index",
    "is_arid",
    "is_mountain",
    "is_hq",
    "is_regional",
]

def fit_ols(df_feat: pd.DataFrame) -> tuple:
    """
    Fit OLS on historical panel. Returns result, R², diagnostics.
    Uses statsmodels for full inference (p-values, conf intervals).
    """
    y = df_feat["energy_cost_kmad"].values
    X = df_feat[FEATURE_COLS].values
    X_const = sm.add_constant(X)

    model  = sm.OLS(y, X_const)
    result = model.fit()

    # Collect coefficient table
    coef_df = pd.DataFrame({
        "feature":  ["const"] + FEATURE_COLS,
        "coef":     result.params.round(4).tolist(),
        "pvalue":   result.pvalues.round(4).tolist(),
        "ci_lower": result.conf_int()[:, 0].round(4).tolist(),
        "ci_upper": result.conf_int()[:, 1].round(4).tolist(),
    })

    return result, coef_df


# ── 4. Building-level 15-year forecast ───────────────────────────────────────

def forecast_building(
    building_row: pd.Series,
    ols_result,
    n_simulations: int = 800,
) -> pd.DataFrame:
    """
    For a single building, project energy cost 15 years forward.
    Uses OLS point prediction + Monte Carlo residual sampling for CI.
    """
    btype    = building_row["building_type"]
    climate  = building_row["climate_zone"]
    age_2024 = building_row["building_age"]
    surface  = building_row["surface_m2"]
    occ_2024 = building_row["occupancy_rate"]

    intensity_drift = CLIMATE_INTENSITY_DRIFT[climate]
    residual_std    = float(np.sqrt(ols_result.mse_resid))

    rows = []
    for i, yr in enumerate(YEARS_FC):
        age_t  = age_2024 + (yr - 2024)
        occ_t  = float(np.clip(occ_2024 + np.random.normal(0, 0.005), 0.15, 0.98))

        # Build feature vector for this year
        feat = {
            "log_surface":    np.log(surface),
            "building_age":   age_t,
            "age_x_arid":     age_t * (1 if climate == "arid" else 0),
            "occupancy_rate": occ_t,
            "onee_index":     ONEE_INDEX[yr] * (1 + intensity_drift * i),
            "is_arid":        1 if climate == "arid"     else 0,
            "is_mountain":    1 if climate == "mountain" else 0,
            "is_hq":          1 if btype == "ministry_hq"      else 0,
            "is_regional":    1 if btype == "regional_office"  else 0,
        }
        x_vec = np.array([1.0] + [feat[f] for f in FEATURE_COLS])

        # Point prediction
        mean_pred = float(x_vec @ ols_result.params)

        # Monte Carlo for CI: sample from residual distribution
        mc_noise  = np.random.normal(0, residual_std, n_simulations)
        mc_preds  = np.clip(mean_pred + mc_noise, 0.1, None)

        rows.append({
            "year":      yr,
            "mean":      round(float(np.clip(mean_pred, 0.1, None)), 3),
            "lower_80":  round(float(np.percentile(mc_preds, 10)), 3),
            "upper_80":  round(float(np.percentile(mc_preds, 90)), 3),
            "lower_95":  round(float(np.percentile(mc_preds,  2.5)), 3),
            "upper_95":  round(float(np.percentile(mc_preds, 97.5)), 3),
            "split":     "forecast",
        })

    return pd.DataFrame(rows)


# ── 5. Main pipeline ──────────────────────────────────────────────────────────

def run_energy_forecasting() -> tuple:
    print("=" * 60)
    print("  Task 4 — Energy Cost Forecasting Model")
    print("=" * 60)

    df_raw  = pd.read_csv(DATA_PATH)
    df_feat = build_feature_matrix(df_raw)

    # ── VIF check ─────────────────────────────────────────────────
    print("\n  VIF check (multicollinearity diagnostic):")
    vif_df = check_vif(df_feat, FEATURE_COLS)
    for _, row in vif_df.iterrows():
        flag = "  ✓" if row["VIF"] < 10 else "  ✗ HIGH"
        print(f"    {row['feature']:<20}  VIF = {row['VIF']:>6.2f}{flag}")

    # ── Fit OLS ───────────────────────────────────────────────────
    print("\n  Fitting OLS on full panel (1,140 observations)...")
    ols_result, coef_df = fit_ols(df_feat)

    r2     = round(float(ols_result.rsquared), 4)
    r2_adj = round(float(ols_result.rsquared_adj), 4)
    rmse   = round(float(np.sqrt(ols_result.mse_resid)), 4)
    aic    = round(float(ols_result.aic), 2)

    print(f"\n  Model fit:  R²={r2}  Adj-R²={r2_adj}  RMSE={rmse}  AIC={aic}")
    print("\n  Coefficients (significant at p<0.05 marked *):")
    for _, row in coef_df.iterrows():
        sig = " *" if row["pvalue"] < 0.05 else "  "
        print(f"    {row['feature']:<20}  coef={row['coef']:>8.4f}  "
              f"p={row['pvalue']:.4f}{sig}")

    # ── Building-level forecasts ──────────────────────────────────
    print("\n  Forecasting 57 buildings × 15 years...")
    snap_2024 = df_feat[df_feat["year"] == 2024].copy()

    all_fc_records  = []
    all_hist_records = []

    for _, brow in snap_2024.iterrows():
        bid = brow["building_id"]

        # Historical records
        hist = df_raw[df_raw["building_id"] == bid].copy()
        for _, hrow in hist.iterrows():
            all_hist_records.append({
                "building_id":   bid,
                "region":        hrow["region"],
                "building_type": hrow["building_type"],
                "climate_zone":  hrow["climate_zone"],
                "year":          int(hrow["year"]),
                "mean":          round(float(hrow["energy_cost_kmad"]), 3),
                "lower_80":      None,
                "upper_80":      None,
                "lower_95":      None,
                "upper_95":      None,
                "split":         "historical",
            })

        # Forecast records
        fc_df = forecast_building(brow, ols_result)
        for _, frow in fc_df.iterrows():
            all_fc_records.append({
                "building_id":   bid,
                "region":        brow["region"],
                "building_type": brow["building_type"],
                "climate_zone":  brow["climate_zone"],
                **frow.to_dict(),
            })

    df_building = pd.concat([
        pd.DataFrame(all_hist_records),
        pd.DataFrame(all_fc_records),
    ], ignore_index=True)

    # ── Portfolio aggregation ──────────────────────────────────────
    # Sum across all buildings → total portfolio energy cost (MAD thousands)
    # Convert to MAD millions for readability
    df_port_fc = (
        df_building[df_building["split"] == "forecast"]
        .groupby("year")[["mean", "lower_80", "upper_80", "lower_95", "upper_95"]]
        .sum()
        .div(1000)            # MAD thousands → MAD millions
        .round(3)
        .reset_index()
    )
    df_port_fc["split"] = "forecast"

    df_port_hist = (
        df_building[df_building["split"] == "historical"]
        .groupby("year")["mean"]
        .sum()
        .div(1000)
        .round(3)
        .reset_index()
    )
    df_port_hist["split"] = "historical"

    df_portfolio = pd.concat([df_port_hist, df_port_fc], ignore_index=True)

    # ── Print forecast summary ─────────────────────────────────────
    port_2024 = float(df_port_hist[df_port_hist["year"] == 2024]["mean"].iloc[0])
    port_2039 = float(df_port_fc[df_port_fc["year"]  == 2039]["mean"].iloc[0])
    lo_2039   = float(df_port_fc[df_port_fc["year"]  == 2039]["lower_95"].iloc[0])
    hi_2039   = float(df_port_fc[df_port_fc["year"]  == 2039]["upper_95"].iloc[0])

    print(f"\n  Portfolio energy cost:")
    print(f"    2024 (baseline):  {port_2024:.2f} MAD M/yr")
    print(f"    2039 (forecast):  {port_2039:.2f} MAD M/yr  "
          f"[{lo_2039:.2f} – {hi_2039:.2f}]  95% CI")
    pct = (port_2039 - port_2024) / port_2024 * 100
    print(f"    Change:           {pct:+.1f}% over horizon  "
          f"(driven by ONEE tariff rise + building aging)")

    # ── By climate zone ───────────────────────────────────────────
    print("\n  2039 mean forecast by climate zone (MAD M/yr):")
    fc_2039 = df_building[
        (df_building["split"] == "forecast") & (df_building["year"] == 2039)
    ]
    for zone in ["coastal", "arid", "mountain"]:
        total = fc_2039[fc_2039["climate_zone"] == zone]["mean"].sum() / 1000
        hist_total = df_building[
            (df_building["split"] == "historical") &
            (df_building["year"]  == 2024) &
            (df_building["climate_zone"] == zone)
        ]["mean"].sum() / 1000
        print(f"    {zone:<12}  2024: {hist_total:.2f}  →  2039: {total:.2f}  "
              f"({(total - hist_total) / hist_total * 100:+.1f}%)")

    # ── Save outputs ──────────────────────────────────────────────
    bldg_path  = f"{OUTPUT_DIR}/energy_forecasts.csv"
    port_path  = f"{OUTPUT_DIR}/energy_portfolio_forecast.csv"
    meta_path  = "model_meta/energy_model_meta.json"

    df_building.to_csv(bldg_path, index=False)
    df_portfolio.to_csv(port_path, index=False)

    meta = {
        "model":         "OLS (statsmodels)",
        "n_obs":         int(ols_result.nobs),
        "features":      FEATURE_COLS,
        "r2":            r2,
        "r2_adj":        r2_adj,
        "rmse_kmad":     rmse,
        "aic":           aic,
        "portfolio_2024_mad_m": round(port_2024, 3),
        "portfolio_2039_mad_m": round(port_2039, 3),
        "pct_change_horizon":   round(pct, 2),
        "coefficients":  {
            row["feature"]: {
                "coef":   row["coef"],
                "pvalue": row["pvalue"],
            }
            for _, row in coef_df.iterrows()
        },
        "vif": {
            row["feature"]: round(float(row["VIF"]), 2)
            for _, row in vif_df.iterrows()
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("  Files saved:")
    print("    energy_forecasts.csv           — per-building, all years")
    print("    energy_portfolio_forecast.csv  — portfolio totals, MAD M/yr")
    print("    energy_model_meta.json         — OLS diagnostics + VIF")
    print("=" * 60)

    return df_building, df_portfolio, meta


if __name__ == "__main__":
    df_building, df_portfolio, meta = run_energy_forecasting()
    print("\nDone. Ready for Task 5 — VAR combination + Monte Carlo engine.")
