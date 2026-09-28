"""
UC2 – SDI Stress-Test Engine
Task 5: VAR Combination + Monte Carlo Scenario Engine

Pipeline:
  1. Build VAR input matrix from Tasks 2/3/4 outputs
  2. Fit VAR(p) — lag order selected by AIC
  3. Stability check — reject unstable models
  4. Define 4 scenario shock distributions (scipy.stats)
  5. Monte Carlo loop — 1000 runs per scenario
  6. Strategy scoring — weighted composite per scenario
  7. Save all outputs for Task 8 (Streamlit dashboard)

Outputs:
  scenario_trajectories.csv    — p10/p25/mean/p75/p90 per scenario × variable
  strategy_rankings.csv        — ranked strategies per scenario
  engine_meta.json             — VAR order, stability, scenario params
"""

import warnings
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "data"
HORIZON    = 15
YEARS_FC   = list(range(2025, 2025 + HORIZON))
N_SIMS     = 1000


# ── 1. Build VAR input matrix ─────────────────────────────────────────────────

def build_var_matrix() -> pd.DataFrame:
    """
    Combine the three historical series into one aligned DataFrame.
    VAR requires a single matrix: rows=years, cols=variables.
    """
    occ = pd.read_csv(f"{OUTPUT_DIR}/occupancy_forecasts.csv")
    wf  = pd.read_csv(f"{OUTPUT_DIR}/workforce_forecasts.csv")
    nrj = pd.read_csv(f"{OUTPUT_DIR}/energy_portfolio_forecast.csv")

    occ_h = (occ[occ["split"] == "historical"]
             .groupby("year")["mean"].mean()
             .rename("occupancy"))

    wf_h  = (wf[wf["split"] == "historical"]
             .groupby("year")["headcount"].sum()
             .rename("workforce"))

    nrj_h = (nrj[nrj["split"] == "historical"]
             .set_index("year")["mean"]
             .rename("energy_cost"))

    df = pd.concat([occ_h, wf_h, nrj_h], axis=1).dropna()
    df.index = pd.RangeIndex(len(df))   # integer index for VAR
    return df


# ── 2. Stationarity + differencing ───────────────────────────────────────────

def prepare_for_var(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    ADF test each series. Difference if non-stationary.
    Returns (prepared_df, differencing_info).
    """
    diff_info = {}
    df_out    = df.copy()

    for col in df.columns:
        pval = adfuller(df[col].values, autolag="AIC")[1]
        if pval > 0.05:
            df_out[col] = df[col].diff().dropna()
            diff_info[col] = {"d": 1, "adf_pval": round(pval, 4)}
        else:
            diff_info[col] = {"d": 0, "adf_pval": round(pval, 4)}

    df_out = df_out.dropna().reset_index(drop=True)
    return df_out, diff_info


# ── 3. Fit VAR ────────────────────────────────────────────────────────────────

def fit_var(df_var: pd.DataFrame) -> tuple:
    """
    Select lag order by AIC (max 4), fit VAR, check stability.
    Falls back to lag=1 if stability fails.
    """
    model = VAR(df_var)
    # Safe max lags: VAR(p) needs (p+1)*k obs minimum; k=n_vars
    n_obs, n_vars = df_var.shape
    safe_maxlags  = max(1, (n_obs // (n_vars + 1)) - 1)
    safe_maxlags  = min(safe_maxlags, 3)
    lag_result    = model.select_order(maxlags=safe_maxlags)
    best_lag      = lag_result.selected_orders["aic"]
    best_lag      = max(1, min(best_lag, safe_maxlags))

    result = model.fit(best_lag)

    # Stability check — all eigenvalues must be inside unit circle
    is_stable = result.is_stable()
    if not is_stable:
        print(f"  Warning: VAR({best_lag}) unstable — falling back to VAR(1)")
        result   = model.fit(1)
        best_lag = 1
        is_stable = result.is_stable()

    return result, best_lag, is_stable


# ── 4. Scenario shock definitions ────────────────────────────────────────────
#
# Each scenario defines a distribution over shock multipliers
# applied to the VAR forecast at each time step (ramped over 5 years).
# Using scipy.stats.norm(loc=mean_shock, scale=uncertainty).
#
# Variables: occupancy, workforce, energy_cost
# Multipliers are cumulative by end of ramp — e.g. 1.20 = +20%

SCENARIOS = {
    "optimistic": {
        "label":       "Optimistic — workforce +20%",
        "occupancy":   stats.norm(loc=1.08,  scale=0.03),
        "workforce":   stats.norm(loc=1.20,  scale=0.05),
        "energy_cost": stats.norm(loc=1.05,  scale=0.04),
        "color":       "#1D9E75",
    },
    "austerity": {
        "label":       "Austerity — budget −10%",
        "occupancy":   stats.norm(loc=0.93,  scale=0.03),
        "workforce":   stats.norm(loc=0.98,  scale=0.03),
        "energy_cost": stats.norm(loc=0.90,  scale=0.04),
        "color":       "#BA7517",
    },
    "climate_shock": {
        "label":       "Climate shock — energy +40%",
        "occupancy":   stats.norm(loc=1.00,  scale=0.03),
        "workforce":   stats.norm(loc=1.01,  scale=0.02),
        "energy_cost": stats.norm(loc=1.40,  scale=0.08),
        "color":       "#D85A30",
    },
    "digitization": {
        "label":       "Digitization — space need −30%",
        "occupancy":   stats.norm(loc=0.70,  scale=0.06),
        "workforce":   stats.norm(loc=0.95,  scale=0.04),
        "energy_cost": stats.norm(loc=0.85,  scale=0.05),
        "color":       "#378ADD",
    },
}

VARIABLES = ["occupancy", "workforce", "energy_cost"]
RAMP_YEARS = 5    # shock reaches full effect by year 5


# ── 5. Monte Carlo engine ─────────────────────────────────────────────────────

def run_monte_carlo(
    var_result,
    df_raw: pd.DataFrame,
    diff_info: dict,
    scenario_name: str,
) -> dict[str, pd.DataFrame]:
    """
    Run N_SIMS simulations for one scenario.
    Returns dict of DataFrames (one per variable):
      columns: year, mean, p10, p25, p75, p90
    """
    scenario = SCENARIOS[scenario_name]
    n_vars   = len(VARIABLES)
    all_runs = np.zeros((N_SIMS, HORIZON, n_vars))

    # Last k_ar rows of the VAR training matrix for initial conditions
    endog_tail = var_result.endog[-var_result.k_ar:]

    for sim in range(N_SIMS):
        # Draw shock multipliers for this simulation
        multipliers = np.array([
            scenario[v].rvs() for v in VARIABLES
        ])

        # VAR point forecast (on differenced series if applicable)
        fc_diff = var_result.forecast(endog_tail, steps=HORIZON)

        # Undo differencing for variables that were differenced
        fc_levels = np.zeros_like(fc_diff)
        last_vals  = np.array([df_raw[v].iloc[-1] for v in VARIABLES])

        for t in range(HORIZON):
            for j, var in enumerate(VARIABLES):
                if diff_info[var]["d"] == 1:
                    # Cumulative sum to undo differencing
                    base = last_vals[j] if t == 0 else fc_levels[t - 1, j]
                    fc_levels[t, j] = base + fc_diff[t, j]
                else:
                    fc_levels[t, j] = fc_diff[t, j]

        # Apply scenario shocks with linear ramp
        shocked = fc_levels.copy()
        for t in range(HORIZON):
            ramp = min(1.0, (t + 1) / RAMP_YEARS)
            for j in range(n_vars):
                shocked[t, j] *= (1.0 + ramp * (multipliers[j] - 1.0))

        # Clip to physically plausible bounds
        shocked[:, 0] = np.clip(shocked[:, 0], 0.15, 0.98)  # occupancy
        shocked[:, 1] = np.clip(shocked[:, 1], 1000, None)   # workforce > 0
        shocked[:, 2] = np.clip(shocked[:, 2], 0.01, None)   # energy cost > 0

        all_runs[sim] = shocked

    # Summarise across simulations
    results = {}
    for j, var in enumerate(VARIABLES):
        arr = all_runs[:, :, j]
        results[var] = pd.DataFrame({
            "year": YEARS_FC,
            "mean": arr.mean(axis=0),
            "p10":  np.percentile(arr, 10, axis=0),
            "p25":  np.percentile(arr, 25, axis=0),
            "p75":  np.percentile(arr, 75, axis=0),
            "p90":  np.percentile(arr, 90, axis=0),
        }).round(4)

    return results


# ── 6. Strategy scoring ───────────────────────────────────────────────────────

# Base profile: how well each strategy performs on 3 dimensions (0–1 scale)
# Grounded in UC2 logic: Rehabilitate good for sustainability,
# Dispose good for cost efficiency under low occupancy, etc.
STRATEGY_BASE = pd.DataFrame({
    "strategy":          ["Rehabilitate", "Pool",  "Rent",  "Build", "Dispose"],
    "cost_efficiency":   [0.55,           0.82,    0.65,    0.28,    0.92],
    "sustainability":    [0.82,           0.60,    0.42,    0.65,    0.18],
    "space_flexibility": [0.38,           0.72,    0.88,    0.48,    0.96],
}).set_index("strategy")


def score_strategies(
    mc_results: dict,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Convert Monte Carlo scenario outcomes into strategy rankings.
    Scenario outcomes modulate the base profile scores before weighting.
    """
    if weights is None:
        weights = {"cost_efficiency": 0.40,
                   "sustainability":  0.30,
                   "space_flexibility": 0.30}

    # Extract final-year mean values as scenario outcome signals
    occ_final    = float(mc_results["occupancy"]["mean"].iloc[-1])
    wf_final     = float(mc_results["workforce"]["mean"].iloc[-1])
    energy_final = float(mc_results["energy_cost"]["mean"].iloc[-1])

    # Normalise signals to [0,1] modifiers
    # Low occupancy → reward flexibility/disposal strategies
    occ_pressure    = float(np.clip(1.0 - occ_final / 0.85, 0, 1))
    # High energy cost → penalise low-sustainability strategies
    energy_pressure = float(np.clip((energy_final - 0.25) / 0.60, 0, 1))
    # Workforce decline → reward space-reducing strategies
    wf_pressure     = float(np.clip(1.0 - wf_final / 11000, 0, 1))

    scores = STRATEGY_BASE.copy().astype(float)

    # Apply scenario-driven adjustments
    scores["space_flexibility"] = np.clip(
        scores["space_flexibility"] + occ_pressure * 0.18 + wf_pressure * 0.10, 0, 1)
    scores["sustainability"]    = np.clip(
        scores["sustainability"]    - energy_pressure * 0.14, 0, 1)
    scores["cost_efficiency"]   = np.clip(
        scores["cost_efficiency"]   + wf_pressure * 0.08, 0, 1)

    # Weighted composite score
    scores["composite"] = (
        scores["cost_efficiency"]   * weights["cost_efficiency"] +
        scores["sustainability"]    * weights["sustainability"]  +
        scores["space_flexibility"] * weights["space_flexibility"]
    )

    scores["rank"] = scores["composite"].rank(ascending=False).astype(int)
    return scores.sort_values("composite", ascending=False).round(3)


# ── 7. Main pipeline ──────────────────────────────────────────────────────────

def run_scenario_engine():
    print("=" * 62)
    print("  Task 5 — VAR Combination + Monte Carlo Scenario Engine")
    print("=" * 62)

    # ── Build VAR matrix ──────────────────────────────────────────
    df_raw = build_var_matrix()
    print(f"\n  VAR input matrix: {len(df_raw)} years × {len(VARIABLES)} variables")
    print(f"  Variables: {VARIABLES}")

    # ── Stationarity + differencing ───────────────────────────────
    print("\n  Stationarity (ADF test):")
    df_var, diff_info = prepare_for_var(df_raw)
    for var, info in diff_info.items():
        status = "stationary" if info["d"] == 0 else "differenced (d=1)"
        print(f"    {var:<15}  ADF p={info['adf_pval']:.4f}  →  {status}")

    # ── Fit VAR ───────────────────────────────────────────────────
    print("\n  Fitting VAR model...")
    var_result, best_lag, is_stable = fit_var(df_var)
    print(f"  Lag order (AIC):  {best_lag}")
    print(f"  Stable:           {is_stable}  ✓" if is_stable
          else f"  Stable:           {is_stable}  ✗")
    print(f"  Log-likelihood:   {var_result.llf:.2f}")
    print(f"  AIC:              {var_result.aic:.2f}")

    # Print Granger causality summary
    print("\n  Granger causality (does X help predict Y?):")
    for caused in VARIABLES:
        others = [v for v in VARIABLES if v != caused]
        gc = var_result.test_causality(caused, others, kind="f")
        sig = "significant" if gc.pvalue < 0.05 else "not significant"
        print(f"    {' + '.join(others)} → {caused}:  "
              f"p={gc.pvalue:.4f}  ({sig})")

    # ── Run Monte Carlo for all scenarios ─────────────────────────
    print(f"\n  Running Monte Carlo ({N_SIMS} simulations × 4 scenarios)...")
    all_mc      = {}
    all_ranked  = {}
    traj_records = []

    for scenario_name in SCENARIOS:
        mc = run_monte_carlo(var_result, df_raw, diff_info, scenario_name)
        ranked = score_strategies(mc)
        all_mc[scenario_name]     = mc
        all_ranked[scenario_name] = ranked

        # Print scenario summary
        label = SCENARIOS[scenario_name]["label"]
        print(f"\n  ── {label} ──")
        for var in VARIABLES:
            m25  = mc[var]["mean"].iloc[0]
            m39  = mc[var]["mean"].iloc[-1]
            lo39 = mc[var]["p10"].iloc[-1]
            hi39 = mc[var]["p90"].iloc[-1]
            print(f"    {var:<15}  2025:{m25:>9.3f}  "
                  f"2039:{m39:>9.3f}  "
                  f"[{lo39:.3f}–{hi39:.3f}] 80% CI")

        print(f"  Strategy ranking:")
        for strat, row in ranked.iterrows():
            bar = "█" * int(row["composite"] * 20)
            print(f"    #{int(row['rank'])} {strat:<15}  "
                  f"score={row['composite']:.3f}  {bar}")

        # Collect trajectory records
        for var in VARIABLES:
            for _, row in mc[var].iterrows():
                traj_records.append({
                    "scenario":  scenario_name,
                    "variable":  var,
                    "year":      int(row["year"]),
                    "mean":      float(row["mean"]),
                    "p10":       float(row["p10"]),
                    "p25":       float(row["p25"]),
                    "p75":       float(row["p75"]),
                    "p90":       float(row["p90"]),
                })

    # Also add historical baseline to trajectory file
    for var in VARIABLES:
        for _, row in df_raw.iterrows():
            traj_records.append({
                "scenario":  "historical",
                "variable":  var,
                "year":      int(2005 + _),
                "mean":      float(row[var]),
                "p10":       None,
                "p25":       None,
                "p75":       None,
                "p90":       None,
            })

    # ── Strategy comparison across scenarios ──────────────────────
    print("\n" + "=" * 62)
    print("  Strategy ranking matrix (composite score per scenario)")
    print("=" * 62)
    strategies = STRATEGY_BASE.index.tolist()
    header = f"  {'Strategy':<15}" + "".join(
        f"  {s[:10]:<10}" for s in SCENARIOS
    )
    print(header)
    print("  " + "-" * (15 + 12 * len(SCENARIOS)))
    for strat in strategies:
        row_str = f"  {strat:<15}"
        for sc in SCENARIOS:
            score = all_ranked[sc].loc[strat, "composite"]
            rank  = int(all_ranked[sc].loc[strat, "rank"])
            row_str += f"  {score:.2f} (#{rank}) "
        print(row_str)

    # ── Save outputs ──────────────────────────────────────────────
    df_traj = pd.DataFrame(traj_records)

    # Rankings: one row per strategy × scenario
    rank_records = []
    for sc in SCENARIOS:
        for strat, row in all_ranked[sc].iterrows():
            rank_records.append({
                "scenario":        sc,
                "strategy":        strat,
                "rank":            int(row["rank"]),
                "composite":       float(row["composite"]),
                "cost_efficiency": float(row["cost_efficiency"]),
                "sustainability":  float(row["sustainability"]),
                "space_flexibility": float(row["space_flexibility"]),
            })
    df_rankings = pd.DataFrame(rank_records)

    traj_path    = f"{OUTPUT_DIR}/scenario_trajectories.csv"
    rank_path    = f"{OUTPUT_DIR}/strategy_rankings.csv"
    meta_path    = "model_meta/engine_meta.json"

    df_traj.to_csv(traj_path, index=False)
    df_rankings.to_csv(rank_path, index=False)

    engine_meta = {
        "var_lag":       int(best_lag),
        "var_stable":    bool(is_stable),
        "var_aic":       round(float(var_result.aic), 4),
        "n_simulations": int(N_SIMS),
        "horizon_years": int(HORIZON),
        "ramp_years":    int(RAMP_YEARS),
        "differencing": {k: {"d": int(v["d"]), "adf_pval": float(v["adf_pval"])}
                         for k, v in diff_info.items()},
        "scenarios": {
            name: {
                "label": cfg["label"],
                "shocks": {
                    v: {"mean": float(cfg[v].mean()),
                        "std":  float(cfg[v].std())}
                    for v in VARIABLES
                }
            }
            for name, cfg in SCENARIOS.items()
        },
    }
    with open(meta_path, "w") as f:
        json.dump(engine_meta, f, indent=2)

    print("\n  Files saved:")
    print(f"    scenario_trajectories.csv  — {len(df_traj)} rows")
    print(f"    strategy_rankings.csv      — {len(df_rankings)} rows")
    print(f"    engine_meta.json           — VAR params + scenario config")
    print("=" * 62)

    return df_traj, df_rankings, all_mc, engine_meta


if __name__ == "__main__":
    df_traj, df_rankings, all_mc, meta = run_scenario_engine()
    print("\nDone. Ready for Task 6 — Strategy scoring + Plotly charts.")
