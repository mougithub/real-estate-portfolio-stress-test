"""
UC2 – SDI Stress-Test Engine
Task 1: Synthetic Dataset Generation
Morocco MEF Real Estate Portfolio
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────

REGIONS = {
    "Rabat-Salé-Kénitra":    {"n": 20, "climate": "coastal",  "urban_factor": 1.10},
    "Casablanca-Settat":     {"n": 15, "climate": "coastal",  "urban_factor": 1.05},
    "Fès-Meknès":            {"n": 10, "climate": "mountain", "urban_factor": 0.95},
    "Marrakech-Safi":        {"n": 8,  "climate": "arid",     "urban_factor": 0.90},
    "Souss-Massa":           {"n": 7,  "climate": "arid",     "urban_factor": 0.88},
}

BUILDING_TYPES = {
    "ministry_hq":      {"occ_base": 0.78, "cost_base": 14.0, "area_range": (3000, 18000)},
    "regional_office":  {"occ_base": 0.65, "cost_base": 8.5,  "area_range": (800,  4000)},
    "annex":            {"occ_base": 0.52, "cost_base": 5.0,  "area_range": (300,  1200)},
}

CLIMATE_ENERGY_FACTOR = {
    "coastal":  0.88,   # mild — lower energy demand
    "mountain": 1.12,   # heating costs
    "arid":     1.22,   # heavy cooling demand
}

# Morocco ONEE electricity price index (2005=100, realistic growth)
ENERGY_PRICE_INDEX = {yr: 100 * (1.038 ** (yr - 2005)) for yr in range(2005, 2026)}

YEARS = list(range(2005, 2025))   # 20 years of history
N_YEARS = len(YEARS)


# ── Helper functions ──────────────────────────────────────────────────────────

def simulate_occupancy_series(base: float, n_years: int) -> np.ndarray:
    """
    Simulate annual occupancy rate with:
    - Slow mean-reverting drift
    - Ramadan / summer dip captured as random shocks
    - Structural break ~2020 (COVID remote work)
    """
    series = np.zeros(n_years)
    series[0] = base + np.random.normal(0, 0.03)

    for t in range(1, n_years):
        year = YEARS[t]
        drift = 0.002 * (base - series[t - 1])         # mean reversion
        shock = np.random.normal(0, 0.018)
        if year == 2020:
            shock -= 0.12                               # COVID drop
        elif year == 2021:
            shock += 0.05                               # partial recovery
        series[t] = series[t - 1] + drift + shock

    return np.clip(series, 0.20, 0.98)


def simulate_headcount_series(base: int, region_decay: float, n_years: int) -> np.ndarray:
    """
    Cohort-inspired headcount projection:
    - Gradual retirement-driven decline modified by hiring policy
    - Decentralization push adds variance post-2018
    """
    series = np.zeros(n_years)
    series[0] = base

    for t in range(1, n_years):
        year = YEARS[t]
        retirement_loss = series[t - 1] * region_decay
        new_hires = retirement_loss * np.random.uniform(0.80, 1.05)
        if year >= 2018:
            new_hires *= np.random.uniform(0.95, 1.08)  # decentralisation policy
        noise = np.random.normal(0, series[t - 1] * 0.01)
        series[t] = max(10, series[t - 1] - retirement_loss + new_hires + noise)

    return series.round().astype(int)


def simulate_energy_cost(
    area_m2: float,
    age: int,
    climate: str,
    occ_series: np.ndarray,
    n_years: int,
) -> np.ndarray:
    """
    Energy cost (MAD thousands/year):
    - Base: area × age degradation × climate factor
    - Scales with occupancy
    - Tracks ONEE price index
    """
    base_intensity = 0.9 + (age / 100) * 0.6          # kWh/m²/yr proxy
    climate_mult   = CLIMATE_ENERGY_FACTOR[climate]
    series = np.zeros(n_years)

    for t in range(n_years):
        year      = YEARS[t]
        price_idx = ENERGY_PRICE_INDEX[year] / 100
        age_t     = age + t
        intensity = base_intensity * (1 + (age_t / 200)) * climate_mult
        raw_cost  = area_m2 * intensity * price_idx * occ_series[t]
        # convert to MAD thousands, add noise
        series[t] = (raw_cost / 1000) + np.random.normal(0, raw_cost * 0.04 / 1000)

    return np.clip(series, 0.5, None).round(2)


def regulatory_status(age: int, year_idx: int) -> str:
    """
    Assign compliance status:
    older buildings more likely non-compliant,
    improves slowly over time as retrofits happen.
    """
    compliance_prob = max(0.1, 0.9 - (age / 120) + (year_idx / 80))
    r = np.random.random()
    if r < compliance_prob * 0.75:
        return "compliant"
    elif r < compliance_prob:
        return "partial"
    else:
        return "non_compliant"


# ── Main generator ────────────────────────────────────────────────────────────

def generate_building_stock(output_dir: str = "data") -> pd.DataFrame:
    records = []
    building_id = 1

    for region, rcfg in REGIONS.items():
        climate      = rcfg["climate"]
        urban_factor = rcfg["urban_factor"]
        n_buildings  = rcfg["n"]

        # distribute building types: 20% HQ, 50% regional, 30% annex
        type_draw = (
            ["ministry_hq"]    * max(1, int(n_buildings * 0.20)) +
            ["regional_office"] * max(1, int(n_buildings * 0.50)) +
            ["annex"]          * max(1, int(n_buildings * 0.30))
        )[:n_buildings]
        np.random.shuffle(type_draw)

        for btype in type_draw:
            bcfg       = BUILDING_TYPES[btype]
            area_m2    = int(np.random.uniform(*bcfg["area_range"]))
            build_year = int(np.random.choice(
                range(1960, 2020),
                p=np.array([1]*30 + [2]*15 + [3]*15) / (30 + 30 + 45)
            ))
            age_2005   = max(0, 2005 - build_year)
            tenure     = np.random.choice(["owned", "leased"], p=[0.70, 0.30])
            base_hc    = int(area_m2 / np.random.uniform(12, 22))   # m² per person
            decay_rate = np.random.uniform(0.030, 0.055)             # 3–5.5% retirement

            occ_series = simulate_occupancy_series(bcfg["occ_base"], N_YEARS)
            hc_series  = simulate_headcount_series(base_hc, decay_rate, N_YEARS)
            nrj_series = simulate_energy_cost(area_m2, age_2005, climate, occ_series, N_YEARS)

            for t, year in enumerate(YEARS):
                age_t = age_2005 + t
                records.append({
                    "building_id":       f"MEF-{building_id:04d}",
                    "year":              year,
                    "region":            region,
                    "building_type":     btype,
                    "climate_zone":      climate,
                    "tenure":            tenure,
                    "surface_m2":        area_m2,
                    "build_year":        build_year,
                    "building_age":      age_t,
                    "occupancy_rate":    round(float(occ_series[t]), 4),
                    "headcount":         int(hc_series[t]),
                    "energy_cost_kmad":  float(nrj_series[t]),  # MAD thousands
                    "regulatory_status": regulatory_status(age_t, t),
                    "urban_factor":      urban_factor,
                })

            building_id += 1

    df = pd.DataFrame(records)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["cost_per_m2"]         = (df["energy_cost_kmad"] * 1000 / df["surface_m2"]).round(2)
    df["sqm_per_person"]      = (df["surface_m2"] / df["headcount"].replace(0, np.nan)).round(1)
    df["is_underutilised"]    = df["occupancy_rate"] < 0.50
    df["is_aging"]            = df["building_age"] > 40
    df["risk_score"]          = (
        (1 - df["occupancy_rate"]) * 0.35 +
        (df["building_age"] / 80).clip(0, 1) * 0.35 +
        df["regulatory_status"].map({"compliant": 0, "partial": 0.5, "non_compliant": 1.0}) * 0.30
    ).round(3)

    # ── Save outputs ──────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main_path = f"{output_dir}/mef_building_stock.csv"
    df.to_csv(main_path, index=False)

    # Latest-year snapshot for dashboard use
    snap = df[df["year"] == 2024].copy()
    snap_path = f"{output_dir}/mef_snapshot_2024.csv"
    snap.to_csv(snap_path, index=False)

    return df


# ── Validation report ─────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    n_buildings = df["building_id"].nunique()
    n_records   = len(df)
    nulls       = df.isnull().sum().sum()

    snap = df[df["year"] == 2024]

    print("=" * 56)
    print("  UC2 Dataset Validation Report")
    print("=" * 56)
    print(f"  Buildings      : {n_buildings}")
    print(f"  Total records  : {n_records}  ({N_YEARS} years × {n_buildings} buildings)")
    print(f"  Null values    : {nulls}  {'✓' if nulls == 0 else '✗ PROBLEM'}")
    print()
    print("  Occupancy rate (2024 snapshot)")
    print(f"    Mean         : {snap['occupancy_rate'].mean():.3f}")
    print(f"    Std          : {snap['occupancy_rate'].std():.3f}")
    print(f"    Min / Max    : {snap['occupancy_rate'].min():.3f} / {snap['occupancy_rate'].max():.3f}")
    print(f"    Under-utilised (<50%): {snap['is_underutilised'].sum()} buildings")
    print()
    print("  Energy cost — MAD thousands/yr (2024)")
    print(f"    Mean         : {snap['energy_cost_kmad'].mean():.1f}")
    print(f"    Std          : {snap['energy_cost_kmad'].std():.1f}")
    print(f"    Min / Max    : {snap['energy_cost_kmad'].min():.1f} / {snap['energy_cost_kmad'].max():.1f}")
    print()
    print("  Buildings by region (2024)")
    for r, cnt in snap["region"].value_counts().items():
        print(f"    {r:<30}: {cnt}")
    print()
    print("  Buildings by type (2024)")
    for t, cnt in snap["building_type"].value_counts().items():
        print(f"    {t:<20}: {cnt}")
    print()
    print("  Regulatory status (2024)")
    for s, cnt in snap["regulatory_status"].value_counts().items():
        pct = cnt / len(snap) * 100
        print(f"    {s:<16}: {cnt}  ({pct:.0f}%)")
    print()
    print("  Risk score distribution (2024)")
    bins = [0, 0.25, 0.50, 0.75, 1.0]
    labels = ["Low (<0.25)", "Medium (0.25–0.50)", "High (0.50–0.75)", "Critical (>0.75)"]
    cuts = pd.cut(snap["risk_score"], bins=bins, labels=labels)
    for label, cnt in cuts.value_counts().sort_index().items():
        print(f"    {label:<22}: {cnt}")
    print()
    print("  Tenure split (2024)")
    for ten, cnt in snap["tenure"].value_counts().items():
        print(f"    {ten:<10}: {cnt}")
    print("=" * 56)
    print("  Files saved:")
    print("    mef_building_stock.csv   — full 20-year panel")
    print("    mef_snapshot_2024.csv    — latest-year slice")
    print("=" * 56)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating MEF building stock dataset...")
    df = generate_building_stock()
    validate(df)
    print("\nDone. Ready for Task 2 — individual forecasting models.")
