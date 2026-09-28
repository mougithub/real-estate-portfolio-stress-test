# UC2 — SDI Stress-Test Engine
### Dynamic and Sustainable Optimisation of Real Estate Assets
**Morocco MEF · Real Estate Master Plan (SDI) · 15-Year Horizon**

---

## How this started

This project grew out of a guest lecture in Professor Macosko's "Finance
through Physics" course, where Daric's CTO walked us through the kinds of
real-world problems his team solves with real estate data in a fintech
context. It was one of the most concrete, practical sessions I'd seen — it
wasn't abstract theory, it was "here's the actual decision a portfolio
manager has to make and here's how data informs it."

That session became the seed for the UC-2 class assignment, and I kept
going well past the assignment's scope because I wanted to build the full
decision pipeline: not just forecast a number, but simulate uncertainty
around it and turn that into a ranked recommendation the way a real
analyst would have to. I later reached out to Daric's CTO directly to
share what I'd built and get his perspective on it.

This repo is the result — a synthetic-data case study, but an end-to-end
one: data generation → forecasting → scenario simulation → decision
scoring → dashboard.

---

## What this project does

This system helps the Morocco Ministry of Economy and Finance (MEF)
make long-term real estate decisions by simulating how the portfolio
will evolve under four future scenarios — Optimistic, Austerity,
Climate shock, and Digitization — and ranking five strategies (Build,
Rent, Rehabilitate, Pool, Dispose) under each one.

It is not just a forecasting tool. It is a **decision system**: it
takes uncertainty seriously (Monte Carlo), captures how variables
influence each other (VAR), and translates model outputs into
actionable strategy scores; the top-ranked strategy (Pool) is robust
across the modeled scenarios, while the runner-up gap narrows or
closes under specific stress conditions (see sensitivity analysis).

---

## Quick start

```bash
# 1. Clone / copy all files into one folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (Tasks 1–6)
python task1_data_generation.py
python task2_occupancy_forecast.py
python task3_workforce_forecast.py
python task4_energy_forecast.py
python task5_scenario_engine.py
python task6_plotly_charts.py
python task8_sensitivity.py    # optional: sensitivity analysis on strategy scores

# 4. Launch the dashboard
streamlit run app.py
```

Each script reads the previous task's outputs from `data/` and writes its
own outputs back into `data/`. Run them in order the first time; after
that you can re-run any single task as long as its inputs already exist.

---

## Project structure

```
.
├── task1_data_generation.py      # Synthetic MEF building stock (Task 1)
├── task2_occupancy_forecast.py   # SARIMA occupancy model (Task 2)
├── task3_workforce_forecast.py   # Cohort + regression workforce model (Task 3)
├── task4_energy_forecast.py      # OLS energy cost model (Task 4)
├── task5_scenario_engine.py      # VAR + Monte Carlo engine (Task 5)
├── task6_plotly_charts.py        # Standalone interactive HTML charts (Task 6)
├── task8_sensitivity.py          # Sensitivity analysis on strategy scores (Task 8)
├── app.py                        # Streamlit dashboard (Task 7)
│
├── data/                         # generated at runtime by tasks 1–5
│   ├── mef_building_stock.csv        # 57 buildings × 20 years (Task 1 output)
│   ├── mef_snapshot_2024.csv         # Latest-year slice for dashboard cards
│   ├── occupancy_forecasts.csv       # SARIMA forecasts per building type
│   ├── workforce_forecasts.csv       # Cohort forecasts per region
│   ├── energy_forecasts.csv          # OLS forecasts per building
│   ├── energy_portfolio_forecast.csv # Portfolio-level energy totals
│   ├── scenario_trajectories.csv     # Monte Carlo p10/p25/mean/p75/p90
│   └── strategy_rankings.csv         # Composite scores per scenario
│
├── model_meta/                   # generated at runtime by tasks 2–5
│   ├── occupancy_model_meta.json     # ARIMA orders, AIC, MAE
│   ├── workforce_model_meta.json     # Cohort params, regression R²
│   ├── energy_model_meta.json        # OLS coefficients, VIF, R²
│   ├── engine_meta.json              # VAR config, scenario shock params
│   └── sensitivity_analysis.json     # Task 8 output
│
└── charts/                       # standalone interactive HTML charts (Task 6)
    ├── chart1_occupancy_trajectories.html
    ├── chart2_workforce_trajectories.html
    ├── chart3_energy_trajectories.html
    ├── chart4_strategy_rankings.html
    └── chart5_dimension_breakdown.html
```

---

## Architecture — 5-layer pipeline

```
Layer 1  Data inputs
         Building stock · workforce headcount · energy · regulatory status
         → mef_building_stock.csv  (57 buildings × 20 years)

Layer 2  Individual forecasting models
         Occupancy    → statsmodels SARIMA per building type
         Workforce    → numpy cohort matrix + sklearn regression per region
         Energy cost  → statsmodels OLS on building panel

Layer 3  VAR combination + Monte Carlo
         VAR(p) captures cross-variable dependencies
         scipy.stats shock distributions applied per scenario
         1,000 runs → p10 / p25 / mean / p75 / p90 trajectories

Layer 4  Scenario outputs (4 scenarios × 3 variables × 15 years)
         Optimistic    : workforce +20%
         Austerity     : budget −10%
         Climate shock : energy +40%
         Digitization  : space need −30%

Layer 5  Ranked strategy recommendations
         Build · Rent · Rehabilitate · Pool · Dispose
         Weighted composite score — tunable via dashboard sliders
```

---

## Library map

| Library | Used for |
|---|---|
| `pandas` + `numpy` | Data wrangling, cohort matrix ops, shock multipliers |
| `statsmodels` | SARIMA (occupancy), OLS (energy), VAR (combination) |
| `scipy.stats` | Monte Carlo shock distributions, percentile bands |
| `scikit-learn` | Workforce regression, strategy scoring, MinMaxScaler |
| `plotly` | Interactive trajectory fan charts, ranking bar charts |
| `streamlit` | Dashboard UI, sliders, caching |
| `pmdarima` | auto_arima — finds SARIMA (p,d,q) order automatically |

---

## Model details

### Task 2 — Occupancy (SARIMA)

ADF stationarity test per building type. auto_arima selects order.
Final orders: ministry_hq → ARIMA(1,0,0), regional_office and
annex → ARIMA(0,1,0). In-sample MAE: 0.039–0.055.

Ministry HQ holds stable at ~80% through 2039.
Annexes decline from 40% to 32% — strongest "Dispose" signal.

### Task 3 — Workforce (Cohort + OLS)

5 age bands (22–30, 30–38, 38–46, 46–54, 54–62).
Leslie transition matrix per region with Morocco HCP-informed
retirement rates (3.8–4.8%) and hiring ratios (0.92–1.05).
Decentralisation boost applied from 2018 onward for secondary cities.

OLS regression (sklearn) adds budget index and year trend as
calibration signals. In-sample R²: 0.57–0.98.

Total MEF workforce: 10,887 (2024) → 10,191 (2039) = −6.4%.
All five regions decline. Digitization scenario amplifies this to
−7.6% before space-demand multiplier is applied.

### Task 4 — Energy cost (OLS panel)

9 features: log_surface, building_age, age×arid interaction,
occupancy_rate, ONEE price index, climate zone dummies,
building type dummies.

R² = 0.797 on 1,140 observations. All VIF < 10.
Significant drivers: log_surface (coef=5.18), ONEE index (1.93),
occupancy_rate (7.16), is_arid (+1.27).

Portfolio energy cost: 0.288 MAD M/yr (2024) → 0.54 (2039, baseline).
Climate shock scenario: 0.501 MAD M/yr by 2039.

### Task 5 — VAR + Monte Carlo

VAR(3) fitted on differenced occupancy and workforce,
stationary energy_cost. Stable (all eigenvalues inside unit circle).
AIC = −12.40. Log-likelihood = 61.09.

Monte Carlo: 1,000 runs per scenario. Each run draws shock
multipliers from scipy.stats.norm distributions, applies them
with a 5-year linear ramp to the VAR forecast.

Shock distributions:
  Optimistic    : occ N(1.08,0.03), wf N(1.20,0.05), nrj N(1.05,0.04)
  Austerity     : occ N(0.93,0.03), wf N(0.98,0.03), nrj N(0.90,0.04)
  Climate shock : occ N(1.00,0.03), wf N(1.01,0.02), nrj N(1.40,0.08)
  Digitization  : occ N(0.70,0.06), wf N(0.95,0.04), nrj N(0.85,0.05)

### Task 6 — Strategy scoring

Weighted composite:
  composite = w_cost × cost_efficiency
            + w_sust × sustainability
            + w_space × space_flexibility

Default weights: 40% / 30% / 30%.
Scenario outcomes modulate base scores before weighting:
  high energy pressure → sustainability scores fall
  low occupancy       → space flexibility scores rise
  workforce decline   → cost efficiency scores rise

Base ranking: Pool (#1) · Dispose (#2) · Rent (#3) ·
              Rehabilitate (#4) · Build (#5)

**Sensitivity note:** Pool vs Dispose gap = 0.014 — FRAGILE.
A sustainability weight above ~55% flips Rehabilitate to #1.
The dashboard sliders surface this tradeoff intentionally.

**Scenario robustness finding:** the strategy scores are robust to
scenario pressure by design — the base profile gaps between strategies
are larger than any plausible shock in most scenarios, so Pool holds
#1 under Optimistic, Austerity, and Digitization. The exception is
energy-cost shocks specifically: under Climate shock, elevated energy
pressure closes the Pool/Dispose gap (Dispose removes floor area
outright rather than continuing to run shared space, so it is the one
strategy whose sustainability position improves rather than degrades
under this pressure). This was a deliberate choice — report the finding
accurately rather than force the model to flip rankings on every
scenario just to look more dynamic.

---

## Known limitations and next steps

**Small dataset.** 20 years of annual data is at the edge of what
VAR needs. Granger causality tests are underpowered. With real MEF
data (30+ years, or quarterly frequency), significance would improve
substantially.

**Synthetic data anchors.** Building stock parameters are calibrated
to ANME and HCP benchmarks but are not real MEF records. Replacing
Task 1 with real data requires only changing the CSV path — the
entire pipeline downstream is unchanged.

**Strategy base profiles are assumed.** The 3×5 STRATEGY_BASE matrix
reflects domain knowledge but has not been validated against real MEF
decision outcomes. A calibration step with MEF planners would
strengthen the scoring function.

**Pool vs Dispose gap is small (0.014).** Rankings are sensitive to
±0.10 perturbations in base scores (52.8% of random perturbations
change the ranking). Present the top two strategies as co-recommended
rather than a strict #1/#2 in the final report.

**Next steps for production:**
1. Replace synthetic data with real MEF building registry
2. Connect ONEE API for live energy price index
3. Add a Pareto frontier view (cost vs sustainability tradeoff)
4. Add tipping-point detection (energy price at which Rehabilitate
   beats Pool)
5. Deploy to Streamlit Community Cloud for ministry access

---

## Authors and context

Built as a Python coding project demonstrating time series
forecasting and scenario simulation for UC2 of the Morocco MEF
DATALAB initiative — Dynamic and Sustainable Optimisation of
Real Estate Assets (SDI).

Stack: Python 3.10+ · pandas · numpy · statsmodels · scipy ·
scikit-learn · plotly · streamlit

---

## License

MIT — see [LICENSE](LICENSE).
