# UC2 Project Report

## Dynamic Real Estate Optimization Using Forecasting and Scenario Simulation

**Author:** Banasree Sarkar Mou  
**Context:** UC2 - Sustainable Real Estate Asset Optimization (Morocco case study)

### 1. Project Overview

This project was designed as a decision-support system for long-term real estate planning under uncertainty. The business objective was to help decision-makers evaluate how a public-sector real estate portfolio may evolve over a 15-year horizon and identify which asset strategies are most appropriate under different future conditions.

The system focuses on three practical planning questions:

- How will building occupancy evolve over time?
- How will workforce demand change across regions?
- How will energy costs develop under different operating conditions?

These forecasts are then translated into strategy recommendations across five actions: **Build, Rent, Rehabilitate, Pool,** and **Dispose**.

### 2. Problem Framing

The use case is motivated by a common portfolio-management challenge: large asset bases often suffer from fragmented visibility, uncertain future demand, rising operating costs, and difficult tradeoffs between keeping, sharing, upgrading, or disposing of assets. A single static forecast is not enough for these decisions because long-term planning depends on uncertainty in staffing, utilization, and costs.

The goal of this project was therefore not just to predict future values, but to build a pipeline that supports **forecasting, stress testing, and decision-making** in one end-to-end workflow.

### 3. Data Strategy

Because real institutional building records were not publicly available, the project used a synthetic time-series dataset calibrated to plausible Moroccan benchmarks. This made it possible to demonstrate the full technical pipeline while still grounding the assumptions in realistic external signals.

The synthetic data covered **57 buildings across 2005-2024** and included:

- building type, age, tenure, and surface area
- occupancy rate
- workforce headcount
- annual energy cost
- regulatory/compliance condition

Calibration assumptions were informed by:

- Morocco HCP-style demographic retirement patterns
- ANME-style building energy benchmarks
- ONEE electricity tariff growth trends

### 4. Methodology

The project was implemented as a multi-stage analytical pipeline:

**1. Synthetic portfolio generation**  
Created a building-level historical panel dataset with occupancy, workforce, energy cost, and asset risk indicators.

**2. Occupancy forecasting**  
Used SARIMA-style time-series forecasting to model occupancy trajectories by building type. Stationarity checks and model-order selection were used before generating 15-year forecasts and uncertainty bands.

**3. Workforce forecasting**  
Used a cohort-based model to simulate retirement and hiring dynamics by region, then calibrated the resulting forecasts with regression features such as budget trend and year effects.

**4. Energy-cost forecasting**  
Used OLS regression on the building panel to estimate how energy costs respond to factors such as building size, age, climate zone, occupancy, and electricity price index.

**5. Scenario simulation**  
Combined occupancy, workforce, and energy signals in a VAR-based scenario engine, then ran Monte Carlo stress tests across four planning scenarios:

- **Optimistic:** workforce growth and additional space demand
- **Austerity:** budget pressure and tighter asset consolidation
- **Climate shock:** energy-cost spike and retrofit pressure
- **Digitization:** reduced space demand due to remote or digital work

**6. Strategy scoring and ranking**  
Mapped scenario outcomes into a weighted decision model that ranks the five strategic actions according to cost efficiency, sustainability, and space flexibility.

**7. Interactive dashboard**  
Presented results in a Streamlit dashboard with scenario selection, ranking views, sensitivity controls, and asset-level recommendations.

### 5. Key Results

The project produced three main analytical outputs:

- long-range occupancy, workforce, and energy forecasts
- scenario-based stress-tested trajectories with uncertainty bands
- ranked strategy recommendations under each scenario

Key findings from the modeling included:

- workforce and occupancy pressures meaningfully change the preferred portfolio strategy
- climate and energy shocks increase the attractiveness of rehabilitation-oriented decisions
- digitization and lower space demand strengthen consolidation and disposal logic
- strategy rankings are sensitive to policy priorities, which makes explicit weighting important for decision-makers

Across the tested scenarios, the project consistently showed that portfolio strategy should be tied to **future operating conditions**, not just current asset status.

### 6. Why This Project Matters

This project is relevant beyond real estate because it demonstrates a transferable analytical pattern often used in risk, operations, and fintech environments:

- build a structured data pipeline
- forecast core business drivers
- simulate uncertainty rather than relying on a single estimate
- convert model outputs into decisions a stakeholder can use

That combination is especially valuable in settings where planning depends on uncertain future behavior, costs, and constraints.

### 7. Technical Takeaways

This work strengthened several applied data and ML skills:

- time-series forecasting and stationarity testing
- regression-based feature modeling
- Monte Carlo simulation for uncertainty analysis
- decision modeling with weighted ranking logic
- dashboard design for communicating model outputs clearly

It also reinforced an important practical lesson: when direct access to production-quality data is limited, a carefully designed synthetic dataset can still be used to prototype a realistic and end-to-end analytical system.

### 8. Conclusion

The final result is a compact decision-support platform for long-horizon asset planning. Rather than treating forecasting as an isolated modeling exercise, the project links data generation, forecasting, simulation, and strategy ranking into one coherent workflow.

In short, the project answers a practical question:

**Given uncertain future demand, cost, and operating conditions, which real-estate strategy is most appropriate, and how confident should a decision-maker be in that recommendation?**

That decision-oriented framing is the core value of the project.
