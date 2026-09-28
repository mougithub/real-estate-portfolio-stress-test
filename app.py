"""
UC2 – SDI Stress-Test Engine
Task 7: Streamlit Dashboard

Run with:
    streamlit run app.py

Requires all output files from Tasks 1–6 in the same directory,
or set DATA_DIR to the correct path.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="UC2 – SDI Stress-Test Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #F1EFE8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 4px;
    }
    .metric-label { font-size: 12px; color: #5F5E5A; margin: 0 0 4px 0; }
    .metric-value { font-size: 26px; font-weight: 500; margin: 0; color: #2C2C2A; }
    .metric-delta { font-size: 12px; margin: 3px 0 0 0; }
    .section-label {
        font-size: 11px; font-weight: 500; color: #888780;
        text-transform: uppercase; letter-spacing: .06em;
        margin: 0 0 6px 0;
    }
    .rank-chip {
        display: inline-block; font-size: 12px; font-weight: 500;
        padding: 3px 10px; border-radius: 20px; margin-right: 6px;
    }
    [data-testid="stSidebar"] { background: #F8F7F3; }
    .stPlotlyChart { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
SCENARIO_STYLE = {
    "optimistic":    {"color": "#1D9E75", "label": "Optimistic",    "long": "Workforce +20%"},
    "austerity":     {"color": "#BA7517", "label": "Austerity",     "long": "Budget −10%"},
    "climate_shock": {"color": "#D85A30", "label": "Climate shock", "long": "Energy +40%"},
    "digitization":  {"color": "#378ADD", "label": "Digitization",  "long": "Space need −30%"},
}

CHIP_CSS = {
    "optimistic":    "background:#E1F5EE;color:#085041",
    "austerity":     "background:#FAEEDA;color:#633806",
    "climate_shock": "background:#FAECE7;color:#712B13",
    "digitization":  "background:#E6F1FB;color:#0C447C",
}

RISK_CSS = {
    "Low":      "color:#1D9E75",
    "Medium":   "color:#BA7517",
    "High":     "color:#D85A30",
    "Critical": "color:#A32D2D",
}

STRATEGY_COLORS = {
    "Pool":         "#534AB7",
    "Dispose":      "#D85A30",
    "Rent":         "#378ADD",
    "Rehabilitate": "#1D9E75",
    "Build":        "#888780",
}


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_trajectories():
    return pd.read_csv(DATA_DIR / "scenario_trajectories.csv")

@st.cache_data
def load_rankings():
    return pd.read_csv(DATA_DIR / "strategy_rankings.csv")

@st.cache_data
def load_snapshot():
    return pd.read_csv(DATA_DIR / "mef_snapshot_2024.csv")

@st.cache_data
def load_workforce():
    return pd.read_csv(DATA_DIR / "workforce_forecasts.csv")

@st.cache_data
def load_engine_meta():
    with open(Path(__file__).parent / "model_meta" / "engine_meta.json") as f:
        return json.load(f)


# ── Strategy re-scoring (responds to weight sliders) ─────────────────────────

def rescore_strategies(
    df_rank: pd.DataFrame,
    scenario: str,
    w_cost: float,
    w_sust: float,
    w_space: float,
) -> pd.DataFrame:
    sc = df_rank[df_rank["scenario"] == scenario].copy()
    total = w_cost + w_sust + w_space
    if total == 0:
        total = 1.0
    sc["composite"] = (
        sc["cost_efficiency"]   * (w_cost  / total) +
        sc["sustainability"]    * (w_sust  / total) +
        sc["space_flexibility"] * (w_space / total)
    ).round(3)
    sc["rank"] = sc["composite"].rank(ascending=False).astype(int)
    return sc.sort_values("composite", ascending=False)


# ── Chart builders ────────────────────────────────────────────────────────────

def hex_rgba(hex_c: str, alpha: float) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def trajectory_chart(df: pd.DataFrame, variable: str, scenario: str) -> go.Figure:
    VAR_LABELS = {
        "occupancy":   "Occupancy rate",
        "workforce":   "Headcount",
        "energy_cost": "MAD M / year",
    }
    VAR_FMT = {
        "occupancy":   ".3f",
        "workforce":   ",.0f",
        "energy_cost": ".3f",
    }
    fmt   = VAR_FMT[variable]
    ytitle = VAR_LABELS[variable]
    traces = []

    # Historical
    hist = df[(df["scenario"] == "historical") & (df["variable"] == variable)]
    traces.append(go.Scatter(
        x=hist["year"], y=hist["mean"],
        mode="lines", name="Historical",
        line=dict(color="#888780", width=1.8),
        hovertemplate=f"<b>Historical</b><br>Year: %{{x}}<br>Value: %{{y:{fmt}}}<extra></extra>",
    ))

    # Chosen scenario fan
    sc_df = df[(df["scenario"] == scenario) & (df["variable"] == variable)].copy()
    color = SCENARIO_STYLE[scenario]["color"]
    label = SCENARIO_STYLE[scenario]["label"]
    yrs_f = sc_df["year"].tolist()
    yrs_b = yrs_f[::-1]

    traces.append(go.Scatter(
        x=yrs_f + yrs_b,
        y=sc_df["p90"].tolist() + sc_df["p10"].tolist()[::-1],
        fill="toself", fillcolor=hex_rgba(color, 0.08),
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI", showlegend=True, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=yrs_f + yrs_b,
        y=sc_df["p75"].tolist() + sc_df["p25"].tolist()[::-1],
        fill="toself", fillcolor=hex_rgba(color, 0.20),
        line=dict(color="rgba(0,0,0,0)"),
        name="50% CI", showlegend=True, hoverinfo="skip",
    ))
    traces.append(go.Scatter(
        x=sc_df["year"], y=sc_df["mean"],
        mode="lines", name=f"{label} — mean",
        line=dict(color=color, width=2.4),
        hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Mean: %{{y:{fmt}}}<extra></extra>",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="#EDEBE3", dtick=5, zeroline=False),
        yaxis=dict(title=ytitle, showgrid=True, gridcolor="#EDEBE3", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(l=50, r=20, t=30, b=80),
        shapes=[dict(type="line", x0=2024, x1=2024, y0=0, y1=1, yref="paper",
                     line=dict(color="#B4B2A9", width=1, dash="dot"))],
    )
    return fig


def rankings_chart(df_scored: pd.DataFrame) -> go.Figure:
    order = df_scored.sort_values("composite", ascending=True)["strategy"].tolist()
    colors = [STRATEGY_COLORS.get(s, "#888780") for s in order]
    scores = [float(df_scored.set_index("strategy").loc[s, "composite"]) for s in order]
    ranks  = [int(df_scored.set_index("strategy").loc[s, "rank"]) for s in order]

    fig = go.Figure(go.Bar(
        x=scores, y=order,
        orientation="h",
        marker=dict(color=colors, opacity=0.88),
        text=[f"#{r}  {s:.3f}" for r, s in zip(ranks, scores)],
        textposition="inside",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#EDEBE3",
                   zeroline=False, title="Composite score"),
        yaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=110, r=20, t=20, b=40),
    )
    return fig


def dimension_chart(df_scored: pd.DataFrame) -> go.Figure:
    order  = df_scored.sort_values("composite", ascending=True)["strategy"].tolist()
    dims   = ["cost_efficiency", "sustainability", "space_flexibility"]
    dlabel = {"cost_efficiency": "Cost efficiency",
               "sustainability": "Sustainability",
               "space_flexibility": "Space flexibility"}
    dcol   = {"cost_efficiency": "#534AB7",
               "sustainability": "#1D9E75",
               "space_flexibility": "#378ADD"}
    traces = []
    for dim in dims:
        vals = [float(df_scored.set_index("strategy").loc[s, dim]) for s in order]
        traces.append(go.Bar(
            name=dlabel[dim], x=vals, y=order,
            orientation="h", marker=dict(color=dcol[dim], opacity=0.85),
            hovertemplate=f"<b>%{{y}}</b><br>{dlabel[dim]}: %{{x:.3f}}<extra></extra>",
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 2.2], showgrid=True, gridcolor="#EDEBE3",
                   zeroline=False, title="Score contribution"),
        yaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        margin=dict(l=110, r=20, t=20, b=70),
    )
    return fig


def workforce_regional_chart(
    df_wf: pd.DataFrame,
    traj: pd.DataFrame,
    scenario: str,
    scenario_label: str,
) -> go.Figure:
    """Bar chart: 2024 actual vs 2039 scenario-adjusted headcount per region."""
    hist = df_wf[df_wf["year"] == 2024].groupby("region")["headcount"].sum()
    fc = df_wf[df_wf["year"] == 2039].groupby("region")["headcount"].sum()
    regions = hist.index.tolist()

    baseline_total = float(fc.sum()) if len(fc) else 0.0
    scenario_total_row = traj[
        (traj["scenario"] == scenario) &
        (traj["variable"] == "workforce") &
        (traj["year"] == 2039)
    ]
    scenario_total = (
        float(scenario_total_row["mean"].iloc[0])
        if not scenario_total_row.empty else baseline_total
    )
    scale = (scenario_total / baseline_total) if baseline_total else 1.0
    fc_adjusted = fc.reindex(regions).fillna(0) * scale

    fig = go.Figure([
        go.Bar(name="2024 (actual)", x=regions,
               y=hist.values, marker_color="#D3D1C7",
               hovertemplate="<b>%{x}</b><br>2024: %{y:,.0f}<extra></extra>"),
        go.Bar(name=f"2039 ({scenario_label})", x=regions,
               y=fc_adjusted.values,
               marker_color="#534AB7", opacity=0.85,
               hovertemplate=(
                   f"<b>%{{x}}</b><br>2039 ({scenario_label}): %{{y:,.0f}}"
                   "<extra></extra>"
               )),
    ])
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-20),
        yaxis=dict(title="Headcount", showgrid=True, gridcolor="#EDEBE3"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=11)),
        margin=dict(l=50, r=20, t=30, b=90),
    )
    return fig


# ── KPI helpers ───────────────────────────────────────────────────────────────

def get_kpis(traj: pd.DataFrame, scenario: str) -> dict:
    def val(variable, year):
        r = traj[(traj["scenario"] == scenario) &
                 (traj["variable"] == variable) &
                 (traj["year"] == year)]
        return float(r["mean"].iloc[0]) if not r.empty else None

    def hist(variable, year):
        r = traj[(traj["scenario"] == "historical") &
                 (traj["variable"] == variable) &
                 (traj["year"] == year)]
        return float(r["mean"].iloc[0]) if not r.empty else None

    occ_now   = hist("occupancy",   2024) or 0.591
    occ_2039  = val("occupancy",    2039) or 0
    wf_now    = hist("workforce",   2024) or 10887
    wf_2039   = val("workforce",    2039) or 0
    nrj_now   = hist("energy_cost", 2024) or 0.288
    nrj_2039  = val("energy_cost",  2039) or 0

    return {
        "occ_now":   occ_now,
        "occ_2039":  occ_2039,
        "occ_delta": occ_2039 - occ_now,
        "wf_now":    wf_now,
        "wf_2039":   wf_2039,
        "wf_delta":  (wf_2039 - wf_now) / wf_now * 100,
        "nrj_now":   nrj_now,
        "nrj_2039":  nrj_2039,
        "nrj_delta": (nrj_2039 - nrj_now) / nrj_now * 100,
    }


def delta_color(val: float, good_direction: str = "up") -> str:
    if good_direction == "up":
        return "color:#1D9E75" if val >= 0 else "color:#D85A30"
    else:
        return "color:#1D9E75" if val <= 0 else "color:#D85A30"


def risk_label(score: float) -> str:
    if score < 0.25:   return "Low"
    if score < 0.50:   return "Medium"
    if score < 0.75:   return "High"
    return "Critical"


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # Load data
    traj    = load_trajectories()
    rank    = load_rankings()
    snap    = load_snapshot()
    wf_full = load_workforce()
    meta    = load_engine_meta()

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## SDI Stress-Test")
        st.markdown("UC2 — Morocco MEF Real Estate")
        st.divider()

        st.markdown('<p class="section-label">Active scenario</p>',
                    unsafe_allow_html=True)
        scenario = st.radio(
            label="scenario",
            options=list(SCENARIO_STYLE.keys()),
            format_func=lambda s: (
                f"{SCENARIO_STYLE[s]['label']} — {SCENARIO_STYLE[s]['long']}"
            ),
            label_visibility="collapsed",
        )
        sc_color = SCENARIO_STYLE[scenario]["color"]

        st.divider()
        st.markdown('<p class="section-label">Policy weights</p>',
                    unsafe_allow_html=True)
        w_cost  = st.slider("Cost efficiency",    0, 100, 40, step=5)
        w_sust  = st.slider("Sustainability",     0, 100, 30, step=5)
        w_space = st.slider("Space flexibility",  0, 100, 30, step=5)
        total_w = w_cost + w_sust + w_space
        if total_w == 0:
            st.warning("Set at least one weight above 0.")

        st.divider()
        st.markdown('<p class="section-label">Chart variable</p>',
                    unsafe_allow_html=True)
        variable = st.selectbox(
            label="variable",
            options=["occupancy", "workforce", "energy_cost"],
            format_func=lambda v: {
                "occupancy":   "Occupancy rate",
                "workforce":   "Workforce demand",
                "energy_cost": "Energy cost (MAD M)",
            }[v],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown('<p class="section-label">Model info</p>',
                    unsafe_allow_html=True)
        st.caption(
            f"VAR({meta['var_lag']})  ·  "
            f"AIC {meta['var_aic']:.2f}"
        )
        st.caption(
            f"{meta['n_simulations']:,} Monte Carlo runs  ·  "
            f"{meta['horizon_years']}-year horizon"
        )

    # ── Header ────────────────────────────────────────────────────
    sc_style = SCENARIO_STYLE[scenario]
    chip_css = CHIP_CSS[scenario]
    st.markdown(
        f"## SDI Stress-Test Engine &nbsp;"
        f'<span class="rank-chip" style="{chip_css}">'
        f'{sc_style["label"]} — {sc_style["long"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption("UC2 — Dynamic and sustainable optimisation of Morocco MEF real estate assets")

    # ── KPI row ───────────────────────────────────────────────────
    kpis = get_kpis(traj, scenario)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        d_col = delta_color(kpis["occ_delta"], "up")
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-label">Avg occupancy rate — 2039</p>
          <p class="metric-value">{kpis['occ_2039']:.1%}</p>
          <p class="metric-delta" style="{d_col}">
            {kpis['occ_delta']:+.1%} vs 2024 baseline
          </p>
        </div>""", unsafe_allow_html=True)

    with k2:
        d_col = delta_color(kpis["wf_delta"], "up")
        arrow = "▲" if kpis["wf_delta"] >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-label">Total workforce — 2039</p>
          <p class="metric-value">{kpis['wf_2039']:,.0f}</p>
          <p class="metric-delta" style="{d_col}">
            {arrow} {abs(kpis['wf_delta']):.1f}% from {kpis['wf_now']:,.0f}
          </p>
        </div>""", unsafe_allow_html=True)

    with k3:
        d_col = delta_color(kpis["nrj_delta"], "down")
        arrow = "▲" if kpis["nrj_delta"] >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-label">Energy cost — 2039 (MAD M)</p>
          <p class="metric-value">{kpis['nrj_2039']:.3f}</p>
          <p class="metric-delta" style="{d_col}">
            {arrow} {abs(kpis['nrj_delta']):.1f}% from {kpis['nrj_now']:.3f}
          </p>
        </div>""", unsafe_allow_html=True)

    with k4:
        n_risk = int((snap["risk_score"] > 0.50).sum())
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-label">High-risk assets (2024)</p>
          <p class="metric-value">{n_risk}</p>
          <p class="metric-delta" style="color:#D85A30">
            risk score &gt; 0.50 — need decision by 2027
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main content: trajectory + rankings ───────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<p class="section-label">Scenario trajectory</p>',
                    unsafe_allow_html=True)
        fig_traj = trajectory_chart(traj, variable, scenario)
        st.plotly_chart(fig_traj, use_container_width=True, key="traj")

    with col_right:
        st.markdown('<p class="section-label">Strategy ranking</p>',
                    unsafe_allow_html=True)
        df_scored = rescore_strategies(rank, scenario, w_cost, w_sust, w_space)
        fig_rank  = rankings_chart(df_scored)
        st.plotly_chart(fig_rank, use_container_width=True, key="rank")

    # ── Score dimensions + workforce regional ─────────────────────
    col_dim, col_wf = st.columns([2, 3], gap="large")

    with col_dim:
        st.markdown('<p class="section-label">Score breakdown by dimension</p>',
                    unsafe_allow_html=True)
        fig_dim = dimension_chart(df_scored)
        st.plotly_chart(fig_dim, use_container_width=True, key="dim")

    with col_wf:
        st.markdown('<p class="section-label">Workforce by region — 2024 vs 2039</p>',
                    unsafe_allow_html=True)
        fig_wf = workforce_regional_chart(wf_full, traj, scenario, sc_style["label"])
        st.plotly_chart(fig_wf, use_container_width=True, key="wf")
        st.caption(
            "2039 regional bars are scaled from the baseline regional forecast "
            "to match the selected scenario's portfolio-level workforce total."
        )

    # ── Asset-level cards ─────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-label">Asset-level recommendations (2024 snapshot)</p>',
                unsafe_allow_html=True)

    # Determine recommendation per building based on risk score + occupancy
    top_row  = df_scored.iloc[0]["strategy"]   # top-ranked strategy
    sec_row  = df_scored.iloc[1]["strategy"]   # second-ranked

    def recommend(row):
        if row["risk_score"] > 0.65:    return "Dispose"
        if row["occupancy_rate"] < 0.35: return sec_row
        if row["building_age"]   > 50:   return "Rehabilitate"
        if row["occupancy_rate"] < 0.55: return top_row
        return "Rehabilitate"

    snap["recommendation"] = snap.apply(recommend, axis=1)

    # Filter controls
    fc1, fc2, fc3 = st.columns([2, 2, 2])
    with fc1:
        region_filter = st.selectbox(
            "Region", ["All regions"] + sorted(snap["region"].unique().tolist()),
            label_visibility="visible",
        )
    with fc2:
        type_filter = st.selectbox(
            "Building type", ["All types"] + sorted(snap["building_type"].unique().tolist()),
            label_visibility="visible",
        )
    with fc3:
        risk_filter = st.selectbox(
            "Min risk level", ["All", "Medium+", "High+", "Critical"],
            label_visibility="visible",
        )

    filtered = snap.copy()
    if region_filter != "All regions":
        filtered = filtered[filtered["region"] == region_filter]
    if type_filter != "All types":
        filtered = filtered[filtered["building_type"] == type_filter]
    if risk_filter == "Medium+":
        filtered = filtered[filtered["risk_score"] >= 0.25]
    elif risk_filter == "High+":
        filtered = filtered[filtered["risk_score"] >= 0.50]
    elif risk_filter == "Critical":
        filtered = filtered[filtered["risk_score"] >= 0.75]

    filtered = filtered.sort_values("risk_score", ascending=False)
    st.caption(f"Showing {len(filtered)} of {len(snap)} buildings")

    # Render cards in rows of 3
    for i in range(0, min(len(filtered), 9), 3):
        cols = st.columns(3, gap="small")
        for j, col in enumerate(cols):
            if i + j >= len(filtered):
                break
            row = filtered.iloc[i + j]
            rl  = risk_label(row["risk_score"])
            rc  = RISK_CSS.get(rl, "color:#444441")
            rec = row["recommendation"]
            rec_color = STRATEGY_COLORS.get(rec, "#888780")

            with col:
                st.markdown(f"""
                <div style="background:white;border:0.5px solid #D3D1C7;
                            border-radius:10px;padding:14px 16px;margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start;margin-bottom:10px">
                    <div>
                      <p style="font-weight:500;font-size:13px;margin:0 0 2px">
                        {row['building_id']}</p>
                      <p style="font-size:11px;color:#888780;margin:0">
                        {row['region']} · {row['building_type'].replace('_',' ')}</p>
                    </div>
                    <span style="font-size:11px;font-weight:500;padding:3px 10px;
                                 border-radius:20px;background:{rec_color}22;
                                 color:{rec_color};white-space:nowrap">
                      {rec}
                    </span>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">
                    <div>
                      <p style="font-size:10px;color:#888780;margin:0">Occupancy</p>
                      <p style="font-size:13px;font-weight:500;margin:0">
                        {row['occupancy_rate']:.0%}</p>
                    </div>
                    <div>
                      <p style="font-size:10px;color:#888780;margin:0">Age</p>
                      <p style="font-size:13px;font-weight:500;margin:0">
                        {int(row['building_age'])} yrs</p>
                    </div>
                    <div>
                      <p style="font-size:10px;color:#888780;margin:0">Risk</p>
                      <p style="font-size:13px;font-weight:500;margin:0;{rc}">
                        {rl}</p>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "UC2 SDI Stress-Test Engine · (Morocco MEF-like) · "
        f"VAR({meta['var_lag']}) + Monte Carlo ({meta['n_simulations']:,} runs) · "
        "Built with Python / statsmodels / scipy / Streamlit"
    )


if __name__ == "__main__":
    main()
