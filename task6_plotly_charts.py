"""
UC2 – SDI Stress-Test Engine
Task 6: Plotly Trajectory Charts + Strategy Scoring Visualisations

Produces four standalone Plotly charts saved as interactive HTML files:

  chart1_occupancy_trajectories.html   — occupancy rate fan chart, all scenarios
  chart2_workforce_trajectories.html   — workforce fan chart, all scenarios
  chart3_energy_trajectories.html      — energy cost fan chart, all scenarios
  chart4_strategy_rankings.html        — grouped bar chart of composite scores

Each chart is fully interactive: hover, zoom, pan, toggle scenarios.
All charts are self-contained HTML — open in any browser, no server needed.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

OUTPUT_DIR = "data"

# ── Colour palette — one per scenario ────────────────────────────────────────
SCENARIO_STYLE = {
    "optimistic":   {"color": "#1D9E75", "label": "Optimistic (workforce +20%)",  "dash": "solid"},
    "austerity":    {"color": "#BA7517", "label": "Austerity (budget −10%)",       "dash": "dot"},
    "climate_shock":{"color": "#D85A30", "label": "Climate shock (energy +40%)",   "dash": "dash"},
    "digitization": {"color": "#378ADD", "label": "Digitization (space −30%)",     "dash": "dashdot"},
}

FORECAST_SCENARIOS = list(SCENARIO_STYLE.keys())

# Variable display config
VAR_CONFIG = {
    "occupancy": {
        "title":  "Occupancy rate — portfolio mean",
        "ytitle": "Occupancy rate",
        "fmt":    ".3f",
        "pct":    True,
        "yrange": [0.0, 1.0],
    },
    "workforce": {
        "title":  "Total workforce demand — all regions",
        "ytitle": "Headcount",
        "fmt":    ",.0f",
        "pct":    False,
        "yrange": None,
    },
    "energy_cost": {
        "title":  "Portfolio energy cost — MAD millions/year",
        "ytitle": "MAD M / year",
        "fmt":    ".3f",
        "pct":    False,
        "yrange": [0, None],
    },
}


# ── Shared layout template ────────────────────────────────────────────────────

def base_layout(title: str, ytitle: str, yrange=None) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#2C2C2A"), x=0.02),
        xaxis=dict(
            title="Year",
            showgrid=True,
            gridcolor="#E8E6DE",
            gridwidth=0.5,
            tickmode="linear",
            dtick=5,
            zeroline=False,
        ),
        yaxis=dict(
            title=ytitle,
            showgrid=True,
            gridcolor="#E8E6DE",
            gridwidth=0.5,
            range=yrange,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D3D1C7",
            borderwidth=0.5,
        ),
        margin=dict(l=60, r=30, t=60, b=120),
        font=dict(family="Arial, sans-serif", size=12, color="#444441"),
        shapes=[
            # Vertical line at 2024 — historical / forecast boundary
            dict(
                type="line", x0=2024, x1=2024,
                y0=0, y1=1, yref="paper",
                line=dict(color="#888780", width=1, dash="dot"),
            )
        ],
        annotations=[
            dict(
                x=2024, y=1.02, yref="paper",
                text="← Historical | Forecast →",
                showarrow=False,
                font=dict(size=10, color="#888780"),
                xanchor="center",
            )
        ],
    )


# ── Chart 1–3: Trajectory fan charts ─────────────────────────────────────────

def make_trajectory_chart(
    df: pd.DataFrame,
    variable: str,
    output_path: str,
):
    """
    Fan chart for one variable across all scenarios.
    Layers per scenario (bottom to top):
      1. 80% CI shaded band  (p10–p90, very light)
      2. 50% CI shaded band  (p25–p75, medium)
      3. Mean line           (solid, labelled)
    Plus: historical line in gray.
    """
    cfg    = VAR_CONFIG[variable]
    traces = []

    # ── Historical series (gray, behind everything) ───────────────
    hist = df[(df["scenario"] == "historical") & (df["variable"] == variable)]
    if not hist.empty:
        traces.append(go.Scatter(
            x=hist["year"], y=hist["mean"],
            mode="lines",
            name="Historical",
            line=dict(color="#888780", width=1.8, dash="solid"),
            hovertemplate=f"<b>Historical</b><br>Year: %{{x}}<br>{variable}: %{{y:{cfg['fmt']}}}<extra></extra>",
        ))

    # ── Forecast traces per scenario ──────────────────────────────
    for sc in FORECAST_SCENARIOS:
        sc_df  = df[(df["scenario"] == sc) & (df["variable"] == variable)].copy()
        style  = SCENARIO_STYLE[sc]
        color  = style["color"]
        label  = style["label"]

        # Parse hex → rgba helper
        def hex_rgba(hex_c: str, alpha: float) -> str:
            h = hex_c.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        years_fwd  = sc_df["year"].tolist()
        years_back = years_fwd[::-1]

        # 80% band (p10–p90)
        traces.append(go.Scatter(
            x=years_fwd + years_back,
            y=sc_df["p90"].tolist() + sc_df["p10"].tolist()[::-1],
            fill="toself",
            fillcolor=hex_rgba(color, 0.08),
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} — 80% CI",
            showlegend=False,
            hoverinfo="skip",
        ))

        # 50% band (p25–p75)
        traces.append(go.Scatter(
            x=years_fwd + years_back,
            y=sc_df["p75"].tolist() + sc_df["p25"].tolist()[::-1],
            fill="toself",
            fillcolor=hex_rgba(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{label} — 50% CI",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Mean line
        traces.append(go.Scatter(
            x=sc_df["year"],
            y=sc_df["mean"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2.2, dash=style["dash"]),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Year: %{{x}}<br>"
                f"Mean: %{{y:{cfg['fmt']}}}<br>"
                f"<extra></extra>"
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**base_layout(cfg["title"], cfg["ytitle"], cfg.get("yrange")))

    # Add range selector buttons
    fig.update_xaxes(
        rangeslider=dict(visible=False),
    )

    # Watermark
    fig.add_annotation(
        text="UC2 SDI Stress-Test Engine",
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=9, color="#B4B2A9"),
        xanchor="right",
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {output_path.split('/')[-1]}")
    return fig


# ── Chart 4: Strategy ranking bar chart ──────────────────────────────────────

def make_strategy_chart(df_rank: pd.DataFrame, output_path: str):
    """
    Grouped horizontal bar chart.
    X = composite score, Y = strategy, grouped by scenario.
    Bars are sorted by mean composite score across scenarios.
    """
    # Sort strategies by mean score
    mean_score = df_rank.groupby("strategy")["composite"].mean()
    strategy_order = mean_score.sort_values(ascending=True).index.tolist()

    traces = []
    for sc in FORECAST_SCENARIOS:
        sc_df  = df_rank[df_rank["scenario"] == sc].set_index("strategy")
        style  = SCENARIO_STYLE[sc]
        scores = [float(sc_df.loc[s, "composite"]) if s in sc_df.index else 0
                  for s in strategy_order]
        ranks  = [int(sc_df.loc[s, "rank"]) if s in sc_df.index else 0
                  for s in strategy_order]

        traces.append(go.Bar(
            name=style["label"],
            x=scores,
            y=strategy_order,
            orientation="h",
            marker=dict(color=style["color"], opacity=0.85),
            customdata=ranks,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Scenario: {style['label']}<br>"
                "Score: %{x:.3f}<br>"
                "Rank: #%{customdata}<extra></extra>"
            ),
            text=[f"#{r}" for r in ranks],
            textposition="inside",
            textfont=dict(size=11, color="white"),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Strategy rankings — composite score per scenario",
            font=dict(size=15, color="#2C2C2A"),
            x=0.02,
        ),
        barmode="group",
        xaxis=dict(
            title="Composite score (0–1)",
            range=[0, 1.0],
            showgrid=True,
            gridcolor="#E8E6DE",
            gridwidth=0.5,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            autorange=True,
            tickfont=dict(size=13),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#D3D1C7",
            borderwidth=0.5,
        ),
        margin=dict(l=120, r=40, t=60, b=130),
        font=dict(family="Arial, sans-serif", size=12, color="#444441"),
        annotations=[
            dict(
                text="UC2 SDI Stress-Test Engine",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                showarrow=False,
                font=dict(size=9, color="#B4B2A9"),
                xanchor="right",
            )
        ],
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {output_path.split('/')[-1]}")
    return fig


# ── Chart 5: Dimension breakdown radar / bar per scenario ─────────────────────

def make_dimension_breakdown(df_rank: pd.DataFrame, output_path: str):
    """
    Stacked bar showing the three score dimensions per strategy × scenario.
    Helps decision-makers see *why* a strategy ranks where it does.
    """
    # Use austerity scenario as the illustrative case (most policy-relevant)
    dims   = ["cost_efficiency", "sustainability", "space_flexibility"]
    colors = {"cost_efficiency": "#534AB7",
               "sustainability": "#1D9E75",
               "space_flexibility": "#378ADD"}
    labels = {"cost_efficiency": "Cost efficiency",
               "sustainability": "Sustainability",
               "space_flexibility": "Space flexibility"}

    mean_score = df_rank.groupby("strategy")["composite"].mean()
    strategy_order = mean_score.sort_values(ascending=True).index.tolist()

    traces = []
    for dim in dims:
        # Average across scenarios for each dim
        dim_mean = (df_rank.groupby("strategy")[dim].mean()
                    .reindex(strategy_order))
        traces.append(go.Bar(
            name=labels[dim],
            x=dim_mean.values,
            y=strategy_order,
            orientation="h",
            marker=dict(color=colors[dim], opacity=0.88),
            hovertemplate=(
                f"<b>%{{y}}</b><br>"
                f"{labels[dim]}: %{{x:.3f}}<extra></extra>"
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Score dimensions — mean across all scenarios",
            font=dict(size=15, color="#2C2C2A"),
            x=0.02,
        ),
        barmode="stack",
        xaxis=dict(
            title="Contribution to composite score",
            range=[0, 2.2],
            showgrid=True,
            gridcolor="#E8E6DE",
            gridwidth=0.5,
            zeroline=False,
        ),
        yaxis=dict(title="", tickfont=dict(size=13)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        margin=dict(l=120, r=40, t=60, b=110),
        font=dict(family="Arial, sans-serif", size=12, color="#444441"),
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Saved: {output_path.split('/')[-1]}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def run_charts():
    print("=" * 58)
    print("  Task 6 — Plotly Trajectory Charts")
    print("=" * 58)

    df_traj = pd.read_csv(f"{OUTPUT_DIR}/scenario_trajectories.csv")
    df_rank = pd.read_csv(f"{OUTPUT_DIR}/strategy_rankings.csv")

    print(f"\n  Building charts from:")
    print(f"    {len(df_traj)} trajectory rows  "
          f"({df_traj['scenario'].nunique()} scenarios × "
          f"{df_traj['variable'].nunique()} variables × "
          f"{df_traj['year'].nunique()} years)")
    print(f"    {len(df_rank)} ranking rows  "
          f"({df_rank['scenario'].nunique()} scenarios × "
          f"{df_rank['strategy'].nunique()} strategies)\n")

    # Chart 1: Occupancy
    fig1 = make_trajectory_chart(
        df_traj, "occupancy",
        f"charts/chart1_occupancy_trajectories.html"
    )

    # Chart 2: Workforce
    fig2 = make_trajectory_chart(
        df_traj, "workforce",
        f"charts/chart2_workforce_trajectories.html"
    )

    # Chart 3: Energy cost
    fig3 = make_trajectory_chart(
        df_traj, "energy_cost",
        f"charts/chart3_energy_trajectories.html"
    )

    # Chart 4: Strategy rankings
    fig4 = make_strategy_chart(
        df_rank,
        f"charts/chart4_strategy_rankings.html"
    )

    # Chart 5: Dimension breakdown
    fig5 = make_dimension_breakdown(
        df_rank,
        f"charts/chart5_dimension_breakdown.html"
    )

    # ── Validation summary ────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Chart validation")
    print("=" * 58)

    for var in ["occupancy", "workforce", "energy_cost"]:
        cfg = VAR_CONFIG[var]
        sub = df_traj[df_traj["variable"] == var]
        hist_end = sub[(sub["scenario"] == "historical") &
                       (sub["year"] == 2024)]["mean"].values
        print(f"\n  {var}")
        print(f"    Historical 2024: {hist_end[0]:{cfg['fmt']}}")
        for sc in FORECAST_SCENARIOS:
            fc = sub[(sub["scenario"] == sc) & (sub["year"] == 2039)]
            if not fc.empty:
                m   = float(fc["mean"].iloc[0])
                lo  = float(fc["p10"].iloc[0])
                hi  = float(fc["p90"].iloc[0])
                print(f"    {sc:<14}  2039: {m:{cfg['fmt']}}  "
                      f"[{lo:{cfg['fmt']}} – {hi:{cfg['fmt']}}]")

    print("\n" + "=" * 58)
    print("  All 5 charts saved as standalone interactive HTML.")
    print("  Open any file in a browser — no server required.")
    print("=" * 58)

    return fig1, fig2, fig3, fig4, fig5


if __name__ == "__main__":
    figs = run_charts()
    print("\nDone. Ready for Task 8 — Streamlit dashboard.")
